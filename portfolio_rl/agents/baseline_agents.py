from abc import abstractmethod
from typing import Optional
import pandas as pd
from numpy.typing import NDArray

from portfolio_rl.gym_env.gym_env import PortfolioEnv
from portfolio_rl.utils import simple_normalize


class AbstractAgent:
    def __init__(self, env: PortfolioEnv):
        self.env = env

    def on_init(self, obs: pd.DataFrame, info: dict[str, pd.DataFrame]):
        """Implement this if you need to do something with the first observation given by `reset`"""
        pass

    def _init(self, **kwargs) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Calls reset on environment with `kwargs`"""
        obs, info = self.env.reset(**kwargs)
        self.on_init(obs, info)
        return obs, info

    @abstractmethod
    def step(self, obs: pd.DataFrame, reward: float) -> NDArray:
        """This should be overriden by child classes to define behaviour to be taken at each environment step"""
        raise NotImplementedError('AbstractAgent needs to implement "step" method')

    def run(self, **kwargs) -> tuple[pd.DataFrame, dict[str, str]]:
        """Runs the environment until termination, `kwargs` are passed to `env.reset`"""
        obs, _ = self._init(**kwargs)
        reward = 0
        terminated = truncated = False
        while not terminated and not truncated:
            action = self.step(obs, reward)
            obs, reward, terminated, truncated, info = self.env.step(action)
            print(info['dt'], reward, terminated, truncated)

        self.results = self.env.calc_results()
        self.res_df = self.env.historical_wallet_mtm()
        return self.res_df, self.results


class EqualWeightedAgent(AbstractAgent):
    def step(self, *args, **kwargs) -> NDArray:
        return self.env.action_space.sample()


class RandomWeightedAgent(AbstractAgent):
    def __init__(self, env: PortfolioEnv, seed: Optional[int] = None):
        super().__init__(env)
        self.set_seed(seed)

    def set_seed(self, seed: Optional[int] = None):
        self.env._set_seed(seed)

    def step(self, *args, **kwargs) -> NDArray:
        return self.env.action_space.sample(equal_weight=False)


class FixedRandomAgent(AbstractAgent):
    def __init__(self, env: PortfolioEnv, seed: Optional[int] = None):
        super().__init__(env)
        self.set_seed(seed)

    def set_seed(self, seed: Optional[int] = None):
        self.env._set_seed(seed)

    def on_init(self, obs: pd.DataFrame, info: dict[str, pd.DataFrame]):
        obs['weight'] = self.env.action_space.sample(equal_weight=False)
        self.weights = obs[['ticker', 'weight']]

    def step(self, obs: pd.DataFrame, reward: float) -> NDArray:
        weights = obs[['ticker']].join(self.weights.set_index('ticker'), on='ticker')
        weights['weight'] = weights['weight'].fillna(0)
        return weights['weight'].to_numpy()


class IndexEtfAgent(AbstractAgent):
    def step(self, obs: pd.DataFrame, reward: float) -> NDArray:
        return simple_normalize(obs['weight'].to_numpy())


class SingleStockAgent(AbstractAgent):
    def __init__(self, env: PortfolioEnv, ticker: str):
        super().__init__(env)
        self.ticker = ticker

    def step(self, obs: pd.DataFrame, reward: float) -> NDArray:
        aux = obs[['ticker', 'weight']]
        aux.loc[aux['ticker'] == self.ticker, 'weight'] = 1
        aux.loc[aux['ticker'] != self.ticker, 'weight'] = 0
        return aux['weight'].to_numpy()


class MovingAverageAgent(AbstractAgent):
    def __init__(self, env: PortfolioEnv, ma_days: int = 50):
        super().__init__(env)
        self.ma_days = ma_days

    def _calc_ma(self):
        self.df['MA'] = self.df.groupby('ticker')['close'].rolling(self.ma_days).mean()\
            .reset_index().drop('ticker', axis=1).set_index('level_1')['close']

    def on_init(self, obs: pd.DataFrame, info: dict[str, pd.DataFrame]):
        self.df = info['warmup']

    def step(self, obs: pd.DataFrame, reward: float) -> NDArray:
        self.df = pd.concat([self.df, obs]).reset_index(drop=True)
        self._calc_ma()
        df = self.df.loc[self.df['date'] == obs['date'].iloc[0], ['weight', 'close', 'MA']]
        df['signal'] = (df['close'] > df['MA']).map(lambda b: 2 if b else 0.5)
        df['weight'] *= df['signal']
        return simple_normalize(df['weight'])

    def run(self, **kwargs) -> tuple[pd.DataFrame, dict[str, str]]:
        if 'options' not in kwargs:
            kwargs['options'] = {}

        kwargs['options'].update({'warmup': True})
        return super().run(**kwargs)
