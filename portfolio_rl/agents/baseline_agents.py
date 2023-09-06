from abc import abstractmethod
from typing import Optional
import pandas as pd
from numpy.typing import NDArray

from portfolio_rl.gym_env.gym_env import PortfolioEnv


class AbstractAgent:
    def __init__(self, env: PortfolioEnv):
        self.env = env

    def on_init(self, obs: pd.DataFrame, info: dict[str, pd.DataFrame]):
        """Implement this if you need to do something with the first observation given by `reset`"""
        pass

    def init(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        obs, info = self.env.reset()
        self.on_init(obs, info)
        return obs, info

    @abstractmethod
    def step(self, obs: pd.DataFrame, reward: float, terminated: bool, truncated: bool) -> NDArray:
        raise NotImplementedError('AbstractAgent needs to implement "step" method')

    def run(self) -> tuple[pd.DataFrame, dict[str, str]]:
        obs, _ = self.init()
        reward = 0
        terminated = truncated = False
        while not terminated and not truncated:
            action = self.step(obs, reward, terminated, truncated)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            print(self.env._cur_date, reward, terminated, truncated)

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
