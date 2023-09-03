from datetime import date
from typing import Optional, Dict, Any, Tuple, SupportsFloat, Iterable

from gymnasium import Env
from gymnasium.spaces import Space, Box
from gymnasium.core import ObsType, ActType
from gymnasium.envs.registration import EnvSpec

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import normalize

DEFAULT_WALLET = 1e6


class NormBox(Box):
    """A gym box space that returns normalized samples (ie. rows sum to 1)"""
    def sample(self, equal_weight: bool = True) -> NDArray[Any]:
        """
        Returns an action space sample for the PortfolioEnv
        
        Args:
            equal_weight (bool): Whether to return an equal weighted action
                (ie, every stock will get the same weight) or a random weighted one
        """
        if not equal_weight:
            samp = super().sample()
            samp = normalize(samp, axis=1, norm='l1')
        else:
            samp = np.ones(self._shape) * (1 / self._shape[1])

        return samp[0]


class MarketObservation(Space):
    def __init__(self, df: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.dates = sorted(list(self.df['date'].unique()))

    def sample(self, random: bool = True) -> pd.DataFrame:
        if random:
            pos = int(np.floor(self.np_random.random() * len(self.dates)))
        else:
            pos = 0

        return self.df.loc[self.df['date'] == self.dates[pos]]


class PortfolioEnv(Env):
    action_space: Space[ActType]
    observation_space: Space[ObsType]
    reward_range: Tuple[float, float] = -1, np.inf
    spec: EnvSpec | None = None
    _last_date: date = None
    _cur_date: date = None
    _iterator: Iterable[date] = None

    def __init__(self,
        data_path: str, warmup_data_path: Optional[str] = None, wallet: float = DEFAULT_WALLET,
        warmup: bool = False, slippage: int = 10
    ):
        """
        Portfolio Env constructor

        Args:
            data_path (str): A valid parquet file path with the S&P 500 portfolio data to be used,
                should be in the shape `(N * 500, 11)`, where `N` is the number of days to be traded.
                Each day should have the top 500 stocks in the index that day.
            warmup_data_path (optional str): A parquet file path with warmup data to be given to the agent
                before trading starts, this is usually to calculate things like moving averages
            wallet (optional float): The size of the trading wallet, 1,000,000 by default
            warmup (optional bool): Whether to pass the agent warmup data on `reset` call or not.
                If `True`, then `warmup_data_path` must resolve to a valid parquet file
            slippage (optional int): how many bps of slippage to apply to each order (default 10 bps)
        """
        self.df = pd.read_parquet(data_path)
        self._validate_data(self.df)
        self.warmup = warmup
        self.warmup_data_path = warmup_data_path
        self._validate_warmup()
        if self.warmup_data_path is not None:
            self.warmup_df = pd.read_parquet(warmup_data_path)

        self.wallet = wallet
        self.slippage = slippage
        self.action_space = NormBox(0, 1, [1, 500], np.float32)
        self.observation_space = MarketObservation(self.df, shape=[500, 11])

    def _validate_data(self, df: pd.DataFrame):
        aux = df.groupby('date').agg({'ticker': 'count'})
        assert aux[aux['ticker'] != 500].empty, 'Every date of data should have exactly 500 rows (top 500 tickers)'

    def _validate_warmup(self):
        assert not self.warmup or self.warmup_data_path is not None, 'Need to specify warmup data path when using warmup = True'

    def __iter__(self):
        return iter(sorted(list(self.df['date'].unique())))

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self)

        self._last_date = self._cur_date
        self._cur_date = next(self._iterator)
        return self.df.loc[self.df['date'] == self._cur_date]

    def _next_obs(self) -> ObsType:
        return next(self)

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Dict[str, Any] = {}
    ) -> tuple[ObsType, Dict[str, Any]]:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict):
                - wallet (float): wallet size (default: 1,000,000)
                - warmup (bool): Whether to pass the agent warmup data on `reset` call or not (default: False)
                - slippage (int): bps of slippage to be applied to each order

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed)
        if 'wallet' in options:
            self.wallet = options['wallet']

        if 'warmup' in options:
            self.warmup = options['warmup']
            self._validate_warmup()

        if 'slippage' in options:
            self.slippage = options['slippage']

        self._iterator = None
        obs = self._next_obs()
        if self.warmup:
            obs = pd.concat([self.warmup_df, obs]).reset_index(drop=True)

        return obs, {}

    def _validate_action(self, action: ActType) -> NDArray[Any]:
        arr = np.array(action)
        assert arr.shape == (500,), f'action shape is {arr.shape}, should be (500,)'
        assert np.isclose(arr.sum(), 1), f'portfolio weights should sum to 1, not {arr.sum()}'
        return arr

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): A list of 500 floats with the weights attributed to each of the 500 tickers in the last observation
                the weights should add to 1 (ie sum(action) == 1) and be ordered by ticker alphabetically (ie the first weight
                will be attributed to the first ticker alphabetically)

        Returns:
            observation (ObsType): A dataframe in the shape (500, 11) with the 500 top stocks of the S&P 500 on the day and market data on each
            reward (SupportsFloat): The return of the portfolio on the day
            terminated (bool): whether the trading period is over (env is finished)
            truncated (bool): whether the agent wallet reached 0 (cumulative return over trading period is -100%)
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging)
        """
        obs = None
        terminated = False
        try:
            obs = self._next_obs()
        except StopIteration:
            terminated = True

        new_weights = self._validate_action(action)
        # TODO: handle action

        reward = 0      # portfolio daily return
        truncated = self.wallet <= 0
        return obs, reward, terminated, truncated, {}
