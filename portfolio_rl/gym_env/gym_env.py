from datetime import date
from typing import Optional, Dict, Any, Tuple, SupportsFloat, Iterable

from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.core import ObsType, ActType
from gymnasium.envs.registration import EnvSpec
import numpy as np
import pandas as pd

DEFAULT_WALLET = 1e6


class PortfolioEnv(Env):
    # TODO: make action_space and observation_space
    action_space: Space[ActType]
    observation_space: Space[ObsType]
    reward_range: Tuple[float, float] = -1, np.inf
    spec: EnvSpec | None = None
    _last_date: date = None
    _iterator: Iterable[date] = None

    def __init__(self, data_path: str, warmup_data_path: Optional[str] = None, wallet: float = DEFAULT_WALLET, warmup: bool = False):
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
        """
        self.df = pd.read_parquet(data_path)
        self._validate_data(self.df)
        self.warmup = warmup
        self.warmup_data_path = warmup_data_path
        self._validate_warmup()
        if self.warmup_data_path is not None:
            self.warmup_df = pd.read_parquet(warmup_data_path)

        self.wallet = wallet

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

        self._last_date = next(self._iterator)
        return self.df.loc[self.df['date'] == self._last_date]

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

        self._iterator = None
        obs = self._next_obs()
        if self.warmup:
            obs = pd.concat([self.warmup_df, obs]).reset_index(drop=True)

        return obs, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): A dataframe in the shape (500, 11) with the 500 top stocks of the S&P 500 on the day and market data on each
            reward (SupportsFloat): The return of the portfolio on the day
            terminated (bool): whether the trading period is over (env is finished)
            truncated (bool): whether the agent wallet reached 0 (cumulative return over trading period is -100%)
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging)
        """
        # TODO: handle action
        obs = None
        reward = 0 # portfolio daily return
        terminated = False
        truncated = self.wallet <= 0
        try:
            obs = self._next_obs()
        except StopIteration:
            terminated = True

        return obs, reward, terminated, truncated, {}
