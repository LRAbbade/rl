import warnings
from datetime import date
from enum import Enum
from typing import Optional, Dict, Any, Tuple, SupportsFloat, Iterable, Union

from gymnasium import Env
from gymnasium.spaces import Space, Box
from gymnasium.core import ObsType, ActType
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils import seeding

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import normalize
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from portfolio_rl.utils import calc_cum_return, round_lot, format_percent

DEFAULT_WALLET = 1e7


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


class RewardType(Enum):
    NOTIONAL = 'notional'
    PERCENT = 'percent'


class PortfolioEnv(Env):
    # Possible TODO's:
    # - add option to short stocks
    # - calculate long and short carry rates and sum/subtract to wallet
    # - add option to hedge wallet on index futures
    # - formalize dataframe format (validate on init) in order to facilitate
    #   running the environment for other indices
    # - add running free float shares information to data and check it before trades
    action_space: NormBox
    observation_space: Space[ObsType]
    reward_range: Tuple[float, float] = -1, np.inf
    spec: EnvSpec | None = None
    _last_date: date = None
    _cur_date: date = None
    _iterator: Iterable[date] = None

    def __init__(self,
        data_path: str, warmup_data_path: Optional[str] = None, start_wallet: float = DEFAULT_WALLET,
        warmup: bool = False, slippage: int = 10, liquidity_limit: float = 0.5,
        reward_type: RewardType = RewardType.PERCENT, seed: Optional[int] = None,
        fail_threshold: float = 0.1
    ):
        """
        Portfolio Env constructor

        Args:
            data_path (str): A valid parquet file path with the S&P 500 portfolio data to be used,
                should be in the shape `(N * 500, 11)`, where `N` is the number of days to be traded.
                Each day should have the top 500 stocks in the index that day.
            warmup_data_path (optional str): A parquet file path with warmup data to be given to the agent
                before trading starts, this is usually to calculate things like moving averages
            start_wallet (optional float): The size of the trading wallet, 10,000,000 by default
            warmup (optional bool): Whether to pass the agent warmup data on `reset` call or not.
                If `True`, then `warmup_data_path` must resolve to a valid parquet file
            slippage (optional int): how many bps of slippage to apply to each order (default 10 bps)
            liquidity_limit (optional float): maximum of daily volume that can be traded (default 50%)
            reward_type (optional RewardType): how the reward will be measured, either in notional or percentage
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            fail_threshold (optional float): % of start_wallet to consider run a failure and truncate (default 10%)
        """
        self.seed = seed
        self.df = pd.read_parquet(data_path)
        self._validate_data(self.df)
        self.warmup = warmup
        self.warmup_data_path = warmup_data_path
        self._validate_warmup()
        if self.warmup_data_path is not None:
            self.warmup_df = pd.read_parquet(warmup_data_path)

        self.start_wallet = start_wallet
        self.slippage = slippage
        self.liquidity_limit = liquidity_limit
        self.reward_type = reward_type
        self.fail_threshold = fail_threshold
        self.action_space = NormBox(0, 1, [1, 500], np.float32)
        self.observation_space = MarketObservation(self.df, shape=[500, 11])
        self._set_seed(seed)

    def _set_seed(self, seed: Optional[int] = None):
        self._np_random, _ = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _validate_data(self, df: pd.DataFrame):
        aux = df.groupby('date').agg({'ticker': 'count'})
        assert aux[aux['ticker'] != 500].empty, 'Every date of data should have exactly 500 rows (top 500 tickers)'

    def _validate_warmup(self):
        assert not self.warmup or self.warmup_data_path is not None, 'Need to specify warmup data path when using warmup = True'

    def _base_wallet_df(self, cash: float, mtm: float) -> pd.DataFrame:
        return pd.DataFrame(data={
                'date': [self._cur_date, self._cur_date],
                'ticker': ['USD.CASH', 'MTM_VALUE'],
                'shares': [cash, mtm],
                'notional': [cash, mtm],
                'weight': [cash / mtm, 1.0]
        })

    def _update_historical_wallet(self):
        self.historical_wallet_df = pd.concat([self.historical_wallet_df, self.wallet_df]).reset_index(drop=True)

    def _day_data(self, dt: date) -> pd.DataFrame:
        return self.df.loc[self.df['date'] == dt]

    def __iter__(self):
        return iter(sorted(list(self.df['date'].unique())))

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self)

        self._last_date = self._cur_date
        self._cur_date = next(self._iterator)
        return self._day_data(self._cur_date)

    def _next_obs(self) -> ObsType:
        return next(self)

    def _set_prop_from_options(self, options: Dict[str, Any], attr: str):
        setattr(self, attr, options.get(attr, getattr(self, attr)))

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
                - start_wallet (float): wallet size (default: 10,000,000)
                - warmup (bool): Whether to pass the agent warmup data on `reset` call or not (default: False)
                - slippage (int): bps of slippage to be applied to each order
                - liquidity_limit (float): maximum of daily volume that can be traded (default 50%)
                - reward_type (optional RewardType): how the reward will be measured, either in notional or percentage
                - fail_threshold (optional float): % of start_wallet to consider run a failure and truncate (default 10%)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary): if warmup is true, than the warmup data is returned as a DF in a 'warmup' key
        """
        self._set_seed(seed)
        for attr in ('start_wallet', 'warmup', 'slippage', 'liquidity_limit', 'reward_type', 'fail_threshold'):
            self._set_prop_from_options(options, attr)

        self._iterator = None
        obs, info = self._next_obs(), {}
        if self.warmup:
            info['warmup'] = self.warmup_df

        self.wallet = self.start_wallet
        self.historical_wallet_df = pd.DataFrame()
        self.wallet_df = self._base_wallet_df(self.wallet, self.wallet)
        self._update_historical_wallet()
        return obs, info

    def _validate_action(self, action: ActType) -> NDArray[Any]:
        arr = np.array(action)
        assert arr.shape == (500,), f'action shape is {arr.shape}, should be (500,)'
        assert np.isclose(arr.sum(), 1) or arr.sum() <= 1, f'sum of weights should be at most equal to 1, not {arr.sum()}'
        assert arr.sum() >= 0, f'sum of weights should be bigger than 0, not {arr.sum()}'
        return arr

    def _update_wallet(self, previous_obs: pd.DataFrame, cash_left: float, final_mtm: float):
        self.wallet = final_mtm
        portfolio_wallet = previous_obs[['ticker', 'shares_new', 'final_notional', 'final_weight']].rename({
            'shares_new': 'shares',
            'final_notional': 'notional',
            'final_weight': 'weight'
        }, axis=1)
        portfolio_wallet['date'] = self._cur_date
        self.wallet_df = pd.concat([
            portfolio_wallet,
            self._base_wallet_df(cash_left, final_mtm)
        ]).reset_index(drop=True)
        self._update_historical_wallet()

    def _calc_reward(self, new_mtm: float, previous_mtm: float) -> float:
        if self.reward_type == RewardType.PERCENT:
            return (new_mtm / previous_mtm) - 1
        else:
            return new_mtm - previous_mtm

    def _wallet_day_mtm_value(self, day: date) -> float:
        return self.historical_wallet_df.loc[
            (self.historical_wallet_df['date'] == day) &
            (self.historical_wallet_df['ticker'] == 'MTM_VALUE'),
            'notional'
        ].iloc[0]

    def _set_nans_for_tickers_out_of_index(self, df: pd.DataFrame) -> pd.DataFrame:
        # symbols that were in the composition but are not anymore
        # need to fill volume, close_new, vwap, and weight
        # will consider vwap = close_new = close, volume = inf, weight = 0
        # in order for model to be able to close the position on the ticker
        # (this is not ideal, but its realistic to assume that anything leaving the S&P 500
        #  will have a large auction in its last day in the index)
        out_of_index_mask = df['weight'].isna()
        df.loc[out_of_index_mask, 'volume'] = np.inf
        df.loc[out_of_index_mask, 'close_new'] = df.loc[out_of_index_mask, 'close']
        df.loc[out_of_index_mask, 'vwap'] = df.loc[out_of_index_mask, 'close']
        df.loc[out_of_index_mask, 'weight'] = 0
        return df

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
            return obs, 0, True, False, {'dt': self._cur_date}

        previous_obs = self._day_data(self._last_date)
        weights = self._validate_action(action)

        # TODO: refactor all this
        previous_obs['portfolio_weights'] = weights
        previous_obs = previous_obs[['ticker', 'close', 'portfolio_weights']].join(
            obs[['ticker', 'volume', 'close', 'vwap', 'weight']].set_index('ticker'),
            on='ticker',
            rsuffix='_new'
        )
        previous_obs = self._set_nans_for_tickers_out_of_index(previous_obs)
        # weight check
        to_zero = list(previous_obs.loc[(previous_obs['weight'] == 0) & (previous_obs['portfolio_weights'] > 0), 'ticker'])
        if len(to_zero) > 0:
            warnings.warn(f'{len(to_zero)} tickers positions will be zeroed due to having weight 0 in the index')

        previous_obs.loc[previous_obs['ticker'].isin(to_zero), 'portfolio_weights'] = 0
        wallet_mtm = self._wallet_day_mtm_value(self._last_date)
        previous_obs['notional'] = wallet_mtm * previous_obs['portfolio_weights']
        previous_obs['shares'] = (previous_obs['notional'] / previous_obs['close']).map(round_lot)
        previous_obs = previous_obs.join(self.wallet_df[['ticker', 'shares']].set_index('ticker'), on='ticker', rsuffix='_old')
        previous_obs['shares_old'] = previous_obs['shares_old'].fillna(0)
        previous_obs['diff'] = previous_obs['shares'] - previous_obs['shares_old']
        previous_obs['max_trade_vol'] = previous_obs['volume'] * self.liquidity_limit
        previous_obs['over_max_volume'] = abs(previous_obs['diff']) > previous_obs['max_trade_vol']
        volume_restricted_tickers = list(previous_obs.loc[previous_obs['over_max_volume'], 'ticker'])
        if len(volume_restricted_tickers) > 0:
            warnings.warn(f'{len(volume_restricted_tickers)} tickers require trades over the daily liquidity limits, will cap trades')

        previous_obs['trade_size'] = abs(previous_obs[['max_trade_vol', 'diff']]).min(axis=1)
        previous_obs['side'] = previous_obs['diff'].map(lambda n: 'buy' if n >= 0 else 'sell')
        previous_obs['slippage'] = previous_obs['side'].map(lambda s: (self.slippage if s == 'buy' else -self.slippage) / 1e4)
        previous_obs['trade_price'] = previous_obs['vwap'] * (1 + previous_obs['slippage'])
        previous_obs['side_sign'] = previous_obs['side'].map(lambda s: 1 if s == 'buy' else -1)
        previous_obs['trade_notional'] = previous_obs['trade_price'] * previous_obs['trade_size'] * previous_obs['side_sign']
        previous_obs['shares_new'] = previous_obs['shares_old'] + (previous_obs['trade_size'] * previous_obs['side_sign'])
        previous_obs['final_notional'] = previous_obs['close_new'] * previous_obs['shares_new']
        prev_cash = self.wallet_df.loc[self.wallet_df['ticker'] == 'USD.CASH', 'notional'].iloc[0]
        total_cost = previous_obs['trade_notional'].sum()
        cash_left = prev_cash - total_cost
        final_mtm = cash_left + previous_obs['final_notional'].sum()
        previous_obs['final_weight'] = previous_obs['final_notional'] / wallet_mtm
        self._update_wallet(previous_obs, cash_left, final_mtm)

        reward = self._calc_reward(final_mtm, wallet_mtm)
        truncated = self.wallet <= self.fail_threshold * self.start_wallet
        return obs, reward, terminated, truncated, {'dt': self._cur_date}

    def historical_wallet_mtm(self) -> pd.DataFrame:
        hist_wallet = self.historical_wallet_df
        return hist_wallet.loc[hist_wallet['ticker'] == 'MTM_VALUE'].reset_index(drop=True)

    def calc_results(self) -> Dict[str, str]:
        df = self.historical_wallet_mtm()
        df['return'] = df['notional'].pct_change()
        df['cum_return'] = (1 + df['return']).cumprod()
        df = df.dropna().reset_index(drop=True)
        mean_return = df['return'].mean()
        vol = df['return'].std()
        # sharpe should take into consideration the risk free rate of return
        # I may come back to this later
        sharpe = mean_return / vol
        # TODO: calculate max drawdown
        return {
            'total_return': format_percent(df['cum_return'].iloc[-1] - 1),
            'mean_return': format_percent(mean_return),
            'vol': format_percent(vol),
            'sharpe': f'{sharpe:.3f}'
        }

    def plot_mtm(self) -> go.Figure:
        return px.line(self.historical_wallet_mtm(), x='date', y='notional')

    def _parse_comp_returns(self, df1: pd.DataFrame, df2: pd.DataFrame, df1_value_col: str, df2_value_col: str) -> Union[pd.DataFrame, str, str]:
        df = df1.join(df2.set_index('date'), on='date')
        df1_return_col, df2_return_col = f'return_{df1_value_col}', f'return_{df2_value_col}'
        df1_cum_ret_col, df2_cum_ret_col = f'cum_{df1_return_col}', f'cum_{df2_return_col}'
        df[df1_return_col] = df[df1_value_col].pct_change()
        df[df2_return_col] = df[df2_value_col].pct_change()
        df = calc_cum_return(df, df1_return_col, df1_cum_ret_col)
        return calc_cum_return(df, df2_return_col, df2_cum_ret_col), df1_cum_ret_col, df2_cum_ret_col

    def _plot_comp(self, df1: pd.DataFrame, df2: pd.DataFrame, df1_value_col: str, df2_value_col: str) -> go.Figure:
        df, cum_ret_col1, cum_ret_col2 = self._parse_comp_returns(df1, df2, df1_value_col, df2_value_col)
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=df['date'], y=df[cum_ret_col1], name=f'{df1_value_col} Return'))
        fig.add_trace(go.Scatter(x=df['date'], y=df[cum_ret_col2], name=f'{df2_value_col} Return'))
        fig.update_layout(title_text=f'{df1_value_col} X {df2_value_col}')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Return')
        return fig

    def plot_stock_comp(self, ticker: str) -> go.Figure:
        wallet_df = self.historical_wallet_mtm()[['date', 'notional']].rename({'notional': 'portfolio'}, axis=1)
        stock_df = self.df.loc[self.df['ticker'] == ticker, ['date', 'close']].rename({'close': ticker}, axis=1)
        return self._plot_comp(wallet_df, stock_df, 'portfolio', ticker)

    def index_comp(self, index_parquet_path: str) -> go.Figure:
        wallet_df = self.historical_wallet_mtm()[['date', 'notional']].rename({'notional': 'portfolio'}, axis=1)
        spx_df = pd.read_parquet(index_parquet_path)[['date', 'close']].rename({'close': 'SPX'}, axis=1)
        return self._plot_comp(wallet_df, spx_df, 'portfolio', 'SPX')
