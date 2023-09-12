from typing import List

import numpy as np


def simple_normalize(l: List[float]) -> List[float]:
    s = sum(l)
    return [i / s for i in l]


def round_lot(n: float) -> float:
    return np.floor(n / 100) * 100


def format_percent(n: float) -> str:
    return f'{100 * n:.2f}%'


def calc_cum_return(df, return_col='return', cum_col_name='cum_ret'):
    df[cum_col_name] = 1
    df[cum_col_name] += df[return_col]
    df[cum_col_name] = df[cum_col_name].cumprod()
    return df
