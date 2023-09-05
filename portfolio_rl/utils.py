from typing import List

import numpy as np


def simple_normalize(l: List[float]) -> List[float]:
    s = sum(l)
    return [i / s for i in l]


def round_lot(n: float) -> float:
    return np.floor(n / 100) * 100


def format_percent(n: float) -> str:
    return f'{100 * n:.2f}%'
