from typing import List


def normalize(l: List[float]) -> List[float]:
    s = sum(l)
    return [i / s for i in l]
