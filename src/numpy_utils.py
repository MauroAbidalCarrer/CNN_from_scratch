from functools import cache

import numpy  as np
from numpy import ndarray

@cache
def cached_zeros(shape) -> ndarray:
    return np.zeros(shape)

@cache
def cached_ones(shape) -> ndarray:
    return np.ones(shape)