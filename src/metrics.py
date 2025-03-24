import numpy as np
from numpy import ndarray


def accuracy(y_pred:ndarray, y_true:ndarray, **kwargs) -> ndarray:
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()