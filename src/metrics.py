from typing import Callable, Dict

# import numpy as np
from numpy import ndarray


metric_func = Callable[[Dict, dict], Dict]

def accuracy(metric_line:dict, y_pred:ndarray, y_true:ndarray, **_) -> dict:
    metric_line["accuracy"] = (y_pred.argmax(1) == y_true.argmax(1)).mean()
    return metric_line

def gradient_stats(metric_line:dict, gradients:list[ndarray], nn:list, **_) -> dict:
    for layer_i, (layer, gradient) in enumerate(zip(nn, gradients)):
        layer_name = f"{type(layer).__name__}_{layer_i}"
        metric_line[f"{layer_name}_grad_mean"] = gradient.mean()
        metric_line[f"{layer_name}_grad_std"] = gradient.mean()
    return metric_line
