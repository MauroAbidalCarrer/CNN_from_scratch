from itertools import accumulate
from typing import Callable, Dict

import numpy as np
from numpy import ndarray

from constants import PARAM_NAMES


metric_func = Callable[[Dict, dict], Dict]

def accuracy(metric_line:dict, y_pred:ndarray, y_true:ndarray, **_) -> dict:
    metric_line["accuracy"] = (y_pred.argmax(1) == y_true.argmax(1)).mean()
    return metric_line

def gradient_stats(metric_line:dict, gradients:list[dict[str, ndarray]], nn:list, **_) -> dict:
    for layer_i, (layer, gradient) in enumerate(zip(nn, gradients)):
        layer_name = f"{type(layer).__name__}_{layer_i}"
        metric_line[f"{layer_name}_grad_wrt_inputs_mean"] = gradient["inputs"].mean()
        metric_line[f"{layer_name}_grad_wrt_inputs_std"] = gradient["inputs"].std()
    return metric_line

def nn_params_stats(metric_line:dict, nn, **_) -> dict:
    for layer_i, layer in enumerate(nn):
        layer_type = type(layer)
        layer_name = f"{layer_type.__name__}_{layer_i}"
        for param_name in PARAM_NAMES:
            if hasattr(layer, param_name):
                metric_line[f"{layer_name}_{param_name}_mean"] = getattr(layer, param_name).mean()
                metric_line[f"{layer_name}_{param_name}_std"] = getattr(layer, param_name).std()
                metric_line[f"{layer_name}_{param_name}_l1"] = np.abs(getattr(layer, param_name)).sum()
                metric_line[f"{layer_name}_{param_name}_l2"] = np.sqrt((getattr(layer, param_name) ** 2).sum())
    return metric_line

def activations_stats(metric_line:dict, nn, activations, **_) -> dict:
    for layer_i, (layer, activation) in enumerate(zip(nn, activations)):
        activation_name = f"{type(layer).__name__}_{layer_i}_activation"
        metric_line[activation_name + "_mean"] = activation.mean(axis=0).mean()
        metric_line[activation_name + "_std"] = activation.std(axis=0).mean()
        metric_line[activation_name + "_l1"] = np.abs(activation).sum(axis=0).mean()
        metric_line[activation_name + "_l2"] = np.sqrt(activation ** 2).sum(axis=0).mean()
    return metric_line
