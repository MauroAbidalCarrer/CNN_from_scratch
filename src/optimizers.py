from functools import reduce

import numpy as np
from numpy import ndarray
from rich.progress import track
from pandas import DataFrame as DF

from layers import Layer
from losses import Loss

class SGD:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate

    # Only works for categorical datasets for now
    def optimize_nn(self, nn:list[Layer], x:ndarray, y:ndarray, epochs:int, batch_size:int, loss:Loss, score_funcs:dict[callable], metric_freq:int=1) -> DF:
        """Optimizes the neural network and returns a dataframe of the training mmetrics."""
        self.iterations = 0
        training_metrics:list[dict] = []
        for epoch in track(range(epochs), description="Training..."):
            # Perfomr steps
            for batch_i in range(0, x.shape[0], batch_size):
                batch_x = x[batch_i:batch_i+batch_size]
                batch_y = y[batch_i:batch_i+batch_size]
                batch_y_pred = SGD.forward(nn, batch_x)
                batch_gradients = loss.backward(batch_y_pred, batch_y)
                self.update_params(batch_gradients, nn)
                self.iterations += 1
            # Compute and store new trainging metrics
            if not epoch % metric_freq:
                y_pred = SGD.forward(nn, x)
                training_metrics.append({
                    "epoch": epoch,
                    "loss": loss.forward(y_pred, y),
                    **{score_func.__name__: score_func(y_pred, y) for score_func in score_funcs},
                })
        return DF.from_records(training_metrics)

    # Forward is implemented in SGD because it's the only place it's used, eventually it will be in a NueralNetwork module.
    @classmethod
    def forward(cls, nn:list[Layer], inputs:ndarray) -> ndarray:
        return reduce(lambda x, l: l.forward(x), nn, inputs)

    def update_params(self, next_layer_gradients:ndarray, nn:list[Layer]):
        for layer in reversed(nn):
            gradients = layer.backward(next_layer_gradients)
            for param_name, grads_wrt_param in gradients.items():
                if param_name != "inputs":
                    param = getattr(layer, param_name)
                    setattr(
                        layer,
                        param_name,
                        param - self.compute_gradient_wrt_param(layer, param_name, grads_wrt_param)
                    )
            next_layer_gradients = gradients["inputs"]

    def compute_gradient_wrt_param(self, layer:Layer, param_name:str, gradient_wrt_param:ndarray) -> ndarray:
        return self.learning_rate * gradient_wrt_param

class SGD_with_decay(SGD):
    def __init__(self, starting_lr:float, lr_decay:float):
        super(SGD, self).__init__(starting_lr)
        self.decay = lr_decay

    @property
    def learning_rate(self) -> float:
        return self.starting_learning_rate / (1 + self.learning_rate_decay * self.iterations)
    
# class SGD_with_momentum(SGD_with_decay):
#     def __init__(self, starting_lr: float, lr_decay: float, momentum:float):
#         super().__init__(starting_lr, lr_decay)
#         self.momentum = momentum

#     def compute_gradient_wrt_param(self, layer: Layer, param_name: str, gradients_wrt_param: ndarray) -> ndarray:
#         momentum = getattr(layer, param_name + "_momentum", np.zeros_like(getattr(layer, param_name)))
#         setattr(layer, param_name + "_momentum", momentum - gradients_wrt_param)
#         return self.learning_rate * (momentum - gradients_wrt_param)