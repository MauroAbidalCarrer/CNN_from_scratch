from functools import reduce, cache

import numpy as np
from numpy import ndarray
from rich.progress import track
from pandas import DataFrame as DF

from losses import Loss
from layers import Layer
from metrics import accuracy
from constants import EPSILON


class SGD:
    def __init__(self, learning_rate:float):
        self._learning_rate = learning_rate
        self.iterations = 0
        self.training_metrics:list[dict] = []

    # Only works for categorical datasets for now
    def optimize_nn(self, nn:list[Layer], x:ndarray, y:ndarray, epochs:int, batch_size:int, loss:Loss, metric_freq:int=1, use_track=True) -> DF:
        """Optimizes the neural network and returns a dataframe of the training metrics."""
        it = track(range(epochs), description="Training...") if use_track else range(epochs)
        for epoch in it:
            # Compute and store new trainging metrics
            if not epoch % metric_freq:
                y_pred = SGD.forward(nn, x)
                self.training_metrics.append({
                    "iteration": self.iterations,
                    "epoch": epoch,
                    "loss": loss.forward(y_pred, y),
                    "learning_rate": self.learning_rate,
                    "accuracy": accuracy(y_pred, y),
                })
            # Perform steps
            for batch_i in range(0, x.shape[0], batch_size):
                batch_x = x[batch_i:batch_i+batch_size]
                batch_y = y[batch_i:batch_i+batch_size]
                batch_y_preds = SGD.forward(nn, batch_x)
                batch_gradients = loss.backward(batch_y_preds, batch_y)
                self.update_params(batch_gradients, nn)
                self.iterations += 1
        return DF.from_records(self.training_metrics)

    # Forward is implemented in SGD because it's the only place it's used, eventually it will be in a NueralNetwork module.
    @classmethod
    def forward(cls, nn:list[Layer], inputs:ndarray) -> ndarray:
        return reduce(lambda x, l: l.forward(x), nn, inputs)

    def update_params(self, outputs_gradients:ndarray, nn:list[Layer]):
        for layer in reversed(nn):
            gradients:dict = layer.backward(outputs_gradients)
            inputs_grad = gradients.pop("inputs")
            for param_name, grads_wrt_param in gradients.items():
                param:ndarray = getattr(layer, param_name)
                gradient_wrt_param = self.postprocess_gradient(layer, param_name, grads_wrt_param)
                setattr(layer, param_name, param - gradient_wrt_param)
            outputs_gradients = inputs_grad

    def postprocess_gradient(self, layer:Layer, param_name:str, gradient_wrt_param:ndarray) -> ndarray:
        return self.learning_rate * gradient_wrt_param

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

class SGD_with_decay(SGD):
    def __init__(self, starting_lr:float, lr_decay:float=0.0):
        self.starting_lr = starting_lr
        self.lr_decay = lr_decay
        super().__init__(starting_lr)

    @property
    def learning_rate(self) -> float:
        return self.starting_lr / (1 + self.lr_decay * self.iterations)

class SGD_with_momentum(SGD_with_decay):
    def __init__(self, starting_lr:float, lr_decay:float, momentum:float):
        self.momentum = momentum
        super().__init__(starting_lr, lr_decay)

    def postprocess_gradient(self, layer:Layer, param_name:str, gradient_wrt_param:ndarray) -> ndarray:
        param_momentum = getattr(layer, param_name + "_momentum", cached_zeros(gradient_wrt_param.shape))
        post_processed_gradient_wrt_param = self.learning_rate * gradient_wrt_param + param_momentum * self.momentum
        setattr(layer, param_name + "_momentum", post_processed_gradient_wrt_param)
        return post_processed_gradient_wrt_param

class RMSprop(SGD_with_decay):
    def __init__(self, starting_lr:float, lr_decay:float, rho:float):
        self.rho = rho
        super().__init__(starting_lr, lr_decay)

    def get_and_set_adapted_grad(self, layer:Layer, param_name:str, gradient_wrt_param:ndarray) -> ndarray:
        param_cache = getattr(layer, param_name + "_cache", cached_zeros(gradient_wrt_param.shape))
        param_cache = self.rho * param_cache + (1 - self.rho) * gradient_wrt_param ** 2
        setattr(layer, param_name + "_cache", param_cache)
        return gradient_wrt_param / (np.sqrt(param_cache) + EPSILON)

    def postprocess_gradient(self, layer:Layer, param_name:str, param:ndarray, gradient_wrt_param:ndarray) -> ndarray:
        return self.learning_rate * self.get_and_set_adapted_grad(layer, param_name, param, gradient_wrt_param)

class Adam(SGD_with_decay):
    def __init__(self, starting_lr:float, lr_decay:float, momentum_weight:float, ada_grad_weight:float):
        self.momentum_weight = momentum_weight
        self.ada_grad_weight = ada_grad_weight
        super().__init__(starting_lr, lr_decay)

    def postprocess_gradient(self, layer:Layer, param_name:str, gradient_wrt_param:ndarray) -> ndarray:
        momentum = self.update_moving_average(layer, param_name + "_momentum", gradient_wrt_param, self.momentum_weight)
        cache = self.update_moving_average(layer, param_name + "_cache", gradient_wrt_param ** 2, self.ada_grad_weight)
        return self.learning_rate * momentum / (np.sqrt(cache) + EPSILON)

    def update_moving_average(self, layer, moving_average_name:str, gradient:ndarray, beta:float) -> ndarray:
        """Updates the moving average and returns it corrected."""
        moving_average = getattr(layer, moving_average_name, cached_zeros(gradient.shape)) # Get the moving averate
        moving_average = lerp(moving_average, gradient, beta) # Move the average
        setattr(layer, moving_average_name, moving_average) # Set the moved average for future use
        return moving_average / (1 - beta ** (self.iterations + 1)) # Correct and return the average 

def lerp(a:ndarray, b:ndarray, t:float) -> ndarray:
    return a * t + (1 - t) * b

@cache
def cached_zeros(shape) -> ndarray:
    return np.zeros(shape)