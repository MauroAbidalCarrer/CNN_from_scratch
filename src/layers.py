from json import load, dump
from functools import partial
from itertools import accumulate
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view as sliding_views

from numpy_utils import cached_zeros, cached_ones
from constants import DEFAULT_WEIGHTS_SCALING, EPSILON#PARAM_NAMES , HYPER_PARAMS_TO_SERIALIZE


# class NeuralNetwork(list):
#     """Implements a sequential Neural network."""

#     def __init__(*layers):
#         super().__init__(*layers)

#     def forward(self, inputs:ndarray) -> list[ndarray]:
#         return list(accumulate(self, lambda x, l: l.forward(x), initial=inputs))

#     def backward(self, gradients:ndarray) -> list[ndarray]:
#         gradients = {"inputs": gradients}
#         return list(accumulate(reversed(self), lambda x, l: l.forward(x), initial=gradients))

#     def serialize(self) -> list[dict]:
#         serialized_self:list[dict] = []
#         for layer in self:
#             layer_dict = {"layer_type": layer.__class__.name}
#             for param_name in filter(partial(hasattr, layer), PARAM_NAMES + HYPER_PARAMS_TO_SERIALIZE):
#                 layer_dict[param_name] = getattr(layer, param_name)
#             serialized_self.append(layer_dict)
#         return serialized_self

#     def to_json(self, path:str):
#         with open(path, "w") as fp:
#             dump(self.serialize(), fp, indent=1)

class Layer(ABC):
    @abstractmethod
    def forward(self, inputs:ndarray) -> ndarray:
        pass

    @abstractmethod
    def backward(self, gradients:ndarray) -> dict[str, ndarray]:
        """Returns a dict of the gradients wrt params and inputs."""
        pass


class BatchNorm(Layer):
    def __init__(self, axis:tuple=(0, 1, 2)):
        self.gamma: ndarray = None  # Scale parameter, to be initialized lazily.
        self.beta: ndarray = None   # Shift parameter, to be initialized lazily.
        self.cache = None  # Will store values needed for backward pass
        self.axis = axis

    def forward(self, inputs: ndarray) -> ndarray:
        # Initialize gamma and beta if not already done.
        if self.gamma is None or self.beta is None:
            param_shape = tuple(1 if idx in self.axis else size for idx, size in enumerate(inputs.shape))
            self.gamma = np.ones(param_shape)
            self.beta = np.zeros(param_shape)
        
        # Compute mean and variance over (N, W, H) for each channel.
        mean = inputs.mean(axis=self.axis, keepdims=True)  # Shape (1,1,1,C)
        var = inputs.var(axis=self.axis, keepdims=True)      # Shape (1,1,1,C)
        
        # Normalize the input.
        x_hat = (inputs - mean) / np.sqrt(var + EPSILON)
        # Scale and shift.
        out = self.gamma * x_hat + self.beta

        # Cache variables for backward pass.
        self.cache = (inputs, x_hat, mean, var)
        return out

    def backward(self, gradients: ndarray) -> dict[str, ndarray]:
        inputs, x_hat, mean, var = self.cache

        shape_to_prod = tuple(size for idx, size in enumerate(inputs.shape) if idx in self.axis)
        nb_elements = np.prod(shape_to_prod)  # Total number of elements per channel

        # Gradients for the learnable parameters.
        beta_gradient = np.sum(gradients, axis=self.axis, keepdims=True)
        gamma_gradient = np.sum(gradients * x_hat, axis=self.axis, keepdims=True)

        # Backprop through normalization.
        normed_input_grad = gradients * self.gamma  # Shape (N, W, H, C)

        var_grad = np.sum(normed_input_grad * (inputs - mean) * -0.5 * (var + EPSILON) ** (-1.5), axis=self.axis, keepdims=True)
        mean_grad = np.sum(normed_input_grad * -1 / np.sqrt(var + EPSILON), axis=self.axis, keepdims=True) + \
                 var_grad * np.sum(-2 * (inputs - mean), axis=self.axis, keepdims=True) / nb_elements

        inputs_grads = normed_input_grad / np.sqrt(var + EPSILON) + var_grad * 2 * (inputs - mean) / nb_elements + mean_grad / nb_elements

        return {
            "gamma": gamma_gradient,
            "beta": beta_gradient,
            "inputs": inputs_grads
        }
        

class Convolutional(Layer):
    def __init__(self, kernels_shape:tuple, weights_scaling:float=DEFAULT_WEIGHTS_SCALING):
        self.kernels = np.random.rand(*kernels_shape) * weights_scaling
        self.biases = np.zeros((1, 1, 1, kernels_shape[0]))

    def forward(self, inputs:ndarray) -> ndarray:
        self.inputs = inputs
        return Convolutional.valid_correlate(inputs, self.kernels) + self.biases

    def backward(self, gradients:ndarray) -> ndarray:
        gradient_wrt_kernels = Convolutional.valid_correlate(
            self.inputs.swapaxes(0, 3),
            gradients.swapaxes(0, 3),
        )
        return {
            "biases": gradients.mean(axis=(0, 1, 2), keepdims=True),
            "kernels": gradient_wrt_kernels.swapaxes(0, 3),
            "inputs": Convolutional.full_convolve(gradients, self.kernels.swapaxes(0, 3)),
        }
    
    @classmethod
    def full_convolve(cls, inputs:ndarray, k:ndarray) -> ndarray:
        pad = ((0, 0), (k.shape[1]-1, k.shape[1]-1), (k.shape[2]-1, k.shape[2]-1), (0, 0))
        return Convolutional.valid_correlate(np.pad(inputs, pad), np.flip(k, (1, 2)))

    @classmethod
    def valid_correlate(cls, inputs:ndarray, k:ndarray) -> ndarray:
        views = sliding_views(inputs, k.shape[1:3], (1, 2))
        correlations = np.tensordot(views, k, axes=([3, 4, 5], [3, 1, 2]))
        return correlations

class MaxPool(Layer):
    def __init__(self, kernel_shape:tuple[int, int]):
        self.pool_height, self.pool_width = kernel_shape
        self.cache = None

    def forward(self, x):
        N, W, H, C = x.shape
        # Ensure that width and height are divisible by the pooling dimensions.
        assert W % self.pool_height == 0, "Input width must be divisible by pool height"
        assert H % self.pool_width == 0, "Input height must be divisible by pool width"
        out_W = W // self.pool_height
        out_H = H // self.pool_width
        # Reshape the input so that we can perform pooling over the appropriate dimensions.
        # New shape: (N, out_W, pool_height, out_H, pool_width, C)
        x_reshaped = x.reshape(N, out_W, self.pool_height, out_H, self.pool_width, C)
        # Compute the maximum over the pooling regions (axes 2 and 4)
        out = np.max(x_reshaped, axis=(2, 4))
        # Cache values needed for the backward pass.
        self.cache = (x, x_reshaped, out)
        return out

    def backward(self, gradients:ndarray) -> ndarray:
        x, x_reshaped, out = self.cache
        N, out_W, pool_h, out_H, pool_w, C = x_reshaped.shape
        # Expand the pooled output to the shape of x_reshaped for comparison.
        # out_expanded has shape: (N, out_W, 1, out_H, 1, C)
        out_expanded = out[:, :, np.newaxis, :, np.newaxis, :]
        # Create a mask: True at positions that contributed to the max.
        mask = (x_reshaped == out_expanded)
        # Count the number of maximum entries in each pooling region for proper gradient distribution.
        mask_sum = np.sum(mask, axis=(2, 4), keepdims=True)
        # Expand the upstream gradient to match the dimensions of the pooling regions.
        dout_expanded = gradients[:, :, np.newaxis, :, np.newaxis, :]
        # Distribute the gradient: if there are ties, the gradient is divided equally.
        dx_reshaped = mask * (dout_expanded / mask_sum)
        # Reshape back to the original input dimensions.
        dx = dx_reshaped.reshape(x.shape)
        return {"inputs": dx}

class Flatten(Layer):
    def forward(self, inputs:ndarray) -> ndarray:
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, gradients:ndarray) -> ndarray:
        return {"inputs": gradients.reshape(*self.inputs_shape)}

class Linear(Layer):
    def __init__(self, input_size:int, output_size:int, weights_scaling:float=DEFAULT_WEIGHTS_SCALING):
        # Since we are using batches of inputs and performing matrix multiplication on them and that
        # because matMul performs the dot product on the rows of the first(input) matrix and the (neurons) columms of the second
        self.weights = np.random.randn(input_size, output_size) * weights_scaling
        self.biases = np.zeros((1, output_size))

    def forward(self, input:ndarray) -> ndarray:
        self.input = input
        return input @ self.weights + self.biases

    def backward(self, gradients:ndarray) -> ndarray:
        return {
            "inputs": gradients @ self.weights.T,
            "weights": self.input.T @ gradients / gradients.shape[0],
            "biases": gradients.mean(axis=0, keepdims=True),
        }

class Relu(Layer):
    def forward(self, input:ndarray) -> ndarray:
        self.input = input
        return np.maximum(input, 0)

    def backward(self, gradients:ndarray) -> ndarray:
        return {"inputs": np.where(self.input <= 0, 0, gradients)}

class Sigmoid(Layer):
    def forward(self, inputs:ndarray) -> ndarray:
        clipped_inputs = np.clip(inputs, -500, 500)  # Clip the values to avoid overflow
        self.outputs = 1 / (1 + np.exp(-clipped_inputs))
        return self.outputs

    def backward(self, gradients:ndarray) -> ndarray:
        return {"inputs": gradients * (1 - self.outputs) * self.outputs}

class Softmax(Layer):
    def forward(self, inputs: ndarray) -> ndarray:
        # Shift inputs by subtracting the maximum value in each row for numerical stability
        exp_shifted = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.out = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.out

    def backward(self, gradients: ndarray) -> ndarray:
        # For each sample in the batch, the gradient of the softmax is:
        # dL/dz = s * (grad - sum(grad * s))
        # where s is the softmax output for that sample.
        sum_grad = np.sum(gradients * self.out, axis=1, keepdims=True)
        grad_input = self.out * (gradients - sum_grad)
        return {"inputs": grad_input}

