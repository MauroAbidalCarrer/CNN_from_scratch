# This import block is beautifull
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_views

DEFAULT_WEIGHS_SCALING = 0.001

class Convolutional:
    def __init__(self, kernels_shape:tuple, weights_scaling:float=DEFAULT_WEIGHS_SCALING):
        self.kernels = np.random.rand(*kernels_shape) * weights_scaling
        self.biases = np.zeros((1, 1, 1, kernels_shape[0]))

    def forward(self, inputs:np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return valid_correlate(inputs, self.kernels) + self.biases

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        self.biases -= gradients.mean(axis=(0, 1, 2), keepdims=True) * learning_rate
        gradient_wrt_kernels = valid_correlate(
            self.inputs.swapaxes(0, 3),
            gradients.swapaxes(0, 3),
        )
        self.kernels -= gradient_wrt_kernels.swapaxes(0, 3) * learning_rate
        return full_convolve(gradients, self.kernels.swapaxes(0, 3))

def full_convolve(inputs:np.ndarray, k:np.ndarray) -> np.ndarray:
    pad = ((0, 0), (k.shape[1]-1, k.shape[1]-1), (k.shape[2]-1, k.shape[2]-1), (0, 0))
    return valid_correlate(np.pad(inputs, pad, "constant"), np.flip(k, (1, 2)))

def valid_correlate(inputs:np.ndarray, k:np.ndarray) -> np.ndarray:
    return np.einsum(
        "bijcxy, kxyc -> bijk",
        sliding_views(inputs, k.shape[1:3], (1, 2)),
        k
    )

class Flatten:
    def forward(self, inputs:np.ndarray) -> np.ndarray:
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        return gradients.reshape(*self.inputs_shape)

class Linear:
    def __init__(self, input_size:int, output_size:int, weights_scaling:float=DEFAULT_WEIGHS_SCALING):
        # Since we are using batches of inputs and performing matrix multiplication on them and that
        # because matMul performs the dot product on the rows of the first(input) matrix and the (neurons) columms of the second
        self.weights = np.random.randn(input_size, output_size) * weights_scaling
        # We declare the biases as a column vector to perform broadcasted addidtion to the batch (matix) output.
        self.biases = np.zeros((1, output_size))

    def forward(self, input:np.ndarray) -> np.ndarray:
        self.input = input
        return input @ self.weights + self.biases

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        #print(gradients.shape)
        self.weights -= self.input.T @ gradients * learning_rate / gradients.shape[0]
        self.biases -= learning_rate * gradients.mean(axis=0, keepdims=True)
        return gradients @ self.weights.T

class Relu:
    def forward(self, input:np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(input, 0)

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        return np.where(self.input <= 0, 0, gradients)

class Sigmoid:
    def forward(self, inputs:np.ndarray) -> np.ndarray:
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        return gradients * (1 - self.outputs) * self.outputs
