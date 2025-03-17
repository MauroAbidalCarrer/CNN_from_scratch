import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_views

from constants import DEFAULT_WEIGHTS_SCALING

class Convolutional:
    def __init__(self, kernels_shape:tuple, weights_scaling:float=DEFAULT_WEIGHTS_SCALING):
        self.kernels = np.random.rand(*kernels_shape) * weights_scaling
        self.biases = np.zeros((1, 1, 1, kernels_shape[0]))

    def forward(self, inputs:np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return Convolutional.valid_correlate(inputs, self.kernels) + self.biases

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        self.biases -= gradients.mean(axis=(0, 1, 2), keepdims=True) * learning_rate
        gradient_wrt_kernels = Convolutional.valid_correlate(
            self.inputs.swapaxes(0, 3),
            gradients.swapaxes(0, 3),
        )
        self.kernels -= gradient_wrt_kernels.swapaxes(0, 3) * learning_rate
        return Convolutional.full_convolve(gradients, self.kernels.swapaxes(0, 3))
    
    @classmethod
    def full_convolve(cls, inputs:np.ndarray, k:np.ndarray) -> np.ndarray:
        pad = ((0, 0), (k.shape[1]-1, k.shape[1]-1), (k.shape[2]-1, k.shape[2]-1), (0, 0))
        return Convolutional.valid_correlate(np.pad(inputs, pad), np.flip(k, (1, 2)))

    @classmethod
    def valid_correlate(cls, inputs:np.ndarray, k:np.ndarray) -> np.ndarray:
        views = sliding_views(inputs, k.shape[1:3], (1, 2))
        correlations = np.tensordot(views, k, axes=([3, 4, 5], [3, 1, 2]))
        return correlations

class MaxPool:
    def __init__(self, kernel_shape):
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

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
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
        return dx

class Flatten:
    def forward(self, inputs:np.ndarray) -> np.ndarray:
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        return gradients.reshape(*self.inputs_shape)

class Linear:
    params = ["weights", "biases"]
    def __init__(self, input_size:int, output_size:int, weights_scaling:float=DEFAULT_WEIGHTS_SCALING):
        # Since we are using batches of inputs and performing matrix multiplication on them and that
        # because matMul performs the dot product on the rows of the first(input) matrix and the (neurons) columms of the second
        self.weights = np.random.randn(input_size, output_size) * weights_scaling
        # We declare the biases as a column vector to perform broadcasted addidtion to the batch (matix) output.
        self.biases = np.zeros((1, output_size))

    def forward(self, input:np.ndarray) -> np.ndarray:
        self.input = input
        return input @ self.weights + self.biases

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
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
        clipped_inputs = np.clip(inputs, -500, 500)  # Clip the values to avoid overflow
        self.outputs = 1 / (1 + np.exp(-clipped_inputs))
        return self.outputs

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        return gradients * (1 - self.outputs) * self.outputs

class Softmax:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Shift inputs by subtracting the maximum value in each row for numerical stability
        exp_shifted = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.out = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.out

    def backward(self, gradients: np.ndarray, learning_rate:float) -> np.ndarray:
        # For each sample in the batch, the gradient of the softmax is:
        # dL/dz = s * (grad - sum(grad * s))
        # where s is the softmax output for that sample.
        sum_grad = np.sum(gradients * self.out, axis=1, keepdims=True)
        grad_input = self.out * (gradients - sum_grad)
        return grad_input

