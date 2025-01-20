import sys

import numpy as np
from scipy.signal import correlate, convolve

class Convolutional:
    def __init__(self, input_depth:tuple, kernel_size:int, num_kernels:int):
        self.kernels = np.random.randn(num_kernels, kernel_size, kernel_size, input_depth)
        self.biases = np.random.randn(num_kernels)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        activation_maps = [correlate(input, k, "valid") + b for k, b in zip(self.kernels, self.biases)]
        self.output = np.stack(activation_maps)
        return self.output

    def backward(self, gradient:np.ndarray, learning_rate:float) -> np.ndarray:
        input_gradient = sum([convolve(k_gradient, k, "full") for k_gradient, k in zip(gradient, self.kernels)])
        self.biases -= gradient.sum(tuple(range(1, gradient.ndim))) * learning_rate
        self.kernels -=  np.stack([correlate(input, k_gradient, "valid") for k_gradient in gradient]) * learning_rate
        return input_gradient

class Linear:
    def __init__(self, input_size:int, output_size:int):
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.biases = np.random.uniform(-1, 1, output_size)

    def forward(self, input:np.ndarray) -> np.ndarray:
        self.input = input
        return input @ self.weights + self.biases
    
    def backward(self, gradient:np.ndarray, learning_rate:float) -> np.ndarray:
        self.weights -= learning_rate * np.outer(self.input, gradient)
        self.biases -= learning_rate * gradient
        return gradient @ self.weights.T

class Relu:
    def forward(self, input:np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(input, 0)
    
    def backward(self, gradient:np.ndarray, learning_rate:float) -> np.ndarray:
        gradient[self.input <= 0] = 0
        return gradient 

class SoftMax:
    def forward(self, input: np.ndarray) -> np.ndarray:
        # Compute the softmax output
        exponantiated_output = np.exp(input - input.max())
        self.input = input
        self.output = exponantiated_output / exponantiated_output.sum()
        return self.output

    def backward(self, d_output: np.ndarray, learning_rate:float) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the input of softmax.
        
        Parameters:
        d_output (np.ndarray): Gradient of the loss with respect to the output of softmax.
        
        Returns:
        np.ndarray: Gradient of the loss with respect to the input of softmax.
        """
        # Reshape softmax output to a column vector
        y = self.output.reshape(-1, 1)  # Shape: (n, 1)
        
        # Compute the Jacobian matrix of softmax
        jacobian = np.diagflat(y) - np.dot(y, y.T)  # Shape: (n, n)
        
        # Compute the gradient with respect to the input
        d_input = np.dot(jacobian, d_output)  # Shape: (n,)
        return d_input
