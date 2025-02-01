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
        # Create a matrix of shape(nb_neurons, nb_inputs) with random values.
        # Since we are using batches of inputs and performing matrix multiplication on them and that
        # because matMul performs the dot product on the rows of the first matrix and the columms of the second instead of row/row,
        # we would need to transpose() the weight matrix for every pass.
        # So instead, we make the matrix of shape (input_size, nb_neurons).
        self.weights = np.random.randn(input_size, output_size)
        # The parameter of the funciton is in parenthesis because it is a tuple of size one.
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

    def backward(self, gradient:np.ndarray, learning_rate:float) -> np.ndarray:
        return np.where(self.input <= 0, 0, gradient)

class Sigmoid:
    def forward(self, inputs:np.ndarray) -> np.ndarray:
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        # Derivative - calculates from output of the sigmoid function
        return gradients * (1 - self.outputs) * self.outputs

#class SoftMax:
#    def forward(self, input: np.ndarray) -> np.ndarray:
#        # Compute the softmax output
#        exponantiated_output = np.exp(input - input.max())
#        self.input = input
#        self.output = exponantiated_output / exponantiated_output.sum()
#        return self.output
#
#    def backward(self, gradient: np.ndarray, learning_rate:float) -> np.ndarray:
#        # Reshape softmax output to a column vector
#        y = self.output.reshape(-1, 1)  # Shape: (n, 1)
#        
#        # Compute the Jacobian matrix of softmax
#        jacobian = np.diagflat(y) - np.dot(y, y.T)  # Shape: (n, n)
#        
#        # Compute the gradient with respect to the input
#        d_input = np.dot(jacobian, gradient)  # Shape: (n,)
#        return d_input
