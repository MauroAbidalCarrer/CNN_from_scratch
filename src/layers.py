import numpy as np


class Linear:
    def __init__(self, input_size:int, output_size:int, weights_scaling:float=0.01):
        # Create a matrix of shape(nb_neurons, nb_inputs) with random values.
        # Since we are using batches of inputs and performing matrix multiplication on them and that
        # because matMul performs the dot product on the rows of the first matrix and the columms of the second instead of row/row,
        # we would need to transpose() the weight matrix for every pass.
        # So instead, we make the matrix of shape (input_size, nb_neurons).
        self.weights = np.random.randn(input_size, output_size) * weights_scaling
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

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        return np.where(self.input <= 0, 0, gradients)

class Sigmoid:
    def forward(self, inputs:np.ndarray) -> np.ndarray:
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        # Derivative - calculates from output of the sigmoid function
        return gradients * (1 - self.outputs) * self.outputs

