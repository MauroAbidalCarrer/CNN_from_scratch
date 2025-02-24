# That import block is beautifull
import numpy as np

DEFAULT_WEIGHS_SCALING = 0.001

class Convolutional2D:
    def __init__(self, input_shape:np.ndarray, kernel_shape:tuple, nb_kernels:int, weights_scaling:float=DEFAULT_WEIGHS_SCALING):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.nb_kernels = nb_kernels
        self.flatten_kernels = np.random.rand(np.prod(kernel_shape), nb_kernels) * weights_scaling
        self.biases = np.zeros((1, 1, 1, nb_kernels)) * weights_scaling
        self.input_and_kernels_corr_idx, self.output_shape = Convolutional2D.compute_correlation_indices(input_shape, kernel_shape, nb_kernels)
        print("output_shape:", self.output_shape)
        padded_output_shape = (self.output_shape[0] + kernel_shape[0] * 2 - 2, self.output_shape[1] + kernel_shape[1] * 2 - 2, input_shape[2])
        self.outputs_and_kernels_conv_idx, _ = Convolutional2D.compute_correlation_indices(padded_output_shape, kernel_shape, nb_kernels)
        swapped_input_shape = (input_shape)
        self.inputs_and_gradients_corr_idx, _ = Convolutional2D.compute_correlation_indices(input_shape, self.output_shape, nb_kernels, True)

    def forward(self, inputs:np.ndarray) -> np.ndarray:
        print('=====forward=====')
        self.inputs = inputs
        return Convolutional2D.valid_correlate(inputs, self.flatten_kernels, self.output_shape, self.input_and_kernels_corr_idx, True) + self.biases

    def backward(self, gradients:np.ndarray, learning_rate:float) -> np.ndarray:
        print('=====backward=====')
        self.biases -= gradients.mean(axis=(0, 1, 2), keepdims=True) * learning_rate
        print("gradients:", gradients.shape)
        flatten_gradients = gradients.reshape(-1, gradients.shape[3])
        self.flatten_kernels -= Convolutional2D.valid_correlate(self.inputs, flatten_gradients, self.input_shape, self.inputs_and_gradients_corr_idx, True) * learning_rate
        pad_with = np.tile(np.asarray((0, self.kernel_shape[0], self.kernel_shape[1], 0)), (1, 1))
        padded_gradients = np.pad(gradients, pad_with, "constant")
        flipped_kernels = np.flip(self.flatten_kernels.reshape(self.nb_kernels, self.kernel_shape[0], self.kernel_shape[1], self.kernel_shape[2]), axis=(1, 2)).reshape(self.nb_kernels, -1)
        return Convolutional2D.valid_correlate(padded_gradients, flipped_kernels, self.input_shape, self.outputs_and_kernels_conv_idx)

    @classmethod
    def valid_correlate(cls, inputs:np.ndarray, flatten_kernels:np.ndarray, output_shape:tuple, corr_idx:np.ndarray, print_shapes=False) -> np.ndarray:
        flatten_inputs = inputs.reshape(inputs.shape[0], -1)
        if print_shapes:
            print("inputs:", inputs.shape)
            print("flatten_inputs:", flatten_inputs.shape)
            print("corr_idx:", corr_idx.shape)
            print("views:", flatten_inputs[:, corr_idx].shape)
            print("flatten_kernels:", flatten_kernels.shape)
        flatten_correlations = flatten_inputs[:, corr_idx] @ flatten_kernels 
        return flatten_correlations.reshape(inputs.shape[0], output_shape[0], output_shape[1], flatten_kernels.shape[1])

    @classmethod
    def compute_correlation_indices(cls, input_shape:tuple, kernel_shape:tuple, nb_kernels:int, print_shapes=False) -> tuple[np.ndarray, np.ndarray]:
        if print_shapes:
            print("input_shape:", input_shape)
            print("kernel_shape:", kernel_shape)
        window_idx = np.arange(kernel_shape[2])
        window_idx = np.tile(window_idx, kernel_shape[1])
        window_idx += np.repeat(np.arange(kernel_shape[1]) * input_shape[2], kernel_shape[2])
        window_idx = np.tile(window_idx, kernel_shape[0])
        window_idx += np.repeat(np.arange(kernel_shape[0]) * input_shape[1] * input_shape[2], kernel_shape[1] * kernel_shape[2])

        nb_x_correlations = 1 + input_shape[0] - kernel_shape[0]
        nb_y_correlations = 1 + input_shape[1] - kernel_shape[1]
        total_nb_correlations = nb_x_correlations * nb_y_correlations

        x_offset_multiplicator = input_shape[2]
        y_offset_multiplicator = input_shape[2] * input_shape[1]
        x_offsets = np.tile(np.arange(nb_x_correlations) * x_offset_multiplicator, nb_y_correlations).reshape(-1, 1)
        y_offsets = np.repeat(np.arange(nb_y_correlations) * y_offset_multiplicator, nb_y_correlations).reshape(-1, 1)
        correlation_indices = np.tile(window_idx, (total_nb_correlations, 1)) + x_offsets + y_offsets

        return correlation_indices, np.array((nb_x_correlations, nb_y_correlations, nb_kernels))

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
