DEFAULT_WEIGHTS_SCALING = 0.001
# Tiny positive number for numerical stability in divisions
EPSILON = 1e-7
PARAM_NAMES = ["weights", "biases", "kernels"]
MAX_NB_SAMPLES = 500
DFLT_NEGATIVE_LEAKY_RELU_SLOPE = 1e-2 # Taken from pytorch's default 