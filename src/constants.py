from metrics import accuracy

DEFAULT_WEIGHTS_SCALING = 0.001
# Tiny positive number for numerical stability in divisions
EPSILON = 1e-7
DFLT_METRICS = [accuracy]