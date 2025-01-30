import numpy as np

class BinaryCrossentropy:
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))

    def backward(self, y_true: np.ndarray) -> np.ndarray:
        return - (y_true / self.y_pred - (1 - y_true) / (1 - self.y_pred))

class MeanAbsoluteError:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true)