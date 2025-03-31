import pickle
from os import makedirs
from os.path import dirname, join, exists
import tarfile
import urllib.request

import numpy as np

# URL to download CIFAR-10
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_DATASET_FILENAME = "cifar-10-python.tar.gz"
dataset_dir = "cifar_10_dataset"
dataset_dir = join(dirname(__file__), dataset_dir)
compressed_dataset_path = join(dataset_dir, TAR_DATASET_FILENAME)
dataset_batches_dir = join(dataset_dir, "cifar-10-batches-py")

if not exists(compressed_dataset_path):
    makedirs(dirname(compressed_dataset_path), exist_ok=True)
    print(f"Downloading CIFAR-10 dataset into {compressed_dataset_path}...")
    urllib.request.urlretrieve(URL, compressed_dataset_path)
    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(compressed_dataset_path, "r:gz") as tar:
        tar.extractall(dataset_dir)

def load_cifar_10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load all training batches
    x_train, y_train = [], []
    for i in range(1, 6):
        file = join(dataset_batches_dir, f"data_batch_{i}")
        data, labels = load_batch(file)
        x_train.append(data)
        y_train.append(labels)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    # Load test batch
    x_test, y_test = load_batch(join(dataset_batches_dir, "test_batch"))

    return x_train, y_train, x_test, y_test

def load_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = one_hot_encode_labels(np.array(batch[b'labels']))
    return data, labels

def one_hot_encode_labels(label_idxs:np.ndarray) -> np.ndarray:
    return np.eye(10)[label_idxs]

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_cifar_10()
    # Print shapes
    print("Train data shape:", x_train.shape)  # (50000, 32, 32, 3)
    print("Train labels shape:", y_train.shape)  # (50000,)
    print("Test data shape:", x_test.shape)  # (10000, 32, 32, 3)
    print("Test labels shape:", y_test.shape)  # (10000,)