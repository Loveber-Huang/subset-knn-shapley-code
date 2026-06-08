import numpy as np


def compute_distances(
    X: np.ndarray, x_test: np.ndarray, metric: str = "l2"
) -> np.ndarray:
    if metric == "l2":
        distances = np.linalg.norm(X - x_test, axis=1)
    elif metric == "l1":
        distances = np.sum(np.abs(X - x_test), axis=1)
    elif metric == "cosine":
        # assume X and x_test are normalized
        distances = 1 - np.dot(X, x_test.T)
    else:
        raise ValueError("Unsupported distance metric")

    return distances
