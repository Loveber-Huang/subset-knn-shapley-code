import numpy as np
import bisect

from helper import compute_distances


def shapley_mc(
    X: np.ndarray,
    Y: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    F_indices: list = None,
    k: int = 1,
    n_perms: int = 1000,
    metric: str = "l2",
):
    """
    Compute KNN Shapley values via Monte Carlo sampling.
    Use conditioning data if provided, i.e., F_indices is not None.
    When enumerating a permutation of X, maintain a dynamic sorted list of k (distance, label) tuples for each test point.

    Args:
        X: np.ndarray, shape (n, d) - training data points
        Y: np.ndarray, shape (n,) - training data labels
        X_test: np.ndarray, shape (n_test, d) - test data points
        Y_test: np.ndarray, shape (n_test,) - test data labels
        F_indices: list of int, optional - indices of conditioning points in X/Y(optional)
        k: int - number of nearest neighbors
        n_perms: int - number of permutations for Monte Carlo sampling
        metric: str - distance metric ('l2', 'l1', or 'cosine')

    Returns:
        shapley_values: np.ndarray, shape (n,)
    """
    X = np.array(X)
    Y = np.array(Y)
    if F_indices:
        X_cond = np.array(X[F_indices])
        Y_cond = np.array(Y[F_indices])

        mask = np.ones(len(X), dtype=bool)
        mask[F_indices] = False

        X = np.array(X[mask])
        Y = np.array(Y[mask])
    else:
        X_cond = None
        Y_cond = None

    n = len(X)
    if n < k or n < 2:
        raise ValueError(f"Number of points must be greater than k and 2: n={n}, k={k}")

    X_text = np.array(X_test)
    Y_text = np.array(Y_test)
    n_test = len(X_text)

    # Initialize Shapley values array for all points
    shapley_values = np.zeros(len(X))

    for i in range(n_test):
        x_test, y_test = X_test[i], Y_test[i]

        distances = compute_distances(X, x_test, metric)
        max_dist = max(distances)
        distances /= max_dist  # normalize the distances

        Y_combined = np.concatenate([Y, Y_cond]) if Y_cond is not None else Y
        unique_labels = set(Y_combined)
        C = len(unique_labels)

        # sort conditioning points by distance to x_test
        if X_cond is not None:
            distances_cond = compute_distances(X_cond, x_test, metric)
            distances_cond /= max_dist  # normalize the distances
            sorted_cond = sorted(zip(distances_cond, Y_cond))

        # Initialize value array across all permutations
        s = np.zeros(n)

        for _ in range(n_perms):
            # Sample a random permutation of training points only
            perm = np.random.permutation(n)

            if X_cond is not None:
                # Start with conditioning points already in the k-NN
                sorted_topk = sorted_cond[: min(k, len(sorted_cond))]
                knn_labels = [y for _, y in sorted_topk]
                utility_prev = sum(knn_labels == y_test) / len(knn_labels)
            else:
                sorted_topk = []  # Maintain a sorted list of (distance, label) tuples
                utility_prev = 1 / C  # Default prediction without any points

            # For each position in the permutation, calculate marginal contribution
            for idx in perm:
                # Insert current point into sorted list using bisect
                current_label = Y[idx]
                current_dist = distances[idx]
                insert_pos = bisect.bisect_left(
                    sorted_topk, (current_dist, current_label)
                )
                sorted_topk.insert(insert_pos, (current_dist, current_label))

                # Keep only the k nearest neighbors
                if len(sorted_topk) > k:
                    sorted_topk = sorted_topk[:k]

                # Calculate utility with the current coalition
                knn_labels = [y for _, y in sorted_topk]
                utility_curr = sum(knn_labels == y_test) / len(knn_labels)

                # Add marginal contribution to Shapley value
                marginal = utility_curr - utility_prev
                s[idx] += marginal

                # Update previous utility for next iteration
                utility_prev = utility_curr

        # Average over all permutations
        s /= n_perms
        shapley_values += s

    return shapley_values / n_test
