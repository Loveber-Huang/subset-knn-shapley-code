import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from algs.helper import compute_distances


def kmeans_f1score(value_array, cluster=False):
    n_data = len(value_array)

    if cluster:
        X = value_array.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        min_cluster = min(kmeans.cluster_centers_.reshape(-1))
        pred = np.zeros(n_data)
        pred[value_array < min_cluster] = 1
    else:
        threshold = np.sort(value_array)[int(0.1 * n_data)]
        pred = np.zeros(n_data)
        pred[value_array < threshold] = 1

    true = np.zeros(n_data)
    true[: int(0.1 * n_data)] = 1
    return f1_score(true, pred)


def kmeans_aucroc(value_array):
    n_data = len(value_array)
    true = np.zeros(n_data)
    true[int(0.1 * n_data) :] = 1
    return roc_auc_score(true, value_array)


def kmeans_aupr(value_array, cluster=False):
    n_data = len(value_array)

    if cluster:
        X = value_array.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        min_cluster = min(kmeans.cluster_centers_.reshape(-1))
        pred = np.zeros(n_data)
        pred[value_array < min_cluster] = 1
    else:
        threshold = np.sort(value_array)[int(0.1 * n_data)]
        pred = np.zeros(n_data)
        pred[value_array < threshold] = 1

    true = np.zeros(n_data)
    true[: int(0.1 * n_data)] = 1
    return average_precision_score(true, pred)


def normalize(val):
    v_max, v_min = np.max(val), np.min(val)
    val = (val - v_min) / (v_max - v_min)
    return val


def rank_neighbor(x_test, x_train, metric="l2"):
    distance = compute_distances(x_test, x_train, metric=metric)
    return np.argsort(distance)


# x_test, y_test are single data point
def knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K):
    N = len(y_train_few)
    sv = np.zeros(N)
    rank = rank_neighbor(x_test, x_train_few, metric="l2")
    sv[int(rank[-1])] += int(y_test == y_train_few[int(rank[-1])]) / N

    for j in range(2, N + 1):
        i = N + 1 - j
        sv[int(rank[-j])] = (
            sv[int(rank[-(j - 1)])]
            + (
                (
                    int(y_test == y_train_few[int(rank[-j])])
                    - int(y_test == y_train_few[int(rank[-(j - 1)])])
                )
                / K
            )
            * min(K, i)
            / i
        )

    return sv


# Original KNN-Shapley proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf
def knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K):
    N = len(y_train_few)
    sv = np.zeros(N)

    n_test = len(y_val_few)
    for i in range(n_test):
        x_test, y_test = x_val_few[i], y_val_few[i]
        sv += knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K)

    return sv / n_test


# x_test, y_test are single data point
def knn_shapley_JW_single(
    x_train_few, y_train_few, x_test, y_test, K, precomputed_rank=None
):

    # Use precomputed rank if provided, otherwise compute it
    if precomputed_rank is not None:
        rank = precomputed_rank.astype(int)
        N = len(rank)
    else:
        rank = rank_neighbor(x_test, x_train_few, metric="l2").astype(int)
        N = len(y_train_few)
    sv = np.zeros(len(y_train_few))

    C = len(np.unique(y_train_few[rank[:N]]))

    c_A = np.sum(y_test == y_train_few[rank[: N - 1]])

    const = np.sum([1 / j for j in range(1, min(K, N) + 1)])

    sv[rank[-1]] = (int(y_test == y_train_few[rank[-1]]) - c_A / (N - 1)) / N * (
        np.sum([1 / (j + 1) for j in range(1, min(K, N))])
    ) + (int(y_test == y_train_few[rank[-1]]) - 1 / C) / N

    for j in range(2, N + 1):
        i = N + 1 - j
        coef = (
            int(y_test == y_train_few[int(rank[-j])])
            - int(y_test == y_train_few[int(rank[-(j - 1)])])
        ) / (N - 1)

        sum_K3 = K

        sv[int(rank[-j])] = sv[int(rank[-(j - 1)])] + coef * (
            const + int(N >= K) / K * (min(i, K) * (N - 1) / i - sum_K3)
        )

    return sv


# Soft-label KNN-Shapley proposed in https://arxiv.org/abs/2304.04258
def knn_shapley_JW(
    x_train_few,
    y_train_few,
    x_val_few,
    y_val_few,
    K,
    precomputed_ranks=None,
    valid_lengths=None,
):
    N = len(y_train_few)
    sv = np.zeros(N)

    n_test = len(y_val_few)
    for i in range(n_test):
        x_test, y_test = x_val_few[i], y_val_few[i]

        # Use precomputed rank if provided
        rank = precomputed_ranks[i] if precomputed_ranks is not None else None
        if rank is not None and valid_lengths is not None:
            rank = rank[: valid_lengths[i]]
            sv += knn_shapley_JW_single(
                x_train_few, y_train_few, x_test, y_test, K, rank
            )
        else:
            sv += knn_shapley_JW_single(
                x_train_few, y_train_few, x_test, y_test, K, rank
            )

    return sv / n_test
