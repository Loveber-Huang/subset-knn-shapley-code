import numpy as np
from algs.baseline import knn_shapley_JW
from algs.helper import compute_distances
import time


def build_faiss_index(D_features, ann_N=128, use_gpu=False):
    import faiss

    """
    构建 Faiss HNSW32 索引（L2）
    """
    d = D_features.shape[1]
    xb = D_features.astype(np.float32)

    # HNSW32 索引
    index = faiss.IndexHNSWFlat(d, 32)  # 32 是邻居数 M
    index.hnsw.efConstruction = ann_N  # 构建图时精度，可调
    index.hnsw.efSearch = 2 * ann_N  # 查询时精度，可调

    index.add(xb)

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index


def cod_subshap(
    D_features,
    D_labels,
    D_test_features,
    D_test_labels,
    k,
    b,
    F_indices=None,
    n_top=None,
    ann_index=None,
    ann_N=None,
):
    """
    Conditional Shapley value computation using numpy arrays.

    Args:
        D_features: numpy array of shape (n, d) - training features
        D_labels: numpy array of shape (n,) - training labels
        D_test_features: numpy array of shape (m, d) - test features
        D_test_labels: numpy array of shape (m,) - test labels
        k: number of nearest neighbors
        b: budget
        F_indices: list of indices of initially selected points
        n_top: top N nearest neighbors (optional)
        ann_N: number of nearest neighbors (optional)

    Returns:
        phi: Shapley values for all points
        T_indices: indices of selected points
    """
    if F_indices is None:
        F_indices = []
    T_indices = F_indices.copy()
    n = len(D_features)
    n_top_more = (
        None if n_top is None else n_top + b + len(T_indices) + 1
    )  # preserve more space for next unselected point after n_top
    ann_N = None if ann_N is None else min(n, max(ann_N, n_top_more))
    selected = np.zeros(n, dtype=bool)
    n_rem = n
    phi = None  # Shapley values
    for idx in F_indices:
        selected[idx] = True
        n_rem -= 1
    if ann_N is not None:
        D_ann, I_ann = ann_index.search(D_test_features.astype(np.float32), ann_N)
    if n_top is None:
        sorted_indices = np.zeros((len(D_test_features), n), dtype=int)
    else:
        valid_lengths = np.zeros(len(D_test_features), dtype=int)
        if ann_N is not None:
            sorted_indices = np.zeros((len(D_test_features), ann_N), dtype=int)
        else:
            sorted_indices = np.zeros(
                (len(D_test_features), min(n, n_top_more)),
                dtype=int,
            )
    s_T = np.zeros(
        len(D_test_features)
    )  # number of points in T that have the same label as ztest
    bar_s = np.zeros(len(D_test_features))
    for i in range(len(D_test_features)):
        ztest_features = D_test_features[i]
        ztest_label = D_test_labels[i]
        if ann_N is not None:
            idxs = I_ann[i]
            idxs = np.unique(idxs[idxs >= 0])
            dists = compute_distances(D_features[idxs], ztest_features, metric="l2")
            order = np.argsort(dists)
            sorted_indices[i, : len(idxs)] = idxs[order]
            valid_lengths[i] = len(idxs)
        else:
            distances = compute_distances(D_features, ztest_features, metric="l2")
            if n_top is None:
                sorted_indices[i, :] = np.argsort(distances)
            else:
                sorted_indices[i, :] = np.argsort(distances)[: min(n, n_top_more)]
        T_labels = D_labels[T_indices]
        s_T[i] = np.sum(T_labels == ztest_label)
        bar_s[i] = np.sum(
            D_labels[~selected] == ztest_label
        )  # number of points in D that have the same label as ztest

    for _ in range(1, b + 1):
        m = len(T_indices)
        phi = np.zeros(n)  # Shapley values

        if m == 0:
            if ann_N is not None:
                phi = knn_shapley_JW(
                    D_features,
                    D_labels,
                    D_test_features,
                    D_test_labels,
                    k,
                    sorted_indices,
                    valid_lengths,
                )
            else:
                phi = knn_shapley_JW(
                    D_features,
                    D_labels,
                    D_test_features,
                    D_test_labels,
                    k,
                    sorted_indices,
                )
        else:

            T_features = D_features[T_indices]
            T_labels = D_labels[T_indices]
            sum_terms_mk = cached_harmonic_sum(
                m, k
            )  # sum_terms = np.sum(1.0 / np.arange(m+1,k+1))
            sum_terms_mk1 = cached_harmonic_sum(m, k - 1)

            for i in range(len(D_test_features)):
                ztest_features = D_test_features[i]
                ztest_label = D_test_labels[i]

                T_distances = compute_distances(T_features, ztest_features, metric="l2")
                T_sorted_indices = np.argsort(T_distances)
                T_sort_labels = T_labels[T_sorted_indices]
                n_less = n
                if n_top is None:
                    k_T = m  # number of points in T that are ranked before the current point j
                    k_T_next = k_T
                    i_val = n_rem  # index of the current unselected point
                    phi_next = None
                    y_next = None
                else:
                    Size = valid_lengths[i]
                    if Size < (n_top + len(T_indices) + 1) and ann_N is not None:
                        phi_next = 0
                        current = min(Size - 1, n_top)
                        if current != n_top:
                            flage = True
                        else:
                            flage = False
                        while selected[sorted_indices[i, current]]:
                            if flage:
                                current -= 1  # find the next unselected point
                            else:
                                current += 1  # find the next unselected point
                            if current == Size - 1:
                                flage = True
                        y_next = D_labels[sorted_indices[i, current]]
                        k_T = np.sum(
                            selected[sorted_indices[i, 0 : min(n_top, current)]]
                        )
                        i_val = (
                            min(n_top, current) - k_T
                        )  # index of the current unselected point
                        k_T_next = np.sum(selected[sorted_indices[i, 0:current]])
                        n_less = current
                    else:
                        k_T = np.sum(
                            selected[sorted_indices[i, 0:n_top]]
                        )  # number of points in T that are ranked before the current point j
                        k_T_next = k_T
                        i_val = n_top - k_T  # index of the current unselected point
                        phi_next = 0
                        current = n_top
                        while selected[sorted_indices[i, current]]:
                            current += 1  # find the next unselected point
                        y_next = D_labels[sorted_indices[i, current]]
                        k_T_next = np.sum(selected[sorted_indices[i, 0:current]])

                max_j = n if n_top is None else min(n, min(n_top, n_less))
                for j in reversed(range(max_j)):
                    idx = sorted_indices[i, j]
                    if selected[idx]:
                        k_T -= 1
                    else:
                        # base case
                        if i_val == n_rem:
                            bar_s[i] -= D_labels[idx] == ztest_label
                            if k_T >= k:
                                phi_j_z = 0

                            elif m >= k > k_T:
                                term_sum = np.sum(
                                    [
                                        indicator(D_labels[idx], ztest_label)
                                        - indicator(
                                            T_sort_labels[k - s - 1], ztest_label
                                        )
                                        for s in range(k - k_T)
                                    ]
                                )
                                phi_j_z = term_sum / (k * n_rem)

                            else:  # k > m
                                term1 = (
                                    indicator(D_labels[idx], ztest_label) * sum_terms_mk
                                    if k >= (m + 1)
                                    else 0
                                )
                                term2 = (
                                    (bar_s[i] / (n_rem - 1)) * sum_terms_mk1
                                    if k > (m + 1)
                                    else 0
                                )
                                term3 = (
                                    (bar_s[i] * (k - m - 1)) / ((n_rem - 1) * k)
                                    if n_rem > 1
                                    else 0
                                )
                                term4 = s_T[i] * (1 / m - 1 / k)
                                term5_sum = 0

                                for s in range(k - m, k - k_T):
                                    if k - s - 1 < len(T_sort_labels):
                                        term5_sum += indicator(
                                            D_labels[idx], ztest_label
                                        ) - indicator(
                                            T_sort_labels[k - s - 1], ztest_label
                                        )
                                term5 = term5_sum / k

                                phi_j_z = (
                                    term1 - term2 + term3 - term4 + term5
                                ) / n_rem
                            bar_s[i] += D_labels[idx] == ztest_label
                        # recursive case
                        else:
                            if k_T >= k:
                                phi_j_z = phi_next
                            else:
                                lbl_idx = D_labels[idx]
                                delta_y = (1 if lbl_idx == ztest_label else 0) - (
                                    1 if y_next == ztest_label else 0
                                )

                                boundary_term = (
                                    max(0, min(k - k_T_next, i_val))
                                    * (n_rem - 1)
                                    / i_val
                                    - max(0, k - m)
                                ) / k

                                if k_T == k_T_next:
                                    phi_j_z = phi_next + (delta_y / (n_rem - 1)) * (
                                        sum_terms_mk + boundary_term
                                    )
                                else:
                                    delta_y1 = 0
                                    for h in range(
                                        max(0, k - k_T_next),
                                        min(i_val - 1, k - k_T - 1) + 1,
                                    ):
                                        delta_y1 += indicator(
                                            lbl_idx, ztest_label
                                        ) - indicator(
                                            T_sort_labels[k - h - 1], ztest_label
                                        )
                                    phi_j_z = (
                                        phi_next
                                        + (delta_y / (n_rem - 1))
                                        * (sum_terms_mk + boundary_term)
                                        + delta_y1 / i_val / k
                                    )

                        k_T_next = k_T
                        phi[idx] += phi_j_z
                        phi_next = phi_j_z
                        i_val -= 1
                        y_next = D_labels[idx]

        unselected_indices = np.where(~selected)[0]
        if len(unselected_indices) == 0:
            break

        best_idx = unselected_indices[np.argmax(phi[unselected_indices])]
        T_indices.append(best_idx)

        selected[best_idx] = True
        n_rem -= 1

        for i in range(len(D_test_features)):
            ztest_label = D_test_labels[i]
            s_T[i] += indicator(D_labels[best_idx], ztest_label)
            bar_s[i] -= indicator(
                D_labels[best_idx], ztest_label
            )  # number of points in D that have the same label as ztest

    return phi, T_indices


class HarmonicPrefixCache:
    """
    H[n] = sum_{i=1..n} 1/i，
    sum_{t=m+1..k} 1/t = H[k] - H[m]。
    """

    def __init__(self):
        self._H = np.array([0.0], dtype=np.float64)  # H[0] = 0

    def ensure(self, n: int) -> None:
        if n < len(self._H):
            return
        start = len(self._H)
        new_vals = 1.0 / np.arange(start, n + 1, dtype=np.float64)
        self._H = np.concatenate([self._H, self._H[-1] + np.cumsum(new_vals)])

    def harmonic_sum(self, m: int, k: int) -> float:
        if k <= m:
            return 0.0
        self.ensure(k)
        return float(self._H[k] - self._H[m])

    def clear(self) -> None:
        self._H = np.array([0.0], dtype=np.float64)


_HARMONIC_CACHE = HarmonicPrefixCache()


def cached_harmonic_sum(m, k):
    return _HARMONIC_CACHE.harmonic_sum(m, k)


def indicator(a, b):
    """Indicator function for numpy arrays."""
    return 1 if a == b else 0


def make_ann_test_data(n=20000, m_test=20, d=20, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)

    D_features = rng.normal(size=(n, d)).astype(np.float32)
    D_labels = rng.integers(0, n_classes, size=n)

    D_test_features = rng.normal(size=(m_test, d)).astype(np.float32)
    D_test_labels = rng.integers(0, n_classes, size=m_test)

    return D_features, D_labels, D_test_features, D_test_labels


def main():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0),
        (np.array([3.0]), 1),
        (np.array([4.0]), 0),
        (np.array([5.0]), 1),
        (np.array([6.0]), 0),
        (np.array([7.0]), 1),
        (np.array([8.0]), 1),
        (np.array([9.0]), 0),
        (np.array([10.0]), 1),
    ]
    X = np.array([x for x, _ in D])
    Y = np.array([y for _, y in D])

    X_test = np.array([0.0])
    Y_test = np.array([1])
    index = build_faiss_index(D_features)
    phi = cod_subshap(
        D_features=X,
        D_labels=Y,
        D_test_features=X_test,
        D_test_labels=Y_test,
        k=5,
        b=1,
        F_indices=[0],
    )


if __name__ == "__main__":
    main()
