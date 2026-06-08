import numpy as np
import pytest
import time

from algs.mc import shapley_mc
from algs.baseline import knn_shapley_JW
from algs.conditional_shapley import cod_subshap, build_faiss_index



def test_shapley_mc():
    D = [(np.array([0.5]), 1), (np.array([2.0]), 1), (np.array([1.0]), 0)]
    X = np.array([x for x, _ in D])
    Y = np.array([y for _, y in D])

    X_test = np.array([0.0])
    Y_test = np.array([1])

    np.random.seed(42)  # Set a fixed seed for reproducibility
    shapley_values = shapley_mc(X, Y, X_test, Y_test, k=1, n_perms=1000)
    expected_approx = [0.6709, 0.1671, -0.338]
    assert np.allclose(shapley_values, expected_approx, atol=1e-01)

    # Also verify that increasing the number of permutations reduces variance
    np.random.seed(42)
    shapley_values_more_perms = shapley_mc(X, Y, X_test, Y_test, k=1, n_perms=5000)
    assert np.allclose(shapley_values_more_perms, expected_approx, atol=1e-03)

    # Test with conditioning data
    np.random.seed(42)
    # Use point at index 0 as conditioning data, remove it from training set
    F_indices = [0]
    shapley_values_cond = shapley_mc(
        X,
        Y,
        X_test,
        Y_test,
        F_indices=F_indices,
        k=1,
        n_perms=1000,
    )
    # Should have one less Shapley value (since we removed one training point)
    assert len(shapley_values_cond) == len(X) - 1

    np.random.seed(42)
    # Use point at index 2 as conditioning data, remove it from training set
    F_indices = [2]
    shapley_values_cond = shapley_mc(
        X,
        Y,
        X_test,
        Y_test,
        F_indices=F_indices,
        k=1,
        n_perms=1000,
    )
    # Should have one less Shapley value (since we removed one training point)
    assert len(shapley_values_cond) == len(X) - 1


def test_shapley_mc_jw():
    D = [(np.array([0.5]), 1), (np.array([2.0]), 1), (np.array([1.0]), 0)]
    X = [x for x, _ in D]
    Y = [y for _, y in D]

    X_test = np.array([0.0])
    Y_test = np.array([1])

    np.random.seed(42)  # Set a fixed seed for reproducibility
    shapley_values = shapley_mc(
        np.array(X), np.array(Y), X_test, Y_test, k=1, n_perms=1000
    )
    shapley_values_jw = knn_shapley_JW(np.array(X), np.array(Y), X_test, Y_test, K=1)
    assert np.allclose(shapley_values, shapley_values_jw, atol=1e-01)

    # Also verify that increasing the number of permutations reduces variance
    np.random.seed(42)
    shapley_values_more_perms = shapley_mc(
        np.array(X), np.array(Y), X_test, Y_test, k=1, n_perms=5000
    )
    assert np.allclose(shapley_values_more_perms, shapley_values_jw, atol=1e-02)


def test_shapley_mc_cod_shap():
    # 1st test
    D = [(np.array([0.5]), 1), (np.array([2.0]), 1), (np.array([1.0]), 0)]
    X = [x for x, _ in D]
    Y = [y for _, y in D]

    X_test = np.array([0.0])
    Y_test = np.array([1])

    np.random.seed(42)  # Set a fixed seed for reproducibility
    shapley_values = shapley_mc(
        np.array(X), np.array(Y), X_test, Y_test, k=1, n_perms=1000
    )
    conditional_shapley_values, _ = cod_subshap(
        np.array(X), np.array(Y), X_test, Y_test, k=1, b=1
    )
    assert np.allclose(shapley_values, conditional_shapley_values, atol=1e-01)

    # Also verify that increasing the number of permutations reduces variance
    np.random.seed(42)
    shapley_values_more_perms = shapley_mc(
        np.array(X), np.array(Y), X_test, Y_test, k=1, n_perms=5000
    )
    assert np.allclose(
        shapley_values_more_perms, conditional_shapley_values, atol=1e-02
    )

    # 2nd test
    # Use point at index 0 as conditioning data, remove it from training set
    np.random.seed(42)
    F_indices = [0]
    shapley_values_cond = shapley_mc(
        np.array(X),
        np.array(Y),
        X_test,
        Y_test,
        F_indices=F_indices,
        k=1,
        n_perms=1000,
    )
    phi, _ = cod_subshap(
        np.array(X), np.array(Y), X_test, Y_test, k=1, b=1, F_indices=F_indices
    )
    selected = np.zeros(len(D), dtype=bool)
    selected[F_indices] = True
    conditional_shapley_values = phi[~selected]
    assert np.allclose(shapley_values_cond, conditional_shapley_values, atol=1e-01)

    # 3rd test
    # Use point at index 2 as conditioning data, remove it from training set
    np.random.seed(42)
    F_indices = [2]
    shapley_values_cond = shapley_mc(
        np.array(X),
        np.array(Y),
        X_test,
        Y_test,
        F_indices=F_indices,
        k=1,
        n_perms=1000,
    )
    phi, _ = cod_subshap(
        np.array(X), np.array(Y), X_test, Y_test, k=1, b=1, F_indices=F_indices
    )
    selected = np.zeros(len(D), dtype=bool)
    selected[F_indices] = True
    conditional_shapley_values = phi[~selected]
    assert np.allclose(shapley_values_cond, conditional_shapley_values, atol=1e-01)

    # 4th test
    # Use two points as conditioning data, remove them from the training set
    D = [
        (np.array([0.4]), 1),
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0),
        (np.array([0.9]), 1),
    ]
    X = [x for x, _ in D]
    Y = [y for _, y in D]

    X_test = np.array([0.0])
    Y_test = np.array([1])

    np.random.seed(42)
    F_indices = [3, 4]
    shapley_values_cond = shapley_mc(
        np.array(X),
        np.array(Y),
        X_test,
        Y_test,
        F_indices=F_indices,
        k=2,
        n_perms=1000,
    )
    phi, _ = cod_subshap(
        np.array(X), np.array(Y), X_test, Y_test, k=2, b=1, F_indices=F_indices
    )
    selected = np.zeros(len(D), dtype=bool)
    selected[F_indices] = True
    conditional_shapley_values = phi[~selected]
    print(conditional_shapley_values)
    assert np.allclose(shapley_values_cond, conditional_shapley_values, atol=1e-01)


# 1
def test_conditional_shapley():
    # example 1
    D_features = np.array([[0.5], [1.0], [2.0], [3.0]])
    D_labels = np.array([1, 0, 1, 0])
    D_test_features = np.array([[0.0]])
    D_test_labels = np.array([1])
    F_indices = [3]  # 对应原来的DataPoint(features=np.array([3.0]), y=0)

    # 条件子集测试（固定第一个点）
    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=1,
        b=1,
        F_indices=F_indices,
    )
    # 获取未选择点的Shapley值（对应原来的T1）
    selected = np.zeros(len(D_features), dtype=bool)
    selected[F_indices] = True
    T1 = phi[~selected]
    assert np.allclose(T1, [5 / 6, -1 / 6, 1 / 3], atol=1e-3)

    # example 2
    D_features = np.array([[0.5], [1.0], [2.0], [3.0]])
    D_labels = np.array([1, 0, 1, 0])
    D_test_features = np.array([[0.0]])
    D_test_labels = np.array([1])
    F_indices = [3]

    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=2,
        b=1,
        F_indices=F_indices,
    )
    selected = np.zeros(len(D_features), dtype=bool)
    selected[F_indices] = True
    T1 = phi[~selected]
    assert np.allclose(T1, [1 / 3, -1 / 6, 1 / 3], atol=1e-3)

    # example 3
    D_features = np.array([[0.5], [1.0], [2.0], [0.7]])
    D_labels = np.array([1, 0, 1, 0])
    D_test_features = np.array([[0.0]])
    D_test_labels = np.array([1])
    F_indices = [3]  # 对应原来的DataPoint(features=np.array([0.7]), y=0)

    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=2,
        b=1,
        F_indices=F_indices,
    )
    selected = np.zeros(len(D_features), dtype=bool)
    selected[F_indices] = True
    T1 = phi[~selected]
    assert np.allclose(T1, [5 / 12, -1 / 12, 1 / 6], atol=1e-3)


def make_ann_test_data(n=20000, m_test=20, d=20, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)

    D_features = rng.normal(size=(n, d)).astype(np.float32)
    D_labels = rng.integers(0, n_classes, size=n)

    D_test_features = rng.normal(size=(m_test, d)).astype(np.float32)
    D_test_labels = rng.integers(0, n_classes, size=m_test)

    return D_features, D_labels, D_test_features, D_test_labels


def test_approx_cod_subshap():
    # example 1
    D_features = np.array([[0.5], [1.0], [2.0], [3.0]])
    D_labels = np.array([1, 0, 1, 0])
    D_test_features = np.array([[0.0]])
    D_test_labels = np.array([1])
    F_indices = [3]  # 对应原来的DataPoint(features=np.array([3.0]), y=0)

    # 条件子集测试（固定第一个点）
    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=1,
        b=1,
        F_indices=F_indices,
        n_top=2,
    )
    # 获取未选择点的Shapley值（对应原来的T1）
    selected = np.zeros(len(D_features), dtype=bool)
    selected[F_indices] = True
    T1 = phi[~selected]
    assert np.allclose(T1, [1 / 2, -1 / 2, 0], atol=1e-3)

    # example 2
    D_features = np.array([[0.5], [1.0], [2.0], [3.0]])
    D_labels = np.array([1, 0, 1, 0])
    D_test_features = np.array([[0.0]])
    D_test_labels = np.array([1])
    F_indices = [3]

    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=2,
        b=1,
        F_indices=F_indices,
        n_top=2,
    )
    selected = np.zeros(len(D_features), dtype=bool)
    selected[F_indices] = True
    T1 = phi[~selected]
    assert np.allclose(T1, [0, -1 / 2, 0], atol=1e-3)

    # example 3
    D_features = np.array([[0.5], [1.0], [2.0], [0.7]])
    D_labels = np.array([1, 0, 1, 0])
    D_test_features = np.array([[0.0]])
    D_test_labels = np.array([1])
    F_indices = [3]  # 对应原来的DataPoint(features=np.array([0.7]), y=0)

    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=2,
        b=1,
        F_indices=F_indices,
        n_top=2,
    )
    selected = np.zeros(len(D_features), dtype=bool)
    selected[F_indices] = True
    T1 = phi[~selected]
    assert np.allclose(T1, [1 / 2, 0, 0], atol=1e-3)


def test_approx_time():
    D_features, D_labels, D_test_features, D_test_labels = make_ann_test_data()

    # Time cod_subshap
    start_time = time.process_time()
    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=10,
        b=1,
    )
    end_time = time.process_time()
    approx_time = end_time - start_time
    print(f"cod_subshap runtime: {approx_time:.6f} seconds")
    print(T_indices)

    start_time = time.process_time()
    phi = knn_shapley_JW(D_features, D_labels, D_test_features, D_test_labels, 10)
    end_time = time.process_time()
    approx_time = end_time - start_time
    print(f"knn_shapley_JW runtime: {approx_time:.6f} seconds")

    # Time approx_cod_subshap
    start_time = time.process_time()
    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=10,
        b=3,
        n_top=100,
    )
    end_time = time.process_time()
    approx_time = end_time - start_time
    print(f"approx_cod_subshap runtime: {approx_time:.6f} seconds")
    print(T_indices)

    # Time approx_cod_subshap with ANN
    start_time = time.process_time()
    index = build_faiss_index(D_features)
    end_time = time.process_time()
    build_time = end_time - start_time
    print(f"build_faiss_index runtime: {build_time:.6f} seconds")
    start_time = time.process_time()
    phi, T_indices = cod_subshap(
        D_features=D_features,
        D_labels=D_labels,
        D_test_features=D_test_features,
        D_test_labels=D_test_labels,
        k=10,
        b=3,
        n_top=100,
        ann_index=index,
        ann_N=200,
    )
    end_time = time.process_time()
    approx_time = end_time - start_time
    print(f"approx_cod_subshap_ann runtime: {approx_time:.6f} seconds")
    print(T_indices)
