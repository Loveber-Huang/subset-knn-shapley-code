# Subset-kNN-Shapley

A Python library for computing Subset-kNN-Shapley classifiers. 
This library provides efficient algorithms for data valuation in machine learning, helping to understand the contribution of individual training samples to model predictions.

Main methods:
- `cod_subshap`: Inc-CkNN values
- `shapley_mc``: Monte Carlo Subset-kNN-Shapley values


## Installation

See `pyproject.toml` for dependencies.
We recommend using `uv` to install a local environment.


## Quick Start

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Import methods
from algs.conditional_shapley import cod_subshap, build_faiss_index
from algs.mc import shapley_mc


# Generate sample data
X, y = make_classification(
    n_samples=200, 
    n_features=8, 
    n_classes=2, 
    n_informative=4,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Normalize features
X_mean, X_std = np.mean(X_train, 0), np.std(X_train, 0)
normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
X_train_norm, X_test_norm = normalizer_fn(X_train), normalizer_fn(X_test)

k = 5
F = [5]
b = 1
n_top = 100
ann_N = 150

print("Computing different value types...")

# 1. Inc-CkNN-E values
Inc-CkNN-E, _=cod_subshap(
    X_train_norm, y_train, X_test_norm, y_test, k, b, F
)
print(f"Inc-CkNN-E values: {Inc-CkNN-E[:5]}")

# 2. Inc-CkNN-Apx values  
Inc-CkNN-Apx, _=cod_subshap(
    X_train_norm, y_train, X_test_norm, y_test, k, b, F, n_top
)
print(f"Inc-CkNN-Apx values: {Inc-CkNN-Apx[:5]}")

# 3. Inc-CkNN-ANN values
ann_index = build_faiss_index(X_train_norm)
Inc-CkNN-ANN, _=cod_subshap(
    X_train_norm, y_train, X_test_norm, y_test, k, b, F, n_top, ann_index, ann_N
)
print(f"Inc-CkNN-ANN values: {Inc-CkNN-ANN[:5]}")

# 4. Monte Carlo subset-knn-shapley approximation
subset_knn_shapley_mc = shapley_mc(
    X_train_norm, y_train, X_test_norm, y_test, F, k, n_perms=1000
    )
print(f"MC subset-knn-shapley values: {subset_knn_shapley_mc[:5]}")   
```

Output:
```
Computing different value types...
Inc-CkNN-E values: [-0.01253532  0.07605505 -0.05771792 -0.096305    0.04437077]
Inc-CkNN-Apx values: [ 0.01044235  0.0608802  -0.0577962  -0.07295734  0.04342374]
Inc-CkNN-ANN values: [ 0.01044235  0.0608802  -0.0577962  -0.07295734  0.04342374]
MC subset-knn-shapley values: [-0.00091833  0.00403667 -0.0036925  -0.00468083  0.002315  ]
```