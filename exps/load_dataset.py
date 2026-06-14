import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_svmlight_file
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def make_binary_imbalance(X, y, IF, random_state=42):
    """
    对二分类数据构造不均衡分布

    IF = majority / minority
    """
    rng = np.random.RandomState(random_state)

    idx0 = np.where(y == 1)[0]
    idx1 = np.where(y == 2)[0]

    # 判断哪个是 majority
    if len(idx0) > len(idx1):
        maj_idx, min_idx = idx0, idx1
    else:
        maj_idx, min_idx = idx1, idx0

    maj_n = len(maj_idx)
    min_n = int(maj_n / IF)

    if min_n < 1:
        min_n = 1

    chosen_min = rng.choice(min_idx, min_n, replace=False)

    new_idx = np.concatenate([maj_idx, chosen_min])
    rng.shuffle(new_idx)

    return X[new_idx], y[new_idx]


def make_multiclass_imbalance(X, y, IF=10, random_state=42):
    """
    构造多分类不均衡数据集

    IF = max_class_samples / min_class_samples

    参数
    ----
    X : ndarray
    y : ndarray
    IF : imbalance factor
    random_state : 随机种子
    """

    rng = np.random.RandomState(random_state)

    classes = np.unique(y)
    K = len(classes)

    # 每类原始样本数
    class_counts = {c: np.sum(y == c) for c in classes}
    n_max = max(class_counts.values())  # 为了保证能采样

    # long-tail 样本数量
    samples_per_class = []
    for i in range(K):
        num = int(n_max * (IF ** (-i / (K - 1))))
        samples_per_class.append(num)

    new_indices = []

    for c, n in zip(classes, samples_per_class):
        idx = np.where(y == c)[0]

        n = min(n, len(idx))  # 防止超出
        chosen = rng.choice(idx, n, replace=False)

        new_indices.append(chosen)

    new_indices = np.concatenate(new_indices)
    rng.shuffle(new_indices)

    return X[new_indices], y[new_indices]


def _require_torch_transformers() -> None:
    if torch is None:
        raise ImportError(
            "Need to install torch to use embedding dataset loaders (e.g., bert/cifar/mnist)."
        )
    if tqdm is None:
        raise ImportError("Need to install tqdm to use embedding dataset loaders.")
    if BertTokenizer is None or BertModel is None:
        raise ImportError(
            "Need to install transformers to use BERT embedding dataset loaders."
        )


def _resolve_data_files_root(dataset_path: str | None = None) -> str:
    if dataset_path is None:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data_files")
        )
    return os.path.abspath(dataset_path)


def _embedding_cache_root(dataset_path: str | None = None) -> str:
    return os.path.join(_resolve_data_files_root(dataset_path), "embeddings_cache")


def _slug(s: str) -> str:
    return (
        str(s).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    )


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    np.save(tmp, arr)
    # np.save adds ".npy" if missing; keep our tmp extension consistent
    if not tmp.endswith(".npy"):
        tmp_npy = tmp + ".npy"
    else:
        tmp_npy = tmp
    os.replace(tmp_npy, path)


def _atomic_save_npz(path: str, **arrays: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    np.savez_compressed(tmp, **arrays)
    if not tmp.endswith(".npz"):
        tmp_npz = tmp + ".npz"
    else:
        tmp_npz = tmp
    os.replace(tmp_npz, path)


def _with_file_lock(lock_path: str):
    import contextlib

    try:
        import fcntl  # Unix only
    except Exception:  # pragma: no cover
        fcntl = None

    @contextlib.contextmanager
    def _ctx():
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "w") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)

    return _ctx()


def load_adult_dataset(
    dataset_path: str,
    train_size: int = 1200,
    test_size: int = 500,
    random_state: int = 42,
):
    """Load Adult dataset

    Args:
        dataset_path: Dataset path
        train_size: Training set sample size
        test_size: Test set sample size
        random_state: Random seed

    Returns:
        X_train: Training set features
        y_train: Training set labels
        X_val: Validation set features
        y_val: Validation set labels
        X_test: Test set features
        y_test: Test set labels
    """
    train_path = os.path.join(dataset_path, "adult", "train.csv")
    test_path = os.path.join(dataset_path, "adult", "test.csv")

    # Set column names (according to Adult dataset standard format)
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    # Read CSV file, specify column names and missing values
    train_df = pd.read_csv(train_path, names=columns, na_values="?")
    test_df = pd.read_csv(test_path, names=columns, na_values="?")

    # Keep rows with missing features; handle missing values via imputation later.
    # Still drop rows with missing labels, if any.
    train_df = train_df.dropna(subset=["income"])
    test_df = test_df.dropna(subset=["income"])

    # Use smaller dataset for testing (randomly select samples)
    train_df = train_df.sample(
        n=train_size, random_state=random_state
    )  # Increase sample size for validation set split
    test_df = test_df.sample(n=test_size, random_state=random_state)

    # Split validation set from training set (90% train, 10% validation)
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=random_state, stratify=train_df["income"]
    )

    # Separate features and labels
    X_train = train_df.drop("income", axis=1)
    y_train = train_df["income"]
    X_val = val_df.drop("income", axis=1)
    y_val = val_df["income"]
    X_test = test_df.drop("income", axis=1)
    y_test = test_df["income"]

    # Clean labels: Remove periods from test set labels
    y_train = y_train.str.strip()
    y_val = y_val.str.strip()
    y_test = y_test.str.strip().str.rstrip(".")

    # Strict preprocessing for Euclidean-distance based methods:
    # - categorical cols: impute + one-hot (handle_unknown="ignore")
    # - numeric cols: impute + standardize
    categorical_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Coerce numeric columns (robust to odd header rows / parse issues).
    for part_df in (X_train, X_val, X_test):
        for col in numeric_cols:
            part_df[col] = pd.to_numeric(part_df[col], errors="coerce")
        for col in categorical_cols:
            part_df[col] = part_df[col].astype("string").str.strip()

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    def _one_hot_encoder():
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
                dtype=np.float32,
            )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X_train).astype(np.float32, copy=False)
    X_val = preprocessor.transform(X_val).astype(np.float32, copy=False)
    X_test = preprocessor.transform(X_test).astype(np.float32, copy=False)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_cifar10_dataset(
    train_size: int = 40000,
    test_size: int = 9000,
    random_state: int = 42,
    batch_size: int = 256,
    device: str | None = None,
):
    """Load CIFAR-10 dataset and return in consistent (train/val/test) format.

    Note: This implementation uses ImageNet pre-trained ResNet50 to extract 2048-dimensional
    embeddings for each image; first run may require downloading CIFAR-10 data and ResNet50 weights.

    Returns:
        X_train: (n_train, 2048) float32, ResNet50 embedding
        y_train: (n_train,) int
        X_val: (n_val, 2048) float32, ResNet50 embedding
        y_val: (n_val,) int
        X_test: (n_test, 2048) float32, ResNet50 embedding
        y_test: (n_test,) int
        X_train_img: (n_train, 3, 32, 32) uint8, original images
        X_val_img: (n_val, 3, 32, 32) uint8, original images
        X_test_img: (n_test, 3, 32, 32) uint8, original images
    """
    cache_dir = os.path.join(_embedding_cache_root(None), "cifar10_resnet50_full_v1")
    x_train_path = os.path.join(cache_dir, "X_train_full.npy")
    y_train_path = os.path.join(cache_dir, "y_train_full.npy")
    x_test_path = os.path.join(cache_dir, "X_test_full.npy")
    y_test_path = os.path.join(cache_dir, "y_test_full.npy")
    x_train_img_path = os.path.join(cache_dir, "X_train_img_full.npy")
    x_test_img_path = os.path.join(cache_dir, "X_test_img_full.npy")
    lock_path = os.path.join(cache_dir, ".lock")

    data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data_files", "cifar10")
    )

    have_cache = all(
        os.path.exists(p)
        for p in (
            x_train_path,
            y_train_path,
            x_test_path,
            y_test_path,
            x_train_img_path,
            x_test_img_path,
        )
    )
    # Older cache versions may store preprocessed tensors as "images" (e.g., 3x224x224).
    # Force regeneration so X_*_img remains true raw CIFAR-10 images (3x32x32 uint8).
    if have_cache:
        try:
            img_probe = np.load(x_train_img_path, mmap_mode="r")
            if img_probe.ndim != 4 or tuple(img_probe.shape[1:]) != (3, 32, 32):
                have_cache = False
        except Exception:
            have_cache = False
    if not have_cache:
        import torch
        import torchvision

        with _with_file_lock(lock_path):
            have_cache = all(
                os.path.exists(p)
                for p in (
                    x_train_path,
                    y_train_path,
                    x_test_path,
                    y_test_path,
                    x_train_img_path,
                    x_test_img_path,
                )
            )
            if have_cache:
                try:
                    img_probe = np.load(x_train_img_path, mmap_mode="r")
                    if img_probe.ndim != 4 or tuple(img_probe.shape[1:]) != (3, 32, 32):
                        have_cache = False
                except Exception:
                    have_cache = False
            if not have_cache:
                weights = torchvision.models.ResNet50_Weights.DEFAULT
                preprocess = weights.transforms()

                train_dataset = torchvision.datasets.CIFAR10(
                    root=data_root, train=True, download=True, transform=preprocess
                )
                test_dataset = torchvision.datasets.CIFAR10(
                    root=data_root, train=False, download=True, transform=preprocess
                )
                train_dataset_raw = torchvision.datasets.CIFAR10(
                    root=data_root, train=True, download=True, transform=None
                )
                test_dataset_raw = torchvision.datasets.CIFAR10(
                    root=data_root, train=False, download=True, transform=None
                )

                model = torchvision.models.resnet50(weights=weights)
                model.fc = torch.nn.Identity()  # (N, 2048)
                model.eval()

                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)

                def extract_full_embeddings(dataset):
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=str(device).startswith("cuda"),
                    )
                    X = np.empty((len(dataset), 2048), dtype=np.float32)
                    y = np.empty((len(dataset),), dtype=np.int64)
                    offset = 0
                    with torch.inference_mode():
                        for images, targets in loader:
                            images = images.to(device, non_blocking=True)
                            emb = model(images).detach().cpu().numpy()
                            emb = emb.astype(np.float32, copy=False)
                            bs = emb.shape[0]
                            X[offset : offset + bs] = emb
                            y[offset : offset + bs] = targets.detach().cpu().numpy()
                            offset += bs
                    return X, y

                def extract_full_images(dataset):
                    # torchvision CIFAR10 stores raw images in `dataset.data` as uint8 HWC.
                    return np.transpose(dataset.data, (0, 3, 1, 2)).astype(
                        np.uint8, copy=False
                    )

                X_train_full, y_train_full = extract_full_embeddings(train_dataset)
                X_test_full, y_test_full = extract_full_embeddings(test_dataset)

                X_train_img_full = extract_full_images(train_dataset_raw)
                X_test_img_full = extract_full_images(test_dataset_raw)

                _atomic_save_npy(x_train_path, X_train_full)
                _atomic_save_npy(y_train_path, y_train_full)
                _atomic_save_npy(x_test_path, X_test_full)
                _atomic_save_npy(y_test_path, y_test_full)
                _atomic_save_npy(x_train_img_path, X_train_img_full)
                _atomic_save_npy(x_test_img_path, X_test_img_full)

    X_train_full = np.load(x_train_path, mmap_mode="r")
    y_train_full = np.load(y_train_path, mmap_mode="r")
    X_test_full = np.load(x_test_path, mmap_mode="r")
    y_test_full = np.load(y_test_path, mmap_mode="r")
    X_train_img_full = np.load(x_train_img_path, mmap_mode="r")
    X_test_img_full = np.load(x_test_img_path, mmap_mode="r")

    rng = np.random.RandomState(random_state)

    train_n = min(train_size, len(X_train_full))
    test_n = min(test_size, len(X_test_full))

    train_idx = rng.choice(len(X_train_full), size=train_n, replace=False)
    test_idx = rng.choice(len(X_test_full), size=test_n, replace=False)

    X_train_all = np.asarray(X_train_full[train_idx]).astype(np.float32, copy=False)
    y_train_all = np.asarray(y_train_full[train_idx]).astype(np.int64, copy=False)
    X_train_img_all = np.asarray(X_train_img_full[train_idx])
    X_test_img = np.asarray(X_test_img_full[test_idx])
    X_test = np.asarray(X_test_full[test_idx]).astype(np.float32, copy=False)
    y_test = np.asarray(y_test_full[test_idx]).astype(np.int64, copy=False)

    # Split validation set from training subset (90% train, 10% validation)
    X_train, X_val, y_train, y_val, X_train_img, X_val_img = train_test_split(
        X_train_all,
        y_train_all,
        X_train_img_all,
        test_size=0.1,
        random_state=random_state,
        stratify=y_train_all,
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_img,
        X_val_img,
        X_test_img,
    )


def load_mnist_dataset(
    dataset_path: str,
    train_size: int = 1200,
    test_size: int = 500,
    random_state: int = 42,
    batch_size: int = 256,
    device: str | None = None,
):
    """Load MNIST dataset and return in consistent (train/val/test) format.

    Note: This implementation uses ImageNet pre-trained ResNet50 to extract 2048-dimensional
    embeddings for each image; first run may require downloading ResNet50 weights.

    Returns:
        X_train: (n_train, 2048) float32, ResNet50 embedding
        y_train: (n_train,) int
        X_val: (n_val, 2048) float32, ResNet50 embedding
        y_val: (n_val,) int
        X_test: (n_test, 2048) float32, ResNet50 embedding
        y_test: (n_test,) int
        X_train_img: (n_train, 1, 28, 28) uint8, original grayscale images
        X_val_img: (n_val, 1, 28, 28) uint8, original grayscale images
        X_test_img: (n_test, 1, 28, 28) uint8, original grayscale images
    """
    cache_dir = os.path.join(
        _embedding_cache_root(dataset_path), "mnist_resnet50_full_v1"
    )
    x_train_path = os.path.join(cache_dir, "X_train_full.npy")
    y_train_path = os.path.join(cache_dir, "y_train_full.npy")
    x_test_path = os.path.join(cache_dir, "X_test_full.npy")
    y_test_path = os.path.join(cache_dir, "y_test_full.npy")
    x_train_img_path = os.path.join(cache_dir, "X_train_img_full.npy")
    x_test_img_path = os.path.join(cache_dir, "X_test_img_full.npy")
    lock_path = os.path.join(cache_dir, ".lock")

    have_cache = all(
        os.path.exists(p)
        for p in (
            x_train_path,
            y_train_path,
            x_test_path,
            y_test_path,
            x_train_img_path,
            x_test_img_path,
        )
    )
    if have_cache:
        X_train_full = np.load(x_train_path, mmap_mode="r")
        y_train_full = np.load(y_train_path, mmap_mode="r")
        X_test_full = np.load(x_test_path, mmap_mode="r")
        y_test_full = np.load(y_test_path, mmap_mode="r")
        X_train_img_full = np.load(x_train_img_path, mmap_mode="r")
        X_test_img_full = np.load(x_test_img_path, mmap_mode="r")

        rng = np.random.RandomState(random_state)
        train_n = min(train_size, len(X_train_full))
        test_n = min(test_size, len(X_test_full))
        train_idx = rng.choice(len(X_train_full), size=train_n, replace=False)
        test_idx = rng.choice(len(X_test_full), size=test_n, replace=False)

        X_train_all = np.asarray(X_train_full[train_idx]).astype(np.float32, copy=False)
        y_train_all = np.asarray(y_train_full[train_idx]).astype(np.int64, copy=False)
        X_train_img_all = np.asarray(X_train_img_full[train_idx])
        X_test_img = np.asarray(X_test_img_full[test_idx])
        X_test = np.asarray(X_test_full[test_idx]).astype(np.float32, copy=False)
        y_test = np.asarray(y_test_full[test_idx]).astype(np.int64, copy=False)

        X_train, X_val, y_train, y_val, X_train_img, X_val_img = train_test_split(
            X_train_all,
            y_train_all,
            X_train_img_all,
            test_size=0.1,
            random_state=random_state,
            stratify=y_train_all,
        )
        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_train_img,
            X_val_img,
            X_test_img,
        )

    import torch
    import torchvision

    # Directly specify local MNIST file paths

    train_images_path = os.path.join(
        dataset_path, "MNIST", "raw", "train-images-idx3-ubyte"
    )
    train_labels_path = os.path.join(
        dataset_path, "MNIST", "raw", "train-labels-idx1-ubyte"
    )
    test_images_path = os.path.join(
        dataset_path, "MNIST", "raw", "t10k-images-idx3-ubyte"
    )
    test_labels_path = os.path.join(
        dataset_path, "MNIST", "raw", "t10k-labels-idx1-ubyte"
    )

    # Read MNIST binary files
    def read_mnist_images(filename):
        with open(filename, "rb") as f:
            import struct

            # Read file header
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read image data
            images = []
            for _ in range(num_images):
                img = f.read(rows * cols)
                img = torch.tensor(list(img), dtype=torch.uint8).view(28, 28)
                images.append(img)
            return images

    def read_mnist_labels(filename):
        with open(filename, "rb") as f:
            import struct

            # Read file header
            magic, num_labels = struct.unpack(">II", f.read(8))
            # Read label data
            labels = list(f.read(num_labels))
            return torch.tensor(labels, dtype=torch.long)

    # Read data
    train_images = read_mnist_images(train_images_path)
    train_labels = read_mnist_labels(train_labels_path)
    test_images = read_mnist_images(test_images_path)
    test_labels = read_mnist_labels(test_labels_path)

    # Define MNIST preprocessing: convert grayscale to RGB and apply ResNet50 preprocessing
    class MNISTTransform:
        def __init__(self):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet_transform = weights.transforms()

        def __call__(self, img):
            # Convert grayscale to RGB
            img = torchvision.transforms.functional.to_pil_image(img)
            img = torchvision.transforms.functional.rgb_to_grayscale(
                img, num_output_channels=3
            )
            # Apply ResNet50 preprocessing
            return self.resnet_transform(img)

    preprocess = MNISTTransform()

    # Create custom dataset
    class MNISTDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = self.images[idx]
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = MNISTDataset(train_images, train_labels, transform=preprocess)
    test_dataset = MNISTDataset(test_images, test_labels, transform=preprocess)

    train_dataset_raw = MNISTDataset(train_images, train_labels, transform=None)
    test_dataset_raw = MNISTDataset(test_images, test_labels, transform=None)

    with _with_file_lock(lock_path):
        have_cache = all(
            os.path.exists(p)
            for p in (
                x_train_path,
                y_train_path,
                x_test_path,
                y_test_path,
                x_train_img_path,
                x_test_img_path,
            )
        )
        if not have_cache:
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            )
            model.fc = torch.nn.Identity()  # (N, 2048)
            model.eval()

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            def extract_full_embeddings(dataset):
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=str(device).startswith("cuda"),
                )
                X = np.empty((len(dataset), 2048), dtype=np.float32)
                y = np.empty((len(dataset),), dtype=np.int64)
                offset = 0
                with torch.inference_mode():
                    for images, targets in loader:
                        images = images.to(device, non_blocking=True)
                        emb = model(images).detach().cpu().numpy()
                        emb = emb.astype(np.float32, copy=False)
                        bs = emb.shape[0]
                        X[offset : offset + bs] = emb
                        y[offset : offset + bs] = targets.detach().cpu().numpy()
                        offset += bs
                return X, y

            def extract_full_images(dataset):
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                X_img = []
                with torch.inference_mode():
                    for images, _ in loader:
                        X_img.append(images.unsqueeze(1).numpy())
                return np.concatenate(X_img, axis=0).astype(np.uint8, copy=False)

            X_train_full, y_train_full = extract_full_embeddings(train_dataset)
            X_test_full, y_test_full = extract_full_embeddings(test_dataset)

            X_train_img_full = extract_full_images(train_dataset_raw)
            X_test_img_full = extract_full_images(test_dataset_raw)

            _atomic_save_npy(x_train_path, X_train_full)
            _atomic_save_npy(y_train_path, y_train_full)
            _atomic_save_npy(x_test_path, X_test_full)
            _atomic_save_npy(y_test_path, y_test_full)
            _atomic_save_npy(x_train_img_path, X_train_img_full)
            _atomic_save_npy(x_test_img_path, X_test_img_full)

    X_train_full = np.load(x_train_path, mmap_mode="r")
    y_train_full = np.load(y_train_path, mmap_mode="r")
    X_test_full = np.load(x_test_path, mmap_mode="r")
    y_test_full = np.load(y_test_path, mmap_mode="r")
    X_train_img_full = np.load(x_train_img_path, mmap_mode="r")
    X_test_img_full = np.load(x_test_img_path, mmap_mode="r")

    rng = np.random.RandomState(random_state)
    train_n = min(train_size, len(X_train_full))
    test_n = min(test_size, len(X_test_full))
    train_idx = rng.choice(len(X_train_full), size=train_n, replace=False)
    test_idx = rng.choice(len(X_test_full), size=test_n, replace=False)

    X_train_all = np.asarray(X_train_full[train_idx]).astype(np.float32, copy=False)
    y_train_all = np.asarray(y_train_full[train_idx]).astype(np.int64, copy=False)
    X_train_img_all = np.asarray(X_train_img_full[train_idx])
    X_test_img = np.asarray(X_test_img_full[test_idx])
    X_test = np.asarray(X_test_full[test_idx]).astype(np.float32, copy=False)
    y_test = np.asarray(y_test_full[test_idx]).astype(np.int64, copy=False)

    X_train, X_val, y_train, y_val, X_train_img, X_val_img = train_test_split(
        X_train_all,
        y_train_all,
        X_train_img_all,
        test_size=0.1,
        random_state=random_state,
        stratify=y_train_all,
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_img,
        X_val_img,
        X_test_img,
    )


def load_covertype_dataset(
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
    val_ratio: float = 0.1,
):
    """Load Forest CoverType (Covertype) dataset.

    Data files typically come from UCI's covtype.data (in this repository as `covtype.data.gz`),
    each line has 55 columns:
    First 54 columns are features (10 continuous features + 4 wilderness one-hot + 40 soil one-hot),
    last 1 column is label (1~7).

    Args:
        dataset_path: Dataset root directory path (default in project is `../data_files`)
        train_size: Training+validation sample size (will be split by val_ratio)
        test_size: Test set sample size
        random_state: Random seed
        val_ratio: Validation set ratio (split from train_size)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    covtype_path = os.path.join(dataset_path, "covertype", "covtype.data.gz")

    df = pd.read_csv(covtype_path, header=None, compression="infer")
    if df.shape[1] != 55:
        raise ValueError(
            f"covertype data should have 55 columns (54 features + 1 label), but got {df.shape[1]} columns: {covtype_path}"
        )

    X = df.iloc[:, :-1].to_numpy(dtype=float, copy=False)
    y = df.iloc[:, -1].to_numpy()

    rng = np.random.RandomState(random_state)

    # First extract test set from the entire dataset (ensure no overlap with train/val)
    test_n = min(test_size, len(X))
    idx_all = np.arange(len(X))
    test_idx = rng.choice(idx_all, size=test_n, replace=False)
    train_val_idx = np.setdiff1d(idx_all, test_idx, assume_unique=False)

    X_test = X[test_idx]
    y_test = y[test_idx]

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]

    # Resample train_val to specified size (while maintaining class distribution)
    train_val_n = min(train_size, len(X_train_val))
    if train_val_n < len(X_train_val):
        X_train_val, _, y_train_val, _ = train_test_split(
            X_train_val,
            y_train_val,
            train_size=train_val_n,
            random_state=random_state,
            stratify=y_train_val,
        )

    # Split validation set from train_val subset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_higgs_dataset(
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
    val_ratio: float = 0.1,
    pool_size: int = 200_000,
):
    """Load HIGGS binary classification dataset.

    Data file: `data_files/higgs/HIGGS.csv.gz` (path in this repository).
    The original HIGGS data is very large (tens of millions of rows), so by default
    we only read the first `pool_size` rows as a sampling pool, then split into
    train/val/test (usually the HIGGS file itself is already shuffled, so the head subset
    can be treated as an approximately random sample).

    File format (UCI HIGGS standard format): each line has 29 columns
    - Column 1: label (0/1)
    - Columns 2-29: continuous features

    Args:
        dataset_path: Dataset root directory path (default in project is `../data_files`)
        train_size: Training+validation sample size (will be split by val_ratio)
        test_size: Test set sample size
        random_state: Random seed
        val_ratio: Validation set ratio (split from train_size)
        pool_size: Sampling pool size (upper limit of rows to read)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    higgs_path = os.path.join(dataset_path, "higgs", "HIGGS.csv.gz")

    pool_n = max(int(train_size + test_size), int(pool_size))
    df = pd.read_csv(higgs_path, header=None, compression="infer", nrows=pool_n)
    if df.shape[1] != 29:
        raise ValueError(
            f"HIGGS data should have 29 columns (1 label + 28 features), but got {df.shape[1]} columns: {higgs_path}"
        )

    y = df.iloc[:, 0].to_numpy()
    # Labels may be floating-point strings (e.g., 1.0/0.0), convert to int here
    y = y.astype(np.int64, copy=False)
    X = df.iloc[:, 1:].to_numpy(dtype=float, copy=False)

    # First split test set, then sample and split validation set within train_val to ensure no overlap
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=min(test_size, len(X)),
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    train_val_n = min(train_size, len(X_train_val))
    if train_val_n < len(X_train_val):
        X_train_val, _, y_train_val, _ = train_test_split(
            X_train_val,
            y_train_val,
            train_size=train_val_n,
            random_state=random_state,
            stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None,
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def load_ag_news_dataset(
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
    model_name: str = "bert-base-uncased",
    device: str = None,
):
    """Load AG_News dataset, generate BERT embeddings, return np.ndarray

    Each text takes the 768-dimensional embedding of the [CLS] token
    """
    _require_torch_transformers()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    max_length = 128
    cache_path = os.path.join(
        _embedding_cache_root(dataset_path),
        f"ag_news_bert_{_slug(model_name)}_ts{int(train_size)}_tes{int(test_size)}_rs{int(random_state)}_ml{max_length}_v3.npz",
    )
    lock_path = cache_path + ".lock"
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return (
            cached["X_train"],
            cached["y_train"],
            cached["X_val"],
            cached["y_val"],
            cached["X_test"],
            cached["y_test"],
        )

    train_path = os.path.join(dataset_path, "AG_News", "train.csv")
    test_path = os.path.join(dataset_path, "AG_News", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_cols = {"Class Index", "Title", "Description"}
    for df_name, df in [("train", train_df), ("test", test_df)]:
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(
                f"{df_name} CSV columns do not meet expectations, require {sorted(required_cols)};"
                f"actual columns={df.columns.tolist()}"
            )

    # Remove missing values
    train_df = train_df.dropna(subset=["Class Index", "Title", "Description"])
    test_df = test_df.dropna(subset=["Class Index", "Title", "Description"])

    # Sampling
    train_n = min(train_size, len(train_df))
    test_n = min(test_size, len(test_df))
    train_df = train_df.sample(n=train_n, random_state=random_state)
    test_df = test_df.sample(n=test_n, random_state=random_state)

    # train/val split
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=random_state,
        stratify=train_df["Class Index"],
    )

    # Normalize labels to 0~3
    y_train = train_df["Class Index"].to_numpy() - 1
    y_val = val_df["Class Index"].to_numpy() - 1
    y_test = test_df["Class Index"].to_numpy() - 1

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    def embed_texts(df: pd.DataFrame) -> np.ndarray:
        texts = (
            df["Title"].astype(str) + " " + df["Description"].astype(str)
        ).to_numpy()
        embs = []
        batch_size = 32
        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch_texts = texts[i : i + batch_size].tolist()
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embs.append(cls_embedding.cpu().numpy())
        return np.vstack(embs).astype(np.float32, copy=False)  # (N, 768)

    with _with_file_lock(lock_path):
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return (
                cached["X_train"],
                cached["y_train"],
                cached["X_val"],
                cached["y_val"],
                cached["X_test"],
                cached["y_test"],
            )

        X_train = embed_texts(train_df)
        X_val = embed_texts(val_df)
        X_test = embed_texts(test_df)

        _atomic_save_npz(
            cache_path,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=np.asarray(y_train).astype(np.int64, copy=False),
            X_val=X_val.astype(np.float32, copy=False),
            y_val=np.asarray(y_val).astype(np.int64, copy=False),
            X_test=X_test.astype(np.float32, copy=False),
            y_test=np.asarray(y_test).astype(np.int64, copy=False),
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_amazon_review_polarity_dataset(
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
    model_name: str = "bert-base-uncased",
    device: str = None,
):
    """Load Amazon Review Polarity dataset (single CSV file), generate BERT embeddings, return np.ndarray.

    CSV format: first line is header with 4 columns:
    - label: 1/2 (binary classification)
    - title: title
    - text: content
    - split: train/test

    Each text takes the 768-dimensional embedding of the [CLS] token; text is title + " " + text.
    """
    _require_torch_transformers()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    max_length = 128
    cache_path = os.path.join(
        _embedding_cache_root(dataset_path),
        f"amazon_review_polarity_bert_{_slug(model_name)}_ts{int(train_size)}_tes{int(test_size)}_rs{int(random_state)}_ml{max_length}_v2.npz",
    )
    lock_path = cache_path + ".lock"
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return (
            cached["X_train"],
            cached["y_train"],
            cached["X_val"],
            cached["y_val"],
            cached["X_test"],
            cached["y_test"],
        )

    csv_path = os.path.join(
        dataset_path, "amazon_review_polarity", "amazon_review_polarity_csv.csv"
    )
    df = pd.read_csv(csv_path)

    required_cols = {"label", "title", "text", "split"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"amazon_review_polarity CSV columns do not meet expectations, require {sorted(required_cols)};"
            f"actual columns={df.columns.tolist()}"
        )

    df = df.dropna(subset=["label", "title", "text", "split"])
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(np.int64, copy=False)
    df = df[df["split"].isin(["train", "test"])]

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"amazon_review_polarity must contain both train/test split;"
            f"current train={len(train_df)}, test={len(test_df)}: {csv_path}"
        )

    # label only allows 1/2 (will be uniformly converted to 0/1 later)
    for df_name, part_df in [("train", train_df), ("test", test_df)]:
        unique_labels = set(pd.unique(part_df["label"]))
        if not unique_labels.issubset({1, 2}):
            raise ValueError(
                f"{df_name} label can only contain 1/2, but got {sorted(unique_labels)}: {csv_path}"
            )

    # Sampling
    train_n = min(train_size, len(train_df))
    test_n = min(test_size, len(test_df))
    train_df = train_df.sample(n=train_n, random_state=random_state)
    test_df = test_df.sample(n=test_n, random_state=random_state)

    # train/val split (try to maintain class distribution)
    stratify = None
    if train_df["label"].nunique() > 1:
        min_count = train_df["label"].value_counts().min()
        if min_count >= 2:
            stratify = train_df["label"]

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=random_state,
        stratify=stratify,
    )

    y_train = (train_df["label"].to_numpy() - 1).astype(np.int64, copy=False)
    y_val = (val_df["label"].to_numpy() - 1).astype(np.int64, copy=False)
    y_test = (test_df["label"].to_numpy() - 1).astype(np.int64, copy=False)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    def embed_texts(part_df: pd.DataFrame) -> np.ndarray:
        texts = (
            part_df["title"].astype(str) + " " + part_df["text"].astype(str)
        ).to_numpy()
        embs = []
        batch_size = 32
        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch_texts = texts[i : i + batch_size].tolist()
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
                embs.append(cls_embedding.cpu().numpy())
        return np.vstack(embs).astype(np.float32, copy=False)

    with _with_file_lock(lock_path):
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return (
                cached["X_train"],
                cached["y_train"],
                cached["X_val"],
                cached["y_val"],
                cached["X_test"],
                cached["y_test"],
            )

        X_train = embed_texts(train_df)
        X_val = embed_texts(val_df)
        X_test = embed_texts(test_df)

        _atomic_save_npz(
            cache_path,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=np.asarray(y_train).astype(np.int64, copy=False),
            X_val=X_val.astype(np.float32, copy=False),
            y_val=np.asarray(y_val).astype(np.int64, copy=False),
            X_test=X_test.astype(np.float32, copy=False),
            y_test=np.asarray(y_test).astype(np.int64, copy=False),
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dbpedia_ontology_dataset(
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
    model_name: str = "bert-base-uncased",
    device: str = None,
):
    """Load DBpedia_Ontology dataset, generate BERT embeddings, return np.ndarray

    Each text takes the 768-dimensional embedding of the [CLS] token.
    DBpedia_Ontology CSV has no header, containing 3 columns: class index (1~14), title, content.
    """
    _require_torch_transformers()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    max_length = 128
    cache_path = os.path.join(
        _embedding_cache_root(dataset_path),
        f"dbpedia_ontology_bert_{_slug(model_name)}_ts{int(train_size)}_tes{int(test_size)}_rs{int(random_state)}_ml{max_length}_v3.npz",
    )
    lock_path = cache_path + ".lock"
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return (
            cached["X_train"],
            cached["y_train"],
            cached["X_val"],
            cached["y_val"],
            cached["X_test"],
            cached["y_test"],
        )

    train_path = os.path.join(dataset_path, "DBpedia_Ontology", "train.csv")
    test_path = os.path.join(dataset_path, "DBpedia_Ontology", "test.csv")

    col_names = ["Class Index", "Title", "Content"]
    train_df = pd.read_csv(train_path, header=None, names=col_names)
    test_df = pd.read_csv(test_path, header=None, names=col_names)

    required_cols = {"Class Index", "Title", "Content"}
    for df_name, df in [("train", train_df), ("test", test_df)]:
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(
                f"{df_name} CSV columns do not meet expectations, require {sorted(required_cols)};"
                f"actual columns={df.columns.tolist()}"
            )

    # Remove missing values
    train_df = train_df.dropna(subset=["Class Index", "Title", "Content"])
    test_df = test_df.dropna(subset=["Class Index", "Title", "Content"])

    # Sampling
    train_n = min(train_size, len(train_df))
    test_n = min(test_size, len(test_df))
    train_df = train_df.sample(n=train_n, random_state=random_state)
    test_df = test_df.sample(n=test_n, random_state=random_state)

    # train/val split (try to maintain class distribution)
    stratify = None
    if train_df["Class Index"].nunique() > 1:
        min_count = train_df["Class Index"].value_counts().min()
        if min_count >= 2:
            stratify = train_df["Class Index"]

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=random_state,
        stratify=stratify,
    )

    # Normalize labels to 0~13
    y_train = (train_df["Class Index"].to_numpy() - 1).astype(np.int64, copy=False)
    y_val = (val_df["Class Index"].to_numpy() - 1).astype(np.int64, copy=False)
    y_test = (test_df["Class Index"].to_numpy() - 1).astype(np.int64, copy=False)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    def embed_texts(df: pd.DataFrame) -> np.ndarray:
        texts = (df["Title"].astype(str) + " " + df["Content"].astype(str)).to_numpy()
        embs = []
        batch_size = 32
        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch_texts = texts[i : i + batch_size].tolist()
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
                embs.append(cls_embedding.cpu().numpy())
        return np.vstack(embs).astype(np.float32, copy=False)  # (N, 768)

    with _with_file_lock(lock_path):
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return (
                cached["X_train"],
                cached["y_train"],
                cached["X_val"],
                cached["y_val"],
                cached["X_test"],
                cached["y_test"],
            )

        X_train = embed_texts(train_df)
        X_val = embed_texts(val_df)
        X_test = embed_texts(test_df)

        _atomic_save_npz(
            cache_path,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=np.asarray(y_train).astype(np.int64, copy=False),
            X_val=X_val.astype(np.float32, copy=False),
            y_val=np.asarray(y_val).astype(np.int64, copy=False),
            X_test=X_test.astype(np.float32, copy=False),
            y_test=np.asarray(y_test).astype(np.int64, copy=False),
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_imdb_dataset(
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
    model_name: str = "bert-base-uncased",
    device: str | None = None,
    max_length: int = 256,
    batch_size: int = 32,
):
    """Load IMDB Dataset (single CSV file), generate BERT embeddings, return np.ndarray.

    File: `data_files/IMDB/IMDB Dataset.csv`
    - Column 1: review (text)
    - Column 2: sentiment (positive/negative), mapped to 1/0

    Each text takes the 768-dimensional embedding of the [CLS] token.
    """
    _require_torch_transformers()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = os.path.join(
        _embedding_cache_root(dataset_path),
        f"imdb_bert_{_slug(model_name)}_ts{int(train_size)}_tes{int(test_size)}_rs{int(random_state)}_ml{int(max_length)}_bs{int(batch_size)}_v4.npz",
    )
    lock_path = cache_path + ".lock"
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return (
            cached["X_train"],
            cached["y_train"],
            cached["X_val"],
            cached["y_val"],
            cached["X_test"],
            cached["y_test"],
        )

    csv_path = os.path.join(dataset_path, "IMDB", "IMDB Dataset.csv")
    df = pd.read_csv(csv_path)

    # Compatibility with header/no header cases: prioritize standard column names, otherwise use the first two columns
    if {"review", "sentiment"}.issubset(set(df.columns)):
        text_col = "review"
        label_col = "sentiment"
    else:
        if df.shape[1] < 2:
            raise ValueError(
                f"IMDB CSV must have at least two columns (text/label), but got {df.shape[1]} columns: {csv_path}"
            )
        text_col = df.columns[0]
        label_col = df.columns[1]

    df = df.dropna(subset=[text_col, label_col])

    labels_raw = df[label_col]
    # Support string sentiment labels and 0/1 numeric labels
    labels_num = pd.to_numeric(labels_raw, errors="coerce")
    if labels_num.notna().all():
        y_series = labels_num.astype(np.int64, copy=False)
    else:
        s = labels_raw.astype(str).str.strip().str.lower()
        mapping = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
        y_series = s.map(mapping)

    if y_series.isna().any():
        bad = df[y_series.isna()][label_col].astype(str).head(5).tolist()
        raise ValueError(
            f"IMDB label cannot be parsed (expected positive/negative or 0/1), examples={bad}: {csv_path}"
        )

    y_series = y_series.astype(np.int64, copy=False)
    y = y_series.to_numpy()
    unique_y = set(np.unique(y).tolist())
    if not unique_y.issubset({0, 1}):
        raise ValueError(
            f"IMDB label only allows 0/1, but got {sorted(unique_y)}: {csv_path}"
        )

    # Sample a pool, then split into train/val/test to avoid overlap between them
    pool_n = min(int(train_size + test_size), len(df))
    if pool_n < 3:
        raise ValueError(
            f"IMDB dataset is too small (pool_n={pool_n}), cannot split into train/val/test: {csv_path}"
        )

    df = df.sample(n=pool_n, random_state=random_state)
    texts = df[text_col].astype(str).to_numpy()
    y = y_series.loc[df.index].to_numpy()

    # First split test, then split val within train_val
    stratify = None
    if len(np.unique(y)) > 1:
        counts = np.bincount(y, minlength=2)
        if counts.min() >= 2:
            stratify = y

    test_n = min(int(test_size), len(texts) - 2)
    if test_n <= 0:
        raise ValueError(
            f"IMDB test_size is too large, resulting in empty train_val (test_size={test_size}, pool_n={pool_n}): {csv_path}"
        )

    X_train_val_text, X_test_text, y_train_val, y_test = train_test_split(
        texts,
        y,
        test_size=test_n,
        random_state=random_state,
        stratify=stratify,
    )

    stratify_tv = None
    if len(np.unique(y_train_val)) > 1:
        counts_tv = np.bincount(y_train_val, minlength=2)
        if counts_tv.min() >= 2:
            stratify_tv = y_train_val

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_val_text,
        y_train_val,
        test_size=0.1,
        random_state=random_state,
        stratify=stratify_tv,
    )

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    def embed_texts(text_arr: np.ndarray) -> np.ndarray:
        embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(text_arr), batch_size), desc="Embedding texts"):
                batch_texts = text_arr[i : i + batch_size].tolist()
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
                embs.append(cls_emb.cpu().numpy())
        return np.vstack(embs).astype(np.float32, copy=False)

    with _with_file_lock(lock_path):
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return (
                cached["X_train"],
                cached["y_train"],
                cached["X_val"],
                cached["y_val"],
                cached["X_test"],
                cached["y_test"],
            )

        X_train = embed_texts(X_train_text)
        X_val = embed_texts(X_val_text)
        X_test = embed_texts(X_test_text)

        _atomic_save_npz(
            cache_path,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=np.asarray(y_train).astype(np.int64, copy=False),
            X_val=X_val.astype(np.float32, copy=False),
            y_val=np.asarray(y_val).astype(np.int64, copy=False),
            X_test=X_test.astype(np.float32, copy=False),
            y_test=np.asarray(y_test).astype(np.int64, copy=False),
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_skin_nonskin_dataset(
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
    val_ratio: float = 0.1,
):
    """Load Skin/NonSkin dataset (libsvm format) and return (train/val/test).

    File: `data_files/skin_nonskin/skin_nonskin.txt`
    - Column 1: label
    - Other columns: features in format `1:74 2:85 3:123`
    """
    file_path = os.path.join(dataset_path, "skin_nonskin", "skin_nonskin.txt")

    X_sparse, y = load_svmlight_file(file_path)
    X = X_sparse.toarray().astype(np.float32, copy=False)
    y = np.asarray(y).astype(np.int64, copy=False)

    stratify = None
    if len(np.unique(y)) > 1:
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 2:
            stratify = y

    # First split test set, then sample and split validation set within train_val to ensure no overlap
    test_n = min(int(test_size), len(X) - 2)
    if test_n <= 0:
        raise ValueError(
            f"skin_nonskin test_size is too large, resulting in empty train_val (test_size={test_size}, n={len(X)}): {file_path}"
        )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_n,
        random_state=random_state,
        stratify=stratify,
    )

    train_val_n = min(int(train_size), len(X_train_val))
    if train_val_n < len(X_train_val):
        stratify_tv = None
        if len(np.unique(y_train_val)) > 1:
            _, counts_tv = np.unique(y_train_val, return_counts=True)
            if counts_tv.min() >= 2:
                stratify_tv = y_train_val

        X_train_val, _, y_train_val, _ = train_test_split(
            X_train_val,
            y_train_val,
            train_size=train_val_n,
            random_state=random_state,
            stratify=stratify_tv,
        )

    stratify_tv2 = None
    if len(np.unique(y_train_val)) > 1:
        _, counts_tv2 = np.unique(y_train_val, return_counts=True)
        if counts_tv2.min() >= 2:
            stratify_tv2 = y_train_val

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_tv2,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_creditcard_dataset(
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
    val_ratio: float = 0.1,
):
    """Load Credit Card Fraud dataset and return (train/val/test).

    File: `data_files/creditcard.csv`
    - Columns: Time, V1-V28, Amount, Class
    - We use only V1-V28 as features and Class as label
    - Class: 0 (normal), 1 (fraud)
    """
    file_path = os.path.join(dataset_path, "creditcard.csv")

    df = pd.read_csv(file_path)

    feature_cols = [f"V{i}" for i in range(1, 29)]
    X = df[feature_cols].values.astype(np.float32, copy=False)
    y = df["Class"].values.astype(np.int64, copy=False)

    stratify = None
    if len(np.unique(y)) > 1:
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 2:
            stratify = y

    test_n = min(int(test_size), len(X) - 2)
    if test_n <= 0:
        raise ValueError(
            f"creditcard test_size is too large, resulting in empty train_val (test_size={test_size}, n={len(X)}): {file_path}"
        )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_n,
        random_state=random_state,
        stratify=stratify,
    )

    train_val_n = min(int(train_size), len(X_train_val))
    if train_val_n < len(X_train_val):
        stratify_tv = None
        if len(np.unique(y_train_val)) > 1:
            _, counts_tv = np.unique(y_train_val, return_counts=True)
            if counts_tv.min() >= 2:
                stratify_tv = y_train_val

        X_train_val, _, y_train_val, _ = train_test_split(
            X_train_val,
            y_train_val,
            train_size=train_val_n,
            random_state=random_state,
            stratify=stratify_tv,
        )

    stratify_tv2 = None
    if len(np.unique(y_train_val)) > 1:
        _, counts_tv2 = np.unique(y_train_val, return_counts=True)
        if counts_tv2.min() >= 2:
            stratify_tv2 = y_train_val

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_tv2,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_api(
    dataset_name: str,
    dataset_path: str,
    train_size: int = 12000,
    test_size: int = 500,
    random_state: int = 42,
):
    if dataset_name == "adult":
        X_train, y_train, X_val, y_val, X_test, y_test = load_adult_dataset(
            dataset_path, train_size, test_size, random_state
        )
    elif dataset_name == "cifar10-embedding":
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_train_img,
            X_val_img,
            X_test_img,
        ) = load_cifar10_dataset(train_size, test_size, random_state)
        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_train_img,
            X_val_img,
            X_test_img,
        )
    elif dataset_name == "mnist-embedding":
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_train_img,
            X_val_img,
            X_test_img,
        ) = load_mnist_dataset(dataset_path, train_size, test_size, random_state)
        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_train_img,
            X_val_img,
            X_test_img,
        )
    elif dataset_name == "covertype":
        X_train, y_train, X_val, y_val, X_test, y_test = load_covertype_dataset(
            dataset_path, train_size, test_size, random_state
        )
    elif dataset_name == "higgs":
        X_train, y_train, X_val, y_val, X_test, y_test = load_higgs_dataset(
            dataset_path, train_size, test_size, random_state
        )
    elif dataset_name == "dbpedia-embedding":
        X_train, y_train, X_val, y_val, X_test, y_test = load_dbpedia_ontology_dataset(
            dataset_path, train_size, test_size, random_state
        )
    elif dataset_name == "ag-news-embedding":
        X_train, y_train, X_val, y_val, X_test, y_test = load_ag_news_dataset(
            dataset_path, train_size, test_size, random_state
        )
    elif dataset_name == "amazon-review-polarity-embedding":
        X_train, y_train, X_val, y_val, X_test, y_test = (
            load_amazon_review_polarity_dataset(
                dataset_path, train_size, test_size, random_state
            )
        )
    elif dataset_name == "imdb-embedding":
        X_train, y_train, X_val, y_val, X_test, y_test = load_imdb_dataset(
            dataset_path, train_size, test_size, random_state
        )
    elif dataset_name == "skin-nonskin":
        X_train, y_train, X_val, y_val, X_test, y_test = load_skin_nonskin_dataset(
            dataset_path, train_size, test_size, random_state
        )
    elif dataset_name == "creditcard":
        X_train, y_train, X_val, y_val, X_test, y_test = load_creditcard_dataset(
            dataset_path, train_size, test_size, random_state
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def inject_noise(
    X,
    y,
    noise_type="label",  # "label" or "feature"
    noise_rate=0.1,  # 噪声比例
    feature_std=0.1,  # feature noise 强度（基于标准化后的尺度）
    random_state=42,
    class_conditional=False,  # 是否只对某些类加 noise
    target_classes=None,  # 指定哪些类被污染（如 [0] 或 [1,2]）
):
    """
    通用 noise 注入函数（适用于数据估值实验）

    参数：
    - X: (n, d) 特征（建议已标准化）
    - y: (n,)
    - noise_type: "label" or "feature"
    - noise_rate: 噪声比例
    - feature_std: 高斯噪声标准差（feature noise）
    - class_conditional: 是否只对某些类加噪声
    - target_classes: 被污染的类别列表

    返回：
    - X_noisy, y_noisy
    """

    rng = np.random.RandomState(random_state)

    X_noisy = X.copy()
    y_noisy = y.copy()

    n = len(y)

    # ===== 选择要加 noise 的 index =====
    if class_conditional and target_classes is not None:
        mask = np.isin(y, target_classes)
        candidate_idx = np.where(mask)[0]
    else:
        candidate_idx = np.arange(n)

    n_noisy = int(len(candidate_idx) * noise_rate)
    noisy_idx = rng.choice(candidate_idx, size=n_noisy, replace=False)

    # ===== 1. Label noise =====
    if noise_type == "label":
        classes = np.unique(y)

        for idx in noisy_idx:
            current_label = y_noisy[idx]
            other_classes = classes[classes != current_label]
            y_noisy[idx] = rng.choice(other_classes)

    # ===== 2. Feature noise =====
    elif noise_type == "feature":
        noise = rng.normal(0, feature_std, size=X_noisy[noisy_idx].shape)
        X_noisy[noisy_idx] += noise

    else:
        raise ValueError("noise_type must be 'label' or 'feature'")

    return X_noisy, y_noisy


def preprocess_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    scaler=None,
    label_encoder=None,
    dataset_name: str = None,
    random_state: int = 42,
):
    """Data preprocessing

    Args:
        X_train: Training set features
        y_train: Training set labels
        X_val: Validation set features
        y_val: Validation set labels
        X_test: Test set features
        y_test: Test set labels
        scaler: Standard scaler, create new if None
        label_encoder: Label encoder, create new if None

    Returns:
        X_train_scaled: Standardized training set features
        y_train_encoded: Encoded training set labels
        X_val_scaled: Standardized validation set features
        y_val_encoded: Encoded validation set labels
        X_test_scaled: Standardized test set features
        y_test_encoded: Encoded test set labels
        scaler: Fitted standard scaler
        label_encoder: Fitted label encoder
    """

    def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        X = np.asarray(X)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return (X / norms).astype(np.float32, copy=False)

    if dataset_name == "adult":
        mode = "none"
    elif dataset_name in ["covertype", "higgs", "skin-nonskin", "creditcard"]:
        mode = "standard"
    elif dataset_name in [
        "cifar10-embedding",
        "mnist-embedding",
        "dbpedia-embedding",
        "ag-news-embedding",
        "amazon-review-polarity-embedding",
        "imdb-embedding",
    ]:
        mode = "l2"
    else:
        mode = None

    if mode is None:
        mode = "standard"

    if mode == "standard":
        if scaler is None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    elif mode == "l2":
        scaler = None
        X_train_scaled = _l2_normalize_rows(X_train)
        X_val_scaled = _l2_normalize_rows(X_val)
        X_test_scaled = _l2_normalize_rows(X_test)
    elif mode == "none":
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test
    else:
        raise ValueError(f"Unknown x_preprocess mode: {mode!r}")

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
    else:
        y_train_encoded = label_encoder.transform(y_train)

    # Encode validation and test set labels
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    X_train_scaled, y_train_encoded = inject_noise(
        X_train_scaled,
        y_train_encoded,
        noise_type="label",  # "label" or "feature"
        noise_rate=0.2,  # 噪声比例
        random_state=random_state,
    )

    return (
        X_train_scaled,
        y_train_encoded,
        X_val_scaled,
        y_val_encoded,
        X_test_scaled,
        y_test_encoded,
        scaler,
        label_encoder,
    )