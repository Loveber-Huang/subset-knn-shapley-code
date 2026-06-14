from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Dict, Any
from load_dataset import *
from cnn import train_and_evaluate
import torch
import os
import sys
from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from pydvl.influence.torch import CgInfluence, DirectInfluence, SecondOrderMode

# Add algs directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../algs"))

from algs.conditional_shapley import cod_subshap, build_faiss_index
from algs.baseline import knn_shapley_JW
import random
import numpy as np


# Set global random seed
def set_seed(seed=42):
    """Set random seeds for all relevant libraries"""
    # Python built-in random number generator
    random.seed(seed)

    # NumPy random number generator
    np.random.seed(seed)

    # PyTorch random number generator
    torch.manual_seed(seed)

    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU cases

        # Ensure CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DataPointSelectionTrainer:
    """Data point selection trainer class"""

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = "adult",
        model_type: str = "random-forest",
        k: int = 5,
        train_size: int = 1200,
        test_size: int = 500,
        seed: int = 42,
    ):
        """
        Initialize trainer

        Args:
            dataset_path: Dataset path
            dataset_name: Dataset name
            model_type: Model type
            k: KNN parameter
            train_size: Training set sample size
            test_size: Test set sample size
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.k = k
        self.train_size = train_size
        self.test_size = test_size
        self.seed = seed

        # Data-related attributes
        self.X_train = None
        self.y_train = None
        self.X_val = None  # Validation set
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.train_img = None
        self.val_img = None
        self.test_img = None
        self.X_scaled = None
        self.y_encoded = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Model-related attributes
        self.model = None

        # conditional_shapley-related attributes
        self.conditional_shapley_indices = []  # Indices sorted by value

        # Approx Conditional Shapley-related attributes
        self.approx_conditional_shapley_indices = []  # Indices sorted by value

        # ann Approx Conditional Shapley-related attributes
        self.ann_approx_conditional_shapley_indices = []  # Indices sorted by value

        # KNN-Shapley-related attributes
        self.knn_shapley_values = None
        self.knn_shapley_indices = []  # Indices sorted by value

        # Influence Function-related attributes
        self.influence_function_values = None
        self.influence_function_indices = []  # Indices sorted by value

        # Random-related attributes
        self.random_indices = []  # Indices in random order

    def load_dataset(self) -> None:
        """Load and preprocess dataset"""
        print(f"Loading dataset: {self.dataset_name}")
        if (
            self.dataset_name == "mnist-embedding"
            or self.dataset_name == "cifar10-embedding"
        ):
            (
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                self.X_test,
                self.y_test,
                self.train_img,
                self.val_img,
                self.test_img,
            ) = load_api(
                self.dataset_name,
                self.dataset_path,
                self.train_size,
                self.test_size,
                random_state=self.seed,
            )
        else:
            (
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                self.X_test,
                self.y_test,
            ) = load_api(
                self.dataset_name,
                self.dataset_path,
                self.train_size,
                self.test_size,
                random_state=self.seed,
            )

        # Data preprocessing
        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            self.scaler,
            self.label_encoder,
        ) = preprocess_data(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            scaler=None,
            label_encoder=None,
            dataset_name=self.dataset_name,
            random_state=self.seed,
        )

        print(
            f"Dataset loaded: training set size={len(self.X_train)}, test set size={len(self.X_test)}"
        )

    def _create_model(self):
        """Create model instance based on model_type

        Returns:
            Created model instance
        """
        if self.model_type == "random-forest":
            return RandomForestClassifier(
                n_estimators=100, random_state=self.seed, max_depth=10
            )
        elif self.model_type == "logistic-regression":
            return LogisticRegression(random_state=self.seed, max_iter=1000)
        elif self.model_type == "svm":
            return SVC(random_state=self.seed, kernel="rbf", probability=True)
        elif self.model_type == "gradient-boosting":
            return GradientBoostingClassifier(
                n_estimators=100, random_state=self.seed, max_depth=3
            )
        elif self.model_type == "decision-tree":
            return DecisionTreeClassifier(random_state=self.seed, max_depth=10)
        elif self.model_type == "cnn":
            return 1
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def initialize_model(self) -> None:
        """Initialize classification model"""
        self.model = self._create_model()
        print(f"Model initialized: {self.model_type}")

    def compute_knn_shapley_values(self) -> tuple[float, float]:
        print("Computing KNN-Shapley values...")

        # Ensure data is loaded
        if self.X_train is None or self.X_val is None:
            raise ValueError("Data not loaded, please call load_dataset method first")

        try:
            wall_start_time = time.time()
            cpu_start_time = time.process_time()
            self.knn_shapley_values = knn_shapley_JW(
                self.X_train, self.y_train, self.X_val, self.y_val, self.k
            )
            cpu_end_time = time.process_time()
            wall_end_time = time.time()
        except Exception as e:
            raise RuntimeError(f"KNN-Shapley computation failed: {e}") from e

        # Sort indices by value in descending order
        self.knn_shapley_indices = np.argsort(self.knn_shapley_values)[::-1]
        return wall_end_time - wall_start_time, cpu_end_time - cpu_start_time


    def compute_random_indices(self) -> tuple[float, float]:
        print("Generating random indices...")

        # Ensure data is loaded
        if self.X_train is None:
            raise ValueError("Data not loaded, please call load_dataset method first")

        try:
            set_seed(self.seed)
            wall_start_time = time.time()
            cpu_start_time = time.process_time()
            # Generate random permutation of indices
            self.random_indices = np.random.permutation(len(self.X_train))
            cpu_end_time = time.process_time()
            wall_end_time = time.time()
        except Exception as e:
            raise RuntimeError(f"Random indices generation failed: {e}") from e

        return wall_end_time - wall_start_time, cpu_end_time - cpu_start_time

    def select_point(
        self, b, F_indices=None, n_top=None, ann_index=None, ann_N=None
    ) -> tuple[float, float]:
        """

        Args:
            F_indices: List of already selected data point indices

        Returns:
            Index of newly selected data points
        """
        # Use conditional_shapley function
        wall_start_time = time.time()
        cpu_start_time = time.process_time()
        phi, T_indices = cod_subshap(
            D_features=self.X_train,
            D_labels=self.y_train,
            D_test_features=self.X_val,
            D_test_labels=self.y_val,
            k=self.k,
            b=b,
            F_indices=F_indices,
            n_top=n_top,
            ann_index=ann_index,
            ann_N=ann_N,
        )
        cpu_end_time = time.process_time()
        wall_end_time = time.time()

        if n_top is None:
            self.conditional_shapley_indices = T_indices
        else:
            if ann_N is None:
                self.approx_conditional_shapley_indices = T_indices
            else:
                self.ann_approx_conditional_shapley_indices = T_indices
        return wall_end_time - wall_start_time, cpu_end_time - cpu_start_time

    def _run_training_sequence(
        self,
        method_name: str,
        selected_indices: List[int],
        results: Dict[str, Any],
        new_index,
        step,
    ) -> None:
        """
        Generic training sequence function for running training processes with different data point selection methods

        Args:
            method_name: Method name
            selected_indices: List of selected indices
            results: Results dictionary
            new_index: Newly selected index
            step: Current step
        """
        # Create new model instance
        model = self._create_model()

        # Train and evaluate model
        X_selected = self.X_train[selected_indices]
        y_selected = self.y_train[selected_indices]

        # Check if training data meets model requirements
        unique_classes = np.unique(y_selected)

        # Special handling: when there are too few data points or insufficient classes
        if len(selected_indices) <= 1:
            # Only one data point, most models cannot be trained
            accuracy = 0.0
            print(
                f"  Warning: less than 2 data point, skipping model training, accuracy set to 0.0"
            )
        elif len(unique_classes) < 2:
            # Only one class, some models cannot be trained
            if self.model_type in ["logistic-regression", "svm", "gradient-boosting"]:
                accuracy = 0.0
                print(
                    f"  Warning: Only {len(unique_classes)} class, {self.model_type} cannot be trained, accuracy set to 0.0"
                )
            elif self.model_type in ["cnn"]:
                if self.dataset_name == "mnist-embedding":
                    in_channels = 1
                elif self.dataset_name == "cifar10-embedding":
                    in_channels = 3
                else:
                    raise ValueError(f"Unknown dataset name: {self.dataset_name}")
                accuracy = train_and_evaluate(
                    X_train_all=self.train_img,
                    y_train_all=self.y_train,
                    train_idx=selected_indices,
                    X_test=self.test_img,
                    y_test=self.y_test,
                    epochs=15,
                    in_channels=in_channels,
                    seed=self.seed,
                )
            else:
                # Other models can be trained but may have poor performance
                model.fit(X_selected, y_selected)
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
        else:
            if self.model_type in ["cnn"]:
                if self.dataset_name == "mnist-embedding":
                    in_channels = 1
                elif self.dataset_name == "cifar10-embedding":
                    in_channels = 3
                else:
                    raise ValueError(f"Unknown dataset name: {self.dataset_name}")
                accuracy = train_and_evaluate(
                    X_train_all=self.train_img,
                    y_train_all=self.y_train,
                    train_idx=selected_indices,
                    X_test=self.test_img,
                    y_test=self.y_test,
                    epochs=15,
                    in_channels=in_channels,
                    seed=self.seed,
                )
            else:
                # Normal training (using validation set for performance monitoring)
                model.fit(X_selected, y_selected)
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)

        results[method_name]["selected_indices"] = selected_indices.copy()
        results[method_name]["accuracies"].append(accuracy)
        results[method_name]["training_history"].append(
            {
                "step": step + 1,
                "selected_index": new_index,
                "accuracy": accuracy,
                "num_points": len(selected_indices),
            }
        )

        print(f"Selected data point {new_index}, accuracy: {accuracy:.4f}")

    def _run_method(
        self,
        method_name,
        description,
        max_points,
        results,
        index_getter,
        need_compute=False,
        compute_func=None,
        F=None,
    ):
        """
        Generic method running function for abstracting repetitive data point selection training logic

        Args:
            method_name: Method name, corresponding to key in results dictionary
            description: Method description for printing
            max_points: Maximum number of data points
            results: Results dictionary
            index_getter: Function that receives step and selected indices list, returns next index
            need_compute: Whether pre-computation is needed
            compute_func: Corresponding computation function if needed
            F: List of already selected data point indices, default is None
        """
        print(f"\n=== {description} ===")

        if need_compute and compute_func:
            print(f"Starting to compute {description} values...")
            wall_time, cpu_time = compute_func()
            results[method_name]["wall_time"].append(wall_time)
            results[method_name]["cpu_time"].append(cpu_time)
            print(f"{description} values computed, starting model training...")

        selected_indices = F.copy() if F else []
        self._run_training_sequence(
            method_name,
            selected_indices,
            results,
            -1,
            len(selected_indices) - (len(F) if F else 0) - 1,
        )
        step = 0
        while len(selected_indices) < max_points + (len(F) if F else 0):
            # Get next index
            new_index = index_getter(step, selected_indices)
            print(self.y_train[new_index])
            step += 1
            # If None is returned, no more data points
            if new_index is None:
                break
            if new_index in selected_indices:
                continue
            print(
                f"{description} - Step {len(selected_indices) - (len(F) if F else 0) + 1}/{max_points}"
            )
            selected_indices.append(new_index)

            # Train and record results
            self._run_training_sequence(
                method_name,
                selected_indices,
                results,
                new_index,
                len(selected_indices) - (len(F) if F else 0) - 1,
            )

        return selected_indices

    def run_sequential_comparison(
        self, max_points: int = 100, method: str = "all", F: list = None
    ) -> Dict[str, Any]:
        """

        Args:
            max_points: Maximum number of data points
            method: Method to run, default is all
            F: List of already selected data point indices, default is None

        Returns:
            Comparison training results
        """
        print(f"Running method: {method}")
        print("Starting sequential comparison training...")

        # Initialize result records
        results = {
            "conditional_shapley": {
                "selected_indices": [],
                "accuracies": [],
                "training_history": [],
                "wall_time": [],
                "cpu_time": [],
                "is_run": False,
            },
            "approx_conditional_shapley": {
                "selected_indices": [],
                "accuracies": [],
                "training_history": [],
                "wall_time": [],
                "cpu_time": [],
                "is_run": False,
            },
            "ann_approx_conditional_shapley": {
                "selected_indices": [],
                "accuracies": [],
                "training_history": [],
                "wall_time": [],
                "cpu_time": [],
                "ann_index_build_time": [],
                "is_run": False,
            },
            "knn_shapley": {
                "selected_indices": [],
                "accuracies": [],
                "training_history": [],
                "wall_time": [],
                "cpu_time": [],
                "is_run": False,
            },
            "random": {
                "selected_indices": [],
                "accuracies": [],
                "training_history": [],
                "wall_time": [],
                "cpu_time": [],
                "is_run": False,
            },
        }

        # Step 1: Run Conditional Shapley method
        if method == "all" or method == "conditional_shapley":
            wall_time, cpu_time = self.select_point(b=max_points, F_indices=F)
            results["conditional_shapley"]["wall_time"].append(wall_time)
            results["conditional_shapley"]["cpu_time"].append(cpu_time)
            self._run_method(
                method_name="conditional_shapley",
                description="Step 1: Run Conditional Shapley method",
                max_points=max_points,
                results=results,
                index_getter=lambda step, selected_indices: (
                    self.conditional_shapley_indices[step]
                    if step < len(self.conditional_shapley_indices)
                    else None
                ),
                need_compute=False,
                compute_func=None,
                F=F,
            )
            results["conditional_shapley"]["is_run"] = True

        # Step 2: Run Approx Conditional Shapley method
        if method == "all" or method == "approx_conditional_shapley":
            wall_time, cpu_time = self.select_point(
                b=max_points,
                F_indices=F,
                n_top=max_points,
            )
            results["approx_conditional_shapley"]["wall_time"].append(wall_time)
            results["approx_conditional_shapley"]["cpu_time"].append(cpu_time)
            self._run_method(
                method_name="approx_conditional_shapley",
                description="Step 2: Run Approx Conditional Shapley method",
                max_points=max_points,
                results=results,
                index_getter=lambda step, selected_indices: (
                    self.approx_conditional_shapley_indices[step]
                    if step < len(self.approx_conditional_shapley_indices)
                    else None
                ),
                need_compute=False,
                compute_func=None,
                F=F,
            )
            results["approx_conditional_shapley"]["is_run"] = True

        # Step 3: Run Approx Conditional Shapley method (ANN)
        if method == "all" or method == "ann_approx_conditional_shapley":
            start_time = time.process_time()
            ann_index = build_faiss_index(self.X_train, max_points)
            end_time = time.process_time()
            results["ann_approx_conditional_shapley"]["ann_index_build_time"].append(
                end_time - start_time
            )
            wall_time, cpu_time = self.select_point(
                b=max_points,
                F_indices=F,
                n_top=max_points,
                ann_index=ann_index,
                ann_N=max_points,
            )
            results["ann_approx_conditional_shapley"]["wall_time"].append(wall_time)
            results["ann_approx_conditional_shapley"]["cpu_time"].append(cpu_time)
            self._run_method(
                method_name="ann_approx_conditional_shapley",
                description="Step 3: Run ann Approx Conditional Shapley method",
                max_points=max_points,
                results=results,
                index_getter=lambda step, selected_indices: (
                    self.ann_approx_conditional_shapley_indices[step]
                    if step < len(self.ann_approx_conditional_shapley_indices)
                    else None
                ),
                need_compute=False,
                compute_func=None,
                F=F,
            )
            results["ann_approx_conditional_shapley"]["is_run"] = True

        # Step 4: Run KNN-Shapley method
        if method == "all" or method == "knn_shapley":
            self._run_method(
                method_name="knn_shapley",
                description="Step 4: Run KNN-Shapley method",
                max_points=max_points,
                results=results,
                index_getter=lambda step, selected_indices: (
                    self.knn_shapley_indices[step]
                    if step < len(self.knn_shapley_indices)
                    else None
                ),
                need_compute=True,
                compute_func=self.compute_knn_shapley_values,
                F=F,
            )
            results["knn_shapley"]["is_run"] = True

        # Step 5: Run Random method
        if method == "all" or method == "random":
            self._run_method(
                method_name="random",
                description="Step 11: Run Random method",
                max_points=max_points,
                results=results,
                index_getter=lambda step, selected_indices: (
                    self.random_indices[step]
                    if step < len(self.random_indices)
                    else None
                ),
                need_compute=True,
                compute_func=self.compute_random_indices,
                F=F,
            )
            results["random"]["is_run"] = True

        return results

    def log_comparison_results(
        self,
        results: Dict[str, Any],
        max_points: int,
        ver: Optional[str] = None,
        F: list = None,
    ) -> None:
        """Write comparison training results to log file - experimental data for nine methods

        Args:
            results: Results dictionary
            max_points: Maximum number of data points
            ver: Experiment version
            F: List of already selected data point indices
        """
        import json
        from datetime import datetime

        def write_log_entry(file_path: str, log_data: Dict[str, Any]) -> None:
            """Write single log entry to file"""
            with open(file_path, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {json.dumps(log_data)}\n")

        def create_method_log(
            version: str,
            method_name: str,
            is_run: bool,
            accuracies: List[float],
            wall_times: List[float],
            cpu_times: List[float],
            ann_index_build_time: Optional[List[float]] = None,
        ) -> Dict[str, Any]:
            """Create log data structure for specific method"""
            log_data = {
                "ver": version,
                "seed": self.seed,
                "method": method_name,
                "is_run": is_run,
                "all_results": {
                    "axis": list(range(len(accuracies))),
                    "add_high": accuracies,
                    "wall_time": wall_times,
                    "cpu_time": cpu_times,
                },
            }
            if ann_index_build_time is not None:
                log_data["all_results"]["ann_index_build_time"] = ann_index_build_time
            return log_data

        # Log file path
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "mylog.txt")

        # Generate version identifier
        if ver is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            ver = f"pointadd_v1_{timestamp}"

        # Get accuracy data and run times for each method
        method_data = {
            "ConditionalShapley": {
                "accuracies": results["conditional_shapley"]["accuracies"],
                "wall_times": results["conditional_shapley"]["wall_time"],
                "cpu_times": results["conditional_shapley"]["cpu_time"],
                "is_run": results["conditional_shapley"]["is_run"],
            },
            "approx_ConditionalShapley": {
                "accuracies": results["approx_conditional_shapley"]["accuracies"],
                "wall_times": results["approx_conditional_shapley"]["wall_time"],
                "cpu_times": results["approx_conditional_shapley"]["cpu_time"],
                "is_run": results["approx_conditional_shapley"]["is_run"],
            },
            "ann_approx_ConditionalShapley": {
                "accuracies": results["ann_approx_conditional_shapley"]["accuracies"],
                "wall_times": results["ann_approx_conditional_shapley"]["wall_time"],
                "cpu_times": results["ann_approx_conditional_shapley"]["cpu_time"],
                "ann_index_build_time": results["ann_approx_conditional_shapley"][
                    "ann_index_build_time"
                ],
                "is_run": results["ann_approx_conditional_shapley"]["is_run"],
            },
            "KNNShapley": {
                "accuracies": results["knn_shapley"]["accuracies"],
                "wall_times": results["knn_shapley"]["wall_time"],
                "cpu_times": results["knn_shapley"]["cpu_time"],
                "is_run": results["knn_shapley"]["is_run"],
            },
            "Random": {
                "accuracies": results["random"]["accuracies"],
                "wall_times": results["random"]["wall_time"],
                "cpu_times": results["random"]["cpu_time"],
                "is_run": results["random"]["is_run"],
            },
        }

        # Write experiment configuration information
        config_log = {
            "ver": ver,
            "seed": self.seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "dataset": self.dataset_name,
            "model": self.model_type,
            "n_train": len(self.X_train),
            "n_valid": len(self.X_val),
            "n_test": len(self.X_test),
            "b": max_points,
            "n_class": len(np.unique(self.y_train)),
            "train_shape": list(self.X_train.shape),
            "valid_shape": list(self.X_val.shape),
            "test_shape": list(self.X_test.shape),
            "metric": "accuracy",
            "F": F if F else [],
            "k": self.k
        }
        write_log_entry(log_file_path, config_log)

        # Batch write results for each method (only if is_run=True)
        for method_name, data in method_data.items():
            if data["is_run"]:
                method_log = create_method_log(
                    ver,
                    method_name,
                    data["is_run"],
                    data["accuracies"],
                    data["wall_times"],
                    data["cpu_times"],
                    data.get("ann_index_build_time"),
                )
                write_log_entry(log_file_path, method_log)

        print(f"Experimental data written to log file: {log_file_path}")


def main():
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run data point selection training experiments"
    )
    parser.add_argument(
        "--ver",
        type=str,
        default=None,
        help='Experiment version number, e.g., "mc-1". If not provided, will be automatically generated',
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed, default is 42"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10-embedding",
        choices=[
            "cifar10-embedding",
            "covertype",
            "imdb-embedding",
        ],
        help="Dataset name, optional: cifar10-embedding/covertype/imdb-embedding, default is cifar10-embedding",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=[
            "logistic-regression",
            "random-forest",
            "gradient-boosting",
            "svm",
            "decision-tree",
            "cnn",
        ],
        help="Model type, default is logistic-regression",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="k value in KNN-Shapley, default is 5"
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=100,
        help="Maximum number of data points to select, default is 100",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../data_files",
        help="Dataset root directory path, default is ../data_files",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=1200,
        help="Training set sample size, default is 1200",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=500,
        help="Test set sample size, default is 500",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="knn_shapley",
        choices=[
            "all",
            "conditional_shapley",
            "approx_conditional_shapley",
            "ann_approx_conditional_shapley",
            "knn_shapley",
            "random",
        ],
        help="Method to run, default is all",
    )
    parser.add_argument(
        "--F",
        type=int,
        nargs="*",
        default=[],
        help="List of already selected data point indices, default is empty",
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create trainer
    trainer = DataPointSelectionTrainer(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset,
        model_type=args.model,
        k=args.k,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Load dataset
    trainer.load_dataset()

    # Initialize model
    trainer.initialize_model()

    # Run sequential comparison training
    results = trainer.run_sequential_comparison(
        max_points=args.max_points, method=args.method, F=args.F
    )

    # Write comparison results to log file, pass ver parameter
    trainer.log_comparison_results(
        results, max_points=args.max_points, ver=args.ver, F=args.F
    )

    print(f"\n=== Comparison Training Completed ===")
    print(f"Experiment version: {args.ver if args.ver else 'Automatically generated'}")
    print(f"Random seed: {args.seed}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"k value: {args.k}")
    print(f"Training set sample size: {args.train_size}")
    print(f"Test set sample size: {args.test_size}")
    print(f"Maximum data points: {args.max_points}")
    print("Experimental data successfully written to log file")


if __name__ == "__main__":
    main()
