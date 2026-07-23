"""Data loading and preprocessing for TBLS experiments.

DataLoader supports feature selection (Lasso/PCA/MutualInfo) and resampling
(SMOTE/ADASYN/etc.), applied on the training split only to avoid leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.utils import shuffle


class DataLoader:
    """Data loader and preprocessor with optional feature selection and resampling."""

    FEATURE_SELECTORS: ClassVar[dict] = {
        "lasso": {"class": Lasso, "params": {"alpha": 0.01}},
        "pca": {"class": PCA, "params": {"n_components": 0.95}},
        "mutual_info": {
            "class": SelectKBest,
            "params": {"score_func": mutual_info_classif, "k": 10},
        },
    }

    RESAMPLERS: ClassVar[dict] = {
        "smote": SMOTE,
        "adasyn": ADASYN,
        "border_smote": BorderlineSMOTE,
        "undersample": RandomUnderSampler,
        "tomek": TomekLinks,
        "smote_tomek": SMOTETomek,
        "smote_enn": SMOTEENN,
    }

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "experiments/datasets",
        feature_selection: str | None = None,
        resampling: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.data_path = self.data_dir / f"{dataset_name}_data.csv"
        self.label_path = self.data_dir / f"{dataset_name}_label.csv"
        self.pkl_path = self.data_dir / f"{dataset_name}.pkl"
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        self.task_type: str | None = None
        self.feature_selection = feature_selection
        self.resampling = resampling
        self.selected_features: np.ndarray | None = None

    def _load_csv(self) -> tuple[np.ndarray, np.ndarray]:
        """Load multi-label CSV data."""
        data = np.loadtxt(self.data_path, delimiter=",", dtype=np.float32)
        labels = np.loadtxt(self.label_path, delimiter=",", dtype=np.int32)
        return data, labels

    def _load_pkl(self) -> tuple[np.ndarray, np.ndarray]:
        """Load pkl data and drop samples with label -1."""
        data = joblib.load(self.pkl_path)

        # Unify the data-dict handling.
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], dict) and "data" in data[key]:
                    x = data[key]["data"]
                    y = data[key].get("target")
                    break
            else:
                raise KeyError("pkl must contain a sub-dict with 'data' and 'target'")
        else:
            raise ValueError(f"Unsupported pkl format: {type(data)}")

        # Filter invalid labels.
        x_arr, y_arr = np.array(x), np.array(y).flatten()
        valid_idx = y_arr != -1
        return x_arr[valid_idx], y_arr[valid_idx]

    def _apply_feature_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Structured feature selection."""
        if self.feature_selection in self.FEATURE_SELECTORS:
            config = self.FEATURE_SELECTORS[self.feature_selection]
            selector = config["class"](**config["params"]).fit(X, y)

            if self.feature_selection == "lasso":
                # Record selected features (nonzero coefs) for test-time use.
                selected = selector.coef_ != 0
                self.selected_features = selected
                return X[:, selected]
            # PCA and mutual_info transform in place.
            return selector.transform(X)
        return X

    def _apply_resampling(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Structured resampling."""
        if self.resampling in self.RESAMPLERS:
            sampler = self.RESAMPLERS[self.resampling]()
            return sampler.fit_resample(X, y)
        return X, y

    def load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load raw data without any preprocessing."""
        if self.data_path.exists() and self.label_path.exists():
            data, labels = self._load_csv()
            self.task_type = "multilabel" if labels.ndim == 2 else "binary"
        elif self.pkl_path.exists():
            data, labels = self._load_pkl()
            self.task_type = "binary"
        else:
            raise FileNotFoundError(f"Dataset {self.dataset_name} not found")
        return data, labels

    def preprocess(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Fit preprocessing on the train split and apply to test (if given)."""
        # Standardize.
        self.scaler.fit(X_train)
        x_train_scaled = self.scaler.transform(X_train)
        x_test_scaled = self.scaler.transform(X_test) if X_test is not None else None

        # Feature selection.
        if self.feature_selection:
            x_train_scaled = self._apply_feature_selection(x_train_scaled, y_train)
            if x_test_scaled is not None and self.feature_selection == "lasso":
                assert self.selected_features is not None
                x_test_scaled = x_test_scaled[:, self.selected_features]
            # PCA and mutual_info: selector.transform already applied above.

        # Resampling (train only).
        if self.resampling:
            x_train_res, y_train_res = self._apply_resampling(x_train_scaled, y_train)
        else:
            x_train_res, y_train_res = x_train_scaled, y_train

        return x_train_res, y_train_res, x_test_scaled

    def _process_labels(self, labels: np.ndarray) -> np.ndarray:
        """Unify label format."""
        labels = labels.flatten()
        valid_idx = labels != -1
        labels = labels[valid_idx]

        if self.task_type == "multilabel":
            return self.mlb.fit_transform(labels)
        return np.where(labels > 0, 1, 0)

    def _split_dataset(
        self, data: np.ndarray, labels: np.ndarray, test_ratio: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """Split dataset by shuffling then cutting at ``test_ratio``."""
        assert self.task_type is not None
        num_samples = data.shape[0]
        indices = shuffle(np.arange(num_samples))
        split_idx = int(num_samples * (1 - test_ratio))

        return (
            data[indices[:split_idx]],
            labels[indices[:split_idx]],
            data[indices[split_idx:]],
            labels[indices[split_idx:]],
            self.task_type,
        )

    def get_task_type(self) -> str | None:
        """Return the detected task type."""
        return self.task_type
