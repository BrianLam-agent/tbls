# classifiers.py
"""Unified classifier factory.

Provides a single classifier-construction interface for multimodal CCA
experiments. Every returned classifier instance conforms to the sklearn
estimator protocol (``fit``, ``predict``, ``predict_proba``), and class
imbalance is handled automatically when the underlying algorithm supports it.

Available classifiers:
    'rf'        Random Forest
    'svm'       RBF-kernel Support Vector Machine
    'xgb'       XGBoost (automatic class weighting)
    'lgb'       LightGBM (automatic class weighting)
    'catboost'  CatBoost (automatic class weighting)
    'knn'       K-Nearest Neighbors
    'lr'        Logistic Regression (with class balancing)
    'lasso'     L1-regularized Logistic Regression (with class balancing)
    'elasticnet' Elastic-Net Logistic Regression (with class balancing)
    'nb'        Gaussian Naive Bayes
    'lda'       Linear Discriminant Analysis
    'cart'      Decision Tree (with class balancing)
    'mlp'       Multi-Layer Perceptron
    'dnn'       Deep fully-connected network
    'extratrees' Extra Trees (with class balancing)
    'gbdt'      Gradient Boosting Decision Tree
    'block_plsda', 'block_splsda'
    'mogonet', 'mogonet_nn'
    'bls'       Broad Learning System (requires bls.py in the same directory)
    'tbls'      Tree Broad Learning System (requires tbls.py / the tbls package)
    'mofa'      Multi-Omics Factor Analysis (unsupervised feature extraction + downstream classifier)
    'diablo'    Multi-block sparse PLS-DA (an enhanced block.splsda with a design matrix)
    'snf'       Similarity Network Fusion (unsupervised fusion + downstream classifier)
"""

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

# ---------- Optional-dependency handling ----------
try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    lgb = None

try:
    import catboost as cb

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False
    cb = None

# Import BLS / TBLS from the installed tbls package.
try:
    from tbls import BroadLearningSystem

    _HAS_BLS = True
except ImportError:
    _HAS_BLS = False
    BroadLearningSystem = None

try:
    from tbls.tbls import TBLS

    _HAS_TBLS = True
except ImportError:
    _HAS_TBLS = False
    TBLS = None

# Detect PyTorch (used by MOGONET).
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Detect MOFA dependencies (muon or mofapy2).
try:
    import muon as mu
    from muon import tl as mutl

    _HAS_MUON = True
    _MOFA_BACKEND = "muon"
except ImportError:
    _HAS_MUON = False
    try:
        import mofapy2
        from mofapy2.run import run_mofa

        _HAS_MOFAPY2 = True
        _MOFA_BACKEND = "mofapy2"
    except ImportError:
        _HAS_MOFAPY2 = False
        _MOFA_BACKEND = None

# Detect SNF dependency (snfpy).
try:
    import snf

    _HAS_SNF = True
except ImportError:
    _HAS_SNF = False
    snf = None


# ---------- Auto-balancing wrappers ----------
class BalancedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper that computes 'balanced' sample weights at ``fit`` time.

    Suitable for class-imbalanced settings.
    """

    def __init__(self, random_state=42, **kwargs):
        self.random_state = random_state
        params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "verbosity": 0,
            "eval_metric": "mlogloss",
        }
        params.update(kwargs)
        self.kwargs = params
        self._estimator = None

    def fit(self, X, y):
        """Fit the wrapped booster with ``'balanced'`` sample weights."""
        self._estimator = xgb.XGBClassifier(random_state=self.random_state, **self.kwargs)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)
        self._estimator.fit(X, y, sample_weight=sample_weights)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        """Predict class labels for ``X``."""
        return self._estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for ``X``."""
        return self._estimator.predict_proba(X)

    def get_params(self, deep=True):
        """Return this estimator's parameters."""
        params = self.kwargs.copy()
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params):
        """Set this estimator's parameters."""
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self.kwargs.update(params)
        return self


class BalancedLGBMClassifier(BaseEstimator, ClassifierMixin):
    """LightGBM wrapper that computes 'balanced' sample weights at ``fit`` time."""

    def __init__(self, random_state=42, **kwargs):
        self.random_state = random_state
        params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multiclass",
            "num_class": 3,
            "verbosity": -1,
        }
        params.update(kwargs)
        self.kwargs = params
        self._estimator = None

    def fit(self, X, y):
        """Fit the wrapped booster with ``'balanced'`` sample weights."""
        self._estimator = lgb.LGBMClassifier(random_state=self.random_state, **self.kwargs)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)
        self._estimator.fit(X, y, sample_weight=sample_weights)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        """Predict class labels for ``X``."""
        return self._estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for ``X``."""
        return self._estimator.predict_proba(X)

    def get_params(self, deep=True):
        """Return this estimator's parameters."""
        params = self.kwargs.copy()
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params):
        """Set this estimator's parameters."""
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self.kwargs.update(params)
        return self


class BalancedCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """CatBoost wrapper that handles class imbalance internally."""

    def __init__(self, random_state=42, **kwargs):
        self.random_state = random_state
        params = {
            "iterations": 200,
            "learning_rate": 0.1,
            "depth": 6,
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_seed": random_state,
        }
        params.update(kwargs)
        self.kwargs = params
        self._estimator = None

    def fit(self, X, y):
        """Fit the wrapped booster; class imbalance is handled internally by CatBoost."""
        self._estimator = cb.CatBoostClassifier(**self.kwargs)
        self._estimator.fit(X, y)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        """Predict class labels for ``X``."""
        return self._estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for ``X``."""
        return self._estimator.predict_proba(X)

    def get_params(self, deep=True):
        """Return this estimator's parameters."""
        params = self.kwargs.copy()
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params):
        """Set this estimator's parameters."""
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self.kwargs.update(params)
        return self


# ---------- Pure-Python Block PLSDA / sPLSDA ----------
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


class MixOmicsBlockPLSDA(BaseEstimator, ClassifierMixin):
    """Pure-Python multi-block PLS-DA / sparse PLS-DA (no R required).

    In ``splsda`` mode each view first selects ``keepX[i]`` features via
    ``SelectKBest``; all views are then concatenated and a standard PLS-DA is
    fitted on the concatenation.
    """

    def __init__(self, ncomp=2, keepX=None, mode="plsda", random_state=42):
        self.ncomp = ncomp
        self.keepX = keepX  # list of int, e.g. [10,10,10] for sPLSDA
        self.mode = mode  # 'plsda' or 'splsda'
        self.random_state = random_state
        self.scalers_ = None
        self.selectors_ = None  # per-view feature selectors (splsda mode only)
        self.pls_ = None
        self.classes_ = None

    def fit(self, X_views, y):
        """Fit the multi-block (sparse) PLS-DA on the per-view matrices in ``X_views``."""
        # Standardize each view.
        self.scalers_ = [StandardScaler() for _ in X_views]
        X_scaled = [
            scaler.fit_transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]

        # Sparse feature selection (only when mode='splsda' and keepX is not None).
        if self.mode == "splsda" and self.keepX is not None:
            if len(self.keepX) != len(X_views):
                raise ValueError("keepX length must equal the number of views.")
            self.selectors_ = []
            X_selected = []
            for _, (X, k) in enumerate(zip(X_scaled, self.keepX, strict=False)):
                if k >= X.shape[1]:
                    # If k is not smaller than the feature count, keep all features.
                    selector = None
                    X_sel = X
                else:
                    selector = SelectKBest(f_classif, k=k)
                    X_sel = selector.fit_transform(X, y)
                self.selectors_.append(selector)
                X_selected.append(X_sel)
            X_concat = np.hstack(X_selected)
        else:
            # Plain PLSDA: concatenate all views directly.
            X_concat = np.hstack(X_scaled)

        # One-hot encode the labels.
        self.classes_ = np.unique(y)
        y_onehot = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            y_onehot[y == c, i] = 1

        # Fit the PLS regression.
        self.pls_ = PLSRegression(n_components=self.ncomp, scale=False)
        self.pls_.fit(X_concat, y_onehot)
        return self

    def predict(self, X_views):
        """Predict class indices for the per-view matrices in ``X_views``."""
        proba = self.predict_proba(X_views)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_views):
        """Predict class probabilities for the per-view matrices in ``X_views``."""
        if self.pls_ is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        # Standardize.
        X_scaled = [scaler.transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)]
        # Feature selection.
        if self.mode == "splsda" and self.selectors_ is not None:
            X_selected = []
            for _i, (X, selector) in enumerate(zip(X_scaled, self.selectors_, strict=False)):
                X_sel = selector.transform(X) if selector is not None else X
                X_selected.append(X_sel)
            X_concat = np.hstack(X_selected)
        else:
            X_concat = np.hstack(X_scaled)
        # PLS prediction.
        y_scores = self.pls_.predict(X_concat)  # shape (n_samples, n_classes)
        return softmax(y_scores, axis=1).astype(np.float32)

    def get_params(self, deep=True):
        """Return this estimator's parameters."""
        return {
            "ncomp": self.ncomp,
            "keepX": self.keepX,
            "mode": self.mode,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set this estimator's parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ================================
# MOGONET implementation, following the original authors' models.py and paper.
# ================================
if _HAS_TORCH:
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def xavier_init(m):
        """Xavier-initialize ``m`` in place (Linear weights + bias)."""
        # Xavier-initialize a Linear module (weights + bias).
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    class GraphConvolution(nn.Module):
        """A single graph convolution layer (GCN)."""

        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features))
            nn.init.xavier_normal_(self.weight.data)
            if self.bias is not None:
                self.bias.data.fill_(0.0)

        def forward(self, x, adj):
            """Apply one graph convolution to node features ``x`` with adjacency ``adj``."""
            support = torch.mm(x, self.weight)
            output = torch.sparse.mm(adj, support)
            if self.bias is not None:
                return output + self.bias
            return output

    class GCN_E(nn.Module):
        """Three-layer GCN encoder used by MOGONET."""

        def __init__(self, in_dim, hgcn_dim, dropout):
            super().__init__()
            self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
            self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
            self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
            self.dropout = dropout

        def forward(self, x, adj):
            """Encode node features ``x`` through the 3-layer GCN stack with adjacency ``adj``."""
            x = self.gc1(x, adj)
            x = F.leaky_relu(x, 0.25)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            x = F.leaky_relu(x, 0.25)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc3(x, adj)
            return F.leaky_relu(x, 0.25)

    class Classifier_1(nn.Module):
        """Single-linear-layer view classifier."""

        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
            self.clf.apply(xavier_init)

        def forward(self, x):
            """Compute per-view class logits for features ``x``."""
            return self.clf(x)

    class VCDN(nn.Module):
        """View Correlation Discovery Network: fuses per-view class probabilities."""

        def __init__(self, num_view, num_cls, hvcdn_dim):
            super().__init__()
            self.num_cls = num_cls
            self.model = nn.Sequential(
                nn.Linear(pow(num_cls, num_view), hvcdn_dim),
                nn.LeakyReLU(0.25),
                nn.Linear(hvcdn_dim, num_cls),
            )
            self.model.apply(xavier_init)

        def forward(self, in_list):
            """Fuse the per-view logit list into VCDN output logits."""
            num_view = len(in_list)
            for i in range(num_view):
                in_list[i] = torch.sigmoid(in_list[i])
            x = torch.reshape(
                torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
                (-1, pow(self.num_cls, 2), 1),
            )
            for i in range(2, num_view):
                x = torch.reshape(
                    torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_cls, i + 1), 1)
                )
            vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
            return self.model(vcdn_feat)

    class MOGONETClassifier(BaseEstimator, ClassifierMixin):
        """MOGONET: multi-view graph-convolution fusion classifier.

        Per-view GCN encoders + per-view classifiers are trained jointly with a
        VCDN fusion head. Supports an optional NN (non-graph) encoder mode.
        """

        def __init__(
            self,
            hidden_dim=64,
            epochs=200,
            lr=1e-4,
            dropout=0.5,
            seed=42,
            use_nn=False,
            pretrain_epochs=50,
            gamma=1.0,
            k_neighbors=5,
        ):
            self.hidden_dim = hidden_dim
            self.epochs = epochs
            self.lr = lr
            self.dropout = dropout
            self.seed = seed
            self.use_nn = use_nn
            self.pretrain_epochs = pretrain_epochs
            self.gamma = gamma
            self.k_neighbors = k_neighbors
            torch.manual_seed(seed)
            self.model_dict = None
            self.optim_dict = None
            self.device = None
            self.n_views = None
            self.n_classes = None
            self.scalers = None  # per-view standardizers

        def _build_adj(self, X):
            # Build a k-NN cosine-similarity graph adjacency (normalized).
            from sklearn.metrics.pairwise import cosine_similarity

            sim = cosine_similarity(X)
            n = sim.shape[0]
            k = min(self.k_neighbors, n - 1)
            adj = np.zeros_like(sim)
            for i in range(n):
                idx = np.argpartition(sim[i], -k)[-k:]
                adj[i, idx] = sim[i, idx]
                adj[i, i] = 1.0
            d = np.sum(adj, axis=1) + 1e-8
            d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
            adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
            indices = np.array(np.nonzero(adj_norm))
            values = adj_norm[adj_norm != 0]
            return torch.sparse_coo_tensor(
                indices, values, size=adj_norm.shape, dtype=torch.float32
            )

        def _build_nn_encoder(self, in_dim, hidden_dim, out_dim, dropout):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.25),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.25),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.LeakyReLU(0.25),
            )

        def fit(self, X_views, y):
            """Fit per-view GCN encoders + classifiers and the VCDN fusion head."""
            if not _HAS_TORCH:
                raise ImportError("PyTorch not installed; cannot use MOGONET.")

            # 1. Standardize the data (key fix: prevents gradient explosions from input-scale mismatch).
            self.scalers = [StandardScaler() for _ in X_views]
            X_scaled = [
                scaler.fit_transform(X) for scaler, X in zip(self.scalers, X_views, strict=False)
            ]

            # 2. Check for NaN/Inf.
            for i, X in enumerate(X_scaled):
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    raise ValueError(f"View {i} contains NaN or Inf; check data preprocessing.")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_views = len(X_views)
            self.classes_ = np.unique(y)
            self.n_classes = len(self.classes_)

            # Build adjacency matrices (from standardized data).
            adjs = [self._build_adj(X) for X in X_scaled]

            # Convert to tensors.
            X_tensors = [torch.tensor(X, dtype=torch.float32).to(self.device) for X in X_scaled]
            adj_tensors = [adj.to(self.device) for adj in adjs]
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

            # Model initialization.
            hgcn_dim = [self.hidden_dim, self.hidden_dim, self.hidden_dim]
            dim_list = [X.shape[1] for X in X_scaled]

            self.model_dict = {}
            for i in range(self.n_views):
                if self.use_nn:
                    encoder = self._build_nn_encoder(
                        dim_list[i], self.hidden_dim, self.hidden_dim, self.dropout
                    )
                    self.model_dict[f"E{i + 1}"] = encoder
                else:
                    self.model_dict[f"E{i + 1}"] = GCN_E(dim_list[i], hgcn_dim, self.dropout)
                self.model_dict[f"C{i + 1}"] = Classifier_1(self.hidden_dim, self.n_classes)

            if self.n_views >= 2:
                self.model_dict["C"] = VCDN(self.n_views, self.n_classes, self.hidden_dim)

            for key in self.model_dict:
                self.model_dict[key] = self.model_dict[key].to(self.device)

            # Optimizers: per-view (VCDN separate).
            self.optim_dict = {}
            for i in range(self.n_views):
                params = list(self.model_dict[f"E{i + 1}"].parameters()) + list(
                    self.model_dict[f"C{i + 1}"].parameters()
                )
                self.optim_dict[f"C{i + 1}"] = torch.optim.Adam(params, lr=self.lr)
            if self.n_views >= 2:
                self.optim_dict["C"] = torch.optim.Adam(
                    self.model_dict["C"].parameters(), lr=self.lr
                )

            # Pre-training (optional).
            if self.pretrain_epochs > 0:
                print(f"  Pre-training each view for {self.pretrain_epochs} epochs...")
                for i in range(self.n_views):
                    optimizer = torch.optim.Adam(
                        list(self.model_dict[f"E{i + 1}"].parameters())
                        + list(self.model_dict[f"C{i + 1}"].parameters()),
                        lr=self.lr,
                    )
                    for epoch in range(self.pretrain_epochs):
                        self.model_dict[f"E{i + 1}"].train()
                        self.model_dict[f"C{i + 1}"].train()
                        optimizer.zero_grad()
                        if self.use_nn:
                            feat = self.model_dict[f"E{i + 1}"](X_tensors[i])
                        else:
                            feat = self.model_dict[f"E{i + 1}"](X_tensors[i], adj_tensors[i])
                        logits = self.model_dict[f"C{i + 1}"](feat)
                        loss = F.cross_entropy(logits, y_tensor)
                        loss.backward()
                        optimizer.step()
                        if (epoch + 1) % 20 == 0:
                            print(
                                f"    view {i + 1} epoch {epoch + 1}/{self.pretrain_epochs}, loss={loss.item():.4f}"
                            )

            # Joint training.
            print(f"  Joint training for {self.epochs} epochs...")
            for epoch in range(self.epochs):
                # ---- Stage 1: update all views (VCDN frozen) ----
                for i in range(self.n_views):
                    for param in self.model_dict[f"E{i + 1}"].parameters():
                        param.requires_grad = True
                    for param in self.model_dict[f"C{i + 1}"].parameters():
                        param.requires_grad = True
                if self.n_views >= 2:
                    for param in self.model_dict["C"].parameters():
                        param.requires_grad = False

                # Zero gradients.
                for i in range(self.n_views):
                    self.optim_dict[f"C{i + 1}"].zero_grad()

                total_view_loss = 0.0
                for i in range(self.n_views):
                    if self.use_nn:
                        feat = self.model_dict[f"E{i + 1}"](X_tensors[i])
                    else:
                        feat = self.model_dict[f"E{i + 1}"](X_tensors[i], adj_tensors[i])
                    logits = self.model_dict[f"C{i + 1}"](feat)
                    loss_i = F.cross_entropy(logits, y_tensor)
                    total_view_loss = total_view_loss + loss_i
                total_view_loss.backward()
                for i in range(self.n_views):
                    self.optim_dict[f"C{i + 1}"].step()

                # ---- Stage 2: update VCDN (views frozen) ----
                if self.n_views >= 2:
                    for i in range(self.n_views):
                        for param in self.model_dict[f"E{i + 1}"].parameters():
                            param.requires_grad = False
                        for param in self.model_dict[f"C{i + 1}"].parameters():
                            param.requires_grad = False
                    for param in self.model_dict["C"].parameters():
                        param.requires_grad = True

                    self.optim_dict["C"].zero_grad()
                    logits_list = []
                    for i in range(self.n_views):
                        with torch.no_grad():
                            if self.use_nn:
                                feat = self.model_dict[f"E{i + 1}"](X_tensors[i])
                            else:
                                feat = self.model_dict[f"E{i + 1}"](X_tensors[i], adj_tensors[i])
                            logits = self.model_dict[f"C{i + 1}"](feat)
                        logits_list.append(logits)
                    vcdn_out = self.model_dict["C"](logits_list)
                    vcdn_loss = F.cross_entropy(vcdn_out, y_tensor)
                    vcdn_loss.backward()
                    self.optim_dict["C"].step()
                else:
                    vcdn_loss = torch.tensor(0.0, device=self.device)

                if (epoch + 1) % 50 == 0:
                    total_loss = total_view_loss.item() + (
                        self.gamma * vcdn_loss.item() if self.n_views >= 2 else 0
                    )
                    print(
                        f"  MOGONET epoch {epoch + 1}/{self.epochs}, view_loss={total_view_loss.item():.4f}, "
                        f"vcdn_loss={vcdn_loss.item():.4f}, total_loss={total_loss:.4f}"
                    )

            return self

        def _predict_proba_tensor(self, X_views):
            # Apply the training-time standardizer to the test data.
            X_scaled = [
                scaler.transform(X) for scaler, X in zip(self.scalers, X_views, strict=False)
            ]
            for _key, module in self.model_dict.items():
                if isinstance(module, nn.Module):
                    module.eval()
            with torch.no_grad():
                adjs = [self._build_adj(X) for X in X_scaled]
                X_tensors = [torch.tensor(X, dtype=torch.float32).to(self.device) for X in X_scaled]
                adj_tensors = [adj.to(self.device) for adj in adjs]

                logits_list = []
                for i in range(self.n_views):
                    if self.use_nn:
                        feat = self.model_dict[f"E{i + 1}"](X_tensors[i])
                    else:
                        feat = self.model_dict[f"E{i + 1}"](X_tensors[i], adj_tensors[i])
                    logits = self.model_dict[f"C{i + 1}"](feat)
                    logits_list.append(logits)

                out = self.model_dict["C"](logits_list) if self.n_views >= 2 else logits_list[0]
                proba = F.softmax(out, dim=1)
            return proba.cpu().numpy()

        def predict(self, X_views):
            """Predict class indices for the multi-view input."""
            proba = self.predict_proba(X_views)
            return np.argmax(proba, axis=1)

        def predict_proba(self, X_views):
            """Predict class probabilities for the multi-view input."""
            return self._predict_proba_tensor(X_views).astype(np.float32)

else:

    class MOGONETClassifier(BaseEstimator, ClassifierMixin):
        """Stub MOGONET classifier used when PyTorch is unavailable."""

        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            """Always raises; MOGONET requires PyTorch."""
            raise ImportError("PyTorch not installed; MOGONET unavailable.")

        def predict(self, X):
            """Always raises; MOGONET requires PyTorch."""
            raise ImportError("PyTorch not installed; MOGONET unavailable.")


# ================================
# MOFA (Multi-Omics Factor Analysis).
# ================================
class MOFAClassifier(BaseEstimator, ClassifierMixin):
    """MOFA (Multi-Omics Factor Analysis) wrapper.

    Uses the ``muon`` or ``mofapy2`` backend for unsupervised factor extraction,
    then a downstream classifier for prediction.

    Args:
        n_factors: Number of factors (K). Default 20.
        downstream_clf: Downstream classifier name or sklearn estimator instance. Default 'rf'.
        downstream_kwargs: Extra kwargs for the downstream classifier. Default None.
        use_gpu: Whether to use GPU (muon backend only). Default False.
        random_state: Random seed. Default 42.
    """

    def __init__(
        self,
        n_factors=20,
        downstream_clf="rf",
        downstream_kwargs=None,
        use_gpu=False,
        random_state=42,
    ):
        self.n_factors = n_factors
        self.downstream_clf = downstream_clf
        self.downstream_kwargs = downstream_kwargs or {}
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model_ = None  # the trained MOFA model
        self.factors_ = None  # training factor matrix (N, n_factors)
        self.downstream_ = None  # downstream classifier
        self.classes_ = None

    def fit(self, X_views, y):
        """Train the MOFA factor model and the downstream classifier on the per-view matrices."""
        if _MOFA_BACKEND is None:
            raise ImportError(
                "muon or mofapy2 is required to use MOFA. "
                "Install with 'pip install muon' or 'pip install mofapy2'."
            )

        # 1. Train the MOFA model with the muon or mofapy2 backend.
        if _MOFA_BACKEND == "muon":
            self._fit_muon(X_views, y)
        else:
            self._fit_mofapy2(X_views, y)

        # 2. Train the downstream classifier.
        if isinstance(self.downstream_clf, str):
            from classifiers import create_classifier  # avoid circular import; use the factory

            self.downstream_ = create_classifier(
                self.downstream_clf, random_state=self.random_state, **self.downstream_kwargs
            )
        else:
            self.downstream_ = clone(self.downstream_clf)
        self.downstream_.fit(self.factors_, y)
        self.classes_ = self.downstream_.classes_
        return self

    def _fit_muon(self, X_views, y):
        """Train MOFA with the muon backend."""
        import anndata as ad
        import muon as mu

        # Build the MuData object.
        mdata = mu.MuData({f"view_{i}": ad.AnnData(X) for i, X in enumerate(X_views)})
        # Run MOFA.
        mutl.mofa(
            mdata,
            use_obs_as_factors=True,
            n_factors=self.n_factors,
            use_gpu=self.use_gpu,
            random_seed=self.random_state,
        )
        # Extract the factor matrix (samples x factors).
        self.factors_ = mdata.obsm["X_mofa"].values.astype(np.float32)
        self.model_ = mdata

    def _fit_mofapy2(self, X_views, y):
        """Train MOFA with the mofapy2 backend (older versions)."""
        from mofapy2.run import run_mofa

        # Build the input data dict.
        data = {}
        for i, X in enumerate(X_views):
            data[f"view_{i}"] = X.T  # mofapy2 expects features x samples
        # Run MOFA.
        model = run_mofa(
            data, k=self.n_factors, use_obs_as_factors=True, random_seed=self.random_state
        )
        # Extract the factor matrix (samples x factors).
        self.factors_ = model.nodes["Z"].get_values().T.astype(np.float32)
        self.model_ = model

    def predict(self, X_views):
        """Predict class indices for the per-view matrices in ``X_views``."""
        proba = self.predict_proba(X_views)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_views):
        """Predict class probabilities via factor projection and the downstream classifier."""
        # Out-of-sample projection of new views onto the factor space, then downstream predict.
        new_factors = self._transform_new(X_views)
        return self.downstream_.predict_proba(new_factors)

    def _transform_new(self, X_views):
        """Project new views into the factor space."""
        if _MOFA_BACKEND == "muon":
            import anndata as ad
            import muon as mu

            mdata_new = mu.MuData({f"view_{i}": ad.AnnData(X) for i, X in enumerate(X_views)})
            mutl.mofa(
                mdata_new,
                model=self.model_,
                use_obs_as_factors=True,
                n_factors=self.n_factors,
                use_gpu=self.use_gpu,
            )
            return mdata_new.obsm["X_mofa"].values.astype(np.float32)
        # mofapy2 does not support direct projection; approximated here (simplified).
        raise NotImplementedError(
            "mofapy2 backend does not support out-of-sample projection; use the muon backend."
        )

    def get_params(self, deep=True):
        """Return this estimator's parameters."""
        return {
            "n_factors": self.n_factors,
            "downstream_clf": self.downstream_clf,
            "downstream_kwargs": self.downstream_kwargs,
            "use_gpu": self.use_gpu,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set this estimator's parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ================================
# DIABLO (enhanced multi-block sparse PLS-DA).
# ================================
class DIABLOClassifier(MixOmicsBlockPLSDA):
    """Pure-Python approximation of DIABLO.

    DIABLO (Data Integration Analysis for Biomarker Discovery using Latent
    cOmponents). Extends :class:`MixOmicsBlockPLSDA` with:

        - a ``design_matrix`` controlling which view-pairs should be highly
          correlated,
        - more complete cross-validation parameters (``ncomp_range``,
          ``keepX_range``),
        - multiple discriminant distances (``dist`` = 'max'/'centroids'/'mahalanobis').

    Args:
        ncomp: Number of latent variables per view (or a list, one per view). Default 2.
        keepX: Number of features kept per view (for sparsity). Default None.
        design_matrix: ``(n_views, n_views)`` 0/1 matrix flagging which view-pairs to model.
            Defaults to the all-ones matrix (fully connected).
        dist: Discriminant distance type: 'max' (max correlation), 'centroids'
            (centroid distance), or 'mahalanobis' (Mahalanobis distance). Default 'max'.
        random_state: Random seed. Default 42.
    """

    def __init__(self, ncomp=2, keepX=None, design_matrix=None, dist="max", random_state=42):
        # Delegate to the parent initializer (mode='splsda' or 'plsda').
        mode = "splsda" if keepX is not None else "plsda"
        super().__init__(
            ncomp=ncomp if isinstance(ncomp, int) else ncomp[0],
            keepX=keepX,
            mode=mode,
            random_state=random_state,
        )
        self.design_matrix = design_matrix
        self.dist = dist
        self.ncomp_list = (
            ncomp
            if isinstance(ncomp, list)
            else [ncomp] * (self.n_views if hasattr(self, "n_views") else 0)
        )

    def fit(self, X_views, y):
        """Fit the DIABLO multi-block sparse PLS-DA model on the per-view matrices."""
        self.n_views = len(X_views)
        # If no design matrix was given, default to all-ones (fully connected).
        if self.design_matrix is None:
            self.design_matrix = np.ones((self.n_views, self.n_views))
        # The design matrix must be symmetric with a zero diagonal (self-pairs are not modeled).
        np.fill_diagonal(self.design_matrix, 0)
        # Per-view ncomp handling: if ncomp is a list, handle each view separately.
        # Simplified here: take the first ncomp as the global value (consistent with the parent).
        if not isinstance(self.ncomp, int):
            self.ncomp = self.ncomp_list[0] if len(self.ncomp_list) > 0 else 2
        # Delegate fit to the parent.
        super().fit(X_views, y)
        # Store extra internal state for the different discriminant distances (placeholders).
        self.train_scores_ = self.pls_.x_scores_  # training-set score matrix
        self.y_ = y
        return self

    def predict(self, X_views):
        """Predict class indices for the per-view matrices in ``X_views``."""
        proba = self.predict_proba(X_views)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_views):
        """Predict class probabilities under the chosen discriminant distance (``self.dist``)."""
        # Get the PLS prediction scores (continuous values).
        if self.pls_ is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        # Standardize + feature selection.
        X_scaled = [scaler.transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)]
        if self.mode == "splsda" and self.selectors_ is not None:
            X_selected = []
            for _i, (X, selector) in enumerate(zip(X_scaled, self.selectors_, strict=False)):
                X_selected.append(selector.transform(X) if selector is not None else X)
            X_concat = np.hstack(X_selected)
        else:
            X_concat = np.hstack(X_scaled)
        y_scores = self.pls_.predict(X_concat)  # shape (n_samples, n_classes)
        # Compute probabilities according to the chosen distance type.
        if self.dist == "max":
            proba = softmax(y_scores, axis=1)
        elif self.dist == "centroids":
            # Euclidean distance from each test score to each class centroid, converted to probabilities.
            centroids = []
            for c in self.classes_:
                idx = self.y_ == c
                centroid = self.train_scores_[idx].mean(axis=0)
                centroids.append(centroid)
            centroids = np.array(centroids)
            distances = []
            for score in y_scores:
                dists = np.linalg.norm(score - centroids, axis=1)
                distances.append(dists)
            distances = np.array(distances)
            # Inverse-distance normalized to probabilities (avoid divide-by-zero).
            inv_dist = 1.0 / (distances + 1e-10)
            proba = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        elif self.dist == "mahalanobis":
            # Simplified: Mahalanobis distance using the training-score covariance.
            cov = np.cov(self.train_scores_.T)
            try:
                inv_cov = np.linalg.pinv(cov)
            except Exception:
                inv_cov = np.eye(cov.shape[0])
            centroids = []
            for c in self.classes_:
                idx = self.y_ == c
                centroids.append(self.train_scores_[idx].mean(axis=0))
            centroids = np.array(centroids)
            distances = []
            for score in y_scores:
                diff = score - centroids
                md = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                distances.append(md)
            distances = np.array(distances)
            inv_dist = 1.0 / (distances + 1e-10)
            proba = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown distance type: {self.dist}")
        return proba.astype(np.float32)


class SNFClassifier(BaseEstimator, ClassifierMixin):
    """Similarity Network Fusion wrapper.

    Fuses multi-view similarity matrices with ``snfpy`` and feeds the fused
    representation to a downstream classifier. Supports out-of-sample projection
    (test set projected into the fused space defined at training time).

    Args:
        K: Number of nearest neighbors used when building each view's similarity graph. Default 20.
        T: Number of fusion iterations. Default 20.
        downstream_clf: Downstream classifier name or sklearn estimator instance. Default 'rf'.
        downstream_kwargs: Extra kwargs passed to the downstream classifier. Default None.
        metric: Distance metric used when building similarity matrices (as accepted by
            ``snf.make_affinity``). Default 'euclidean'.
        mu: Similarity-scaling parameter. Default 0.5.
        random_state: Random seed. Default 42.
    """

    def __init__(
        self,
        K=20,
        T=20,
        downstream_clf="rf",
        downstream_kwargs=None,
        metric="euclidean",
        mu=0.5,
        random_state=42,
    ):
        self.K = K
        self.T = T
        self.downstream_clf = downstream_clf
        self.downstream_kwargs = downstream_kwargs or {}
        self.metric = metric
        self.mu = mu
        self.random_state = random_state

        self.X_train_views_ = None  # training raw data (kept for projection)
        self.scalers_ = None  # per-view standardizers
        self.fused_matrix_ = None  # training fused similarity matrix (N_train, N_train)
        self.downstream_ = None
        self.classes_ = None

    def fit(self, X_views, y):
        """Train the SNF fusion model and downstream classifier."""
        if not _HAS_SNF:
            raise ImportError("snfpy is required. Install with 'pip install snfpy'.")

        # 1. Save the training data (raw, unscaled) for projection later.
        self.X_train_views_ = [X.copy() for X in X_views]
        self.classes_ = np.unique(y)

        # 2. Standardize each view (using training-set mean/std).
        self.scalers_ = [StandardScaler() for _ in X_views]
        X_scaled = [
            scaler.fit_transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]

        # 3. Build a similarity (affinity) matrix for each view.
        affinities = []
        for X in X_scaled:
            # snf.make_affinity returns an (N, N) similarity matrix.
            aff = snf.make_affinity(X, metric=self.metric, K=self.K, mu=self.mu)
            affinities.append(aff)

        # 4. Fuse the similarity matrices (iterative update).
        self.fused_matrix_ = snf.snf(affinities, K=self.K, t=self.T)

        # 5. Use each row of the fused matrix as a sample's feature vector (N x N dims).
        X_feat = self.fused_matrix_.astype(np.float32)

        # 6. Train the downstream classifier.
        if isinstance(self.downstream_clf, str):
            from classifiers import create_classifier

            self.downstream_ = create_classifier(
                self.downstream_clf, random_state=self.random_state, **self.downstream_kwargs
            )
        else:
            self.downstream_ = clone(self.downstream_clf)
        self.downstream_.fit(X_feat, y)

        return self

    def predict(self, X_views):
        """Predict class indices for the per-view matrices in ``X_views``."""
        proba = self.predict_proba(X_views)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_views):
        """Project new views out-of-sample into fused features, then predict."""
        if self.fused_matrix_ is None:
            raise RuntimeError("Model not fitted; call fit().")

        N_train = self.fused_matrix_.shape[0]
        X_views[0].shape[0]

        # 1. Standardize training and test sets separately (using the training scaler: no leakage).
        X_train_scaled = [
            scaler.transform(self.X_train_views_[i]) for i, scaler in enumerate(self.scalers_)
        ]
        X_test_scaled = [
            scaler.transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]

        # 2. Stack the standardized training and test sets to build a joint matrix.
        combined_views = []
        for i in range(len(X_views)):
            combined = np.vstack([X_train_scaled[i], X_test_scaled[i]])
            combined_views.append(combined)

        # 3. Compute each view's similarity matrix on the joint matrix (same params as at training).
        affinities_comb = []
        for X in combined_views:
            aff = snf.make_affinity(X, metric=self.metric, K=self.K, mu=self.mu)
            affinities_comb.append(aff)

        # 4. Fuse the joint similarity matrices.
        fused_comb = snf.snf(affinities_comb, K=self.K, t=self.T)

        # 5. Extract the rows corresponding to the test set (the last N_test rows).
        test_fused = fused_comb[N_train:, :].astype(np.float32)

        # 6. Downstream probability prediction.
        proba = self.downstream_.predict_proba(test_fused)
        return proba.astype(np.float32)

    def get_params(self, deep=True):
        """Return this estimator's parameters."""
        return {
            "K": self.K,
            "T": self.T,
            "downstream_clf": self.downstream_clf,
            "downstream_kwargs": self.downstream_kwargs,
            "metric": self.metric,
            "mu": self.mu,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set this estimator's parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class _SNFClassifierFixed(SNFClassifier):
    """Revised SNFClassifier with out-of-sample projection support."""

    def fit(self, X_views, y):
        if not _HAS_SNF:
            raise ImportError("snfpy is required to use SNF. Install with 'pip install snfpy'.")
        self.X_train_views_ = [X.copy() for X in X_views]
        self.classes_ = np.unique(y)
        # Standardize the training views.
        self.scalers_ = [StandardScaler() for _ in X_views]
        X_scaled = [
            scaler.fit_transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]
        # Build similarity matrices.
        affinities = []
        for X in X_scaled:
            aff = snf.make_affinity(X, metric="euclidean", K=self.K, mu=0.5)
            affinities.append(aff)
        self.fused_matrix_ = snf.snf(affinities, K=self.K, t=self.T)
        # Features.
        X_feat = self.fused_matrix_.astype(np.float32)
        # Downstream classifier.
        if isinstance(self.downstream_clf, str):
            from classifiers import create_classifier

            self.downstream_ = create_classifier(
                self.downstream_clf, random_state=self.random_state, **self.downstream_kwargs
            )
        else:
            self.downstream_ = clone(self.downstream_clf)
        self.downstream_.fit(X_feat, y)
        return self

    def predict_proba(self, X_views):
        # Out-of-sample projection: stack new samples with the training set, compute the fused
        # matrix, then extract the rows corresponding to the new samples.
        N_train = self.fused_matrix_.shape[0]
        X_views[0].shape[0]
        # Standardize the new views.
        X_test_scaled = [
            scaler.transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]
        # Stack training and test sets.
        combined_views = []
        for v_idx in range(len(X_views)):
            np.vstack([self.X_train_views_[v_idx], X_views[v_idx]])
            # Re-standardize (global standardization, or map via the training scaler then stack?).
            # For consistency, standardize the whole stack. But fitting the scaler on the combined
            # set would leak test information into the scaler. Better: transform training and test
            # separately with the training scaler, then stack the standardized data.
            X_train_scaled = self.scalers_[v_idx].transform(self.X_train_views_[v_idx])
            X_test_scaled = self.scalers_[v_idx].transform(X_views[v_idx])
            X_comb_scaled = np.vstack([X_train_scaled, X_test_scaled])
            combined_views.append(X_comb_scaled)
        # Compute the similarity matrix over all samples.
        affinities_comb = []
        for X in combined_views:
            aff = snf.make_affinity(X, metric="euclidean", K=self.K, mu=0.5)
            affinities_comb.append(aff)
        fused_comb = snf.snf(affinities_comb, K=self.K, t=self.T)
        # Extract the rows corresponding to the test set (the last N_test rows).
        test_fused = fused_comb[N_train:, :].astype(np.float32)
        # Downstream prediction.
        proba = self.downstream_.predict_proba(test_fused)
        return proba.astype(np.float32)


# Use the fixed SNFClassifier as the final implementation.
SNFClassifier = _SNFClassifierFixed


# ---------- Classifier factory ----------
def create_classifier(
    name: str,
    random_state: int = 42,
    **kwargs,
) -> BaseEstimator:
    """Create a classifier instance by name, with imbalance handling configured.

    Args:
        name: Classifier identifier. One of 'rf', 'svm', 'xgb', 'lgb',
            'catboost', 'knn', 'lr', 'lasso', 'elasticnet', 'nb', 'lda',
            'cart', 'mlp', 'dnn', 'extratrees', 'gbdt', 'bls', 'tbls',
            'block_plsda', 'block_splsda', 'mogonet', 'mogonet_nn', 'mofa',
            'diablo', 'snf'.
        random_state: Random seed for reproducibility.
        **kwargs: Extra parameters forwarded to the classifier constructor;
            these override the defaults.

    Returns:
        A classifier instance conforming to the sklearn estimator interface.

    Raises:
        ImportError: The requested classifier's optional dependency is not installed.
        ValueError: ``name`` is not a known classifier identifier.
    """
    name = name.lower()

    if name == "rf":
        params = {
            "n_estimators": 200,
            "class_weight": "balanced_subsample",
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)

    if name == "svm":
        params = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": random_state,
        }
        params.update(kwargs)
        return SVC(**params)

    if name == "xgb":
        if not _HAS_XGB:
            raise ImportError("XGBoost is not installed. Install with `pip install xgboost`.")
        return BalancedXGBClassifier(random_state=random_state, **kwargs)

    if name == "lgb":
        if not _HAS_LGB:
            raise ImportError("LightGBM is not installed. Install with `pip install lightgbm`.")
        return BalancedLGBMClassifier(random_state=random_state, **kwargs)

    if name == "catboost":
        if not _HAS_CATBOOST:
            raise ImportError("CatBoost is not installed. Install with `pip install catboost`.")
        return BalancedCatBoostClassifier(random_state=random_state, **kwargs)

    if name == "knn":
        params = {
            "n_neighbors": 5,
            "weights": "distance",
            "n_jobs": -1,
        }
        params.update(kwargs)
        return KNeighborsClassifier(**params)

    if name == "lr":
        params = {
            "C": 1.0,
            "class_weight": "balanced",
            "multi_class": "multinomial",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": random_state,
        }
        params.update(kwargs)
        return LogisticRegression(**params)

    if name == "lasso":
        params = {
            "penalty": "l1",
            "solver": "saga",
            "C": 1.0,
            "class_weight": "balanced",
            "multi_class": "multinomial",
            "max_iter": 1000,
            "random_state": random_state,
        }
        params.update(kwargs)
        return LogisticRegression(**params)

    if name == "elasticnet":
        params = {
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": 0.5,
            "C": 1.0,
            "class_weight": "balanced",
            "multi_class": "multinomial",
            "max_iter": 1000,
            "random_state": random_state,
        }
        params.update(kwargs)
        return LogisticRegression(**params)

    if name == "nb":
        params = {}
        params.update(kwargs)
        return GaussianNB(**params)

    if name == "lda":
        params = {
            "solver": "svd",
        }
        params.update(kwargs)
        return LinearDiscriminantAnalysis(**params)

    if name == "cart":
        params = {
            "class_weight": "balanced",
            "random_state": random_state,
        }
        params.update(kwargs)
        return DecisionTreeClassifier(**params)

    if name == "mlp":
        params = {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 300,
            "random_state": random_state,
        }
        params.update(kwargs)
        return MLPClassifier(**params)

    if name == "dnn":
        params = {
            "hidden_layer_sizes": (100, 100, 100),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "shuffle": True,
            "random_state": random_state,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
        }
        params.update(kwargs)
        return MLPClassifier(**params)

    if name == "extratrees":
        params = {
            "n_estimators": 200,
            "class_weight": "balanced_subsample",
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return ExtraTreesClassifier(**params)

    if name == "gbdt":
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": random_state,
        }
        params.update(kwargs)
        return GradientBoostingClassifier(**params)

    if name == "bls":
        if not _HAS_BLS:
            raise ImportError(
                "BLS module not found. Ensure bls.py is in the current directory or models/."
            )
        params = {
            "n_feature_groups": 10,
            "n_feature_nodes_per_group": 100,
            "n_enhancement_groups": 10,
            "n_enhancement_nodes_per_group": 100,
            "map_func": "relu",
            "enhance_func": "relu",
            "reg_param": 1e-8,
            "class_weights": "auto",
            "random_state": random_state,
        }
        params.update(kwargs)
        return BroadLearningSystem(**params)

    if name == "tbls":
        if not _HAS_TBLS:
            raise ImportError(
                "TBLS module not found. Ensure tbls.py is in the current directory or models/."
            )
        params = {
            "n_map_trees": 20,
            "n_enhance_trees": 20,
            "n_increment_layers": 0,
            "tree_max_depth": 5,
            "tree_min_samples_split": 3,
            "tree_max_features_ratio": 0.7,
            "reg_param": 1e-4,
            "use_if_weights": False,
            "if_sigma": 1.0,
            "graph_gamma": 0.0,
            "graph_alpha_in": 1.0,
            "graph_alpha_p": 1.0,
            "graph_knn": 10,
            "graph_threshold": 1.0,
            "class_sensitive": False,
            "use_kernel_for_graph": True,
            "random_state": random_state,
        }
        params.update(kwargs)
        # Backward-compatible legacy parameter names.
        if "n_map_nodes" in params and "n_map_trees" not in kwargs:
            params["n_map_trees"] = params.pop("n_map_nodes")
        if "n_enhance_nodes" in params and "n_enhance_trees" not in kwargs:
            params["n_enhance_trees"] = params.pop("n_enhance_nodes")
        params.pop("class_weights", None)
        params.pop("incremental_method", None)
        params.pop("tree_params", None)
        return TBLS(**params)

    # Multi-view classifiers.
    if name == "block_plsda":
        return MixOmicsBlockPLSDA(mode="plsda", ncomp=2, random_state=random_state, **kwargs)
    if name == "block_splsda":
        return MixOmicsBlockPLSDA(
            mode="splsda", ncomp=3, keepX=None, random_state=random_state, **kwargs
        )
    if name == "mogonet":
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to use MOGONET.")
        # Paper-recommended parameters: lr=1e-4, hidden_dim=64, k_neighbors=5 (overridable via kwargs).
        return MOGONETClassifier(
            hidden_dim=64,
            epochs=100,
            lr=1e-4,
            dropout=0.5,
            seed=random_state,
            use_nn=False,
            pretrain_epochs=50,
            gamma=1.0,
            k_neighbors=5,
            **kwargs,
        )
    if name == "mogonet_nn":
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to use MOGONET_NN.")
        return MOGONETClassifier(
            hidden_dim=64,
            epochs=200,
            lr=1e-4,
            dropout=0.5,
            seed=random_state,
            use_nn=True,
            pretrain_epochs=50,
            gamma=1.0,
            k_neighbors=5,
            **kwargs,
        )

    # Additional algorithms.
    if name == "mofa":
        return MOFAClassifier(
            n_factors=20, downstream_clf="rf", random_state=random_state, **kwargs
        )
    if name == "diablo":
        return DIABLOClassifier(
            ncomp=2, keepX=None, design_matrix=None, dist="max", random_state=random_state, **kwargs
        )
    if name == "snf":
        return SNFClassifier(K=20, T=20, downstream_clf="rf", random_state=random_state, **kwargs)

    raise ValueError(
        f"Unknown classifier '{name}'. Options: 'rf', 'svm', 'xgb', 'lgb', 'catboost', "
        "'knn', 'lr', 'lasso', 'elasticnet', 'nb', 'lda', 'cart', 'mlp', 'dnn', "
        "'extratrees', 'gbdt', 'bls', 'tbls', "
        "'block_plsda', 'block_splsda', 'mogonet', 'mogonet_nn', "
        "'mofa', 'diablo', 'snf'."
    )
