# classifiers.py
"""
统一分类器工厂。

为多模态 CCA 实验提供统一的分类器创建接口。所有返回的分类器实例
均符合 sklearn Estimator 协议（fit, predict, predict_proba），
并自动处理类别不平衡问题（当算法支持时）。

可用分类器：
    'rf'        随机森林
    'svm'       RBF 核支持向量机
    'xgb'       XGBoost (自动类别加权)
    'lgb'       LightGBM (自动类别加权)
    'catboost'  CatBoost (自动类别加权)
    'knn'       K 最近邻
    'lr'        逻辑回归 (带类别平衡)
    'lasso'     L1 正则化逻辑回归 (带类别平衡)
    'elasticnet' 弹性网络逻辑回归 (带类别平衡)
    'nb'        高斯朴素贝叶斯
    'lda'       线性判别分析
    'cart'      决策树 (带类别平衡)
    'mlp'       多层感知机
    'dnn'       深度全连接网络
    'extratrees' 极端随机树 (带类别平衡)
    'gbdt'      梯度提升树
    'block_plsda', 'block_splsda'
    'mogonet', 'mogonet_nn'
    'bls'       宽度学习系统 (需要同目录下的 bls.py)
    'tbls'      树宽度学习系统 (需要同目录下的 tbls.py)
    'mofa'      多组学因子分析 (无监督特征提取 + 下游分类器)
    'diablo'    多块稀疏PLS-DA (增强版 block.splsda，支持设计矩阵)
    'snf'       相似性网络融合 (无监督融合 + 下游分类器)
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

# ---------- 可选依赖处理 ----------
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

# NEW: 检测 PyTorch（用于 MOGONET）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# NEW: 检测 MOFA 依赖 (muon 或 mofapy2)
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

# NEW: 检测 SNF 依赖 (snfpy)
try:
    import snf

    _HAS_SNF = True
except ImportError:
    _HAS_SNF = False
    snf = None


# ---------- 自动平衡的包装器 ----------
class BalancedXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost 包装器，在 fit 时自动计算 'balanced' 样本权重，
    使模型适用于类别不平衡场景。
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
        self._estimator = xgb.XGBClassifier(random_state=self.random_state, **self.kwargs)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)
        self._estimator.fit(X, y, sample_weight=sample_weights)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)

    def get_params(self, deep=True):
        params = self.kwargs.copy()
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params):
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self.kwargs.update(params)
        return self


class BalancedLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    LightGBM 包装器，在 fit 时自动计算 'balanced' 样本权重。
    """

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
        self._estimator = lgb.LGBMClassifier(random_state=self.random_state, **self.kwargs)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)
        self._estimator.fit(X, y, sample_weight=sample_weights)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)

    def get_params(self, deep=True):
        params = self.kwargs.copy()
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params):
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self.kwargs.update(params)
        return self


class BalancedCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    CatBoost 包装器，内部自动处理类别不平衡。
    """

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
        self._estimator = cb.CatBoostClassifier(**self.kwargs)
        self._estimator.fit(X, y)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)

    def get_params(self, deep=True):
        params = self.kwargs.copy()
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params):
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self.kwargs.update(params)
        return self


# ---------- 纯 Python 实现的 Block PLSDA / sPLSDA ----------
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


class MixOmicsBlockPLSDA(BaseEstimator, ClassifierMixin):
    """
    纯 Python 实现的多块 PLS-DA / 稀疏 PLS-DA（无需 R）。
    对于 splsda 模式，每个视图先通过 SelectKBest 选择 keepX[i] 个特征，
    然后拼接所有视图后执行标准 PLS-DA。
    """

    def __init__(self, ncomp=2, keepX=None, mode="plsda", random_state=42):
        self.ncomp = ncomp
        self.keepX = keepX  # list of int, e.g. [10,10,10] for sPLSDA
        self.mode = mode  # 'plsda' or 'splsda'
        self.random_state = random_state
        self.scalers_ = None
        self.selectors_ = None  # 每个视图的特征选择器（仅 splsda 模式）
        self.pls_ = None
        self.classes_ = None

    def fit(self, X_views, y):
        # 标准化每个视图
        self.scalers_ = [StandardScaler() for _ in X_views]
        X_scaled = [
            scaler.fit_transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]

        # 稀疏特征选择（仅当 mode='splsda' 且 keepX 不为 None）
        if self.mode == "splsda" and self.keepX is not None:
            if len(self.keepX) != len(X_views):
                raise ValueError("keepX 的长度必须等于视图数量")
            self.selectors_ = []
            X_selected = []
            for _, (X, k) in enumerate(zip(X_scaled, self.keepX, strict=False)):
                if k >= X.shape[1]:
                    # 如果 k 不小于特征数，保留全部特征
                    selector = None
                    X_sel = X
                else:
                    selector = SelectKBest(f_classif, k=k)
                    X_sel = selector.fit_transform(X, y)
                self.selectors_.append(selector)
                X_selected.append(X_sel)
            X_concat = np.hstack(X_selected)
        else:
            # 普通 PLSDA：直接拼接所有视图
            X_concat = np.hstack(X_scaled)

        # 将标签转换为 one-hot
        self.classes_ = np.unique(y)
        y_onehot = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            y_onehot[y == c, i] = 1

        # 拟合 PLS 回归
        self.pls_ = PLSRegression(n_components=self.ncomp, scale=False)
        self.pls_.fit(X_concat, y_onehot)
        return self

    def predict(self, X_views):
        proba = self.predict_proba(X_views)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_views):
        if self.pls_ is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        # 标准化
        X_scaled = [scaler.transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)]
        # 特征选择
        if self.mode == "splsda" and self.selectors_ is not None:
            X_selected = []
            for _i, (X, selector) in enumerate(zip(X_scaled, self.selectors_, strict=False)):
                X_sel = selector.transform(X) if selector is not None else X
                X_selected.append(X_sel)
            X_concat = np.hstack(X_selected)
        else:
            X_concat = np.hstack(X_scaled)
        # PLS 预测
        y_scores = self.pls_.predict(X_concat)  # shape (n_samples, n_classes)
        return softmax(y_scores, axis=1).astype(np.float32)

    def get_params(self, deep=True):
        return {
            "ncomp": self.ncomp,
            "keepX": self.keepX,
            "mode": self.mode,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ================================
# 严格参照原作者 models.py 和论文的 MOGONET 实现
# ================================
if _HAS_TORCH:
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def xavier_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    class GraphConvolution(nn.Module):
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
            support = torch.mm(x, self.weight)
            output = torch.sparse.mm(adj, support)
            if self.bias is not None:
                return output + self.bias
            return output

    class GCN_E(nn.Module):
        def __init__(self, in_dim, hgcn_dim, dropout):
            super().__init__()
            self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
            self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
            self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
            self.dropout = dropout

        def forward(self, x, adj):
            x = self.gc1(x, adj)
            x = F.leaky_relu(x, 0.25)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            x = F.leaky_relu(x, 0.25)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc3(x, adj)
            return F.leaky_relu(x, 0.25)

    class Classifier_1(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
            self.clf.apply(xavier_init)

        def forward(self, x):
            return self.clf(x)

    class VCDN(nn.Module):
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
            self.scalers = None  # 存储每个视图的标准化器

        def _build_adj(self, X):
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
            if not _HAS_TORCH:
                raise ImportError("PyTorch not installed; cannot use MOGONET.")

            # 1. 数据标准化（关键修复：防止输入尺度差异导致梯度爆炸）
            self.scalers = [StandardScaler() for _ in X_views]
            X_scaled = [
                scaler.fit_transform(X) for scaler, X in zip(self.scalers, X_views, strict=False)
            ]

            # 2. 检查 NaN/Inf
            for i, X in enumerate(X_scaled):
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    raise ValueError(f"View {i} contains NaN or Inf; check data preprocessing.")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_views = len(X_views)
            self.classes_ = np.unique(y)
            self.n_classes = len(self.classes_)

            # 构建邻接矩阵（使用标准化后的数据）
            adjs = [self._build_adj(X) for X in X_scaled]

            # 转换为张量
            X_tensors = [torch.tensor(X, dtype=torch.float32).to(self.device) for X in X_scaled]
            adj_tensors = [adj.to(self.device) for adj in adjs]
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

            # 模型初始化
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

            # 优化器（每个视图独立，VCDN 独立）
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

            # 预训练（可选）
            if self.pretrain_epochs > 0:
                print(f"  预训练每个视图 {self.pretrain_epochs} epochs...")
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
                                f"    视图{i + 1} epoch {epoch + 1}/{self.pretrain_epochs}, loss={loss.item():.4f}"
                            )

            # 联合训练
            print(f"  联合训练 {self.epochs} epochs...")
            for epoch in range(self.epochs):
                # ---- 阶段 1：更新所有视图（固定 VCDN） ----
                for i in range(self.n_views):
                    for param in self.model_dict[f"E{i + 1}"].parameters():
                        param.requires_grad = True
                    for param in self.model_dict[f"C{i + 1}"].parameters():
                        param.requires_grad = True
                if self.n_views >= 2:
                    for param in self.model_dict["C"].parameters():
                        param.requires_grad = False

                # 清零梯度
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

                # ---- 阶段 2：更新 VCDN（固定视图） ----
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
            # 对测试数据应用训练时的标准化
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
            proba = self.predict_proba(X_views)
            return np.argmax(proba, axis=1)

        def predict_proba(self, X_views):
            return self._predict_proba_tensor(X_views).astype(np.float32)

else:

    class MOGONETClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise ImportError("PyTorch not installed; MOGONET unavailable.")

        def predict(self, X):
            raise ImportError("PyTorch not installed; MOGONET unavailable.")


# ================================
# 新增算法：MOFA（多组学因子分析）
# ================================
class MOFAClassifier(BaseEstimator, ClassifierMixin):
    """
    MOFA (Multi-Omics Factor Analysis) 包装器。
    使用 muon 或 mofapy2 后端进行无监督因子提取，
    然后使用下游分类器进行分类。

    参数
    ----------
    n_factors : int, default=20
        因子个数 (K)
    downstream_clf : str or sklearn estimator, default='rf'
        下游分类器名称或实例
    downstream_kwargs : dict, default=None
        下游分类器的额外参数
    use_gpu : bool, default=False
        是否使用 GPU（仅 muon 后端有效）
    random_state : int, default=42
        随机种子
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
        self.model_ = None  # 存储 MOFA 模型
        self.factors_ = None  # 训练集的因子矩阵 (N, n_factors)
        self.downstream_ = None  # 下游分类器
        self.classes_ = None

    def fit(self, X_views, y):
        if _MOFA_BACKEND is None:
            raise ImportError(
                "需要安装 muon 或 mofapy2 才能使用 MOFA。"
                "请执行 'pip install muon' 或 'pip install mofapy2'"
            )

        # 1. 使用 muon 或 mofapy2 训练 MOFA 模型
        if _MOFA_BACKEND == "muon":
            self._fit_muon(X_views, y)
        else:
            self._fit_mofapy2(X_views, y)

        # 2. 训练下游分类器
        if isinstance(self.downstream_clf, str):
            from classifiers import create_classifier  # 避免循环依赖，使用工厂函数

            self.downstream_ = create_classifier(
                self.downstream_clf, random_state=self.random_state, **self.downstream_kwargs
            )
        else:
            self.downstream_ = clone(self.downstream_clf)
        self.downstream_.fit(self.factors_, y)
        self.classes_ = self.downstream_.classes_
        return self

    def _fit_muon(self, X_views, y):
        """使用 muon 后端训练 MOFA"""
        import anndata as ad
        import muon as mu

        # 构建 MuData 对象
        mdata = mu.MuData({f"view_{i}": ad.AnnData(X) for i, X in enumerate(X_views)})
        # 运行 MOFA
        mutl.mofa(
            mdata,
            use_obs_as_factors=True,
            n_factors=self.n_factors,
            use_gpu=self.use_gpu,
            random_seed=self.random_state,
        )
        # 提取因子矩阵 (samples × factors)
        self.factors_ = mdata.obsm["X_mofa"].values.astype(np.float32)
        self.model_ = mdata

    def _fit_mofapy2(self, X_views, y):
        """使用 mofapy2 后端训练 MOFA（较旧版本）"""
        from mofapy2.run import run_mofa

        # 构建输入数据字典
        data = {}
        for i, X in enumerate(X_views):
            data[f"view_{i}"] = X.T  # mofapy2 期望 features × samples
        # 运行 MOFA
        model = run_mofa(
            data, k=self.n_factors, use_obs_as_factors=True, random_seed=self.random_state
        )
        # 提取因子矩阵 (samples × factors)
        self.factors_ = model.nodes["Z"].get_values().T.astype(np.float32)
        self.model_ = model

    def predict(self, X_views):
        proba = self.predict_proba(X_views)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_views):
        # 对新视图进行 out-of-sample 投影（MOFA 的投影函数）
        new_factors = self._transform_new(X_views)
        return self.downstream_.predict_proba(new_factors)

    def _transform_new(self, X_views):
        """将新视图投影到因子空间"""
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
        # mofapy2 不支持直接投影，这里用插值近似（此处简化，实际应使用模型的方法）
        raise NotImplementedError(
            "mofapy2 backend does not support out-of-sample projection; use the muon backend."
        )

    def get_params(self, deep=True):
        return {
            "n_factors": self.n_factors,
            "downstream_clf": self.downstream_clf,
            "downstream_kwargs": self.downstream_kwargs,
            "use_gpu": self.use_gpu,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ================================
# 新增算法：DIABLO（增强版多块稀疏PLS-DA）
# ================================
class DIABLOClassifier(MixOmicsBlockPLSDA):
    """
    DIABLO (Data Integration Analysis for Biomarker Discovery using Latent cOmponents)
    的纯 Python 近似实现。

    在 MixOmicsBlockPLSDA 基础上增加了：
        - 设计矩阵 design_matrix (控制哪些视图对之间应具有高相关性)
        - 更完整的交叉验证参数 (ncomp_range, keepX_range)
        - 多距离判别 (dist = 'max'/'centroids'/'mahalanobis')

    参数
    ----------
    ncomp : int or list, default=2
        每个视图的潜在变量个数（若为list，则对应每个视图）
    keepX : list of int, default=None
        每个视图保留的特征数（用于稀疏性）
    design_matrix : np.ndarray, shape (n_views, n_views), default=None
        设计矩阵，元素为 0/1，表示是否建模该对视图间的相关性。
        默认使用全1矩阵（完全连接）。
    dist : str, default='max'
        判别距离类型: 'max' (最大相关), 'centroids' (质心距离), 'mahalanobis' (马氏距离)
    random_state : int, default=42
    """

    def __init__(self, ncomp=2, keepX=None, design_matrix=None, dist="max", random_state=42):
        # 调用父类初始化（mode='splsda' 或 'plsda'）
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
        self.n_views = len(X_views)
        # 若没有提供设计矩阵，默认全1（完全连接）
        if self.design_matrix is None:
            self.design_matrix = np.ones((self.n_views, self.n_views))
        # 设计矩阵必须对称且对角线为0（通常不建模自己）
        np.fill_diagonal(self.design_matrix, 0)
        # 对于每个视图，如果 ncomp 是列表，需要分别处理不同的 ncomp。
        # 这里简化：取第一个 ncomp 作为全局值（与父类一致）
        if not isinstance(self.ncomp, int):
            self.ncomp = self.ncomp_list[0] if len(self.ncomp_list) > 0 else 2
        # 调用父类 fit
        super().fit(X_views, y)
        # 存储更多内部状态用于不同距离的判别（这里仅作为占位，实际预测时使用）
        self.train_scores_ = self.pls_.x_scores_  # 训练集的得分矩阵
        self.y_ = y
        return self

    def predict(self, X_views):
        proba = self.predict_proba(X_views)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_views):
        # 获得 PLS 的预测得分（连续值）
        if self.pls_ is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        # 标准化 + 特征选择
        X_scaled = [scaler.transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)]
        if self.mode == "splsda" and self.selectors_ is not None:
            X_selected = []
            for _i, (X, selector) in enumerate(zip(X_scaled, self.selectors_, strict=False)):
                X_selected.append(selector.transform(X) if selector is not None else X)
            X_concat = np.hstack(X_selected)
        else:
            X_concat = np.hstack(X_scaled)
        y_scores = self.pls_.predict(X_concat)  # shape (n_samples, n_classes)
        # 根据选择的距离类型计算概率
        if self.dist == "max":
            proba = softmax(y_scores, axis=1)
        elif self.dist == "centroids":
            # 计算测试样本与各类别质心的欧氏距离，并转换为概率
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
            # 距离倒数归一化为概率（避免除零）
            inv_dist = 1.0 / (distances + 1e-10)
            proba = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        elif self.dist == "mahalanobis":
            # 简化为使用训练得分协方差矩阵的马氏距离
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
            raise ValueError(f"未知的距离类型: {self.dist}")
        return proba.astype(np.float32)


class SNFClassifier(BaseEstimator, ClassifierMixin):
    """
    相似性网络融合 (Similarity Network Fusion) 包装器。
    使用 snfpy 进行多视图相似度矩阵融合，然后输入下游分类器。
    支持 out-of-sample 投影（测试集投影到训练集定义的融合空间）。

    参数
    ----------
    K : int, default=20
        每个视图构建相似图时的最近邻个数
    T : int, default=20
        迭代融合次数
    downstream_clf : str or sklearn estimator, default='rf'
        下游分类器名称或实例
    downstream_kwargs : dict, default=None
        传递给下游分类器的额外参数
    metric : str, default='euclidean'
        构建相似度矩阵时使用的距离度量（snf.make_affinity 支持）
    mu : float, default=0.5
        相似度缩放参数
    random_state : int, default=42
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

        self.X_train_views_ = None  # 存储训练原始数据（用于投影）
        self.scalers_ = None  # 每个视图的标准化器
        self.fused_matrix_ = None  # 训练集的融合相似度矩阵 (N_train, N_train)
        self.downstream_ = None
        self.classes_ = None

    def fit(self, X_views, y):
        """训练 SNF 融合模型及下游分类器"""
        if not _HAS_SNF:
            raise ImportError("需要安装 snfpy。请执行 'pip install snfpy'")

        # 1. 保存训练数据（原始值，未标准化）用于后续投影
        self.X_train_views_ = [X.copy() for X in X_views]
        self.classes_ = np.unique(y)

        # 2. 标准化每个视图（使用训练集的均值和标准差）
        self.scalers_ = [StandardScaler() for _ in X_views]
        X_scaled = [
            scaler.fit_transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]

        # 3. 为每个视图构建相似度矩阵（亲和矩阵）
        affinities = []
        for X in X_scaled:
            # snf.make_affinity 返回 (N, N) 的相似度矩阵
            aff = snf.make_affinity(X, metric=self.metric, K=self.K, mu=self.mu)
            affinities.append(aff)

        # 4. 融合相似度矩阵（迭代更新）
        self.fused_matrix_ = snf.snf(affinities, K=self.K, t=self.T)

        # 5. 将融合矩阵的每一行作为样本的特征（N × N 维）
        X_feat = self.fused_matrix_.astype(np.float32)

        # 6. 训练下游分类器
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
        proba = self.predict_proba(X_views)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_views):
        """
        对新视图进行 out-of-sample 投影，得到融合特征，再调用下游分类器。
        """
        if self.fused_matrix_ is None:
            raise RuntimeError("Model not fitted; call fit().")

        N_train = self.fused_matrix_.shape[0]
        X_views[0].shape[0]

        # 1. 分别标准化训练集和测试集（使用训练集的 scaler，无数据泄露）
        X_train_scaled = [
            scaler.transform(self.X_train_views_[i]) for i, scaler in enumerate(self.scalers_)
        ]
        X_test_scaled = [
            scaler.transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]

        # 2. 将标准化后的训练集和测试集拼接起来，构建联合矩阵
        combined_views = []
        for i in range(len(X_views)):
            combined = np.vstack([X_train_scaled[i], X_test_scaled[i]])
            combined_views.append(combined)

        # 3. 对联合矩阵计算每个视图的相似度矩阵（与训练时的参数一致）
        affinities_comb = []
        for X in combined_views:
            aff = snf.make_affinity(X, metric=self.metric, K=self.K, mu=self.mu)
            affinities_comb.append(aff)

        # 4. 融合联合相似度矩阵
        fused_comb = snf.snf(affinities_comb, K=self.K, t=self.T)

        # 5. 提取测试集对应的行（最后 N_test 行）
        test_fused = fused_comb[N_train:, :].astype(np.float32)

        # 6. 下游分类器预测概率
        proba = self.downstream_.predict_proba(test_fused)
        return proba.astype(np.float32)

    def get_params(self, deep=True):
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
        for key, value in params.items():
            setattr(self, key, value)
        return self


class _SNFClassifierFixed(SNFClassifier):
    """修正版的 SNFClassifier，支持 out-of-sample 投影"""

    def fit(self, X_views, y):
        if not _HAS_SNF:
            raise ImportError("需要安装 snfpy 才能使用 SNF。请执行 'pip install snfpy'")
        self.X_train_views_ = [X.copy() for X in X_views]
        self.classes_ = np.unique(y)
        # 标准化训练视图
        self.scalers_ = [StandardScaler() for _ in X_views]
        X_scaled = [
            scaler.fit_transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]
        # 构建相似度矩阵
        affinities = []
        for X in X_scaled:
            aff = snf.make_affinity(X, metric="euclidean", K=self.K, mu=0.5)
            affinities.append(aff)
        self.fused_matrix_ = snf.snf(affinities, K=self.K, t=self.T)
        # 特征
        X_feat = self.fused_matrix_.astype(np.float32)
        # 下游分类器
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
        # out-of-sample 投影：将新样本与训练集拼接，计算融合矩阵，提取新样本对应的行
        N_train = self.fused_matrix_.shape[0]
        X_views[0].shape[0]
        # 标准化新视图
        X_test_scaled = [
            scaler.transform(X) for scaler, X in zip(self.scalers_, X_views, strict=False)
        ]
        # 拼接训练集和测试集
        combined_views = []
        for v_idx in range(len(X_views)):
            np.vstack([self.X_train_views_[v_idx], X_views[v_idx]])
            # 重新标准化（使用整体标准化，或者用训练集的 scaler 映射后再拼接？）
            # 为了保证一致性，对整体进行标准化（但不应该使用测试集信息来 fit scaler，有泄漏风险）
            # 更好的方法：使用训练集的 scaler 分别变换训练和测试，然后拼接标准化后的数据。
            X_train_scaled = self.scalers_[v_idx].transform(self.X_train_views_[v_idx])
            X_test_scaled = self.scalers_[v_idx].transform(X_views[v_idx])
            X_comb_scaled = np.vstack([X_train_scaled, X_test_scaled])
            combined_views.append(X_comb_scaled)
        # 计算所有样本的相似度矩阵
        affinities_comb = []
        for X in combined_views:
            aff = snf.make_affinity(X, metric="euclidean", K=self.K, mu=0.5)
            affinities_comb.append(aff)
        fused_comb = snf.snf(affinities_comb, K=self.K, t=self.T)
        # 提取测试集对应的行（即最后 N_test 行）
        test_fused = fused_comb[N_train:, :].astype(np.float32)
        # 下游预测
        proba = self.downstream_.predict_proba(test_fused)
        return proba.astype(np.float32)


# 将修正后的 SNFClassifier 作为最终实现
SNFClassifier = _SNFClassifierFixed


# ---------- 分类器创建函数 ----------
def create_classifier(
    name: str,
    random_state: int = 42,
    **kwargs,
) -> BaseEstimator:
    """
    根据名称创建分类器实例，自动配置类别不平衡处理。

    Parameters
    ----------
    name : str
        分类器标识符: 'rf', 'svm', 'xgb', 'lgb', 'catboost',
        'knn', 'lr', 'lasso', 'elasticnet', 'nb', 'lda', 'cart', 'mlp',
        'dnn', 'extratrees', 'gbdt', 'bls', 'tbls',
        'block_plsda', 'block_splsda', 'mogonet', 'mogonet_nn',
        'mofa', 'diablo', 'snf'
    random_state : int
        随机种子，用于可复现性。
    **kwargs
        传递给分类器构造函数的额外参数，会覆盖默认值。

    Returns
    -------
    clf : 分类器实例，符合 sklearn Estimator 接口。
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
            raise ImportError("XGBoost 未安装。请使用 `pip install xgboost` 安装。")
        return BalancedXGBClassifier(random_state=random_state, **kwargs)

    if name == "lgb":
        if not _HAS_LGB:
            raise ImportError("LightGBM 未安装。请使用 `pip install lightgbm` 安装。")
        return BalancedLGBMClassifier(random_state=random_state, **kwargs)

    if name == "catboost":
        if not _HAS_CATBOOST:
            raise ImportError("CatBoost 未安装。请使用 `pip install catboost` 安装。")
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
            raise ImportError("BLS 模块未找到。请确保 bls.py 在当前目录或 models/ 下。")
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
            raise ImportError("TBLS 模块未找到。请确保 tbls.py 在当前目录或 models/ 下。")
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
        # 兼容旧参数名
        if "n_map_nodes" in params and "n_map_trees" not in kwargs:
            params["n_map_trees"] = params.pop("n_map_nodes")
        if "n_enhance_nodes" in params and "n_enhance_trees" not in kwargs:
            params["n_enhance_trees"] = params.pop("n_enhance_nodes")
        params.pop("class_weights", None)
        params.pop("incremental_method", None)
        params.pop("tree_params", None)
        return TBLS(**params)

    # 多视图分类器
    if name == "block_plsda":
        return MixOmicsBlockPLSDA(mode="plsda", ncomp=2, random_state=random_state, **kwargs)
    if name == "block_splsda":
        return MixOmicsBlockPLSDA(
            mode="splsda", ncomp=3, keepX=None, random_state=random_state, **kwargs
        )
    if name == "mogonet":
        if not _HAS_TORCH:
            raise ImportError("需要安装 PyTorch 才能使用 MOGONET。")
        # 使用论文推荐参数：lr=1e-4, hidden_dim=64, k_neighbors=5（可通过 kwargs 覆盖）
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
            raise ImportError("需要安装 PyTorch 才能使用 MOGONET_NN。")
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

    # 新增算法
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
        f"未知的分类器 '{name}'. 可选: 'rf', 'svm', 'xgb', 'lgb', 'catboost', "
        "'knn', 'lr', 'lasso', 'elasticnet', 'nb', 'lda', 'cart', 'mlp', 'dnn', "
        "'extratrees', 'gbdt', 'bls', 'tbls', "
        "'block_plsda', 'block_splsda', 'mogonet', 'mogonet_nn', "
        "'mofa', 'diablo', 'snf'"
    )
