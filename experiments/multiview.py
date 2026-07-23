"""Multi-view data loading and CCA/GFCCA feature fusion for experiments/.

See ``docs/usage-multiview-fusion.md`` for the pkl contract, fusion-group
config, and resampling restrictions this module implements. Do not change the
contract here without updating that document first.
"""

from __future__ import annotations

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from dataprocess import DataLoader
from tbls.cca import build_cca_features, project_cca_features as project_cca
from tbls.gfcca import build_gfcca_features, project_cca_features as project_gfcca

# SMOTE-family resamplers synthesize new feature vectors by interpolating in
# one feature space; there is no way to do that consistently across multiple
# different per-view feature spaces. They are unsupported for multi-view data
# (see docs/usage-multiview-fusion.md Section 3).
_SMOTE_FAMILY = {"smote", "adasyn", "border_smote", "smote_tomek", "smote_enn"}

# Index-only resamplers: decide which rows to keep/duplicate from class counts
# alone, so the same row decision applies unambiguously to every view.
_INDEX_RESAMPLERS = {"oversample": RandomOverSampler, "undersample": RandomUnderSampler}


def load_multiview_cohort(pkl_path, cohort_key: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load one cohort's ``{"views": {...}, "target": y}`` multi-view data.

    Applies the same label filtering as
    :func:`experiments.train._load_subsets`: samples with ``target == -1`` are
    dropped, remaining labels are binarized to ``{0, 1}`` (``(y > 0)``), and
    every view's feature matrix is ``nan_to_num``-cleaned.

    Args:
        pkl_path: Path to the dataset pkl (a dict of cohort keys).
        cohort_key: Top-level dict key whose value is the cohort to load.

    Returns:
        A pair ``(views, y)`` where ``views`` maps view name to a
        ``(n_samples, n_features_view)`` ``float64`` matrix and ``y`` is an
        ``(n_samples,)`` ``int64`` label array.

    Raises:
        ValueError: If the cohort dict has both or neither of ``"data"`` /
            ``"views"`` (a multi-view cohort must use ``"views"``), or is
            missing ``"views"`` / ``"target"``.
    """
    data = joblib.load(pkl_path)
    cohort = data[cohort_key]
    if not isinstance(cohort, dict):
        raise ValueError(f"cohort {cohort_key!r} is not a dict (got {type(cohort).__name__})")
    has_data = "data" in cohort
    has_views = "views" in cohort
    if has_data and has_views:
        raise ValueError(
            f"cohort {cohort_key!r} has both 'data' and 'views' - a cohort must "
            "have exactly one (use 'views' for multi-view data)."
        )
    if not has_views:
        raise ValueError(
            f"cohort {cohort_key!r} is not a multi-view cohort (no 'views' key) - "
            "load_multiview_cohort only handles multi-view cohorts; single-view "
            "cohorts use the 'data' path."
        )
    if "target" not in cohort:
        raise ValueError(f"cohort {cohort_key!r} is missing 'target'")

    raw_views = cohort["views"]
    y = np.asarray(cohort["target"]).ravel()
    valid = y != -1
    y = (y[valid] > 0).astype(np.int64)

    views: dict[str, np.ndarray] = {}
    for name, x in raw_views.items():
        x_arr = np.asarray(x, dtype=np.float64)
        views[name] = np.nan_to_num(x_arr[valid], nan=0.0, posinf=0.0, neginf=0.0)
    return views, y


class MultiViewDataLoader:
    """Per-view preprocessing + row-aligned resampling for multi-view cohorts.

    Each view gets its own fitted ``StandardScaler`` and (optional) feature
    selector, keyed by view name so train/test use matching instances.
    Resampling is row-aligned across all views (see
    ``docs/usage-multiview-fusion.md`` Section 3 for the restriction rationale).

    Args:
        feature_selection: ``"lasso"`` / ``"pca"`` / ``"mutual_info"`` or
            ``None`` (reuses :class:`experiments.dataprocess.DataLoader`'s
            ``FEATURE_SELECTORS`` map, applied once per view).
        resampling: ``"oversample"`` / ``"undersample"`` (index-only) or
            ``"tomek"`` (reference-view). SMOTE-family resamplers raise for
            multi-view data.
        fusion_reference_view: View name whose feature space ``"tomek"``
            computes links against. Required when ``resampling == "tomek"`` and
            there is more than one view.
    """

    def __init__(
        self,
        feature_selection: str | None = None,
        resampling: str | None = None,
        fusion_reference_view: str | None = None,
    ) -> None:
        self.feature_selection = feature_selection
        self.resampling = resampling
        self.fusion_reference_view = fusion_reference_view
        # Per-view fitted artifacts, keyed by view name.
        self.scalers_: dict[str, StandardScaler] = {}
        self.selected_features_: dict[str, np.ndarray] = {}
        self.selectors_: dict[str, object] = {}

    def _fit_transform_one_view(self, name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit this view's scaler (+selector) on x and return the transformed x."""
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        self.scalers_[name] = scaler

        if self.feature_selection in DataLoader.FEATURE_SELECTORS:
            config = DataLoader.FEATURE_SELECTORS[self.feature_selection]
            selector = config["class"](**config["params"]).fit(x_scaled, y)
            self.selectors_[name] = selector
            if self.feature_selection == "lasso":
                mask = selector.coef_ != 0
                self.selected_features_[name] = mask
                return x_scaled[:, mask]
            return selector.transform(x_scaled)
        return x_scaled

    def _transform_one_view(self, name: str, x: np.ndarray) -> np.ndarray:
        """Apply this view's already-fitted scaler (+selector) to held-out x."""
        x_scaled = self.scalers_[name].transform(x)
        if self.feature_selection == "lasso":
            return x_scaled[:, self.selected_features_[name]]
        if self.feature_selection in ("pca", "mutual_info"):
            return self.selectors_[name].transform(x_scaled)
        return x_scaled

    def _resample(
        self,
        views: dict[str, np.ndarray],
        y: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Row-aligned resampling across all views + y (see Section 3)."""
        if self.resampling is None:
            return views, y
        if self.resampling in _SMOTE_FAMILY:
            raise ValueError(
                f"resampling={self.resampling!r} is a SMOTE-family resampler that "
                "synthesizes new feature vectors by interpolation; it cannot be "
                "applied consistently across multiple per-view feature spaces and "
                "is unsupported for multi-view data. Use 'oversample'/'undersample' "
                "(index-only) or 'tomek' (reference-view). See "
                "docs/usage-multiview-fusion.md Section 3."
            )

        n = y.shape[0]
        if self.resampling in _INDEX_RESAMPLERS:
            sampler = _INDEX_RESAMPLERS[self.resampling]()
            # Decide rows from class counts alone, against a dummy 1-column X.
            sampler.fit_resample(np.arange(n).reshape(-1, 1), y)
            keep_idx = np.asarray(sampler.sample_indices_, dtype=np.intp)
        elif self.resampling == "tomek":
            if self.fusion_reference_view is None and len(views) > 1:
                raise ValueError(
                    "resampling='tomek' needs preprocess.fusion_reference_view to "
                    "name which view's feature space to compute Tomek links "
                    "against when there is more than one view."
                )
            ref_name = self.fusion_reference_view or next(iter(views))
            ref_x = views[ref_name]
            tomek = TomekLinks()
            tomek.fit_resample(ref_x, y)
            keep_idx = np.asarray(tomek.sample_indices_, dtype=np.intp)
        else:
            raise ValueError(f"Unsupported resampling: {self.resampling!r}")

        return {name: x[keep_idx] for name, x in views.items()}, y[keep_idx]

    def preprocess_views(
        self,
        X_views_train: dict[str, np.ndarray],
        y_train: np.ndarray,
        X_views_test: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray] | None]:
        """Per-view scale + feature-selection, then row-aligned resampling (train).

        Args:
            X_views_train: Train views, name -> ``(n, f)`` matrix.
            y_train: Train labels ``(n,)``.
            X_views_test: Optional held-out views, same names/feature dims.

        Returns:
            ``(views_train, y_train, views_test)`` where test views (if given)
            are preprocessed with the train-fitted scalers/selectors. Resampling
            is applied to the train side only (the index decision is computed on
            train and applied to train views + y; test views are untouched).
        """
        processed_train = {
            name: self._fit_transform_one_view(name, x, y_train)
            for name, x in X_views_train.items()
        }
        processed_test = (
            {name: self._transform_one_view(name, x) for name, x in X_views_test.items()}
            if X_views_test is not None
            else None
        )

        processed_train, y_train = self._resample(processed_train, y_train)
        return processed_train, y_train, processed_test


def _validate_view_groups(
    view_groups: list[list[str]] | None, view_names: list[str]
) -> list[list[str]]:
    """Validate ``view_groups`` is a partition of ``view_names``; default if None."""
    if view_groups is None:
        # Default: fuse every view present in the cohort together as one group
        # (docs/usage-multiview-fusion.md Section 4's documented default -- the
        # existing all-pairs build_cca_features/build_gfcca_features behavior,
        # unchanged). A *singleton* group is passthrough (see below); the
        # unqualified default must be one group containing all views, not one
        # singleton group per view (which would silently skip fusion entirely).
        return [sorted(view_names)]
    flat: list[str] = []
    for group in view_groups:
        flat.extend(group)
    counts = {name: flat.count(name) for name in set(flat)}
    duplicated = [name for name, c in counts.items() if c > 1]
    if duplicated:
        raise ValueError(
            f"fusion.view_groups is not a partition: view(s) {duplicated} appear "
            "in more than one group."
        )
    missing = [name for name in view_names if name not in counts]
    if missing:
        raise ValueError(
            f"fusion.view_groups is not a partition: view(s) {missing} are "
            "present in the cohort but missing from every group."
        )
    extra = [name for name in counts if name not in view_names]
    if extra:
        raise ValueError(
            f"fusion.view_groups is not a partition: view(s) {extra} appear in "
            "groups but are not present in the cohort."
        )
    return view_groups


def fuse_views(
    X_views: dict[str, np.ndarray],
    y: np.ndarray | None,
    X_views_test: dict[str, np.ndarray] | None,
    method: str,
    view_groups: list[list[str]] | None,
    **fusion_kwargs: object,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fuse per-view feature matrices into one matrix via CCA/GFCCA groups.

    Args:
        X_views: Train views, name -> ``(n, f)`` matrix (already per-view
            preprocessed).
        y: Train labels ``(n,)``. Required for ``method="gfcca"`` (it uses class
            labels); ignored for ``method="cca"``.
        X_views_test: Optional held-out views, same names/feature dims.
        method: ``"cca"`` or ``"gfcca"``.
        view_groups: Partition of ``X_views.keys()`` into fusion groups. ``None``
            means one group of all views (sorted by name). A singleton group is
            pure passthrough (no CCA/GFCCA call).
        **fusion_kwargs: Forwarded to ``build_cca_features`` /
            ``build_gfcca_features`` (``cca_k``, ``cca_lambda``, ``kernel_gamma``,
            ``graph_gamma``, ...). Keyword names must match those signatures.

    Returns:
        ``(F_train, F_test)`` where each is the row-wise concatenation of every
        group's fused block (``F_test`` is ``None`` if ``X_views_test`` is
        ``None``).

    Raises:
        ValueError: Unknown ``method``, or ``view_groups`` is not a partition.
    """
    if method == "cca":
        build_fn = build_cca_features
        project_fn = project_cca
    elif method == "gfcca":
        build_fn = build_gfcca_features
        project_fn = project_gfcca
    else:
        raise ValueError(f"Unknown fusion method: {method!r}. Expected 'cca' or 'gfcca'.")

    view_names = list(X_views.keys())
    groups = _validate_view_groups(view_groups, view_names)

    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] | None = [] if X_views_test is not None else None
    for group in groups:
        if len(group) == 1:
            # Singleton group: pure passthrough, no CCA/GFCCA call.
            name = group[0]
            train_blocks.append(X_views[name])
            if test_blocks is not None:
                test_blocks.append(X_views_test[name])  # type: ignore[index]
            continue
        group_train = [X_views[name] for name in group]
        if method == "cca":
            f_train, models = build_fn(group_train, **fusion_kwargs)  # type: ignore[arg-type]
        else:
            f_train, models = build_fn(group_train, y, **fusion_kwargs)  # type: ignore[arg-type]
        train_blocks.append(f_train)
        if test_blocks is not None:
            group_test = [X_views_test[name] for name in group]  # type: ignore[index]
            f_test = project_fn(group_test, models)
            test_blocks.append(f_test)

    f_train_out = np.hstack(train_blocks)
    f_test_out = np.hstack(test_blocks) if test_blocks is not None else None
    return f_train_out, f_test_out
