from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import gc
import warnings
from zipfile import ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC

RANDOM_STATE = 42


def custom_ratio(x: np.ndarray) -> np.ndarray:
    """Apply a bounded ratio transform: ``x / (1 + x)``.

    Args:
        x: Input numeric array.

    Returns:
        Transformed array with the same shape as ``x``.
    """
    return x / (1 + x)


def recall_precision_accuracy_score(
    y_true: pd.Series, y_pred: np.ndarray
) -> float:
    """Compute a weighted composite score that heavily prioritises recall.

    Formula: ``(100 * recall) + (10 * precision) + accuracy``.
    Designed for clinical-deterioration tasks where missing a positive case
    (false negative) is more costly than a false alarm.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        A single float composite score (higher is better).
    """
    rec = recall_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return (100.0 * rec) + (10.0 * pre) + acc


def _finite_param_space_size(
    param_distributions: dict[str, Any] | list[dict[str, Any]],
) -> int | None:
    """Return finite parameter-space size when distributions are list-like.

    If any parameter uses a continuous/scipy-style distribution (supports
    ``.rvs``), return ``None`` to indicate unknown/infinite size.
    """
    grids = (
        param_distributions
        if isinstance(param_distributions, list)
        else [param_distributions]
    )

    total = 0
    for grid in grids:
        if not isinstance(grid, dict):
            return None

        normalized_grid: dict[str, list[Any]] = {}
        for key, values in grid.items():
            if hasattr(values, "rvs"):
                return None
            if isinstance(values, str):
                return None
            try:
                normalized_grid[key] = list(values)
            except TypeError:
                return None

        total += len(ParameterGrid(normalized_grid))

    return total


def _format_model_params(params: dict[str, Any]) -> str:
    """Return deterministic string representation for model parameters."""
    if not params:
        return "{}"
    ordered = {k: params[k] for k in sorted(params)}
    return str(ordered)


def balance_binary_classes_equal_false_true(
    X: pd.DataFrame,
    y: pd.Series,
    true_label: int = 1,
    false_label: int = 0,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Under-sample the majority (false) class so both classes have equal size.

    Randomly samples ``len(true_class)`` rows from the false class without
    replacement, then shuffles and returns the combined balanced dataset.

    Args:
        X: Feature DataFrame aligned with ``y``.
        y: Binary target Series.
        true_label: Integer value that identifies the minority/positive class.
            Defaults to ``1``.
        false_label: Integer value that identifies the majority/negative class.
            Defaults to ``0``.
        random_state: Seed for the NumPy random generator to ensure
            reproducibility. Defaults to ``RANDOM_STATE`` (42).

    Returns:
        A ``(X_balanced, y_balanced)`` tuple where both classes have equal
        sample counts.

    Raises:
        ValueError: If either class is absent from ``y``.
    """
    y_arr = np.asarray(y)
    true_idx = np.where(y_arr == true_label)[0]
    false_idx = np.where(y_arr == false_label)[0]

    if len(true_idx) == 0 or len(false_idx) == 0:
        raise ValueError("Both classes must be present to balance data.")

    take_false = min(len(false_idx), len(true_idx))
    rng = np.random.default_rng(random_state)
    sampled_false = rng.choice(false_idx, size=take_false, replace=False)
    kept_idx = np.concatenate([true_idx, sampled_false])
    rng.shuffle(kept_idx)
    return X.iloc[kept_idx].copy(), y.iloc[kept_idx].copy()


def balance_binary_classes_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Balance classes with SMOTE and a safe k-neighbors setting.

    Falls back to under-sampling when the minority class has fewer than
    2 examples, where SMOTE cannot run.
    """
    y_series = pd.Series(y).reset_index(drop=True)
    class_counts = y_series.value_counts()
    if class_counts.shape[0] < 2:
        raise ValueError("Both classes must be present to balance data.")

    minority_count = int(class_counts.min())
    if minority_count < 2:
        return balance_binary_classes_equal_false_true(
            X, y_series, random_state=random_state
        )

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y_series)

    return (
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled).reset_index(drop=True),
    )


class IQRClipper(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that clips numeric features at IQR-based bounds.

    During ``fit`` the per-feature lower and upper fences are computed as:
    ``Q1 - iqr_multiplier * IQR`` and ``Q3 + iqr_multiplier * IQR``.
    During ``transform`` values outside those fences are clipped to the fence
    value, reducing the influence of extreme outliers without discarding rows.

    Attributes:
        iqr_multiplier (float): Multiplier applied to the IQR to determine
            fence width. Defaults to ``1.5`` (Tukey fences).
        ``lower_bounds_`` (np.ndarray | None): Per-feature lower fence values set
            after ``fit``.
        ``upper_bounds_`` (np.ndarray | None): Per-feature upper fence values set
            after ``fit``.
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> IQRClipper:
        """Compute per-feature IQR fences from the training data.

        Args:
            X: Array-like of shape ``(n_samples, n_features)`` containing
                numeric values. NaNs are ignored during percentile computation.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            The fitted ``IQRClipper`` instance (``self``).
        """
        arr = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - (self.iqr_multiplier * iqr)
        self.upper_bounds_ = q3 + (self.iqr_multiplier * iqr)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Clip feature values to the fitted IQR fences.

        Args:
            X: Array-like of shape ``(n_samples, n_features)`` to transform.

        Returns:
            A NumPy array of the same shape with outlier values clipped to the
            lower or upper fence.

        Raises:
            RuntimeError: If ``transform`` is called before ``fit``.
        """
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("IQRClipper must be fit before transform.")
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)


@dataclass
class ModelSpec:
    model_type: str
    variant: str
    estimator: Any
    random_dist: dict[str, list[Any]]
    explicit_grid: dict[str, list[Any]]
    n_random_iterations: int


def _load_source_frames(
    archive_path: Path,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Read all five source CSVs from the project ZIP archive into DataFrames.

    Args:
        archive_path: Absolute ``Path`` to the ZIP archive that contains the
            following CSV files: ``hospital_deterioration_hourly_panel.csv``,
            ``hospital_deterioration_ml_ready.csv``, ``labs_timeseries.csv``,
            ``patients.csv``, and ``vitals_timeseries.csv``.

    Returns:
        A 5-tuple of DataFrames in the order:
        ``(deterioration_hourly_panel, deterioration_ml_ready,
        labs_timeseries, patients, vitals_timeseries)``.
    """
    with ZipFile(archive_path, mode="r") as project_data:
        with project_data.open(
            "hospital_deterioration_hourly_panel.csv", mode="r"
        ) as deterioration_hourly_panel:
            deterioration_hourly_panel = pd.read_csv(
                deterioration_hourly_panel
            )
        with project_data.open(
            "hospital_deterioration_ml_ready.csv", mode="r"
        ) as deterioration_ml_ready:
            deterioration_ml_ready = pd.read_csv(deterioration_ml_ready)
        with project_data.open(
            "labs_timeseries.csv", mode="r"
        ) as labs_timeseries:
            labs_timeseries = pd.read_csv(labs_timeseries)
        with project_data.open("patients.csv", mode="r") as patients:
            patients = pd.read_csv(patients)
        with project_data.open(
            "vitals_timeseries.csv", mode="r"
        ) as vitals_timeseries:
            vitals_timeseries = pd.read_csv(vitals_timeseries)
    return (
        deterioration_hourly_panel,
        deterioration_ml_ready,
        labs_timeseries,
        patients,
        vitals_timeseries,
    )


def _prepare_hcd_data(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str, list[str]]:
    """Load, merge, and filter the HCD dataset into modelling-ready DataFrames.

    Loads raw source CSVs from the project archive, merges them on
    ``patient_id`` and ``hour_from_admission``, then restricts rows to hours
    *before* clinical deterioration (or cases with no deterioration event).
    The ML-ready column schema is applied to both the filtered and the
    unfiltered joined frames so downstream code can evaluate on both splits.

    Args:
        base_dir: Directory that contains ``archive (10).zip`` and
            ``HCD_utils.py``.

    Returns:
        A 4-tuple ``(df, joined_data, target, features)`` where:

        - ``df`` – filtered DataFrame (pre-deterioration rows only).
        - ``joined_data`` – full merged DataFrame (all hours, ML columns).
        - ``target`` – name of the binary target column
          (``"deterioration_next_12h"``).
        - ``features`` – list of feature column names (all ML-ready columns
          except the target).
    """
    from HCD_utils import get_merged_data_w_ids

    archive_path = base_dir / "archive (10).zip"
    (
        deterioration_hourly_panel,
        deterioration_ml_ready,
        labs_timeseries,
        patients,
        vitals_timeseries,
    ) = _load_source_frames(archive_path)

    datasets = [
        deterioration_hourly_panel,
        labs_timeseries,
        patients,
        vitals_timeseries,
    ]
    joined_data, _report = get_merged_data_w_ids(
        kwargs=datasets,
        join_cols=["patient_id", "hour_from_admission"],
        how="left",
        strict=True,
        return_report=True,
    )

    df = joined_data.copy().loc[
        (
            joined_data["hour_from_admission"]
            < joined_data["deterioration_hour"]
        )
        | (joined_data["deterioration_hour"] == -1)
    ]
    df = df[deterioration_ml_ready.columns].copy()

    target = "deterioration_next_12h"
    features = [c for c in deterioration_ml_ready.columns if c != target]
    return (
        df,
        joined_data[deterioration_ml_ready.columns].copy(),
        target,
        features,
    )


def _split_df_joined(
    df: pd.DataFrame,
    joined_data: pd.DataFrame,
    target: str,
    features: list[str],
) -> dict[str, Any]:
    """Create stratified 80/20 train-test splits for both datasets.

    Both ``df`` (pre-deterioration filter) and ``joined_data`` (full data) are
    split independently using the same ``RANDOM_STATE`` and stratification on
    the target column so class proportions are preserved.

    Args:
        df: Filtered DataFrame (pre-deterioration rows only).
        joined_data: Full merged DataFrame (all hours, ML columns).
        target: Name of the binary target column.
        features: List of feature column names to keep.

    Returns:
        A dictionary with eight keys:
        ``X_train_df``, ``X_test_df``, ``y_train_df``, ``y_test_df``
        (from ``df``)
        and ``X_train_joined``, ``X_test_joined``, ``y_train_joined``,
        ``y_test_joined`` (from ``joined_data``).
    """
    y_df = df[target].astype(int)
    X_df = df[features].copy()

    y_joined = joined_data[target].astype(int)
    X_joined = joined_data[features].copy()

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        X_df,
        y_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_df,
    )
    X_train_joined, X_test_joined, y_train_joined, y_test_joined = (
        train_test_split(
            X_joined,
            y_joined,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_joined,
        )
    )
    return {
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "y_train_df": y_train_df,
        "y_test_df": y_test_df,
        "X_train_joined": X_train_joined,
        "X_test_joined": X_test_joined,
        "y_train_joined": y_train_joined,
        "y_test_joined": y_test_joined,
    }


def _numeric_transformers() -> dict[str, Any]:
    """Return numeric transformers used for preprocessing benchmarking.

    Returns:
        A dictionary with seven entries:

        - ``"log_transformer"`` – ``FunctionTransformer(np.log1p)``
        - ``"power_transformer"`` –
          ``PowerTransformer(method='yeo-johnson')``
        - ``"quantile_transformer"`` –
          ``QuantileTransformer(output_distribution='normal')``
        - ``"robust_scaler"`` – ``RobustScaler()``
        - ``"standard_scaler"`` – ``StandardScaler()``
        - ``"custom_transformer"`` – ``FunctionTransformer(custom_ratio)``
        - ``"sqrt_transformer"`` – ``FunctionTransformer(np.sqrt)``
    """
    return {
        "log_transformer": FunctionTransformer(np.log1p, validate=True),
        "power_transformer": PowerTransformer(method="yeo-johnson"),
        "quantile_transformer": QuantileTransformer(
            output_distribution="normal",
            random_state=RANDOM_STATE,
        ),
        "robust_scaler": RobustScaler(),
        "standard_scaler": StandardScaler(),
        "custom_transformer": FunctionTransformer(custom_ratio, validate=True),
        "sqrt_transformer": FunctionTransformer(np.sqrt, validate=True),
    }


def _apply_polynomial_features(
    X: pd.DataFrame,
    base_numeric_cols: list[str],
    use_squared: bool,
    use_cubed: bool,
) -> pd.DataFrame:
    """Append squared and/or cubed versions of base numeric columns to X.

    Args:
        X: Feature DataFrame.
        base_numeric_cols: The *original* (pre-polynomial) numeric column names.
            Must not include previously generated polynomial columns to avoid
            chained expansions like ``col^2^2``.
        use_squared: If ``True``, append a ``col^2`` column for each base
            numeric column present in ``X``.
        use_cubed: If ``True``, append a ``col^3`` column for each base
            numeric column present in ``X``.

    Returns:
        A copy of ``X`` with new polynomial columns appended.
    """
    if not use_squared and not use_cubed:
        return X.copy()
    X = X.copy()
    for col in base_numeric_cols:
        if col not in X.columns:
            continue
        if use_squared:
            X[f"{col}^2"] = X[col] ** 2
        if use_cubed:
            X[f"{col}^3"] = X[col] ** 3
    return X


def _categorical_transformers() -> dict[str, Any]:
    """Return a mapping of name -> fitted sklearn categorical encoder instances.

    Returns:
        A dictionary with two entries:

        - ``"onehotencoder"`` –
          :class:`sklearn.preprocessing.OneHotEncoder` configured with
          ``handle_unknown='ignore'`` and ``drop='first'``.
        - ``"ordinalencoder"`` –
          :class:`sklearn.preprocessing.OrdinalEncoder` configured with
          ``handle_unknown='use_encoded_value'`` and ``unknown_value=-1``.
    """
    return {
        "onehotencoder": OneHotEncoder(handle_unknown="ignore", drop="first"),
        "ordinalencoder": OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ),
    }


def _build_model_spec(
    model_type: str, variant: str, threads_per_notebook: int
) -> ModelSpec:
    """Construct a :class:`ModelSpec` with default estimator and hyperparameter grids.

    Supported ``model_type`` values and their ``variant`` options:

    - ``"xgboost"`` – variant ignored; requires the ``xgboost`` package.
    - ``"random_forest"`` – variant ignored.
        - ``"logistic"`` – variant ignored.
        - ``"adaboost"`` – variant must be one of ``"logistic"``, ``"mlp"``,
      ``"perceptron"``, or ``"svc"`` (base estimator type).

    Args:
        model_type: Family of model to build (case-insensitive).
        variant: Sub-type within ``model_type``; only meaningful for
            ``"adaboost"``.
        threads_per_notebook: Number of CPU threads allocated to this notebook;
            passed as ``n_jobs`` wherever the estimator supports parallelism.

    Returns:
        A fully populated :class:`ModelSpec` dataclass containing the
        estimator instance, random-search distribution, explicit grid, and
        iteration budget.

    Raises:
        ValueError: If ``model_type`` or ``variant`` is not recognised.
        ImportError: If ``model_type='xgboost'`` and the package is absent.
    """
    model_type = model_type.lower()
    variant = variant.lower()

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model_type='xgboost'. "
                "Install it with: pip install xgboost"
            ) from exc

        estimator = XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=threads_per_notebook,
            eval_metric="logloss",
            tree_method="hist",
        )
        random_dist = {
            "n_estimators": [80, 120],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "random_state": [RANDOM_STATE],
            "n_jobs": [threads_per_notebook],
        }
        explicit_grid = {
            "n_estimators": [80, 120],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "random_state": [RANDOM_STATE],
            "n_jobs": [threads_per_notebook],
        }
        return ModelSpec(
            model_type,
            variant,
            estimator,
            random_dist,
            explicit_grid,
            n_random_iterations=6,
        )

    if model_type == "random_forest":
        estimator = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=threads_per_notebook
        )
        random_dist = {
            "n_estimators": [100, 150],
            "max_depth": [None, 16],
            "min_samples_split": [2, 8],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt"],
            "class_weight": ["balanced", None],
            "random_state": [RANDOM_STATE],
            "n_jobs": [threads_per_notebook],
        }
        explicit_grid = {
            "n_estimators": [100, 150],
            "max_depth": [None, 16],
            "min_samples_split": [2, 8],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt"],
            "class_weight": ["balanced", None],
            "random_state": [RANDOM_STATE],
            "n_jobs": [threads_per_notebook],
        }
        return ModelSpec(
            model_type,
            variant,
            estimator,
            random_dist,
            explicit_grid,
            n_random_iterations=6,
        )

    if model_type == "logistic":
        estimator = LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight=None,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        random_dist = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear"],
            "class_weight": [None, "balanced"],
            "max_iter": [1000],
            "random_state": [RANDOM_STATE],
        }
        explicit_grid = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear"],
            "class_weight": [None, "balanced"],
            "max_iter": [1000],
            "random_state": [RANDOM_STATE],
        }
        return ModelSpec(
            model_type,
            variant,
            estimator,
            random_dist,
            explicit_grid,
            n_random_iterations=6,
        )

    if model_type != "adaboost":
        raise ValueError(f"Unsupported model_type: {model_type}")

    if variant == "logistic":
        base_estimator = LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        )
        n_random_iterations = 6
    elif variant == "mlp":
        base_estimator = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            early_stopping=True,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
        n_random_iterations = 4
    elif variant == "perceptron":
        base_estimator = Perceptron(
            penalty=None,
            alpha=1e-6,
            max_iter=2000,
            tol=1e-4,
            random_state=RANDOM_STATE,
        )
        n_random_iterations = 6
    elif variant == "svc":
        base_estimator = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        )
        n_random_iterations = 4
    else:
        raise ValueError(f"Unsupported AdaBoost variant: {variant}")

    try:
        estimator = AdaBoostClassifier(
            estimator=base_estimator,
            random_state=RANDOM_STATE,
        )
    except TypeError:
        estimator = AdaBoostClassifier(
            base_estimator=base_estimator,
            random_state=RANDOM_STATE,
        )
    random_dist = {
        "n_estimators": [20, 40],
        "learning_rate": [0.5, 1.0],
        "random_state": [RANDOM_STATE],
    }
    explicit_grid = {
        "n_estimators": [20, 40],
        "learning_rate": [0.5, 1.0],
        "random_state": [RANDOM_STATE],
    }
    return ModelSpec(
        model_type,
        variant,
        estimator,
        random_dist,
        explicit_grid,
        n_random_iterations=n_random_iterations,
    )


def _build_preprocessor(
    X_train: pd.DataFrame,
    numeric_transformer: Any,
    categorical_transformer: Any,
    outliers_enabled: bool,
    iqr_multiplier: float = 1.5,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Assemble a :class:`~sklearn.compose.ColumnTransformer` preprocessing pipeline.

    Numeric columns receive median imputation, optional IQR clipping, and the
    supplied scaler.  Categorical columns receive most-frequent imputation and
    the supplied encoder.  Any column not detected as numeric or categorical is
    dropped via ``remainder='drop'``.

    Args:
        X_train: Training feature DataFrame used to identify numeric and
            categorical columns (column names and dtypes must match the data
            that will later be passed to ``fit_transform`` / ``transform``).
        numeric_transformer: A fitted-or-unfitted sklearn numeric
            transformer (e.g. ``PowerTransformer`` or
            ``FunctionTransformer(np.log1p)``) applied after imputation and
            optional outlier clipping.
        categorical_transformer: A fitted-or-unfitted sklearn encoder
            (e.g. ``OneHotEncoder``) applied after imputation.
        outliers_enabled: If ``True``, inserts an :class:`IQRClipper` step
            between imputation and scaling for numeric columns.

    Returns:
        A 3-tuple ``(preprocessor, numeric_cols, categorical_cols)`` where
        ``preprocessor`` is the unfitted :class:`~sklearn.compose.ColumnTransformer`,
        ``numeric_cols`` is the list of numeric column names, and
        ``categorical_cols`` is the list of categorical column names.
    """
    numeric_cols = X_train.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if outliers_enabled:
        numeric_steps.append(
            ("outlier", IQRClipper(iqr_multiplier=iqr_multiplier))
        )
    numeric_steps.append(("scaler", numeric_transformer))

    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", categorical_transformer),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_cols),
            (
                "categorical",
                Pipeline(steps=categorical_steps),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def _predict_scores(model: Any, X_values: np.ndarray) -> np.ndarray:
    """Extract continuous prediction scores from a fitted estimator.

    Tries prediction methods in priority order:
    1. ``predict_proba`` – returns the positive-class probability column.
    2. ``decision_function`` – returns the raw decision scores.
    3. ``predict`` – falls back to hard binary labels.

    Args:
        model: A fitted sklearn-compatible estimator.
        X_values: Pre-processed feature array of shape
            ``(n_samples, n_features)``.

    Returns:
        A 1-D NumPy array of continuous scores with length ``n_samples``.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_values)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs.ravel()
    if hasattr(model, "decision_function"):
        return model.decision_function(X_values)
    return model.predict(X_values)


def _compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """Compute the standard binary-classification evaluation metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels (0 or 1).
        y_score: Continuous prediction scores used for ROC-AUC computation
            (e.g. positive-class probabilities).

    Returns:
        A dictionary with keys ``"recall"``, ``"precision"``,
        ``"accuracy"``, ``"f1"``, and ``"roc_auc"``; all values are floats
        in ``[0, 1]``.
    """
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def _save_metric_best_models(
    results_df: pd.DataFrame,
    strategy_artifacts: dict[str, dict[str, Any]],
    output_dir: Path,
    model_type: str,
    metrics_to_save: list[str] | None = None,
) -> None:
    """Persist the best-performing model artifact for each evaluation metric.

    For every metric in ``[recall, precision, accuracy, roc_auc, f1]`` the
    run with the highest metric value is identified in ``results_df`` and its
    artifact dict is serialised to disk with ``joblib.dump``.  The output
    filename encodes the metric, transformer names, and outlier flag.

    Args:
        results_df: DataFrame of benchmark results; must contain columns
            ``run_id``, ``outliers_enabled`` (or ``outliers_clipped``),
            ``numeric_transformer``, ``categorical_transformer``, and one
            column per metric.
        strategy_artifacts: Mapping ``{run_id -> artifact_dict}`` produced
            during benchmarking.
        output_dir: Directory where ``.joblib`` files will be written;
            created automatically if it does not exist.
        model_type: Model family string (e.g. ``"xgboost"``) used in
            output filenames.
        metrics_to_save: Optional subset of metrics to persist. If ``None``,
            all standard metrics are considered.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_metrics = ["recall", "precision", "accuracy", "f1", "roc_auc"]
    requested_metrics = canonical_metrics
    if metrics_to_save:
        requested_metrics = []
        for metric in metrics_to_save:
            normalized = (
                metric.strip().lower().replace("-", "_").replace(" ", "_")
            )
            if normalized == "rocauc":
                normalized = "roc_auc"
            requested_metrics.append(normalized)

    for metric in requested_metrics:
        test_metric_col = f"test_{metric}"
        if test_metric_col not in results_df.columns:
            continue

        best_row = results_df.sort_values(
            test_metric_col, ascending=False
        ).iloc[0]
        run_id = str(best_row["run_id"])
        artifact = strategy_artifacts.get(run_id)
        if artifact is None:
            continue

        outlier_flag = best_row.get(
            "outliers_clipped", best_row.get("outliers_enabled")
        )
        file_name = (
            f"{model_type}_outliers_{outlier_flag}_"
            f"{best_row['numeric_transformer']}_"
            f"{best_row['categorical_transformer']}_best_{metric}.joblib"
        )
        save_path = output_dir / file_name
        joblib.dump(artifact, save_path)
        print(f"Saved {metric} best model -> {save_path}")


def _print_top_10(
    results_df: pd.DataFrame,
    score_priority: list[str] | None = None,
) -> None:
    """Print the top-10 benchmark runs for each evaluation metric to stdout.

    For each metric in ``score_priority`` (defaults to
    ``[recall, precision, accuracy, roc_auc, f1]``) the results DataFrame is
    sorted descending by that metric and the top 10 rows are printed with the
    columns: metric value, best hyperparameters, transformer names, outlier
    flag, polynomial flags, and run ID.

    Args:
        results_df: DataFrame of benchmark results returned by
            :func:`run_preprocessing_benchmark`.
        score_priority: Ordered list of metric names to display.  Defaults to
            ``['recall', 'precision', 'accuracy', 'roc_auc', 'f1']``.
    """
    metrics_list = score_priority if score_priority is not None else [
        "recall", "precision", "accuracy", "roc_auc", "f1"
    ]
    for metric in metrics_list:
        test_metric_col = f"test_{metric}"
        if test_metric_col not in results_df.columns:
            continue
        cols = [
            test_metric_col,
            "best_params",
            "numeric_transformer",
            "categorical_transformer",
            "outliers_clipped",
            "outliers_enabled",
            "use_squared",
            "use_cubed",
            "run_id",
        ]
        cols = [c for c in cols if c in results_df.columns]
        print(f"\nTop 10 models by {test_metric_col}:")
        print(
            results_df.sort_values(test_metric_col, ascending=False)
            .head(10)[cols]
            .to_string(index=False)
        )


def _plot_metric_best_models(
    results_df: pd.DataFrame,
    metric_best_artifacts: dict[str, dict[str, Any]],
) -> None:
    """Render diagnostic plots for the best model under each evaluation metric.

        For each metric the best run is selected and the following matplotlib
        figures are displayed:

        - Confusion matrix (``pre_deterioration_test`` set).
        - Confusion matrix (``full_timeline_test`` set).
        - Combined ROC-AUC plot with both test sets overlaid and labelled.
        - Combined PR-AUC plot with both test sets overlaid and labelled.
    - Feature importance bar chart (only if the estimator exposes
      ``feature_importances_``, e.g. tree-based models; top-20 features shown).

    Args:
        results_df: DataFrame of benchmark results returned by
            :func:`run_preprocessing_benchmark`.
        metric_best_artifacts: Nested mapping
            ``{metric_name -> {run_id -> artifact_dict}}`` produced during
            benchmarking.  Each artifact dict must contain the keys:
            ``model``, ``y_test``, ``y_pred``, ``y_score``,
            ``y_test_joined``, ``y_pred_joined``, ``y_score_joined``.
    """
    for metric in metric_best_artifacts.keys():
        test_metric_col = f"test_{metric}"
        best_row = results_df.sort_values(
            test_metric_col, ascending=False
        ).iloc[0]
        artifact = metric_best_artifacts[metric][best_row["run_id"]]
        model = artifact["model"]
        y_test = artifact["y_test"]
        y_pred = artifact["y_pred"]
        y_score = artifact["y_score"]
        y_test_joined = artifact["y_test_joined"]
        y_pred_joined = artifact["y_pred_joined"]
        y_score_joined = artifact["y_score_joined"]
        outlier_flag = best_row.get(
            "outliers_clipped", best_row.get("outliers_enabled")
        )
        descriptor = (
            f"{metric.upper()} best | {best_row['numeric_transformer']} + "
            f"{best_row['categorical_transformer']} | "
            f"outliers={outlier_flag}"
        )
        pre_descriptor = f"{descriptor} | pre_deterioration_test"
        full_descriptor = f"{descriptor} | full_timeline_test"

        plt.figure(figsize=(5.4, 3.8))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title(f"Confusion Matrix - {pre_descriptor}", fontsize=10)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5.4, 3.8))
        ConfusionMatrixDisplay.from_predictions(y_test_joined, y_pred_joined)
        plt.title(f"Confusion Matrix - {full_descriptor}", fontsize=10)
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        RocCurveDisplay.from_predictions(
            y_test,
            y_score,
            name="pre_deterioration_test",
            ax=ax,
        )
        RocCurveDisplay.from_predictions(
            y_test_joined,
            y_score_joined,
            name="full_timeline_test",
            ax=ax,
        )
        ax.set_title(f"ROC-AUC Curve - {descriptor}", fontsize=10)
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_score,
            name="pre_deterioration_test",
            ax=ax,
        )
        PrecisionRecallDisplay.from_predictions(
            y_test_joined,
            y_score_joined,
            name="full_timeline_test",
            ax=ax,
        )
        ax.set_title(f"PR-AUC Curve - {descriptor}", fontsize=10)
        ax.legend(loc="lower left")
        fig.tight_layout()
        plt.show()

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
            top_n = min(20, len(importances))
            top_idx = np.argsort(importances)[-top_n:]
            plt.figure(figsize=(6, 4))
            plt.barh(range(top_n), importances[top_idx])
            plt.yticks(range(top_n), [f"feature_{i}" for i in top_idx])
            plt.title(f"Feature Importance - {descriptor}", fontsize=10)
            plt.tight_layout()
            plt.show()


def _get_train_size_reduction_factor(model_type: str, variant: str) -> float:
    """Determine the train size reduction factor based on model complexity.

    SVC and MLP have high computational complexity (O(n²-n³) and O(n*h)
    respectively) and benefit from training on a subset of data. Other models
    scale better and use the full training set.

    Args:
        model_type: Model family (e.g., 'xgboost', 'random_forest', 'adaboost').
        variant: Sub-type within model_type (e.g., 'svc', 'mlp' for adaboost).

    Returns:
        A float in (0.0, 1.0] indicating the fraction of training data to use.
        1.0 means no reduction (use full training set).
    """
    model_type = model_type.lower()
    variant = variant.lower()

    # SVC and MLP have poor scalability; use n% of training data
    if model_type == "adaboost" and variant in ("svc", "mlp"):
        return 0.15

    # All other models use the full training set
    return 1.0


def _reduce_train_size(
    X: pd.DataFrame,
    y: pd.Series,
    reduction_factor: float,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply stratified random sampling to reduce training set size.

    If reduction_factor >= 1.0, returns the data unchanged.
    Otherwise, samples a fraction of the data while preserving class proportions.

    Args:
        X: Feature DataFrame.
        y: Target Series aligned with X.
        reduction_factor: Fraction of data to retain (0.0, 1.0].
        random_state: Seed for reproducibility.

    Returns:
        A (X_reduced, y_reduced) tuple of the same type as inputs.
    """
    if reduction_factor >= 1.0:
        return X, y

    # Use stratified sampling to preserve class proportions
    X_reduced, _, y_reduced, _ = train_test_split(
        X,
        y,
        train_size=reduction_factor,
        random_state=random_state,
        stratify=y,
    )
    return X_reduced, y_reduced


def _cap_train_observations(
    X: pd.DataFrame,
    y: pd.Series,
    max_train_observations: int | None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Cap training samples using stratified sampling when needed."""
    if max_train_observations is None:
        return X, y
    if max_train_observations <= 0:
        raise ValueError("max_train_observations must be a positive integer")
    if len(X) <= max_train_observations:
        return X, y
    reduction_factor = max_train_observations / len(X)
    return _reduce_train_size(
        X,
        y,
        reduction_factor=reduction_factor,
        random_state=random_state,
    )


def print_best_model_metrics(
    results_df: pd.DataFrame,
    metrics_to_show: list[str] | None = None,
) -> None:
    """Print best model parameters and score table for selected metrics.

    For each metric in ``metrics_to_show``, this function finds the row with
    the highest ``test_<metric>`` score and prints:
    1) the winning preprocessing configuration,
    2) the winning hyperparameters,
    3) a compact table of Recall, Precision, Accuracy, F1, and ROC-AUC across
       train and test datasets.
    """
    if metrics_to_show is None:
        metrics_to_show = ["recall", "precision", "f1", "roc_auc"]

    valid_metrics = {"recall", "precision", "accuracy", "f1", "roc_auc"}
    invalid_metrics = set(metrics_to_show) - valid_metrics
    if invalid_metrics:
        raise ValueError(
            "Unknown metrics_to_show values: "
            f"{sorted(invalid_metrics)}"
        )

    for metric in metrics_to_show:
        test_metric_col = f"test_{metric}"
        if test_metric_col not in results_df.columns:
            raise ValueError(
                f"Column '{test_metric_col}' not found in results_df. "
                "Run the updated benchmark function first."
            )

        best_row = results_df.sort_values(
            test_metric_col, ascending=False
        ).iloc[0]
        score_table = pd.DataFrame(
            [
                {
                    "dataset": "train",
                    "recall": best_row["train_recall"],
                    "precision": best_row["train_precision"],
                    "accuracy": best_row["train_accuracy"],
                    "f1": best_row["train_f1"],
                    "roc_auc": best_row["train_roc_auc"],
                },
                {
                    "dataset": "test",
                    "recall": best_row["test_recall"],
                    "precision": best_row["test_precision"],
                    "accuracy": best_row["test_accuracy"],
                    "f1": best_row["test_f1"],
                    "roc_auc": best_row["test_roc_auc"],
                },
            ]
        )

        poly_info = ""
        if "use_squared" in best_row.index or "use_squared" in best_row:
            poly_info = (
                f", squared={best_row.get('use_squared', False)}"
                f", cubed={best_row.get('use_cubed', False)}"
            )
        print(f"\nBest model by test_{metric}:")
        outlier_flag = best_row.get(
            "outliers_clipped", best_row.get("outliers_enabled")
        )
        print(
            "config="
            f"num={best_row['numeric_transformer']}, "
            f"cat={best_row['categorical_transformer']}, "
            f"outliers={outlier_flag}"
            f"{poly_info}"
        )
        print(f"best_params={best_row['best_params']}")
        print(score_table.to_string(index=False))


def run_preprocessing_benchmark(
    model_type: str,
    variant: str,
    output_dir_name: str,
    run_randomized: bool = True,
    run_explicit: bool = True,
    use_imbalance_options: bool = False,
    balance_classes_before_training: bool = True,
    total_threads: int = 16,
    num_current_notebooks: int = 4,
    iqr_multiplier: float = 1.5,
    numeric_transformer_names: list[str] | None = None,
    categorical_transformer_names: list[str] | None = None,
    outliers_enabled_override: bool | None = None,
    outliers_clipped_override: bool | None = None,
    train_size_reduction_factor_override: float | None = None,
    max_train_observations: int | None = None,
    metrics_to_save: list[str] | None = None,
    max_combinations: int | None = None,
    n_random_iterations_override: int | None = None,
    add_polynomial: bool = True,
    score_priority: list[str] | None = None,
    top_n: int = 5,
    save_prediction_artifacts: bool = False,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Enumerate preprocessing + polynomial strategies, fit models, and return ranked results.

    Evaluates all combinations of:

    - numeric transformer,
    - categorical encoder,
    - outlier clipping enabled/disabled,
    - polynomial features (none / squared / cubed / squared+cubed).

    For each combination a fresh model clone is fitted and evaluated on the
    pre-deterioration test set and the full-timeline test set.  Scores are rounded to 3 decimal places.

    Args:
        model_type: Model family (e.g. ``"xgboost"``).
        variant: Sub-type within ``model_type`` (e.g. ``"logistic"`` for
            AdaBoost).
        output_dir_name: Directory name under ``base_dir`` for saved artifacts.
        run_randomized: If ``True``, run ``RandomizedSearchCV`` to tune
            hyperparameters.
        run_explicit: If ``True``, run ``GridSearchCV`` to tune
            hyperparameters.
        use_imbalance_options: Reserved for future imbalance strategies.
        balance_classes_before_training: If ``True``, apply SMOTE before
            fitting (with safe fallback to under-sampling for tiny minority
            classes).
        total_threads: Total CPU threads available across all notebooks.
        num_current_notebooks: Number of notebooks running in parallel;
            ``total_threads // num_current_notebooks`` is passed to the
            estimator as ``n_jobs``.
        iqr_multiplier: IQR fence multiplier for :class:`IQRClipper`.
        numeric_transformer_names: Optional subset of numeric transformer
            names to test.
        categorical_transformer_names: Optional subset of categorical
            transformer names to test.
        outliers_enabled_override: Deprecated alias for
            ``outliers_clipped_override``. If provided, only test this
            outlier-clipping setting instead of both ``True`` and ``False``.
        outliers_clipped_override: Preferred name for the outlier-clipping
            toggle. If provided, only test this setting instead of both
            ``True`` and ``False``.
        train_size_reduction_factor_override: Override the automatic
            reduction factor for slow models.
        max_train_observations: Hard cap on training observations.
        metrics_to_save: Metrics for which the best model is saved to disk.
        max_combinations: Stop after this many combinations (useful for
            quick smoke tests).
        n_random_iterations_override: Override ``n_iter`` for
            ``RandomizedSearchCV``.
        add_polynomial: If ``True``, test two polynomial modes in addition to
            the shared baseline logic: baseline (no polynomial) and
            squared+cubed together. Defaults to ``True``.
        score_priority: Ordered list of metric names used to sort the output.
            Defaults to ``['recall', 'precision', 'accuracy', 'f1',
            'roc_auc']``.  The first entry is the primary sort key.
        top_n: Number of top rows to collect per metric when building
            ``top_combined``.  Defaults to ``5``.
        save_prediction_artifacts: If ``True``, retain per-run prediction
            arrays (``y_test``, ``y_pred``, ``y_score`` and joined variants)
            inside ``strategy_artifacts`` for downstream plotting. Defaults
            to ``False`` to reduce RAM usage during large benchmark runs.

    Returns:
        A 6-tuple ``(results_df, strategy_artifacts, top_per_metric,
        top_scores_df, top_combined, rest_df)`` where:

        - ``results_df`` – full results DataFrame (one row per run).
        - ``strategy_artifacts`` – dict of per-run artifacts keyed by
          ``run_id``.
        - ``top_per_metric`` – ``{metric_name: best_row_Series}`` for each
          metric in ``score_priority``.
                - ``top_scores_df`` – one top-1 row per metric with a ``metric``
                    column indicating the selecting metric.
                - ``top_combined`` – deduplicated top-N rows across all metrics,
                    sorted by ``test_roc_auc`` descending.
        - ``rest_df`` – remaining rows sorted by ``score_priority``.
    """
    alias_map = {
        "roc-auc": "roc_auc",
        "roc_auc": "roc_auc",
        "roc auc": "roc_auc",
        "precision": "precision",
        "precission": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
        "f1": "f1",
    }
    _VALID_METRICS = {"recall", "precision", "accuracy", "f1", "roc_auc"}
    if score_priority is None:
        score_priority = ["recall", "precision", "accuracy", "f1", "roc_auc"]
    normalized_score_priority: list[str] = []
    for metric in score_priority:
        metric_key = metric.strip().lower().replace("_", " ")
        metric_key = " ".join(metric_key.split())
        mapped = alias_map.get(metric_key)
        if mapped is None:
            mapped = alias_map.get(metric.strip().lower())
        if mapped is None:
            mapped = metric.strip().lower().replace("-", "_")
        normalized_score_priority.append(mapped)
    score_priority = normalized_score_priority

    invalid = set(score_priority) - _VALID_METRICS
    if invalid:
        raise ValueError(f"Unknown score_priority values: {sorted(invalid)}")

    threads_per_notebook = max(1, total_threads // max(1, num_current_notebooks))

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / output_dir_name
    df, joined_data, target, features = _prepare_hcd_data(base_dir)

    splits = _split_df_joined(df, joined_data, target, features)
    X_train_df = splits["X_train_df"]
    X_test_df = splits["X_test_df"]
    y_train_df = splits["y_train_df"]
    y_test_df = splits["y_test_df"]
    X_train_joined = splits["X_train_joined"]
    X_test_joined = splits["X_test_joined"]
    y_test_joined = splits["y_test_joined"]

    spec = _build_model_spec(model_type, variant, threads_per_notebook)

    # Determine base numeric cols ONCE from original X_train_df (pre-poly)
    base_numeric_cols = X_train_df.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()

    numeric_transformers = _numeric_transformers()
    categorical_transformers = _categorical_transformers()

    if numeric_transformer_names is not None:
        missing_numeric = set(numeric_transformer_names) - set(
            numeric_transformers.keys()
        )
        if missing_numeric:
            raise ValueError(
                f"Unknown numeric transformers: {sorted(missing_numeric)}"
            )
        numeric_transformers = {
            key: value
            for key, value in numeric_transformers.items()
            if key in numeric_transformer_names
        }

    if categorical_transformer_names is not None:
        missing_categorical = set(categorical_transformer_names) - set(
            categorical_transformers.keys()
        )
        if missing_categorical:
            raise ValueError(
                "Unknown categorical transformers: "
                f"{sorted(missing_categorical)}"
            )
        categorical_transformers = {
            key: value
            for key, value in categorical_transformers.items()
            if key in categorical_transformer_names
        }

    if not numeric_transformers:
        raise ValueError("No numeric transformers selected for benchmarking.")
    if not categorical_transformers:
        raise ValueError(
            "No categorical transformers selected for benchmarking."
        )

    if outliers_enabled_override is not None:
        warnings.warn(
            "'outliers_enabled_override' is deprecated; use "
            "'outliers_clipped_override' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if (
        outliers_clipped_override is not None
        and outliers_enabled_override is not None
        and outliers_clipped_override != outliers_enabled_override
    ):
        raise ValueError(
            "Received conflicting values for outlier clipping aliases: "
            "'outliers_enabled_override' and 'outliers_clipped_override'."
        )

    effective_outliers_clipped_override = (
        outliers_clipped_override
        if outliers_clipped_override is not None
        else outliers_enabled_override
    )

    outlier_options = (
        [effective_outliers_clipped_override]
        if effective_outliers_clipped_override is not None
        else [True, False]
    )

    poly_options: list[tuple[bool, bool]] = [(False, False)]
    if add_polynomial:
        poly_options.append((True, True))

    reduction_factor = (
        train_size_reduction_factor_override
        if train_size_reduction_factor_override is not None
        else _get_train_size_reduction_factor(model_type, variant)
    )

    n_iter = (
        n_random_iterations_override
        if n_random_iterations_override is not None
        else spec.n_random_iterations
    )

    results: list[dict[str, Any]] = []
    strategy_artifacts: dict[str, dict[str, Any]] = {}

    run_counter = 0
    done = False
    for use_squared, use_cubed in poly_options:
        if done:
            break
        for numeric_name, numeric_transformer in numeric_transformers.items():
            if done:
                break
            for (
                categorical_name,
                categorical_transformer,
            ) in categorical_transformers.items():
                if done:
                    break
                for outliers_clipped in outlier_options:
                    run_counter += 1
                    run_id = f"run_{run_counter:03d}"

                    # Apply polynomial feature augmentation to all splits
                    X_train_poly = _apply_polynomial_features(
                        X_train_df, base_numeric_cols, use_squared, use_cubed
                    )
                    X_test_poly = _apply_polynomial_features(
                        X_test_df, base_numeric_cols, use_squared, use_cubed
                    )
                    X_train_joined_poly = _apply_polynomial_features(
                        X_train_joined, base_numeric_cols, use_squared, use_cubed
                    )
                    X_test_joined_poly = _apply_polynomial_features(
                        X_test_joined, base_numeric_cols, use_squared, use_cubed
                    )

                    preprocessor, numeric_cols, categorical_cols = _build_preprocessor(
                        X_train=X_train_poly,
                        numeric_transformer=clone(numeric_transformer),
                        categorical_transformer=clone(categorical_transformer),
                        outliers_enabled=outliers_clipped,
                        iqr_multiplier=iqr_multiplier,
                    )

                    X_train_proc = preprocessor.fit_transform(X_train_poly)
                    X_test_proc = preprocessor.transform(X_test_poly)
                    X_train_joined_proc = preprocessor.transform(X_train_joined_poly)
                    X_test_joined_proc = preprocessor.transform(X_test_joined_poly)

                    transformed_features = X_train_proc.shape[1]

                    # Class balancing
                    X_fit = pd.DataFrame(X_train_proc)
                    y_fit = y_train_df.reset_index(drop=True)
                    if balance_classes_before_training:
                        X_fit, y_fit = balance_binary_classes_smote(
                            X_fit, y_fit, random_state=RANDOM_STATE
                        )

                    # Cap and reduce training size
                    X_fit, y_fit = _cap_train_observations(
                        X_fit, y_fit, max_train_observations
                    )
                    X_fit, y_fit = _reduce_train_size(
                        X_fit, y_fit, reduction_factor
                    )

                    # Fit model with optional hyperparameter tuning
                    estimator = clone(spec.estimator)
                    best_params: dict[str, Any] = {}

                    if run_randomized and spec.random_dist:
                        finite_space_size = _finite_param_space_size(spec.random_dist)
                        effective_n_iter = n_iter
                        if finite_space_size is not None and finite_space_size > 0:
                            effective_n_iter = min(n_iter, finite_space_size)

                        search = RandomizedSearchCV(
                            clone(spec.estimator),
                            param_distributions=spec.random_dist,
                            n_iter=effective_n_iter,
                            cv=3,
                            scoring="recall",
                            random_state=RANDOM_STATE,
                            n_jobs=threads_per_notebook,
                            refit=True,
                        )
                        search.fit(X_fit, y_fit)
                        estimator = search.best_estimator_
                        best_params = search.best_params_
                    elif run_explicit and spec.explicit_grid:
                        search = GridSearchCV(
                            clone(spec.estimator),
                            param_grid=spec.explicit_grid,
                            cv=3,
                            scoring="recall",
                            n_jobs=threads_per_notebook,
                            refit=True,
                        )
                        search.fit(X_fit, y_fit)
                        estimator = search.best_estimator_
                        best_params = search.best_params_
                    else:
                        estimator.fit(X_fit, y_fit)

                    # Compute metrics on test split (pre-deterioration)
                    y_pred_test = estimator.predict(X_test_proc)
                    y_score_test = _predict_scores(estimator, X_test_proc)
                    test_metrics = _compute_metrics(y_test_df, y_pred_test, y_score_test)

                    # Compute metrics on train split (pre-deterioration)
                    y_pred_train = estimator.predict(X_train_proc)
                    y_score_train = _predict_scores(estimator, X_train_proc)
                    train_metrics = _compute_metrics(y_train_df, y_pred_train, y_score_train)

                    # Compute metrics on full-timeline test split
                    y_pred_joined = estimator.predict(X_test_joined_proc)
                    y_score_joined = _predict_scores(estimator, X_test_joined_proc)
                    joined_metrics = _compute_metrics(y_test_joined, y_pred_joined, y_score_joined)

                    results.append(
                        {
                            "run_id": run_id,
                            "numeric_transformer": numeric_name,
                            "categorical_transformer": categorical_name,
                            "outliers_enabled": outliers_clipped,
                            "outliers_clipped": outliers_clipped,
                            "use_squared": use_squared,
                            "use_cubed": use_cubed,
                            "iqr_multiplier": iqr_multiplier,
                            "n_numeric_cols": len(numeric_cols),
                            "n_categorical_cols": len(categorical_cols),
                            "n_transformed_features": transformed_features,
                            "best_params": best_params,
                            "model_params": _format_model_params(best_params),
                            "train_recall": train_metrics["recall"],
                            "train_precision": train_metrics["precision"],
                            "train_accuracy": train_metrics["accuracy"],
                            "train_f1": train_metrics["f1"],
                            "train_roc_auc": train_metrics["roc_auc"],
                            "test_recall": test_metrics["recall"],
                            "test_precision": test_metrics["precision"],
                            "test_accuracy": test_metrics["accuracy"],
                            "test_f1": test_metrics["f1"],
                            "test_roc_auc": test_metrics["roc_auc"],
                            "joined_recall": joined_metrics["recall"],
                            "joined_precision": joined_metrics["precision"],
                            "joined_accuracy": joined_metrics["accuracy"],
                            "joined_f1": joined_metrics["f1"],
                            "joined_roc_auc": joined_metrics["roc_auc"],
                        }
                    )

                    strategy_artifacts[run_id] = {
                        "preprocessor": preprocessor,
                        "model": estimator,
                        "numeric_columns": numeric_cols,
                        "categorical_columns": categorical_cols,
                    }
                    if save_prediction_artifacts:
                        strategy_artifacts[run_id].update(
                            {
                                "y_test": y_test_df,
                                "y_pred": y_pred_test,
                                "y_score": y_score_test,
                                "y_test_joined": y_test_joined,
                                "y_pred_joined": y_pred_joined,
                                "y_score_joined": y_score_joined,
                            }
                        )

                    # Release large per-iteration objects to reduce peak RAM.
                    del (
                        X_train_poly,
                        X_test_poly,
                        X_train_joined_poly,
                        X_test_joined_poly,
                        X_train_proc,
                        X_test_proc,
                        X_train_joined_proc,
                        X_test_joined_proc,
                        X_fit,
                        y_fit,
                        estimator,
                        y_pred_test,
                        y_score_test,
                        y_pred_train,
                        y_score_train,
                        y_pred_joined,
                        y_score_joined,
                    )
                    gc.collect()

                    if (
                        max_combinations is not None
                        and run_counter >= max_combinations
                    ):
                        done = True
                        break

    if not results:
        raise RuntimeError("No preprocessing strategies were produced.")

    results_df = pd.DataFrame(results)

    # Round all score columns to 3 decimal places
    score_prefixes = ("train_", "test_", "joined_")
    score_suffixes = ("recall", "precision", "accuracy", "f1", "roc_auc")
    score_cols_all = [
        f"{p}{s}"
        for p in score_prefixes
        for s in score_suffixes
        if f"{p}{s}" in results_df.columns
    ]
    results_df[score_cols_all] = results_df[score_cols_all].round(3)

    # Build ranked output tables
    test_score_cols = [
        f"test_{m}" for m in score_priority if f"test_{m}" in results_df.columns
    ]
    if not test_score_cols:
        raise RuntimeError("No test score columns available for ranking output.")

    top_frames = [results_df.nlargest(top_n, col) for col in test_score_cols]
    top_concat = pd.concat(top_frames, axis=0)
    selected_idx = pd.Index(top_concat.index).drop_duplicates()
    top_combined = (
        results_df.loc[selected_idx]
        .sort_values("test_roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    combined_front_cols = [
        "run_id",
        "numeric_transformer",
        "categorical_transformer",
        "outliers_clipped",
        "outliers_enabled",
        "use_squared",
        "use_cubed",
        "model_params",
        "best_params",
    ]
    combined_front_cols = [
        col for col in combined_front_cols if col in top_combined.columns
    ]
    combined_other_cols = [
        col for col in top_combined.columns if col not in combined_front_cols
    ]
    top_combined = top_combined[combined_front_cols + combined_other_cols]

    top_per_metric: dict[str, Any] = {
        m: results_df.nlargest(1, f"test_{m}").iloc[0]
        for m in score_priority
        if f"test_{m}" in results_df.columns
    }

    top_scores_rows: list[pd.Series] = []
    for metric_name in score_priority:
        metric_col = f"test_{metric_name}"
        if metric_col not in results_df.columns:
            continue
        metric_top = results_df.nlargest(1, metric_col).iloc[0].copy()
        metric_top["metric"] = metric_name
        top_scores_rows.append(metric_top)

    top_scores_df = (
        pd.DataFrame(top_scores_rows).reset_index(drop=True)
        if top_scores_rows
        else pd.DataFrame(columns=[*results_df.columns, "metric"])
    )

    top_scores_front_cols = [
        "metric",
        "run_id",
        "numeric_transformer",
        "categorical_transformer",
        "outliers_clipped",
        "outliers_enabled",
        "use_squared",
        "use_cubed",
        "model_params",
        "best_params",
    ]
    top_scores_front_cols = [
        col for col in top_scores_front_cols if col in top_scores_df.columns
    ]
    top_scores_other_cols = [
        col for col in top_scores_df.columns if col not in top_scores_front_cols
    ]
    top_scores_df = top_scores_df[top_scores_front_cols + top_scores_other_cols]

    rest_df = (
        results_df.loc[~results_df.index.isin(selected_idx)]
        .sort_values(test_score_cols, ascending=False)
        .reset_index(drop=True)
    )

    if metrics_to_save:
        _save_metric_best_models(
            results_df=results_df,
            strategy_artifacts=strategy_artifacts,
            output_dir=output_dir,
            model_type=model_type,
            metrics_to_save=metrics_to_save,
        )

    return (
        results_df,
        strategy_artifacts,
        top_per_metric,
        top_scores_df,
        top_combined,
        rest_df,
    )
