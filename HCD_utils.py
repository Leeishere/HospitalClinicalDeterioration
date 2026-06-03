
import pandas as pd
import numpy as np
from numpy.linalg import pinv
from scipy.stats import normaltest

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import hdbscan





def _signed_sqrt(values):
    return np.sign(values) * np.sqrt(np.abs(values))


def _x_over_one_plus_abs(values):
    return values / (1.0 + np.abs(values))


def build_numeric_transformer(transform_type, random_state):
    if transform_type == "robust_scaler":
        transformer = RobustScaler()
        params = {"type": "RobustScaler"}
    elif transform_type == "power":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        params = {"type": "PowerTransformer", "method": "yeo-johnson", "random_state": random_state}
    elif transform_type == "sqrt":
        transformer = FunctionTransformer(_signed_sqrt, validate=False)
        params = {"type": "FunctionTransformer", "mode": "signed_sqrt", "validate": False}
    elif transform_type == "custom":
        transformer = FunctionTransformer(_x_over_one_plus_abs, validate=False)
        params = {"type": "FunctionTransformer", "mode": "x_over_1_plus_abs_x", "validate": False}
    elif transform_type == "identity":
        transformer = FunctionTransformer(func=None, validate=False)
        params = {"type": "FunctionTransformer", "mode": "identity", "validate": False}
    else:
        raise ValueError(f"Unknown numeric transform: {transform_type}")

    return transformer, params


def select_best_scaler_by_normaltest(
    train_series,
    candidate_transforms,
    random_state,
    min_normaltest_n=8,
):
    column_name = getattr(train_series, "name", None) or "unnamed_column"
    train_values = np.asarray(train_series, dtype=float).reshape(-1, 1)
    finite_values = train_values[np.isfinite(train_values[:, 0])]
    unique_count = int(np.unique(finite_values).shape[0]) if finite_values.size else 0

    if finite_values.size == 0:
        transformer, params = build_numeric_transformer("identity", random_state=random_state)
        transformer.fit(np.zeros((1, 1)))
        return {
            "column": column_name,
            "scaler": "identity",
            "transformer": transformer,
            "params": params,
            "normaltest_pvalue": None,
            "normaltest_statistic": None,
            "reason": "no_finite_values",
            "candidates": [],
        }

    if unique_count <= 1:
        transformer, params = build_numeric_transformer("identity", random_state=random_state)
        transformer.fit(train_values)
        return {
            "column": column_name,
            "scaler": "identity",
            "transformer": transformer,
            "params": params,
            "normaltest_pvalue": None,
            "normaltest_statistic": None,
            "reason": "constant_column",
            "candidates": [],
        }

    candidate_results = []
    for transform_name in candidate_transforms:
        transformer, params = build_numeric_transformer(transform_name, random_state=random_state)
        try:
            transformed = transformer.fit_transform(train_values)
            transformed_flat = np.asarray(transformed, dtype=float).reshape(-1)
            finite_transformed = transformed_flat[np.isfinite(transformed_flat)]
            if finite_transformed.size < min_normaltest_n or np.unique(finite_transformed).shape[0] <= 1:
                statistic = None
                pvalue = None
                reason = "insufficient_values_for_normaltest"
            else:
                statistic, pvalue = normaltest(finite_transformed)
                statistic = float(statistic)
                pvalue = float(pvalue)
                reason = "normaltest"

            candidate_results.append(
                {
                    "name": transform_name,
                    "transformer": transformer,
                    "params": params,
                    "normaltest_statistic": statistic,
                    "normaltest_pvalue": pvalue,
                    "reason": reason,
                }
            )
        except Exception as exc:
            candidate_results.append(
                {
                    "name": transform_name,
                    "transformer": None,
                    "params": params,
                    "normaltest_statistic": None,
                    "normaltest_pvalue": None,
                    "reason": f"transform_failed: {type(exc).__name__}",
                }
            )

    successful_candidates = [
        candidate
        for candidate in candidate_results
        if candidate["transformer"] is not None
    ]
    if not successful_candidates:
        transformer, params = build_numeric_transformer("identity", random_state=random_state)
        transformer.fit(train_values)
        return {
            "column": column_name,
            "scaler": "identity",
            "transformer": transformer,
            "params": params,
            "normaltest_pvalue": None,
            "normaltest_statistic": None,
            "reason": "all_candidates_failed",
            "candidates": candidate_results,
        }

    def sort_key(candidate):
        pvalue = candidate["normaltest_pvalue"]
        statistic = candidate["normaltest_statistic"]
        return (
            pvalue is None,
            -(pvalue if pvalue is not None else float("-inf")),
            statistic if statistic is not None else float("inf"),
            candidate["name"],
        )

    best_candidate = sorted(successful_candidates, key=sort_key)[0]
    serialized_candidates = []
    for candidate in candidate_results:
        serialized_candidates.append(
            {
                "name": candidate["name"],
                "params": candidate["params"],
                "normaltest_statistic": candidate["normaltest_statistic"],
                "normaltest_pvalue": candidate["normaltest_pvalue"],
                "reason": candidate["reason"],
            }
        )

    return {
        "column": column_name,
        "scaler": best_candidate["name"],
        "transformer": best_candidate["transformer"],
        "params": best_candidate["params"],
        "normaltest_pvalue": best_candidate["normaltest_pvalue"],
        "normaltest_statistic": best_candidate["normaltest_statistic"],
        "reason": best_candidate["reason"],
        "candidates": serialized_candidates,
    }


def fit_columnwise_numeric_scalers(
    X_train_num,
    X_other_frames,
    candidate_transforms,
    random_state,
    min_normaltest_n=8,
):
    if not hasattr(X_train_num, "columns"):
        raise TypeError("X_train_num must be a pandas DataFrame")

    transformed_frames = {"train": pd.DataFrame(index=X_train_num.index)}
    for frame_name, frame in X_other_frames.items():
        if not hasattr(frame, "columns"):
            raise TypeError(f"{frame_name} must be a pandas DataFrame")
        transformed_frames[frame_name] = pd.DataFrame(index=frame.index)

    scaler_map = {}
    scaler_details = {}

    for column in X_train_num.columns:
        selection = select_best_scaler_by_normaltest(
            X_train_num[column],
            candidate_transforms=candidate_transforms,
            random_state=random_state,
            min_normaltest_n=min_normaltest_n,
        )
        transformer = selection.pop("transformer")
        train_column_values = X_train_num[[column]].to_numpy(dtype=float)
        transformed_frames["train"][column] = np.asarray(
            transformer.transform(train_column_values),
            dtype=float,
        ).reshape(-1)

        for frame_name, frame in X_other_frames.items():
            frame_column_values = frame[[column]].to_numpy(dtype=float)
            transformed_frames[frame_name][column] = np.asarray(
                transformer.transform(frame_column_values),
                dtype=float,
            ).reshape(-1)

        scaler_map[column] = selection["scaler"]
        scaler_details[column] = selection

    return transformed_frames, scaler_map, scaler_details

class ClusterFeatureModels:
    """
    Fit/predict hard cluster labels for KMeans, GaussianMixture, and HDBSCAN.

    Input contract behavior:
    - If fit() receives a DataFrame, the fitted feature column names are saved.
    - predict() then requires those trained columns to exist and reorders input
      columns to match training order.
    - Extra columns are allowed by default.
    - If fit() used non-DataFrame input, only feature-count validation is applied.
    """

    def __init__(self, n_kmeans=None, 
                 n_gmm=None, 
                 random_state=42):
        self.n_kmeans = n_kmeans
        self.n_gmm = n_gmm
        self.random_state = random_state

        self.kmeans = None
        self.gmm = None
        self.hdb = None

        self.hdb_noise_label_ = None
        self.hdb_params_ = None
        self.selection_results_ = {}

        self.feature_names_ = [
            "cluster_kmeans",
            "cluster_gmm",
            "cluster_hdbscan",
        ]

        # Input contract metadata (persisted via joblib)
        self.expected_input_columns_ = None
        self.n_input_features_ = None

        # Policy: allow extras, require DataFrame when named contract exists
        self.allow_extra_columns = True
        self.require_dataframe_for_named_contract = True

    @staticmethod
    def _to_dense(X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    @staticmethod
    def _safe_silhouette(X, labels):
        unique = np.unique(labels)
        if unique.shape[0] < 2:
            return np.nan
        return float(silhouette_score(X, labels))

    def _capture_input_contract(self, X):
        if hasattr(X, "columns"):
            expected_cols = list(X.columns)
            self.expected_input_columns_ = expected_cols
            self.n_input_features_ = len(expected_cols)
            return

        Xd = self._to_dense(X)
        if Xd.ndim != 2:
            raise ValueError("Input data must be 2D")
        self.expected_input_columns_ = None
        self.n_input_features_ = int(Xd.shape[1])

    def _validate_and_align_input(self, X, stage="predict"):
        expected_cols = getattr(self, "expected_input_columns_", None)
        n_input_features = getattr(self, "n_input_features_", None)
        allow_extra = getattr(self, "allow_extra_columns", True)
        require_df_named = getattr(self, "require_dataframe_for_named_contract", True)

        if expected_cols is not None:
            if not hasattr(X, "columns"):
                if require_df_named:
                    raise TypeError(
                        f"{stage} expects a pandas DataFrame with trained columns in order-aware mode"
                    )
                Xd = self._to_dense(X)
            else:
                missing = [col for col in expected_cols if col not in X.columns]
                if missing:
                    raise ValueError(f"Missing required columns at {stage} time: {missing}")
                if not allow_extra:
                    extra = [col for col in X.columns if col not in expected_cols]
                    if extra:
                        raise ValueError(f"Unexpected extra columns at {stage} time: {extra}")
                Xd = self._to_dense(X.loc[:, expected_cols])
        else:
            Xd = self._to_dense(X)

        if Xd.ndim != 2:
            raise ValueError(f"Input data at {stage} time must be 2D")

        if n_input_features is not None and Xd.shape[1] != int(n_input_features):
            raise ValueError(
                f"Expected {int(n_input_features)} input features at {stage} time, got {Xd.shape[1]}"
            )

        return Xd

    def choose_kmeans_k(self, X_train, k_values=range(2, 13)):
        Xd = self._validate_and_align_input(X_train, stage="choose_kmeans_k")
        rows = []

        for k in k_values:
            km = KMeans(
                n_clusters=int(k),
                random_state=self.random_state,
                n_init="auto",
            )
            labels = km.fit_predict(Xd)
            rows.append(
                {
                    "k": int(k),
                    "inertia": float(km.inertia_),
                    "silhouette": self._safe_silhouette(Xd, labels),
                }
            )

        df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
        best_row = df.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]
        best_k = int(best_row["k"])

        self.selection_results_["kmeans"] = df
        self.n_kmeans = best_k
        return best_k, df

    def choose_gmm_components(self, X_train, n_values=range(2, 13), covariance_types=("full", "diag")):
        Xd = self._validate_and_align_input(X_train, stage="choose_gmm_components")
        rows = []

        for n in n_values:
            for cov in covariance_types:
                gm = GaussianMixture(
                    n_components=int(n),
                    covariance_type=cov,
                    random_state=self.random_state,
                )
                gm.fit(Xd)
                rows.append(
                    {
                        "n_components": int(n),
                        "covariance_type": cov,
                        "bic": float(gm.bic(Xd)),
                        "aic": float(gm.aic(Xd)),
                    }
                )

        df = pd.DataFrame(rows).sort_values(["bic", "aic"], ascending=[True, True]).reset_index(drop=True)
        best = df.iloc[0]

        self.selection_results_["gmm"] = df
        self.n_gmm = int(best["n_components"])
        return int(best["n_components"]), str(best["covariance_type"]), df

    def choose_hdbscan_params(
        self,
        X_train,
        min_cluster_size_values=(20, 40, 60),
        min_samples_values=(5, 10, 15),
    ):
        Xd = self._validate_and_align_input(X_train, stage="choose_hdbscan_params")
        rows = []

        for min_cluster_size in min_cluster_size_values:
            for min_samples in min_samples_values:
                model = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples),
                    prediction_data=True,
                )
                model.fit(Xd)
                labels = model.labels_

                noise_ratio = float(np.mean(labels == -1))
                non_noise_mask = labels != -1
                non_noise_labels = labels[non_noise_mask]

                n_clusters = int(np.unique(non_noise_labels).shape[0])
                if non_noise_mask.sum() > 2 and n_clusters > 1:
                    sil = self._safe_silhouette(Xd[non_noise_mask], non_noise_labels)
                else:
                    sil = np.nan

                rows.append(
                    {
                        "min_cluster_size": int(min_cluster_size),
                        "min_samples": int(min_samples),
                        "n_clusters": n_clusters,
                        "noise_ratio": noise_ratio,
                        "silhouette_non_noise": sil,
                    }
                )

        df = pd.DataFrame(rows)
        df["silhouette_non_noise"] = df["silhouette_non_noise"].fillna(-1)
        df = df.sort_values(["silhouette_non_noise", "noise_ratio", "n_clusters"], ascending=[False, True, False]).reset_index(drop=True)

        best = df.iloc[0]
        best_params = {
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": int(best["min_samples"]),
        }

        self.selection_results_["hdbscan"] = df
        self.hdb_params_ = best_params
        return best_params, df

    def fit(
        self,
        X_train,
        auto_select=True,
        k_values=range(2, 13),
        gmm_n_values=range(2, 13),
        gmm_covariance_types=("full", "diag"),
        hdbscan_min_cluster_size_values=(20, 40, 60),
        hdbscan_min_samples_values=(5, 10, 15),
    ):
        self._capture_input_contract(X_train)
        Xd = self._validate_and_align_input(X_train, stage="fit")

        if auto_select:
            if self.n_kmeans is None:
                self.choose_kmeans_k(X_train, k_values=k_values)
            if self.n_gmm is None:
                best_gmm_n, best_cov, _ = self.choose_gmm_components(
                    X_train,
                    n_values=gmm_n_values,
                    covariance_types=gmm_covariance_types,
                )
            else:
                best_cov = "full"

            if self.hdb_params_ is None:
                self.choose_hdbscan_params(
                    X_train,
                    min_cluster_size_values=hdbscan_min_cluster_size_values,
                    min_samples_values=hdbscan_min_samples_values,
                )
        else:
            best_cov = "full"
            if self.n_kmeans is None:
                raise ValueError("Set n_kmeans when auto_select=False")
            if self.n_gmm is None:
                raise ValueError("Set n_gmm when auto_select=False")
            if self.hdb_params_ is None:
                self.hdb_params_ = {"min_cluster_size": 40, "min_samples": 15}

        self.kmeans = KMeans(
            n_clusters=int(self.n_kmeans),
            random_state=self.random_state,
            n_init="auto",
        )
        self.kmeans.fit(Xd)

        self.gmm = GaussianMixture(
            n_components=int(self.n_gmm),
            covariance_type=best_cov,
            random_state=self.random_state,
        )
        self.gmm.fit(Xd)

        self.hdb = hdbscan.HDBSCAN(
            min_cluster_size=int(self.hdb_params_["min_cluster_size"]),
            min_samples=int(self.hdb_params_["min_samples"]),
            prediction_data=True,
        )
        self.hdb.fit(Xd)
        self.hdb.generate_prediction_data()

        max_hdb_label = self.hdb.labels_[self.hdb.labels_ >= 0].max() if np.any(self.hdb.labels_ >= 0) else 0
        self.hdb_noise_label_ = int(max_hdb_label + 1)
        return self

    def predict(self, X):
        Xd = self._validate_and_align_input(X, stage="predict")

        kmeans_labels = self.kmeans.predict(Xd).astype(int)
        gmm_labels = self.gmm.predict(Xd).astype(int)

        hdb_labels, _ = hdbscan.approximate_predict(self.hdb, Xd)
        hdb_labels = hdb_labels.astype(int)
        hdb_labels[hdb_labels < 0] = self.hdb_noise_label_

        return pd.DataFrame(
            {
                "cluster_kmeans": kmeans_labels,
                "cluster_gmm": gmm_labels,
                "cluster_hdbscan": hdb_labels,
            }
        )

    def get_feature_names_out(self):
        return list(self.feature_names_)

    def get_expected_input_columns(self):
        expected_cols = getattr(self, "expected_input_columns_", None)
        return None if expected_cols is None else list(expected_cols)

    def save(self, path="cluster_feature_models.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="cluster_feature_models.joblib"):
        return joblib.load(path)
    

##############################################################################################################

class ColumnValidatedScaler:
    def __init__(self, feature_columns=None, with_mean=False, with_std=True):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        self.expected_columns_ = None

    def fit(self, data):
        if not hasattr(data, "columns"):
            raise TypeError("fit expects a pandas DataFrame with named columns")

        if self.feature_columns is None:
            self.expected_columns_ = list(data.columns)
        else:
            missing = [c for c in self.feature_columns if c not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns at fit time: {missing}")
            self.expected_columns_ = list(self.feature_columns)

        self.scaler.fit(data.loc[:, self.expected_columns_])
        return self

    def transform(self, data):
        if self.expected_columns_ is None:
            raise ValueError("Scaler is not fitted yet")
        if not hasattr(data, "columns"):
            raise TypeError("transform expects a pandas DataFrame with named columns")

        missing = [c for c in self.expected_columns_ if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns at transform time: {missing}")

        # enforce exact training order
        ordered = data.loc[:, self.expected_columns_]
        return self.scaler.transform(ordered)

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def get_feature_names_out(self):
        if self.expected_columns_ is None:
            raise ValueError("Scaler is not fitted yet")
        return self.expected_columns_

"""
# ---- Usage ----
scaler_wrapper = ColumnValidatedScaler(with_mean=False)
scaler_wrapper.fit(Xtrain)

joblib.dump(scaler_wrapper, "HCD_StandardScaler.joblib")

loaded_scaler = joblib.load("HCD_StandardScaler.joblib")
Xtrain_scaled = loaded_scaler.transform(Xtrain)
Xtest_scaled = loaded_scaler.transform(Xtest)
"""

#########################################################################################################################################

class ClassEncoder(OneHotEncoder):
    def __init__(self,
                 categoric_columns:list):
        self.categoric_columns = categoric_columns
        self.expected          = {}

        self.one_hot = OneHotEncoder(drop='first', handle_unknown='ignore')

    def fit(self,data):
        """
        calls sklearn.preprocessing.OneHotEncoder
        and stores unique column values
        
        input is a dataframe that includes the categorical columns passed to __init__
        """
        missing = [col for col in self.categoric_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.one_hot.fit(data[self.categoric_columns])
        for col in self.categoric_columns:
            self.expected[col] = data[col].unique()
        return self

    def transform(self,data):
        """
        transform a dataframe that includes self.categoric_columns
        """
        missing = [col for col in self.categoric_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        hot_cols = self.one_hot.transform(data[self.categoric_columns])
        return hot_cols

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.categoric_columns
        return self.one_hot.get_feature_names_out(input_features)
    

"""
hot_encoder = joblib.load('HCD_Categoric_to_OneHot.joblib')
hot_cols = hot_encoder.transform(new_data)
col_headers = hot_encoder.get_feature_names_out()
print(col_headers)
new_data[col_headers] = hot_cols.toarray()
new_data[col_headers] = new_data[col_headers].astype('object')
new_data = new_data.drop(columns=hot_encoder.categoric_columns)
print('sum is null: ',new_data.isna().sum().sum())
hot_encoder.categoric_columns
"""



##################################################################################################################

def get_predictions(model,Xtrain,ytrain,Xtest):
    """"
    calls model.fit and model.predict
    returns fitted_model, train_predictions, test_predictions
    """
    model.fit(Xtrain,ytrain)
    train_pred = model.predict(Xtrain)
    test_pred  = model.predict(Xtest)
    return model, train_pred, test_pred

#--------------------------------------------------------------------------------


def get_score_metrics(ytest,predtest,ytrain:None=None,predtrain:None=None,
                      average='binary', pos_label=True,
                      first_title:str='Test Confusion Matrix',
                      second_title:str='Train Confusion Matrix'):
    if (ytrain is not None) and (predtrain is not None):
        scores = pd.Series(
                    [
                    accuracy_score(ytest,predtest),
                    precision_score(ytest,predtest,average=average,pos_label=pos_label),
                    recall_score(ytest,predtest,average=average,pos_label=pos_label),
                    f1_score(ytest,predtest,average=average,pos_label=pos_label),
                    accuracy_score(ytrain,predtrain),
                    precision_score(ytrain,predtrain,average=average,pos_label=pos_label),
                    recall_score(ytrain,predtrain,average=average,pos_label=pos_label),
                    f1_score(ytrain,predtrain,average=average,pos_label=pos_label),
                    ],
                    
                    index=[                    
                        'test_accuracy_score',
                        'test_precision_score',
                        'test_recall_score',
                        'test_f1_score',
                        'train_accuracy_score',
                        'train_precision_score',
                        'train_recall_score',
                        'train_f1_score'
                    ]
                )
        labels = sorted(set(ytest) | set(ytrain))
        testconfusionmatrix  = confusion_matrix(ytest,predtest,labels=labels)
        trainconfusionmatrix = confusion_matrix(ytrain,predtrain,labels=labels)

        plt.figure(figsize=(6, 4))
        sns.heatmap(testconfusionmatrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(first_title)
        plt.show()
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(trainconfusionmatrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(second_title)
        plt.show()

        return scores, testconfusionmatrix , trainconfusionmatrix

    else:
        scores = pd.Series(
                    [
                    accuracy_score(ytest,predtest),
                    precision_score(ytest,predtest,average=average,pos_label=pos_label),
                    recall_score(ytest,predtest,average=average,pos_label=pos_label),
                    f1_score(ytest,predtest,average=average,pos_label=pos_label),
                    ],
                    
                    index=[                    
                        'test_accuracy_score',
                        'test_precision_score',
                        'test_recall_score',
                        'test_f1_score',
                    ]
                )

        labels = sorted(set(ytest) )
        testconfusionmatrix  = confusion_matrix(ytest,predtest,labels=labels)

        plt.figure(figsize=(6, 4))
        sns.heatmap(testconfusionmatrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(first_title)
        plt.show()

        return scores, testconfusionmatrix



def plot_auc_curves(
    y_true_sets,
    y_prob_sets=None,
    names=None,
    curve="roc",                  # "roc" or "pr"
    pos_label=1,                  # used for binary tasks
    positive_class_index=1,       # used when probs are shape (n_samples, 2)
    classes=None,                 # optional class order for multiclass
    title=None,
    figsize=(8, 6),
    ax=None,
):
    """
    Plot ROC or Precision-Recall curves for one or many datasets and return AUC scores.

    Accepted input styles:
    1) Dict style:
       y_true_sets = {
           "train": (y_train, p_train),
           "test":  (y_test,  p_test),
       }
       and leave y_prob_sets=None

    2) Parallel lists style:
       y_true_sets = [y_train, y_test]
       y_prob_sets = [p_train, p_test]
       names       = ["train", "test"]

    Probability inputs:
    - Binary: shape (n_samples,) OR (n_samples, 1) OR (n_samples, 2)
    - Multiclass: shape (n_samples, n_classes)

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    auc_scores : dict
        Example: {"train": 0.91, "test": 0.88}
    """

    curve = curve.lower().strip()
    if curve not in {"roc", "pr"}:
        raise ValueError("curve must be 'roc' or 'pr'")

    # Normalize inputs to dict: {name: (y_true, y_prob)}
    if isinstance(y_true_sets, dict):
        data = y_true_sets
    else:
        if y_prob_sets is None:
            raise ValueError("y_prob_sets is required when y_true_sets is not a dict")
        if len(y_true_sets) != len(y_prob_sets):
            raise ValueError("y_true_sets and y_prob_sets must have the same length")

        if names is None:
            names = [f"set_{i}" for i in range(len(y_true_sets))]
        if len(names) != len(y_true_sets):
            raise ValueError("names length must match y_true_sets length")

        data = {n: (y, p) for n, y, p in zip(names, y_true_sets, y_prob_sets)}

    # Create figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    auc_scores = {}

    for split_name, (y_true, y_prob) in data.items():
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if y_true.ndim != 1:
            raise ValueError(f"{split_name}: y_true must be 1D")

        # Binary case
        if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] <= 2):
            if y_prob.ndim == 2:
                if y_prob.shape[1] == 1:
                    scores = y_prob[:, 0]
                else:
                    scores = y_prob[:, positive_class_index]
            else:
                scores = y_prob

            if curve == "roc":
                x, y, _ = roc_curve(y_true, scores, pos_label=pos_label)
                area = roc_auc_score(y_true, scores)
                ax.plot(x, y, lw=2, label=f"{split_name} (AUC={area:.4f})")
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true, scores, pos_label=pos_label
                )
                area = average_precision_score(y_true, scores, pos_label=pos_label)
                ax.plot(recall, precision, lw=2, label=f"{split_name} (AP={area:.4f})")

        # Multiclass case
        elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
            if classes is None:
                class_order = np.unique(y_true)
            else:
                class_order = np.asarray(classes)

            y_true_bin = label_binarize(y_true, classes=class_order)

            if y_true_bin.shape[1] != y_prob.shape[1]:
                raise ValueError(
                    f"{split_name}: y_prob has {y_prob.shape[1]} columns but "
                    f"binarized y_true has {y_true_bin.shape[1]}"
                )

            if curve == "roc":
                # Plot micro-average ROC curve; report macro AUC
                x, y, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
                area = roc_auc_score(
                    y_true_bin, y_prob, multi_class="ovr", average="macro"
                )
                ax.plot(x, y, lw=2, label=f"{split_name} (macro AUC={area:.4f})")
            else:
                # Plot micro-average PR curve; report macro AP
                precision, recall, _ = precision_recall_curve(
                    y_true_bin.ravel(), y_prob.ravel()
                )
                area = average_precision_score(y_true_bin, y_prob, average="macro")
                ax.plot(recall, precision, lw=2, label=f"{split_name} (macro AP={area:.4f})")
        else:
            raise ValueError(f"{split_name}: unsupported y_prob shape {y_prob.shape}")

        auc_scores[split_name] = float(area)

    # Decorations
    if curve == "roc":
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "ROC Curve")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or "Precision-Recall Curve")

    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    return fig, ax, auc_scores

#--------------------------------------------------------------------------------

def model_tester(model,
                 Xtrain,
                 Xtest,
                 ytrain,
                 ytest,
                 predtrain:None=None,
                 predtest:None=None,
                 average='binary',
                 pos_label=True):

    if (predtrain is None) or (predtest is None):
        model, predtrain, predtest = get_predictions(model,Xtrain,ytrain,Xtest)

    print(f'[sanity] ytrue positives: {(ytest==pos_label).sum()}  ypred positives: {(predtest==pos_label).sum()}')
    scores, testconfusionmatrix , trainconfusionmatrix = get_score_metrics(ytest,predtest,ytrain,predtrain,average=average,pos_label=pos_label)

    print('. '*30)
    print(scores)
    print('. '*30)
    print('test')
    print(testconfusionmatrix)
    print('test probas')
    test_probas = confusion_matrix_to_bayes_theorem_probas(testconfusionmatrix)
    print(test_probas)
    print('. '*30)
    print('train')
    print(trainconfusionmatrix)
    print('train probas')
    train_probas = confusion_matrix_to_bayes_theorem_probas(trainconfusionmatrix)
    print(train_probas)
    print('. '*30)
    return model

##################################################################################################################

def confusion_matrix_to_bayes_theorem_probas(confusion_matrix):
    """
    where probabilty (p), hypothosis (h), evidence (e)
    p(h|e) = p(e|h) * p(h) / p(e)

    returns p(h|e) , p(e|h)
    where the denominator is p(h)*p(e|h + p(not h)*p(e|not h)
    p(e|h) is determined by counts, not by multiplying p(h)*p(e)/p(e)

    returns returns p(true|pred) , p(pred|true) 
    both as diagonal of matrix from top-left to lower-right
    """
    matrix = np.array(confusion_matrix.copy())

    #prior
    sum_observations = matrix.sum(axis=1)    
    p_h     = sum_observations / sum_observations.sum()
    p_not_h = 1 - p_h

    #new evidence
    sum_predictions  = matrix.sum(axis=0)

    sum_population   = matrix.sum(axis=-1).sum()


    p_e_given_h = []
    p_e_given_not_h = []
    for e_location in range(len(sum_observations)):
        n_true_e        = matrix[e_location][e_location]
        n_h             = sum_observations[e_location] 
        n_not_h         = sum_observations.sum() - sum_observations[e_location]
        n_e_given_not_h = sum_predictions[e_location] - matrix[e_location][e_location]
        if n_h==0:
            pegh   = 0
        else:
            pegh   = n_true_e / n_h
        p_e_given_h.append(pegh)
        if n_not_h==0:
            pegnh = 0
        else:
            pegnh = n_e_given_not_h / n_not_h
        p_e_given_not_h.append(pegnh)
    p_e_given_h = np.array(p_e_given_h)
    p_e_given_not_h = np.array(p_e_given_not_h)
    try:
        p_observed     = p_h * p_e_given_h
        p_not_observed = p_not_h * p_e_given_not_h
        p_h_given_e  = p_observed / (p_observed + p_not_observed)

    except:
        logph, logpeh, logpnh, logpenh = np.log(p_h) , np.log(p_e_given_h) , np.log(p_not_h) , np.log(p_e_given_not_h) 

        log_obs, log_un_observed  =  (logph+logpeh) , (logpnh+logpenh)

        m               = np.maximum(log_obs, log_unobs)

        log_denominator = m + np.log( np.exp(log_obs - m) + np.exp(log_unobs - m) )

        p_h_given_e     = np.exp(log_obs - log_denominator)

    return p_h_given_e, p_e_given_h
                          




#################################################################################################################

# adds boolean columns at different outlier multiplier thresholds
def determine_outlier_boundaries(data,
                                 target_col,
                                 iqr_multiplier=1.5):
    
    """
    """
    
    q1 = data[target_col].quantile(0.25)
    q3 = data[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (iqr_multiplier * iqr)
    upper = q3 + (iqr_multiplier * iqr)
    return lower, upper

def outlier_map(data,
                cols_to_map:list|None,
                iqr_multiplier=1.5):
    outlier_map_ = {}
    if cols_to_map is None:
        cols_to_map = data.select_dtypes('number').columns
    for col in cols_to_map:
        lower, upper = determine_outlier_boundaries(data,
                                 target_col=col,
                                 iqr_multiplier=iqr_multiplier)
        outlier_map_[col] = {'lower': lower,
                             'upper': upper}
    return outlier_map_
    

def add_outlier_columns(data_,
                        cols_to_map:None|list=None,
                        iqr_multiplier=1.5,
                        return_map = True,
                        return_new_data = True,
                        add_bool_col=False,
                        clip=True,
                        outliers:None|dict=None):
    """
    where outliers is a map: {column_i:{'lower': lower,
                                      'upper': upper},
                             column_n:...}
    """
    suffix = '_is_outlier_'+str(iqr_multiplier)
    data = data_.copy()
    if not outliers:

        outliers = outlier_map(data,
                    cols_to_map=cols_to_map,
                    iqr_multiplier=iqr_multiplier)
    if return_new_data==True:
        for k, v in outliers.items():
            if clip==True:
                data[k] = data[k].clip(upper=v['upper'],lower=v['lower'])
            if add_bool_col==True:
                data[k+'_upper_'+suffix] = (data[k] > v['upper'])
                data[k+'_lower_'+suffix] = (data[k] < v['lower']) 
                for new in [k+'_upper_'+suffix,k+'_lower_'+suffix]:
                    if data[new].sum()>0:
                        data[new] = data[new].astype('object')
                    else:
                        data = data.drop(columns=new)
        if return_map==True:
            return data, outliers
        else:
            return data
    return outliers


###################################################################################################################


# Pseudo pca, reduces numeric to errors of linreg with one feature predicting each of the others
def linreg_theta(X,y):    
    X_b = np.c_[np.ones((X.shape[0],1)),X]
    return pinv(X_b.T@X_b)@X_b.T@y
def linreg_predict(X,theta):
    return np.c_[np.ones((X.shape[0],1)),X]@theta
def get_error(X,y):
    theta = linreg_theta(X,y)
    pred  = linreg_predict(X,theta)
    return y-pred,theta
def reduce_coliniarity_based_on_one_column(data_,
                                           highly_correlated_column='lactate',
                                           avoid_columns=['deterioration_next_12h','deterioration_next_12h_is_outlier']):
    data=data_.copy()
    col_theta_dict = {}
    for col in data.select_dtypes('number').columns:
        if (col !=highly_correlated_column) and (col not in avoid_columns):
            errors,theta = get_error(data[highly_correlated_column],data[col])
            data['mutated_'+col] = errors
            data=data.drop(columns=col)
            col_theta_dict[col] = theta
    return data, col_theta_dict

##################################################################################################################################################################

# MERGE DATASETS

def get_merged_data_w_ids(kwargs, join_cols=['patient_id', 'hour_from_admission'], how='left', strict=True, return_report=False):
    """
    Merge multiple datasets by using any available subset of `join_cols` per dataset.

    Parameters
    ----------
    kwargs : dict | tuple | list | set
        Collection of pandas DataFrames. If a dict is provided, keys are used as dataset names.
    join_cols : list[str]
        Priority join columns to use when overlaps exist.
    how : str
        Merge strategy passed to pandas.merge (e.g., 'left', 'inner', 'outer').
    strict : bool
        If True, raise on merge-cardinality validation failures.
    return_report : bool
        If True, return (merged_df, report_df).
    """
    if not isinstance(join_cols, (list, tuple)) or len(join_cols) == 0:
        raise ValueError("join_cols must be a non-empty list/tuple of column names")

    # Normalize accepted container types into a named list of (name, dataframe) pairs.
    if isinstance(kwargs, dict):
        datasets = list(kwargs.items())
    elif isinstance(kwargs, (tuple, list, set)):
        datasets = [(f'df_{i}', df) for i, df in enumerate(kwargs)]
    else:
        raise TypeError("kwargs must be a dict, tuple, list, or set of pandas DataFrames")

    if len(datasets) == 0:
        raise ValueError("No datasets provided")

    cleaned = []
    for name, df in datasets:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Dataset '{name}' is not a pandas DataFrame")
        usable = [c for c in join_cols if c in df.columns]
        cleaned.append((str(name), df, usable))

    # Start from the dataset with the most available join columns (then most rows).
    cleaned.sort(key=lambda x: (len(x[2]), x[1].shape[0]), reverse=True)
    base_name, merged, base_join_cols = cleaned[0]
    merged = merged.copy()

    merge_log = [{
        'dataset': base_name,
        'action': 'base',
        'join_cols_used': base_join_cols,
        'relationship': None,
        'rows_before': merged.shape[0],
        'rows_after': merged.shape[0],
        'note': 'base dataset'
    }]

    for name, df, _ in cleaned[1:]:
        common_cols = [c for c in join_cols if c in merged.columns and c in df.columns]

        if not common_cols:
            merge_log.append({
                'dataset': name,
                'action': 'skipped_no_common_join_cols',
                'join_cols_used': [],
                'relationship': None,
                'rows_before': merged.shape[0],
                'rows_after': merged.shape[0],
                'note': 'no overlap with join_cols'
            })
            continue

        # Keep only join keys and columns not yet present in the merged base.
        cols_to_merge = [col for col in df.columns if (col not in merged.columns) or (col in common_cols)]
        dropped_duplicates = [col for col in df.columns if col not in cols_to_merge]
        df_to_merge = df[cols_to_merge].copy()

        left_unique = not merged.duplicated(subset=common_cols).any()
        right_unique = not df_to_merge.duplicated(subset=common_cols).any()
        if left_unique and right_unique:
            relationship = 'one_to_one'
        elif left_unique and not right_unique:
            relationship = 'one_to_many'
        elif not left_unique and right_unique:
            relationship = 'many_to_one'
        else:
            relationship = 'many_to_many'

        rows_before = merged.shape[0]
        try:
            merged = merged.merge(
                df_to_merge,
                on=common_cols,
                how=how,
                validate=relationship,
                suffixes=('', f'_{name}')
            )
            action = 'merged'
        except pd.errors.MergeError:
            if strict:
                raise
            merged = merged.merge(
                df_to_merge,
                on=common_cols,
                how=how,
                suffixes=('', f'_{name}')
            )
            action = 'merged_without_validation'

        if dropped_duplicates:
            note = (
                'dropped duplicate non-join columns from incoming dataset: '
                + ', '.join(dropped_duplicates)
            )
        else:
            note = 'no duplicate non-join columns dropped'

        merge_log.append({
            'dataset': name,
            'action': action,
            'join_cols_used': common_cols,
            'relationship': relationship,
            'rows_before': rows_before,
            'rows_after': merged.shape[0],
            'note': note
        })

    if return_report:
        return merged, pd.DataFrame(merge_log)
    return merged


#####################################

# re-merge dataset with train_test_split partial dataset

def re_merge(original,
             xtrain,
             xtest,
             test_predicitons,
             train_predictions,
             re_join_columns: list,
             inverse_transformations: list = [],
             ordinal_encoders: list = [],
             ohe_encoders: list = []):
    """
    Reconstruct a full, human-readable DataFrame by reversing preprocessing
    transformations and reinstating columns that were excluded from model training.

    Workflow (in order)
    -------------------
    1. Undo numeric transformations (e.g. StandardScaler, PCA) in reverse order.
    2. Undo OrdinalEncoder(s) — restores encoded integer columns to original categories.
    3. Undo OneHotEncoder(s) — collapses dummy columns back to the original column.
    4. Attach model predictions and a 'split' label ('train' or 'test').
    5. Join columns that were held out of training (e.g. patient_id) from `original`
       using the shared pandas index.

    Parameters
    ----------
    original : pd.DataFrame
        The unencoded, untransformed source DataFrame. Its pandas index must match
        xtrain/xtest (i.e. do not call reset_index() between loading data and
        train_test_split).

    xtrain : pd.DataFrame
        Training feature matrix from train_test_split. Must be a DataFrame
        (so column names and index are preserved).

    xtest : pd.DataFrame
        Test feature matrix from train_test_split. Same requirement as xtrain.

    test_predicitons : array-like
        Model predictions for xtest rows, in the same order as xtest.

    train_predictions : array-like
        Model predictions for xtrain rows, in the same order as xtrain.

    re_join_columns : list of str
        Columns to reinstate from `original` that were excluded from training,
        e.g. ['patient_id', 'hour_from_admission']. All other columns present
        in `original` but absent from the feature matrix are also joined
        automatically.

    inverse_transformations : list of fitted transformers, default []
        Numeric transformers applied before encoding, e.g. [StandardScaler(), PCA()].
        Pass them in the ORDER they were applied — they are undone in reverse.
        Pass [] if none were applied.

    ordinal_encoders : list of (fitted_encoder, column_names) tuples, default []
        Each tuple is (OrdinalEncoder, ['col_a', 'col_b', ...]) where column_names
        are the feature columns the encoder was fit on (same order as when you
        called encoder.fit()). Multiple encoders are processed in list order.

        Example:
            enc = OrdinalEncoder()
            enc.fit(X_train[['gender', 'season']])
            ordinal_encoders = [(enc, ['gender', 'season'])]

    ohe_encoders : list of (fitted_encoder, original_column_names) tuples, default []
        Each tuple is (OneHotEncoder, ['col_a', 'col_b', ...]) where
        original_column_names are the pre-encoding column names the encoder
        was fit on. The function calls encoder.get_feature_names_out(original_cols)
        to locate the expanded dummy columns in xtrain/xtest automatically.

        Example:
            enc = OneHotEncoder(sparse_output=False)
            enc.fit(X_train[['city', 'payment_type']])
            ohe_encoders = [(enc, ['city', 'payment_type'])]

    Returns
    -------
    combined : pd.DataFrame
        All rows (train + test) sorted by original index, with decoded feature
        columns, a 'prediction' column, a 'split' column ('train'/'test'), and
        all reinstated columns from `original`.
    train_df : pd.DataFrame
        Training rows only, same structure as combined.
    test_df : pd.DataFrame
        Test rows only, same structure as combined.
    """
    train_idx = xtrain.index
    test_idx  = xtest.index

    arr_train = xtrain.values.copy()
    arr_test  = xtest.values.copy()

    # Step 1: undo numeric transformations in reverse order
    for transform in reversed(inverse_transformations):
        arr_train = transform.inverse_transform(arr_train)
        arr_test  = transform.inverse_transform(arr_test)

    feature_cols = list(xtrain.columns)
    train_df = pd.DataFrame(arr_train, index=train_idx, columns=feature_cols)
    test_df  = pd.DataFrame(arr_test,  index=test_idx,  columns=feature_cols)

    # Step 2: undo OrdinalEncoder(s)
    for enc, cols in ordinal_encoders:
        decoded_train = enc.inverse_transform(train_df[cols].values)
        decoded_test  = enc.inverse_transform(test_df[cols].values)
        train_df[cols] = decoded_train
        test_df[cols]  = decoded_test

    # Step 3: undo OneHotEncoder(s)
    for enc, original_cols in ohe_encoders:
        dummy_cols = list(enc.get_feature_names_out(original_cols))
        decoded_train = enc.inverse_transform(train_df[dummy_cols].values)
        decoded_test  = enc.inverse_transform(test_df[dummy_cols].values)
        # Drop dummy columns and insert original columns
        train_df = train_df.drop(columns=dummy_cols)
        test_df  = test_df.drop(columns=dummy_cols)
        train_df[original_cols] = decoded_train
        test_df[original_cols]  = decoded_test

    # Step 4: attach predictions and split label
    train_df['prediction'] = np.array(train_predictions)
    test_df['prediction']  = np.array(test_predicitons)
    train_df['split']      = 'train'
    test_df['split']       = 'test'

    # Step 5: reinstate held-out columns from original via index join
    existing_cols = set(train_df.columns)
    cols_to_add = []
    for col in list(re_join_columns) + [c for c in original.columns if c not in existing_cols]:
        if col in original.columns and col not in existing_cols and col not in cols_to_add:
            cols_to_add.append(col)

    if cols_to_add:
        train_df = train_df.join(original[cols_to_add], how='left')
        test_df  = test_df.join(original[cols_to_add],  how='left')

    combined = pd.concat([train_df, test_df]).sort_index()
    return combined, train_df, test_df


#################################################################################################################
# one hot and scaling wrapped in a one function call

def one_hot_dataset_modifications(dataset:pd.DataFrame,
                                  categoric_transformer,
                                  drop_original:bool=True):
    
    original_headers = list(categoric_transformer.feature_names_in_)
    
    new_vector = categoric_transformer.transform(dataset[original_headers])
    new_vector = new_vector.toarray()

    one_hot_headers = list(categoric_transformer.get_feature_names_out())

    dataset[one_hot_headers] = new_vector

    # drop original
    if drop_original==True:
        
        dataset = dataset.drop(columns= original_headers)
    
    return dataset


def numeric_scaler_dataset_modifications(dataset:pd.DataFrame,
                                         numeric_transformer):
    feature_names = numeric_transformer.get_feature_names_out()
    dataset[feature_names] = numeric_transformer.transform(dataset[feature_names])
    return dataset


def transformed_dataset(dataset:pd.DataFrame,
                        categoric_transformer,
                        numeric_transformer,
                        drop_original:bool|str='auto'):
    if drop_original=='auto':
        duplicates = sorted(list(categoric_transformer.get_feature_names_out()))==sorted(list(categoric_transformer.feature_names_in_))
        if duplicates==True:
            drop_original=False
        else:
            drop_original=True
    dataset = one_hot_dataset_modifications(dataset, categoric_transformer,drop_original)
    dataset = numeric_scaler_dataset_modifications(dataset, numeric_transformer)

    return dataset

#########################################################################