import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,   # <-- IMPORTANT: was missing earlier
)
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

# Silence convergence and other annoying warnings
warnings.filterwarnings("ignore", message="Stochastic Optimizer: Maximum iterations", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------

def compute_classification_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    # Optional ROC-AUC (only for binary)
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            pass
    return metrics


def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


# ----------------------------------------------------------
# PLOTS
# ----------------------------------------------------------

def plot_and_save_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_and_save_roc_curve(y_true, y_proba, title, out_path):
    from sklearn.metrics import RocCurveDisplay

    try:
        plt.figure()
        RocCurveDisplay.from_predictions(y_true, y_proba)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        # If ROC cannot be computed, just skip
        pass


def plot_and_save_regression_scatter(y_true, y_pred, title, out_path):
    plt.figure(figsize=(4, 4))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------------------------------------
# MAIN AUTOML FUNCTION
# ----------------------------------------------------------

def run_automl_training(
    data_dir: str,
    target_column: Optional[str] = None,
    results_dir_name: str = "results",
    n_iter_per_model: int = 20,
    random_state: int = 42,
    problem_type: str = "auto",  # "auto", "classification", "regression"
):
    """
    AutoML-style runner for preprocessed train/test CSVs.

    Requirements:
    - data_dir must contain: train_data.csv, test_data.csv
    - target_column is the column to predict (SalePrice, label, etc.)
    - If problem_type = "auto", task is inferred from target.
    """

    train_path = os.path.join(data_dir, "train_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("train_data.csv and/or test_data.csv not found in given directory")

    results_root = os.path.join(data_dir, results_dir_name)
    per_model_dir = os.path.join(results_root, "per_model")
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(per_model_dir, exist_ok=True)

    # ---------- Load data ----------
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column is None:
        target_column = train_df.columns[-1]

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in train_data.csv")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # ---------- Task type detection ----------
    n_unique = y_train.nunique()
    is_numeric = np.issubdtype(y_train.dtype, np.number)

    if problem_type == "auto":
        # Heuristic: numeric + many unique values => regression
        if is_numeric and n_unique > max(20, int(0.1 * len(y_train))):
            task = "regression"
        else:
            task = "classification"
    elif problem_type in ("classification", "regression"):
        task = problem_type
    else:
        raise ValueError("problem_type must be 'auto', 'classification', or 'regression'")

    print(f"\n[AutoML] Detected task type: {task} (unique target values = {n_unique})")

    # ---------- Model search spaces ----------
    if task == "classification":
        # Encode labels for classifiers
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = label_encoder.transform(y_test)

        model_spaces = {
            "LogisticRegression": (
                LogisticRegression(max_iter=5000, n_jobs=-1),
                {
                    "C": np.logspace(-3, 3, 10),
                    "penalty": ["l2"],
                    "class_weight": [None, "balanced"],
                },
            ),
            "RandomForest": (
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                {
                    "n_estimators": [100, 200, 400],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
            "XGBoost": (
                xgb.XGBClassifier(
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                {
                    "n_estimators": [200, 400],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.7, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.9, 1.0],
                },
            ),
            "LightGBM": (
                lgb.LGBMClassifier(random_state=random_state,
                                n_jobs=-1,
                                verbose=-1,          # hide training logs
                                force_col_wise=True,
                ),
                {
                    "n_estimators": [200, 400],
                    "max_depth": [-1, 5, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.7, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.9, 1.0],
                },
            ),
            "CatBoost": (
                CatBoostClassifier(random_state=random_state, verbose=False),
                {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [200, 400],
                },
            ),
            "SVM": (
                SVC(probability=True, random_state=random_state),
                {
                    "C": np.logspace(-2, 2, 8),
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"],
                },
            ),
            "KNN": (
                KNeighborsClassifier(),
                {
                    "n_neighbors": list(range(3, 16, 2)),
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
            ),
            "MLP": (
                MLPClassifier(max_iter=1000, random_state=random_state, early_stopping=True,),
                {
                    "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                    "activation": ["relu", "tanh"],
                    "alpha": [1e-4, 1e-3, 1e-2],
                    "learning_rate_init": [1e-3, 5e-3],
                },
            ),
            "NaiveBayes": (
                GaussianNB(),
                {
                    "var_smoothing": np.logspace(-9, -7, 5),
                },
            ),
        }

        scoring = "f1_macro"
        primary_metric_name = "f1_macro"
        maximize = True

    else:  # regression
        y_train_enc = y_train.values
        y_test_enc = y_test.values

        model_spaces = {
            "RandomForestRegressor": (
                RandomForestRegressor(random_state=random_state, n_jobs=-1),
                {
                    "n_estimators": [200, 400, 600],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
            "XGBRegressor": (
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                {
                    "n_estimators": [300, 500, 800],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.7, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.9, 1.0],
                },
            ),
            "LGBMRegressor": (
                lgb.LGBMRegressor(
                    objective="regression",
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,          # hide training logs
                    force_col_wise=True,
                ),
                {
                    "n_estimators": [300, 500, 800],
                    "max_depth": [-1, 5, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.7, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.9, 1.0],
                },
            ),
            "CatBoostRegressor": (
                CatBoostRegressor(
                    loss_function="RMSE",
                    random_state=random_state,
                    verbose=False,
                ),
                {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [300, 600],
                },
            ),
            "SVR": (
                SVR(),
                {
                    "C": np.logspace(-1, 2, 6),
                    "kernel": ["rbf", "linear"],
                    "epsilon": [0.1, 0.2, 0.3],
                },
            ),
            "KNNRegressor": (
                KNeighborsRegressor(),
                {
                    "n_neighbors": list(range(3, 16, 2)),
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
            ),
            "MLPRegressor": (
                MLPRegressor(max_iter=10000, random_state=random_state, early_stopping=True,),
                {
                    "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                    "activation": ["relu", "tanh"],
                    "alpha": [1e-4, 1e-3, 1e-2],
                    "learning_rate_init": [1e-3, 5e-3],
                },
            ),
        }

        scoring = "neg_mean_squared_error"
        primary_metric_name = "rmse"
        maximize = False

    # ---------- Training loop ----------
    summary_rows = []
    best_model_name = None
    best_metric_val = None
    best_model_obj = None

    for model_name, (base_model, param_dist) in model_spaces.items():
        print(f"\n========== {model_name} ==========")
        model_out_dir = os.path.join(per_model_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        try:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                n_iter=n_iter_per_model,
                scoring=scoring,
                n_jobs=-1,
                cv=3,
                verbose=1,
                random_state=random_state,
            )
            search.fit(X_train, y_train_enc)
        except Exception as e:
            print(f"[WARN] Skipping {model_name} due to error: {e}")
            continue

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.to_csv(
            os.path.join(model_out_dir, f"{model_name}_tuning_results.csv"),
            index=False,
        )

        best_model = search.best_estimator_

        # Evaluate on test
        y_pred = best_model.predict(X_test)

        if task == "classification":
            try:
                y_proba = best_model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    y_proba_for_roc = y_proba[:, 1]
                else:
                    y_proba_for_roc = None
            except Exception:
                y_proba_for_roc = None

            metrics = compute_classification_metrics(y_test_enc, y_pred, y_proba_for_roc)

            # Plots
            plot_and_save_confusion_matrix(
                y_test_enc,
                y_pred,
                title=f"Confusion Matrix - {model_name}",
                out_path=os.path.join(model_out_dir, f"{model_name}_confusion_matrix.png"),
            )
            if y_proba_for_roc is not None and len(np.unique(y_test_enc)) == 2:
                plot_and_save_roc_curve(
                    y_test_enc,
                    y_proba_for_roc,
                    title=f"ROC Curve - {model_name}",
                    out_path=os.path.join(model_out_dir, f"{model_name}_roc_curve.png"),
                )

        else:  # regression
            metrics = compute_regression_metrics(y_test_enc, y_pred)

            # Predicted vs True scatter
            plot_and_save_regression_scatter(
                y_test_enc,
                y_pred,
                title=f"Predicted vs True - {model_name}",
                out_path=os.path.join(model_out_dir, f"{model_name}_pred_vs_true.png"),
            )

        # Save model
        joblib.dump(
            best_model,
            os.path.join(model_out_dir, f"{model_name}_best_model.pkl"),
        )

        row = {"model": model_name}
        row.update(metrics)
        summary_rows.append(row)

        metric_val = metrics.get(primary_metric_name, None)
        if metric_val is not None:
            if best_metric_val is None:
                best_metric_val = metric_val
                best_model_name = model_name
                best_model_obj = best_model
            else:
                if (maximize and metric_val > best_metric_val) or (
                    not maximize and metric_val < best_metric_val
                ):
                    best_metric_val = metric_val
                    best_model_name = model_name
                    best_model_obj = best_model

    if not summary_rows:
        print("\n[AutoML] No model was successfully trained.")
        return None

    # ---------- Leaderboard ----------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(
        by=primary_metric_name,
        ascending=not maximize,
        inplace=True,
    )
    summary_df.to_csv(os.path.join(results_root, "model_summary.csv"), index=False)

    # ---------- Best model globally ----------
    if best_model_obj is not None:
        joblib.dump(best_model_obj, os.path.join(results_root, "best_model.pkl"))
        with open(
            os.path.join(results_root, "best_model.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(f"Best model: {best_model_name}\n")
            f.write(f"Best {primary_metric_name}: {best_metric_val:.4f}\n")

    # ---------- Simple HTML summary ----------
    html_path = os.path.join(results_root, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>AutoML Model Report</title></head><body>")
        f.write("<h1>AutoML Model Report</h1>")
        f.write(f"<h2>Task Type: {task}</h2>")
        f.write("<h2>Model Leaderboard</h2>")
        f.write(summary_df.to_html(index=False))
        if best_model_name is not None:
            f.write(
                f"<h2>Best Model</h2><p>{best_model_name} "
                f"({primary_metric_name}={best_metric_val:.4f})</p>"
            )
        f.write("</body></html>")

    print("\n[AutoML] Done. Results saved to:", results_root)
    return summary_df


# ----------------------------------------------------------
# CLI ENTRY
# ----------------------------------------------------------

if __name__ == "__main__":
    data_dir = input("Press Enter to data path: ").strip()
    if not data_dir:
        data_dir = "."

    target_column = input("Press Enter to target column: ").strip()
    if target_column == "":
        target_column = None

    run_automl_training(
        data_dir=data_dir,
        target_column=target_column,
        problem_type="auto",  # auto-detects regression for SalePrice
    )
    
    print("\n[AutoML] Training complete.")