# `vllama train` — AutoML Training

Automatically trains and compares multiple ML models on preprocessed data with hyperparameter tuning, and produces a ranked leaderboard.

---

## Syntax

```bash
vllama train --path <data_folder> --target <column>
```

---

## Parameters

| Parameter | Short | Description |
|---|---|---|
| `--path` | `-p` | Path to folder containing `train_data.csv` and `test_data.csv` (output from `vllama data`) |
| `--target` | `-t` | Name of the target column |

---

## What It Does

1. **Auto-detects task type** — classification or regression based on the target column
2. **Trains all models** with `RandomizedSearchCV` hyperparameter tuning on each
3. **Evaluates** every model on the test set with comprehensive metrics
4. **Ranks** models in a leaderboard
5. **Saves** all models and generates visualizations
6. **Produces an HTML report** with all results

---

## Models Trained

=== "Classification"
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - SVM
    - KNN
    - MLP (Neural Net)
    - Naive Bayes

=== "Regression"
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - SVR
    - KNN
    - MLP (Neural Net)

---

## Examples

```bash
# Standard usage after vllama data
vllama train --path ./output_folder_20240101_120000 --target price

# Short form
vllama train -p ./output_folder_20240101_120000 -t label
```

---

## Output Structure

```
results/
├── model_summary.csv           ← Leaderboard: all models ranked by metric
├── best_model.pkl              ← Best model, loadable with joblib
├── best_model.txt              ← Best model name and score
├── report.html                 ← Full interactive HTML report ← open this
└── per_model/
    ├── RandomForest/
    │   ├── RandomForest_best_model.pkl
    │   ├── RandomForest_tuning_results.csv
    │   ├── RandomForest_confusion_matrix.png  (classification)
    │   └── RandomForest_roc_curve.png         (binary classification)
    ├── XGBoost/
    │   └── ...
    └── ...
```

!!! tip "Open report.html"
    After training, open `results/report.html` in your browser. It contains the full leaderboard, per-model metrics, and all visualizations in one place.

---

## Loading the Best Model

```python
import joblib

model = joblib.load("results/best_model.pkl")
predictions = model.predict(X_test)
```

---

## Full Pipeline

```bash
vllama data --path raw_data.csv --target price
vllama train --path ./output_folder_YYYYMMDD_HHMMSS --target price
# Open results/report.html
```

See the [AutoML Pipeline Guide](../guides/automl-pipeline.md) for a detailed walkthrough with a real dataset.
