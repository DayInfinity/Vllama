# Full AutoML Pipeline Guide

A complete walkthrough: from a raw CSV to a trained, saved model — using only Vllama.

---

## Overview

```
raw_data.csv
     ↓
vllama data     → cleaned data + visualizations
     ↓
vllama train    → 9 trained models + leaderboard + best_model.pkl
     ↓
results/report.html
```

---

## Example Dataset

We'll use the classic **Titanic survival dataset** to demonstrate.

```bash
# Download the dataset
curl -O https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

---

## Step 1: Preprocess the Data

```bash
vllama data --path titanic.csv --target Survived --test_size 0.2
```

**What Vllama does automatically:**

- Detects `Survived` is a binary classification target
- Drops irrelevant columns (`Name`, `Ticket`, `Cabin`)
- Fills missing `Age` values using KNN imputation
- Encodes `Sex` and `Embarked` as numeric features
- Scales numeric features
- Splits into 80% train / 20% test

**After it finishes**, you'll see something like:

```
✓ Loaded dataset: 891 rows × 12 columns
✓ Removed duplicates: 0
✓ Handled missing values: Age (177), Cabin (687), Embarked (2)
✓ Encoded categoricals: Sex, Embarked
✓ Scaled features: Age, Fare, SibSp, Parch
✓ Feature selection: removed 2 low-variance features
✓ Saved train/test split to: output_folder_20240101_120000/
```

---

## Step 2: Train All Models

```bash
# Use the folder name printed in Step 1
vllama train --path ./output_folder_20240101_120000 --target Survived
```

Vllama trains all 9 classification models (each with hyperparameter tuning via RandomizedSearchCV):

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- SVM
- KNN
- MLP
- Naive Bayes

This takes a few minutes depending on your machine.

---

## Step 3: Review Results

Open the generated report:

```bash
# macOS
open results/report.html

# Linux
xdg-open results/report.html

# Windows
start results/report.html
```

The report contains:

- **Leaderboard** — all models ranked by accuracy / F1
- **Per-model metrics** — precision, recall, AUC, confusion matrices
- **Best model** — highlighted at the top

You can also check the leaderboard in the terminal:

```bash
cat results/model_summary.csv
```

---

## Step 4: Use the Best Model

```python
import joblib
import pandas as pd

# Load the best model
model = joblib.load("results/best_model.pkl")

# Predict on new data
new_data = pd.DataFrame({
    "Pclass": [3], "Sex": [1], "Age": [22],
    "SibSp": [1], "Parch": [0], "Fare": [7.25]
})
prediction = model.predict(new_data)
print("Survived:", prediction[0])
```

---

## Tips

**On a small dataset (< 1000 rows)?**
Training finishes in under a minute. Larger datasets (50k+ rows) may take 10–20 minutes due to hyperparameter search.

**Getting a bad best model score?**
Check the visualizations in `output_folder_*/visualizations/` — the correlation matrix and mutual information plots can tell you if features are informative.

**Want to reproduce the preprocessing later?**
`transformation_metadata.json` in the output folder stores all encoders and scalers. You can load them to preprocess new inference data the same way.
