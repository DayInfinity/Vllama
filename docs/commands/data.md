# `vllama data` — Data Preprocessing

Automatically cleans, encodes, scales, and splits any tabular dataset, and generates visualizations of the data.

---

## Syntax

```bash
vllama data --path <dataset> [--target <column>] [--test_size <float>] [--output_dir <dir>]
```

---

## Parameters

| Parameter | Short | Default | Description |
|---|---|---|---|
| `--path` | | required | Path to dataset file |
| `--target` | | auto-detect | Name of the target column |
| `--test_size` | `-t` | `0.2` | Fraction of data to hold out for testing |
| `--output_dir` | `-o` | current dir | Where to save output files |

**Supported input formats:** CSV, Excel (`.xlsx`), JSON, Parquet.

---

## What It Does

When you run `vllama data`, it performs the following steps automatically:

1. **Loads and inspects** the dataset — shape, dtypes, missing value rates
2. **Removes duplicates** and rows with excessive missing data
3. **Handles missing values** — KNN imputation for numerics, mode filling for categoricals
4. **Handles outliers** — detects and caps extreme values using IQR
5. **Encodes categoricals** — label encoding, one-hot encoding, or frequency encoding based on cardinality
6. **Scales numeric features** using `RobustScaler` (outlier-resistant)
7. **Feature selection** — removes zero-variance and highly correlated (>0.95) features
8. **Splits** into train / test sets
9. **Generates visualizations** — missing values heatmap, correlation matrix, target distribution, mutual information scores

---

## Examples

```bash
# Minimal — auto-detect target column
vllama data --path sales_data.csv

# Specify target column and 25% test split
vllama data --path housing.csv --target price --test_size 0.25

# Custom output directory
vllama data --path data.csv --target label -t 0.3 -o ./processed
```

---

## Output Structure

All outputs go into a timestamped folder:

```
output_folder_YYYYMMDD_HHMMSS/
├── train_data.csv              ← Training set (80% by default)
├── test_data.csv               ← Test set (20% by default)
├── processed_full_data.csv     ← Full preprocessed dataset
├── preprocessing_log.json      ← Detailed JSON log of every step
├── preprocessing_log.txt       ← Human-readable log
├── summary_report.json         ← Summary statistics
├── transformation_metadata.json  ← Encoders & scalers (for inference later)
└── visualizations/
    ├── 01_missing_initial.png
    ├── 02_dtypes.png
    ├── 03_corr_processed.png
    ├── 04_target_processed.png
    └── 05_mi.png
```

!!! note "Save the folder name"
    The timestamped output folder is what you pass to `vllama train`. Copy it after `vllama data` finishes.

---

## Next Step

Pass the output folder to [`vllama train`](train.md) to automatically train and compare ML models on your preprocessed data.

```bash
vllama train --path ./output_folder_YYYYMMDD_HHMMSS --target price
```
