"""
Autonomous Data Preprocessing Agent (Simplified & Optimized)
- Handles: loading, cleaning, encoding, scaling, feature selection, train/test split, saving.
- Output: clean numeric ML-ready CSVs (no True/False, no objects in features).
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

warnings.filterwarnings("ignore")


class AutonomousDataPreprocessor:
    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        target_column: str = None,
        output_dir: str = "."
    ):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.target_column = target_column
        self.user_specified_target = target_column is not None

        self.df: pd.DataFrame = None
        self.original_shape: tuple[int, int] = None

        self.numerical_columns: list[str] = []
        self.categorical_columns: list[str] = []

        self.encoders: dict[str, LabelEncoder] = {}
        self.target_encoder: LabelEncoder = None
        self.scaler: RobustScaler = None

        self.log: list[dict] = []
        self.output_folder = f"{output_dir}/output_folder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(f"{self.output_folder}/visualizations", exist_ok=True)

    # ---------------------- logging ---------------------- #
    def log_step(self, message: str, details=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"timestamp": ts, "step": message, "details": details}
        self.log.append(entry)
        print(f"[{ts}] {message}")
        if details is not None:
            print(f"  Details: {details}")

    # ---------------------- loading ---------------------- #
    def load_data(self) -> bool:
        self.log_step("Loading dataset...", self.dataset_path)
        try:
            if self.dataset_path.endswith(".csv"):
                self.df = pd.read_csv(self.dataset_path)
            elif self.dataset_path.endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(self.dataset_path)
            elif self.dataset_path.endswith(".json"):
                self.df = pd.read_json(self.dataset_path)
            elif self.dataset_path.endswith(".parquet"):
                self.df = pd.read_parquet(self.dataset_path)
            else:
                raise ValueError("Unsupported file format")

            self.original_shape = self.df.shape
            self.log_step(
                "Data loaded successfully",
                {"shape": self.df.shape, "columns": list(self.df.columns)},
            )
            return True
        except Exception as e:
            self.log_step(f"Error loading data: {e}")
            return False

    # ---------------------- analysis ---------------------- #
    def basic_analysis(self):
        if self.df is None:
            return
        info = {
            "shape": self.df.shape,
            "dtypes": {c: str(t) for c, t in self.df.dtypes.items()},
            "missing_per_col": self.df.isnull().sum().to_dict(),
            "duplicates": int(self.df.duplicated().sum()),
        }
        self.log_step("Basic analysis complete", info)

        # Simple visualizations (guarded for huge data)
        if self.df.shape[1] <= 100:
            missing = self.df.isnull()
            if missing.sum().sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.heatmap(missing, cbar=True)
                plt.title("Missing Values Heatmap (Initial)")
                plt.tight_layout()
                plt.savefig(
                    f"{self.output_folder}/visualizations/01_missing_initial.png",
                    dpi=200,
                )
                print("Saved missing values heatmap.")
                plt.close()

        dtype_counts = self.df.dtypes.value_counts()
        plt.figure(figsize=(6, 4))
        dtype_counts.plot(kind="bar")
        plt.title("Data Types Distribution")
        plt.tight_layout()
        plt.savefig(
            f"{self.output_folder}/visualizations/02_dtypes.png",
            dpi=200,
        )
        print("Saved data types distribution plot.")
        plt.close()

    # ---------------------- target & types ---------------------- #
    def detect_target(self):
        if self.df is None:
            return

        if self.user_specified_target and self.target_column in self.df.columns:
            self.log_step(f"Using user-specified target: {self.target_column}")
            return

        if self.user_specified_target and self.target_column not in self.df.columns:
            self.log_step(
                f"User-specified target '{self.target_column}' not found. Auto-detecting."
            )
            self.target_column = None

        common_names = [
            "target",
            "label",
            "class",
            "y",
            "output",
            "price",
            "value",
            "category",
            "result",
        ]
        for col in self.df.columns[::-1]:
            if any(k in col.lower() for k in common_names):
                self.target_column = col
                self.log_step(f"Auto-detected target column: {self.target_column}")
                return

        # fallback: last column
        self.target_column = self.df.columns[-1]
        self.log_step(f"Using last column as target: {self.target_column}")

    def identify_column_types(self):
        if self.df is None:
            return

        # drop obvious IDs
        for id_name in ["id", "Id", "ID"]:
            if id_name in self.df.columns:
                self.df.drop(columns=[id_name], inplace=True)
                self.log_step(f"Dropped ID-like column: {id_name}")

        self.numerical_columns = []
        self.categorical_columns = []

        for col in self.df.columns:
            if col == self.target_column:
                continue

            series = self.df[col]

            # try parse datetime (lightweight)
            if series.dtype == "object":
                try:
                    parsed = pd.to_datetime(series, errors="raise")
                    self.df[col] = parsed
                    self.log_step(f"Converted {col} to datetime")
                except Exception:
                    pass

            if np.issubdtype(self.df[col].dtype, np.number):
                unique = series.nunique(dropna=True)
                if unique < 10 or unique / max(len(series), 1) < 0.05:
                    self.categorical_columns.append(col)
                else:
                    self.numerical_columns.append(col)
            elif np.issubdtype(self.df[col].dtype, np.datetime64):
                # we'll extract numeric parts later
                pass
            else:
                self.categorical_columns.append(col)

        self.log_step(
            "Column type detection complete",
            {"numeric": self.numerical_columns, "categorical": self.categorical_columns},
        )

    # ---------------------- cleaning ---------------------- #
    def handle_missing_values(self):
        if self.df is None:
            return

        self.log_step("Handling missing values...")
        total_before = int(self.df.isnull().sum().sum())

        # numerical
        num_cols = [c for c in self.numerical_columns if c in self.df.columns]
        if num_cols:
            missing_pct = self.df[num_cols].isnull().mean()

            # drop >50% missing
            drop_cols = missing_pct[missing_pct > 0.5].index.tolist()
            if drop_cols:
                self.df.drop(columns=drop_cols, inplace=True)
                self.numerical_columns = [
                    c for c in self.numerical_columns if c not in drop_cols
                ]
                self.log_step("Dropped numeric columns (>50% missing)", drop_cols)

            num_cols = [c for c in self.numerical_columns if c in self.df.columns]
            missing_pct = self.df[num_cols].isnull().mean()

            # KNN for moderate (10–50%)
            knn_cols = missing_pct[(missing_pct > 0.1) & (missing_pct <= 0.5)].index.tolist()
            if knn_cols:
                imputer = KNNImputer(n_neighbors=5)
                self.df[knn_cols] = imputer.fit_transform(self.df[knn_cols])
                self.log_step("KNN imputation for numeric columns", knn_cols)

            # median for small (<=10%)
            med_cols = missing_pct[missing_pct <= 0.1].index.tolist()
            for c in med_cols:
                if self.df[c].isnull().any():
                    self.df[c].fillna(self.df[c].median(), inplace=True)
                    self.log_step(f"Median imputation for {c}")

        # categorical
        cat_cols = [c for c in self.categorical_columns if c in self.df.columns]
        if cat_cols:
            missing_pct_cat = self.df[cat_cols].isnull().mean()
            drop_cat = missing_pct_cat[missing_pct_cat > 0.5].index.tolist()
            if drop_cat:
                self.df.drop(columns=drop_cat, inplace=True)
                self.categorical_columns = [
                    c for c in self.categorical_columns if c not in drop_cat
                ]
                self.log_step("Dropped categorical columns (>50% missing)", drop_cat)

            cat_cols = [c for c in self.categorical_columns if c in self.df.columns]
            for c in cat_cols:
                if self.df[c].isnull().any():
                    mode_vals = self.df[c].mode()
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                    self.df[c].fillna(fill_val, inplace=True)
                    self.log_step(f"Mode/Unknown imputation for {c}")

        total_after = int(self.df.isnull().sum().sum())
        self.log_step(f"Missing values handled: {total_before} -> {total_after}")

    def remove_duplicates(self):
        if self.df is None:
            return
        dup = int(self.df.duplicated().sum())
        if dup > 0:
            self.df.drop_duplicates(inplace=True)
            self.log_step(f"Removed duplicate rows: {dup}")

    def handle_outliers(self):
        if self.df is None:
            return
        self.log_step("Capping outliers using IQR...")
        capped = 0

        for col in self.numerical_columns:
            if col not in self.df.columns or col == self.target_column:
                continue
            s = self.df[col].astype(float)
            if s.nunique() < 5:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lb = q1 - 1.5 * iqr
            ub = q3 + 1.5 * iqr
            before = s.copy()
            s = s.clip(lb, ub)
            capped += int((before != s).sum())
            self.df[col] = s

        self.log_step(f"Total outlier values capped: {capped}")

    # ---------------------- feature engineering ---------------------- #
    def datetime_features(self):
        if self.df is None:
            return

        dt_cols = self.df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
        new_cols = []
        for col in dt_cols:
            self.df[f"{col}_year"] = self.df[col].dt.year
            self.df[f"{col}_month"] = self.df[col].dt.month
            self.df[f"{col}_dayofweek"] = self.df[col].dt.dayofweek
            new_cols.extend([f"{col}_year", f"{col}_month", f"{col}_dayofweek"])
            self.df.drop(columns=[col], inplace=True)
        if new_cols:
            self.log_step("Extracted basic datetime features", new_cols)

        # refresh numeric list
        self.numerical_columns = [
            c
            for c in self.df.columns
            if c != self.target_column and np.issubdtype(self.df[c].dtype, np.number)
        ]

    # ---------------------- encoding & scaling ---------------------- #
    def encode_categoricals(self):
        if self.df is None:
            return
        self.log_step("Encoding categorical variables...")

        updated_numeric = set(self.numerical_columns)

        # encode target if categorical
        if self.target_column is not None and self.df[self.target_column].dtype == "object":
            self.target_encoder = LabelEncoder()
            self.df[self.target_column] = self.target_encoder.fit_transform(
                self.df[self.target_column].astype(str)
            )
            self.log_step("Label-encoded target column", self.target_column)

        for col in list(self.categorical_columns):
            if col not in self.df.columns or col == self.target_column:
                continue

            unique = self.df[col].nunique(dropna=True)
            if unique == 2:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                updated_numeric.add(col)
                self.log_step(f"Binary label-encoded {col}")
            elif unique <= 10:
                dummies = pd.get_dummies(self.df[col].astype(str), prefix=col, drop_first=True)
                self.df.drop(columns=[col], inplace=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                updated_numeric.update(dummies.columns)
                self.log_step(f"One-hot encoded {col} ({unique} categories)")
            else:
                freq = self.df[col].value_counts(normalize=True)
                self.df[col] = self.df[col].map(freq).fillna(0.0)
                updated_numeric.add(col)
                self.log_step(f"Frequency encoded {col} ({unique} categories)")

        self.categorical_columns = []
        self.numerical_columns = [
            c
            for c in self.df.columns
            if c != self.target_column and np.issubdtype(self.df[c].dtype, np.number)
        ]
        self.log_step(
            "Categorical encoding done",
            {"numeric_columns": self.numerical_columns},
        )

    def scale_features(self):
        if self.df is None:
            return
        self.log_step("Scaling numeric features with RobustScaler...")
        feat_cols = [
            c
            for c in self.numerical_columns
            if c in self.df.columns and c != self.target_column
        ]
        if not feat_cols:
            self.log_step("No numeric features to scale.")
            return

        self.scaler = RobustScaler()
        self.df[feat_cols] = self.scaler.fit_transform(self.df[feat_cols])
        self.df[feat_cols] = self.df[feat_cols].astype("float32")
        self.log_step(f"Scaled {len(feat_cols)} numeric features.")

    # ---------------------- feature selection ---------------------- #
    def feature_selection(self):
        if self.df is None or self.target_column not in self.df.columns:
            return
        self.log_step("Running simple feature selection...")

        feat_cols = [
            c
            for c in self.df.columns
            if c != self.target_column and np.issubdtype(self.df[c].dtype, np.number)
        ]
        if not feat_cols:
            self.log_step("No numeric features for selection; skipping.")
            return

        X = self.df[feat_cols]

        # remove zero-variance features
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(X)
        keep_mask = vt.get_support()
        keep_cols = [c for c, keep in zip(feat_cols, keep_mask) if keep]
        drop_zero = list(set(feat_cols) - set(keep_cols))
        if drop_zero:
            self.df.drop(columns=drop_zero, inplace=True)
            self.log_step("Dropped zero-variance features", drop_zero)

        # drop highly correlated features (|corr| > 0.95)
        feat_cols = [
            c
            for c in self.df.columns
            if c != self.target_column and np.issubdtype(self.df[c].dtype, np.number)
        ]
        if len(feat_cols) > 1:
            corr = self.df[feat_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
            if to_drop:
                self.df.drop(columns=to_drop, inplace=True)
                self.log_step("Dropped highly correlated features", to_drop)

    # ---------------------- visuals (processed) ---------------------- #
    def visualize_processed(self):
        if self.df is None or self.target_column not in self.df.columns:
            return

        self.log_step("Generating simple processed-data visualizations...")

        feat_cols = [
            c
            for c in self.df.columns
            if c != self.target_column and np.issubdtype(self.df[c].dtype, np.number)
        ]

        # correlation heatmap (top 20)
        top = feat_cols[:20]
        if len(top) > 1:
            plt.figure(figsize=(12, 8))
            corr = self.df[top].corr()
            sns.heatmap(corr, cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap (Top 20 Numeric Features)")
            plt.tight_layout()
            plt.savefig(
                f"{self.output_folder}/visualizations/03_corr_processed.png",
                dpi=200,
            )
            plt.close()

        # target distribution
        y = self.df[self.target_column]
        plt.figure(figsize=(6, 4))
        if np.issubdtype(y.dtype, np.number) and y.nunique() > 20:
            y.hist(bins=40)
            plt.ylabel("Frequency")
        else:
            y.value_counts().plot(kind="bar")
            plt.ylabel("Count")
        plt.title(f"Target Distribution: {self.target_column}")
        plt.tight_layout()
        plt.savefig(
            f"{self.output_folder}/visualizations/04_target_processed.png",
            dpi=200,
        )
        plt.close()

        # simple MI plot (optional, top 20)
        try:
            if feat_cols:
                X = self.df[feat_cols]
                y = self.df[self.target_column]
                if np.issubdtype(y.dtype, np.integer) and y.nunique() <= 20:
                    mi = mutual_info_classif(X, y, random_state=self.random_state)
                else:
                    mi = mutual_info_regression(X, y, random_state=self.random_state)
                mi = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
                plt.figure(figsize=(8, 6))
                mi.head(20).plot(kind="barh")
                plt.title("Mutual Information (Top 20)")
                plt.tight_layout()
                plt.savefig(
                    f"{self.output_folder}/visualizations/05_mi.png",
                    dpi=200,
                )
                plt.close()
        except Exception as e:
            self.log_step(f"MI plot skipped due to error: {e}")

    # ---------------------- splitting & saving ---------------------- #
    def _convert_bool_to_int(self):
        bool_cols = self.df.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            self.df[bool_cols] = self.df[bool_cols].astype("int8")
            self.log_step("Converted bool columns to int8", bool_cols)

    def split_and_save(self):
        if self.df is None or self.target_column not in self.df.columns:
            raise ValueError("Target column not present for splitting.")

        self.log_step("Splitting into train/test and saving CSVs...")

        self._convert_bool_to_int()

        feat_cols = [c for c in self.df.columns if c != self.target_column]
        X = self.df[feat_cols]
        y = self.df[self.target_column]

        stratify = None
        if (np.issubdtype(y.dtype, np.integer) or y.dtype == "object") and y.nunique() <= 20:
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_path = os.path.join(self.output_folder, "train_data.csv")
        test_path = os.path.join(self.output_folder, "test_data.csv")
        full_path = os.path.join(self.output_folder, "processed_full_data.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        self.df.to_csv(full_path, index=False)

        self.log_step(
            "Data saved",
            {
                "train_shape": train_df.shape,
                "test_shape": test_df.shape,
                "train_path": train_path,
                "test_path": test_path,
                "full_path": full_path,
            },
        )

    # ---------------------- logging & summary ---------------------- #
    def save_logs_and_summary(self):
        log_path = os.path.join(self.output_folder, "preprocessing_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log, f, indent=4)

        txt_log_path = os.path.join(self.output_folder, "preprocessing_log.txt")
        with open(txt_log_path, "w") as f:
            for entry in self.log:
                f.write(f"[{entry['timestamp']}] {entry['step']}\n")
                if entry["details"] is not None:
                    f.write(f"  Details: {entry['details']}\n")
                f.write("\n")

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": self.dataset_path,
            "output_folder": self.output_folder,
            "original_shape": str(self.original_shape),
            "final_shape": str(self.df.shape if self.df is not None else None),
            "target_column": self.target_column,
        }
        summary_path = os.path.join(self.output_folder, "summary_report.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        self.log_step("Saved logs and summary", summary)

    def save_transformation_metadata(self):
        """
        Save the encoders, scalers, and other transformation metadata to a JSON file.
        This metadata can be used to apply the same transformations to new datasets.
        """
        metadata = {
            "encoders": {
                col: encoder.classes_.tolist() for col, encoder in self.encoders.items()
            },
            "target_encoder": self.target_encoder.classes_.tolist()
            if self.target_encoder else None,
            "scaler": {
                "center_": self.scaler.center_.tolist() if self.scaler else None,
                "scale_": self.scaler.scale_.tolist() if self.scaler else None,
                "feature_names": self.numerical_columns,  # Save the feature names used for scaling
            },
            "one_hot_columns": [
                col for col in self.df.columns if col not in self.numerical_columns + self.categorical_columns + [self.target_column]
            ],  # Save one-hot encoded columns
        }

        metadata_path = os.path.join(self.output_folder, "transformation_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        self.log_step("Saved transformation metadata", metadata_path)

    def load_transformation_metadata(self, metadata_path):
        """
        Load the encoders, scalers, and other transformation metadata from a JSON file.
        This metadata can be used to apply the same transformations to new datasets.
        """
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load encoders
        self.encoders = {
            col: LabelEncoder().fit(classes)
            for col, classes in metadata.get("encoders", {}).items()
        }

        # Load target encoder
        if metadata.get("target_encoder"):
            self.target_encoder = LabelEncoder()
            self.target_encoder.classes_ = np.array(metadata["target_encoder"])

        # Load scaler
        if metadata.get("scaler"):
            self.scaler = RobustScaler()
            self.scaler.center_ = np.array(metadata["scaler"].get("center_", []))
            self.scaler.scale_ = np.array(metadata["scaler"].get("scale_", []))

        self.log_step("Loaded transformation metadata", metadata_path)

    # ---------------------- orchestrator ---------------------- #
    def run(self) -> bool:
        print("\n" + "=" * 80)
        print("AUTONOMOUS DATA PREPROCESSING AGENT - RUN")
        print("=" * 80 + "\n")

        try:
            if not self.load_data():
                return False

            self.basic_analysis()
            self.detect_target()
            self.identify_column_types()
            self.handle_missing_values()
            self.remove_duplicates()
            self.handle_outliers()
            self.datetime_features()
            self.encode_categoricals()
            self.scale_features()
            self.feature_selection()
            self.visualize_processed()
            self.split_and_save()
            self.save_logs_and_summary()
            self.save_transformation_metadata()

            print("\n" + "=" * 80)
            print("PREPROCESSING COMPLETE")
            print(f"Output folder: {self.output_folder}")
            print("=" * 80 + "\n")
            return True

        except Exception as e:
            self.log_step(f"CRITICAL ERROR: {e}")
            import traceback

            traceback.print_exc()
            return False


# ---------------------- Public API ---------------------- #
def autonomous_data_preprocessing(
    dataset_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = None,
    output_dir: str = "."
) -> bool:
    agent = AutonomousDataPreprocessor(
        dataset_path=dataset_path,
        test_size=test_size,
        random_state=random_state,
        target_column=target_column,
        output_dir = output_dir
    )
    success = agent.run()

    if success:
        print(f"\n✅ SUCCESS! ML-ready data saved in: {agent.output_folder}")
        train_path = os.path.join(agent.output_folder, "train_data.csv")
        test_path = os.path.join(agent.output_folder, "test_data.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(f"📊 Train shape: {train_df.shape}")
            print(f"📊 Test shape:  {test_df.shape}")
            print(
                f"📈 #Features: {len([c for c in train_df.columns if c != agent.target_column])}"
            )
            print(f"🎯 Target: {agent.target_column}")
    return success


def interactive_preprocessing():
    print("\n" + "=" * 80)
    print("AUTONOMOUS DATA PREPROCESSING AGENT - INTERACTIVE MODE")
    print("=" * 80 + "\n")

    dataset_path = input("Enter dataset path: ").strip()
    if not os.path.exists(dataset_path):
        print(f"\n❌ File not found: {dataset_path}")
        return False

    # quick preview
    try:
        if dataset_path.endswith(".csv"):
            temp_df = pd.read_csv(dataset_path, nrows=5)
        elif dataset_path.endswith((".xlsx", ".xls")):
            temp_df = pd.read_excel(dataset_path, nrows=5)
        elif dataset_path.endswith(".json"):
            temp_df = pd.read_json(dataset_path).head(5)
        else:
            print("Unsupported format in interactive mode.")
            return False
        print("\nColumns:")
        for i, c in enumerate(temp_df.columns, 1):
            print(f"  {i}. {c}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    tgt_input = input(
        "\nTarget column (name or number, Enter for auto-detect): "
    ).strip()
    target_column = None
    if tgt_input:
        if tgt_input.isdigit():
            idx = int(tgt_input) - 1
            if 0 <= idx < len(temp_df.columns):
                target_column = temp_df.columns[idx]
        else:
            if tgt_input in temp_df.columns:
                target_column = tgt_input

    ts_input = input("Test size % (default 20): ").strip()
    try:
        test_size = float(ts_input) / 100 if ts_input else 0.2
        test_size = max(0.1, min(0.5, test_size))
    except Exception:
        test_size = 0.2

    print("\n" + "=" * 80)
    print("CONFIG:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Target:  {target_column or 'AUTO'}")
    print(f"  Test %:  {int(test_size * 100)}")
    print("=" * 80 + "\n")

    confirm = input("Proceed? (y/N): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Cancelled.")
        return False

    return autonomous_data_preprocessing(
        dataset_path=dataset_path,
        test_size=test_size,
        target_column=target_column,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        interactive_preprocessing()
    elif len(sys.argv) == 2:
        autonomous_data_preprocessing(sys.argv[1])
    else:
        print("Usage:")
        print("  python script.py                # interactive mode")
        print("  python script.py <dataset.csv>  # direct mode")
