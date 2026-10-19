"""
preprocessing.py
Comprehensive data cleaning, outlier handling, feature selection, and leakage-free
scaling pipeline for the CIC-IDS2017 network intrusion dataset.

Key Highlights:
- Infinite & NaN value resolution (handling Flow Bytes/s and Flow Packets/s)
- Duplicate row detection and removal with audit tracking
- Statistical outlier analysis using IQR without discarding legitimate attack patterns
- Removal of zero-variance (constant) features and non-predictive identifiers
- Stratified train-val-test partitioning
- Scikit-learn compliant preprocessor fitting ONLY on training data to prevent leakage
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class NetworkTrafficPreprocessor:
    """
    Leakage-free preprocessor for network intrusion flow data.
    Fits scalers and column masks exclusively on training splits.
    """

    def __init__(
        self,
        target_col: str = "Label",
        binary_mode: bool = False,
        drop_duplicates: bool = True,
        variance_threshold: float = 0.0,
        random_state: int = 42
    ):
        """
        Args:
            target_col: Name of the label column.
            binary_mode: If True, merges all non-BENIGN attacks into 'ATTACK'.
            drop_duplicates: Whether to drop duplicate network flow rows.
            variance_threshold: Minimum variance threshold to keep a feature.
            random_state: Seed for reproducibility.
        """
        self.target_col = target_col
        self.binary_mode = binary_mode
        self.drop_duplicates = drop_duplicates
        self.variance_threshold = variance_threshold
        self.random_state = random_state

        # Stateful components fitted ONLY on train set
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.dropped_constant_columns: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.impute_medians: Dict[str, float] = {}
        self.is_fitted: bool = False

        # Preprocessing audit report
        self.audit_log: Dict[str, Union[int, float, List[str], Dict[str, int]]] = {}

    def clean_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans invalid numeric values, infinities, NaNs, and duplicates.

        Args:
            df: Raw input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        data = df.copy()
        initial_rows = len(data)

        # 1. Standardize column names
        data.columns = data.columns.str.strip()

        # 2. Check and remove duplicate rows
        if self.drop_duplicates:
            dup_count = int(data.duplicated().sum())
            data = data.drop_duplicates().reset_index(drop=True)
            self.audit_log["initial_rows"] = initial_rows
            self.audit_log["duplicate_rows_removed"] = dup_count
            self.audit_log["post_dedup_rows"] = len(data)
            print(f"[Preprocessing] Removed {dup_count:,} duplicate rows ({len(data):,} remaining).")

        # 3. Handle Target Label
        if self.target_col not in data.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in DataFrame.")

        # Clean label strings (strip whitespace and unprintable characters)
        data[self.target_col] = data[self.target_col].astype(str).str.strip()
        # In CIC-IDS2017, web attacks sometimes have special non-breaking spaces or characters
        data[self.target_col] = data[self.target_col].replace({
            r"Web Attack\s+\s+Brute Force": "Web Attack - Brute Force",
            r"Web Attack\s+\s+XSS": "Web Attack - XSS",
            r"Web Attack\s+\s+Sql Injection": "Web Attack - Sql Injection",
        }, regex=True)

        if self.binary_mode:
            data[self.target_col] = data[self.target_col].apply(
                lambda x: "BENIGN" if x.upper() == "BENIGN" else "ATTACK"
            )

        # 4. Handle Infinite & NaN values in numeric columns
        # In CIC-IDS2017, 'Flow Bytes/s' and 'Flow Packets/s' frequently contain +/- inf
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        inf_counts = 0
        for col in numeric_cols:
            inf_mask = np.isinf(data[col])
            count = int(inf_mask.sum())
            if count > 0:
                inf_counts += count
                # Replace +/- inf with NaN so it can be handled via statistical imputation
                data.loc[inf_mask, col] = np.nan

        self.audit_log["infinite_values_replaced_with_nan"] = inf_counts
        if inf_counts > 0:
            print(f"[Preprocessing] Detected & replaced {inf_counts:,} +/-inf values with NaN.")

        # Null value count before imputation
        null_counts = int(data[numeric_cols].isnull().sum().sum())
        self.audit_log["null_values_detected"] = null_counts

        return data

    @staticmethod
    def compute_iqr_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Computes statistical summary of outliers using the Interquartile Range (IQR) rule:
        [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

        Note: In network intrusion detection, extreme traffic spikes (e.g. volumetric DDoS,
        port scanning bursts) are legitimate attack markers. This function audits outlier
        proportions per class to document cyberattack footprints rather than indiscriminately
        deleting valid attack flows.

        Args:
            df: Cleaned dataframe.
            columns: List of numeric features to audit.

        Returns:
            pd.DataFrame: Table with Q1, Q3, IQR, lower bound, upper bound, and outlier counts.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        records = []
        for col in columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            records.append({
                "Feature": col,
                "Q1 (25%)": round(q1, 4),
                "Median (50%)": round(float(series.median()), 4),
                "Q3 (75%)": round(q3, 4),
                "IQR": round(iqr, 4),
                "Lower Bound": round(lower_bound, 4),
                "Upper Bound": round(upper_bound, 4),
                "Outlier Count": len(outliers),
                "Outlier %": round((len(outliers) / len(series)) * 100, 2)
            })

        return pd.DataFrame(records)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits scalers, encoders, and feature masks ONLY on the training split
        to ensure zero data leakage.

        Args:
            X_train: Training features DataFrame.
            y_train: Training target labels Series.
        """
        # Exclude non-predictive or metadata columns and any label columns if present
        cols_to_exclude = {self.target_col, "Label", "Binary_Label", "Source_File", "Timestamp", "Flow ID", "Source IP", "Destination IP"}
        candidate_cols = [c for c in X_train.columns if c not in cols_to_exclude and np.issubdtype(X_train[c].dtype, np.number)]

        # Impute missing values using training set medians
        self.impute_medians = {}
        for c in candidate_cols:
            median_val = float(X_train[c].median())
            self.impute_medians[c] = 0.0 if np.isnan(median_val) else median_val

        X_imputed = X_train[candidate_cols].fillna(self.impute_medians)

        # Detect constant (zero variance) features
        variances = X_imputed.var()
        constant_cols = variances[variances <= self.variance_threshold].index.tolist()
        self.dropped_constant_columns = constant_cols
        self.feature_columns = [c for c in candidate_cols if c not in constant_cols]

        print(f"[Preprocessing] Selected {len(self.feature_columns)} features. Dropped {len(constant_cols)} constant/zero-variance features.")

        # Fit Scaler on training features only
        self.scaler.fit(X_imputed[self.feature_columns])

        # Fit LabelEncoder on training targets
        self.label_encoder.fit(y_train)
        self.classes_ = self.label_encoder.classes_

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms input features and labels using pre-fitted parameters.

        Args:
            X: Features DataFrame.
            y: Optional target labels Series.

        Returns:
            Tuple of (scaled_X: np.ndarray, encoded_y: Optional[np.ndarray])
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted using fit() before transform().")

        # Impute using training medians
        X_sub = X[self.feature_columns].copy()
        for col in self.feature_columns:
            if col in self.impute_medians:
                X_sub[col] = X_sub[col].fillna(self.impute_medians[col])

        # Scale features
        X_scaled = self.scaler.transform(X_sub)

        # Encode labels if provided
        y_encoded = None
        if y is not None:
            # Handle unseen labels gracefully if any
            y_clean = y.astype(str).str.strip()
            # Map unseen labels to the closest known class or raise warning
            known_classes = set(self.classes_)
            unknown_labels = set(y_clean.unique()) - known_classes
            if unknown_labels:
                print(f"[Preprocessing] Warning: Unseen test labels detected: {unknown_labels}. Assigning majority class.")
                y_clean = y_clean.apply(lambda val: val if val in known_classes else self.classes_[0])
            y_encoded = self.label_encoder.transform(y_clean)

        return X_scaled, y_encoded

    def split_and_preprocess(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, "NetworkTrafficPreprocessor"]:
        """
        Performs stratified train/validation/test split followed by strictly leakage-free
        fitting on train and transformation on validation and test.

        Returns:
            Tuple: (X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
        """
        cleaned_df = self.clean_raw_dataframe(df)

        y = cleaned_df[self.target_col]
        X = cleaned_df.drop(columns=[self.target_col])

        # Step 1: Split into Train+Val and Test using Stratification
        # Calculate relative test and val sizes
        combined_val_test = test_size + val_size
        X_train_val, X_test_raw, y_train_val, y_test_raw = train_test_split(
            X, y,
            test_size=combined_val_test,
            stratify=y,
            random_state=self.random_state
        )

        # Step 2: Split Train+Val into Train and Val
        val_fraction = val_size / combined_val_test
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_train_val, y_train_val,
            test_size=val_fraction,
            stratify=y_train_val,
            random_state=self.random_state
        )

        print(f"[Preprocessing] Stratified Split: Train={len(X_train_raw):,}, Val={len(X_val_raw):,}, Test={len(X_test_raw):,}")

        # Step 3: Fit exclusively on training split
        self.fit(X_train_raw, y_train_raw)

        # Step 4: Transform each split
        X_train, y_train = self.transform(X_train_raw, y_train_raw)
        X_val, y_val = self.transform(X_val_raw, y_val_raw)
        X_test, y_test = self.transform(X_test_raw, y_test_raw)

        return X_train, y_train, X_val, y_val, X_test, y_test, self
