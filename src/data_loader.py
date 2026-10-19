"""
data_loader.py
Module for discovering, validating, and loading CIC-IDS2017 dataset CSV files.

Supports:
- Configurable data directories
- Pattern-based or explicit file selection
- Column name sanitization (stripping whitespace)
- Missing file warnings with download guidance
- Memory-safe loading with optional row subsampling
"""

import os
import glob
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class CICIDSDataLoader:
    """
    Data loader for the Canadian Institute for Cybersecurity CIC-IDS2017 dataset.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader with a base directory.

        Args:
            data_dir: Relative or absolute path to the data folder.
        """
        self.data_dir = os.path.abspath(data_dir)

    def list_available_files(self) -> List[str]:
        """
        Scans the data directory and returns all found CSV files.
        """
        if not os.path.exists(self.data_dir):
            return []
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        return sorted(csv_files)

    def check_dataset_status(self) -> Tuple[bool, List[str], str]:
        """
        Verifies if valid dataset CSV files exist in the data directory.

        Returns:
            Tuple of (is_present: bool, files: List[str], message: str)
        """
        files = self.list_available_files()
        if not files:
            sep = "=" * 60
            msg = (
                f"\n{sep}\n"
                f"[WARNING] NO CIC-IDS2017 CSV FILES DETECTED IN DATA DIRECTORY!\n"
                f"Directory checked: {self.data_dir}\n"
                f"{sep}\n"
                f"To run with the real dataset:\n"
                f"1. Download one or more CSV files from either:\n"
                f"   - Official UNB portal: https://www.unb.ca/cic/datasets/ids-2017.html\n"
                f"   - Kaggle mirror: https://www.kaggle.com/datasets/cicdataset/cicids2017\n"
                f"2. Recommended starting files (single daily capture):\n"
                f"   - 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' (~225,000 flows)\n"
                f"   - 'Wednesday-workingHours.pcap_ISCX.csv' (DoS / Heartbleed)\n"
                f"3. Place the CSV file(s) into: {self.data_dir}\n"
                f"{sep}\n"
            )
            return False, [], msg
        
        file_list_str = "\n".join([f"  - {os.path.basename(f)} ({os.path.getsize(f) / (1024*1024):.2f} MB)" for f in files])
        msg = f"Found {len(files)} CSV file(s) in {self.data_dir}:\n{file_list_str}"
        return True, files, msg

    def load_dataset(
        self,
        file_names: Optional[Union[str, List[str]]] = None,
        nrows: Optional[int] = None,
        use_binary_label: bool = False
    ) -> pd.DataFrame:
        """
        Load and concatenate specified CSV file(s).

        Args:
            file_names: Single filename/pattern or list of filenames. If None,
                        loads the first available CSV or all matches.
            nrows: Optional maximum rows to load per file (for quick experimentation).
            use_binary_label: If True, adds a 'Binary_Label' column ('BENIGN' vs 'ATTACK').

        Returns:
            pd.DataFrame: Loaded and sanitized dataset.

        Raises:
            FileNotFoundError: If no matching CSV files are present.
        """
        is_present, available_files, status_msg = self.check_dataset_status()
        if not is_present:
            print(status_msg)
            raise FileNotFoundError(
                f"No CIC-IDS2017 CSV files found in '{self.data_dir}'. "
                f"Please follow the download instructions in {os.path.join(self.data_dir, 'README.md')}."
            )

        # Resolve files to load
        target_files = []
        if file_names is None:
            # Default to the first found file to preserve system RAM unless instructed
            target_files = [available_files[0]]
            print(f"[DataLoader] No specific file provided. Auto-selected: {os.path.basename(target_files[0])}")
        elif isinstance(file_names, str):
            # Check if exact path or pattern
            if os.path.isabs(file_names) and os.path.exists(file_names):
                target_files = [file_names]
            else:
                candidate = os.path.join(self.data_dir, file_names)
                if os.path.exists(candidate):
                    target_files = [candidate]
                else:
                    # Try glob match
                    matched = glob.glob(os.path.join(self.data_dir, file_names))
                    if matched:
                        target_files = matched
                    else:
                        raise FileNotFoundError(f"File or pattern '{file_names}' not found in {self.data_dir}")
        elif isinstance(file_names, list):
            for fn in file_names:
                candidate = fn if os.path.isabs(fn) else os.path.join(self.data_dir, fn)
                if os.path.exists(candidate):
                    target_files.append(candidate)
                else:
                    raise FileNotFoundError(f"File '{candidate}' does not exist.")

        # Read CSVs
        dfs = []
        for fpath in target_files:
            fname = os.path.basename(fpath)
            print(f"[DataLoader] Reading '{fname}' (nrows={nrows})...")
            # CIC-IDS2017 CSVs may have mixed encodings or trailing spaces in column headers
            df = pd.read_csv(
                fpath,
                nrows=nrows,
                encoding="utf-8",
                low_memory=False
            )
            # Standardize header names: strip spaces
            df.columns = df.columns.str.strip()
            df["Source_File"] = fname
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # Standardize Label column
        if "Label" in combined_df.columns:
            combined_df["Label"] = combined_df["Label"].astype(str).str.strip()
            if use_binary_label:
                combined_df["Binary_Label"] = combined_df["Label"].apply(
                    lambda x: "BENIGN" if x.upper() == "BENIGN" else "ATTACK"
                )

        print(f"[DataLoader] Successfully loaded {combined_df.shape[0]:,} rows and {combined_df.shape[1]} columns.")
        return combined_df


def inspect_raw_dataset(df: pd.DataFrame) -> dict:
    """
    Computes initial structural inspection summary of the raw dataframe.

    Returns:
        dict: Inspection dictionary with shapes, dtypes, nulls, inf counts.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Count inf values
    inf_counts = {}
    for col in numeric_cols:
        count = np.isinf(df[col]).sum()
        if count > 0:
            inf_counts[col] = int(count)

    summary = {
        "shape": df.shape,
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_values_per_col": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "infinite_values_per_col": inf_counts,
        "duplicate_rows": int(df.duplicated().sum()),
        "labels": df["Label"].value_counts().to_dict() if "Label" in df.columns else {}
    }
    return summary
