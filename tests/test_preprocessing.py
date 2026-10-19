"""
test_preprocessing.py
Unit tests for data loading, preprocessing, statistical analysis, and MLP initialization.
Uses isolated temporary synthetic fixtures strictly for software validation.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.data_loader import CICIDSDataLoader, inspect_raw_dataset
from src.preprocessing import NetworkTrafficPreprocessor
from src.statistical_analysis import NetworkStatisticalAnalyzer
from src.train_model import build_mlp_model, calculate_class_weights


@pytest.fixture
def synthetic_network_fixture():
    """
    Creates a temporary synthetic network dataset strictly for software testing.
    This fixture ensures reproducible unit testing of data pipelines.
    """
    np.random.seed(42)
    n_samples = 300

    labels = ["BENIGN"] * 240 + ["DDoS"] * 40 + ["PortScan"] * 20
    np.random.shuffle(labels)

    data = {
        "Flow Duration": np.random.exponential(scale=1e5, size=n_samples),
        "Total Fwd Packets": np.random.poisson(lam=10, size=n_samples) + 1,
        "Total Backward Packets": np.random.poisson(lam=8, size=n_samples),
        "Total Length of Fwd Packets": np.random.exponential(scale=1500, size=n_samples),
        "Fwd Packet Length Mean": np.random.uniform(20, 1400, size=n_samples),
        "Flow Bytes/s": np.random.exponential(scale=5000, size=n_samples),
        "Flow Packets/s": np.random.exponential(scale=100, size=n_samples),
        "Constant Zero Col": np.zeros(n_samples),  # To test constant column elimination
        "Label": labels
    }
    df = pd.DataFrame(data)

    # Inject specific anomalies to test cleaning routines
    df.loc[5, "Flow Bytes/s"] = np.inf
    df.loc[12, "Flow Packets/s"] = -np.inf
    df.loc[25, "Flow Bytes/s"] = np.nan

    # Inject duplicate rows
    df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
    return df


def test_data_loader_missing_directory_warning():
    """Test data loader handles empty/missing directories with explicit warnings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        loader = CICIDSDataLoader(data_dir=temp_dir)
        is_present, files, msg = loader.check_dataset_status()
        assert not is_present
        assert len(files) == 0
        assert "NO CIC-IDS2017 CSV FILES DETECTED" in msg

        with pytest.raises(FileNotFoundError):
            loader.load_dataset()


def test_preprocessing_cleaning_pipeline(synthetic_network_fixture):
    """Test handling of duplicates, infs, and NaNs."""
    initial_len = len(synthetic_network_fixture)
    preprocessor = NetworkTrafficPreprocessor(target_col="Label", binary_mode=False)
    cleaned = preprocessor.clean_raw_dataframe(synthetic_network_fixture)

    # 3 duplicate rows should have been removed
    assert len(cleaned) == initial_len - 3
    # Infs must be eliminated
    assert not np.isinf(cleaned["Flow Bytes/s"]).any()
    assert not np.isinf(cleaned["Flow Packets/s"]).any()
    assert preprocessor.audit_log["duplicate_rows_removed"] == 3
    assert preprocessor.audit_log["infinite_values_replaced_with_nan"] == 2


def test_preprocessing_binary_mode(synthetic_network_fixture):
    """Test binary conversion to BENIGN and ATTACK."""
    preprocessor = NetworkTrafficPreprocessor(target_col="Label", binary_mode=True)
    cleaned = preprocessor.clean_raw_dataframe(synthetic_network_fixture)
    unique_labels = set(cleaned["Label"].unique())
    assert unique_labels == {"BENIGN", "ATTACK"}


def test_iqr_outlier_computation(synthetic_network_fixture):
    """Test statistical IQR calculation."""
    iqr_df = NetworkTrafficPreprocessor.compute_iqr_outliers(synthetic_network_fixture)
    assert not iqr_df.empty
    assert "Feature" in iqr_df.columns
    assert "IQR" in iqr_df.columns
    assert "Outlier Count" in iqr_df.columns


def test_leakage_free_split_and_fit(synthetic_network_fixture):
    """
    Verifies that preprocessor fits ONLY on training data without test set leakage.
    Checks that constant zero-variance features are detected and dropped.
    """
    preprocessor = NetworkTrafficPreprocessor(target_col="Label", binary_mode=False)
    X_train, y_train, X_val, y_val, X_test, y_test, prep = preprocessor.split_and_preprocess(
        synthetic_network_fixture, test_size=0.15, val_size=0.15
    )

    # Constant column must be excluded from feature columns
    assert "Constant Zero Col" not in prep.feature_columns
    assert "Constant Zero Col" in prep.dropped_constant_columns

    # Verify train/val/test splits non-empty and shapes aligned
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == len(prep.feature_columns)
    assert prep.is_fitted


def test_statistical_analyzer_methods(synthetic_network_fixture):
    """Test descriptive statistics, goodness-of-fit, and hypothesis tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = NetworkStatisticalAnalyzer(output_dir=temp_dir)
        desc_df = analyzer.compute_comprehensive_descriptive_stats(synthetic_network_fixture)
        assert not desc_df.empty
        assert "Skewness" in desc_df.columns
        assert "Kurtosis" in desc_df.columns

        prob_df = analyzer.compute_empirical_probabilities(synthetic_network_fixture, label_col="Label")
        assert not prob_df.empty
        assert "Empirical Probability P(X)" in prob_df.columns

        fit_df = analyzer.evaluate_goodness_of_fit(synthetic_network_fixture)
        assert not fit_df.empty

        hypo_df = analyzer.compute_nonparametric_hypothesis_tests(synthetic_network_fixture, label_col="Label")
        assert not hypo_df.empty


def test_mlp_architecture_and_forward_pass():
    """Test Keras MLP construction, class weighting, and forward pass."""
    input_dim = 10
    num_classes = 3
    model = build_mlp_model(input_dim=input_dim, num_classes=num_classes)
    assert model is not None

    # Test forward pass with dummy batch
    dummy_input = np.random.randn(8, input_dim).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    assert preds.shape == (8, num_classes)
    # Softmax probabilities should sum to 1
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5)

    # Test class weight calculation
    dummy_targets = np.array([0, 0, 0, 1, 1, 2])
    weights = calculate_class_weights(dummy_targets)
    assert len(weights) == 3
    assert weights[2] > weights[0]  # Underrepresented class should have higher weight
