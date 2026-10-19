"""
main.py
End-to-end command-line pipeline execution script for Network Intrusion Detection & Analysis.
Satisfies Assignment II requirements: CO3 (Visualization), CO4 (Statistical Analysis), CO5 (MLP Application).

Usage:
  python main.py
  python main.py --file_name "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" --nrows 50000
  python main.py --binary
"""

import os
import sys
import argparse

from src.data_loader import CICIDSDataLoader, inspect_raw_dataset
from src.preprocessing import NetworkTrafficPreprocessor
from src.eda import NetworkIntrusionEDA
from src.statistical_analysis import NetworkStatisticalAnalyzer
from src.train_model import (
    train_mlp_classifier,
    run_hyperparameter_experiments,
    compute_permutation_feature_importance
)
from src.evaluate_model import ModelEvaluator
from src.generate_academic_report import create_academic_docx_report


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Network Intrusion Detection & EDA Pipeline (CIC-IDS2017)"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset CSV files.")
    parser.add_argument("--file_name", type=str, default=None, help="Specific CSV file or pattern to analyze.")
    parser.add_argument("--nrows", type=int, default=None, help="Max rows to load (useful for rapid testing).")
    parser.add_argument("--binary", action="store_true", help="Enable binary classification (BENIGN vs ATTACK).")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for MLP model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--skip_experiments", action="store_true", help="Skip hyperparameter grid experiments.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print("=" * 80)
    print("NETWORK INTRUSION DETECTION & CYBERATTACK PATTERN ANALYSIS")
    print("Course: U21ADP05 – Exploratory Data Analysis and Visualization (Assignment II)")
    print("Institution: KPR Institute of Engineering and Technology")
    print("=" * 80)

    # 1. Check & Load Dataset
    loader = CICIDSDataLoader(data_dir=args.data_dir)
    is_present, available_files, status_msg = loader.check_dataset_status()
    print(status_msg)

    if not is_present:
        print("[HALT] Real dataset CSV file is required to run the production research pipeline.")
        print("Please follow the instructions above or in 'data/README.md' to place a CSV in 'data/'.")
        print("Note: To verify software logic without real data, run unit tests via: python -m pytest tests/")
        sys.exit(0)

    print(f"\n[STEP 1/6] Loading data from '{args.data_dir}' (nrows={args.nrows})...")
    df = loader.load_dataset(file_names=args.file_name, nrows=args.nrows, use_binary_label=args.binary)
    inspection = inspect_raw_dataset(df)
    print(f"Dataset Shape: {inspection['shape']}")
    print(f"Class Distribution: {inspection['labels']}")

    target_col = "Binary_Label" if (args.binary and "Binary_Label" in df.columns) else "Label"

    # 2. Exploratory Data Analysis (CO3)
    print("\n[STEP 2/6] Running Exploratory Data Analysis & Visualization (CO3)...")
    eda = NetworkIntrusionEDA(output_dir="results/eda")
    eda.generate_all_visualizations(df, label_col=target_col)

    # 3. Statistical & Probability-Distribution Analysis (CO4)
    print("\n[STEP 3/6] Running Statistical & Distribution Analysis (CO4)...")
    analyzer = NetworkStatisticalAnalyzer(output_dir="results/tables")
    analyzer.run_full_statistical_pipeline(df, label_col=target_col)

    # 4. Leakage-Free Preprocessing & Stratified Splitting
    print("\n[STEP 4/6] Executing Anti-Leakage Preprocessing & Stratified Splitting...")
    preprocessor = NetworkTrafficPreprocessor(
        target_col=target_col,
        binary_mode=args.binary,
        drop_duplicates=True,
        random_state=42
    )
    X_train, y_train, X_val, y_val, X_test, y_test, prep = preprocessor.split_and_preprocess(
        df, test_size=0.15, val_size=0.15
    )

    # Save IQR outlier audit table
    iqr_table = preprocessor.compute_iqr_outliers(df)
    iqr_path = os.path.join("results/tables", "iqr_outliers_summary.csv")
    iqr_table.to_csv(iqr_path, index=False)
    print(f"[Preprocessing] Saved IQR outlier audit table to: {iqr_path}")

    # 5. Deep Learning Model Training (CO5)
    print("\n[STEP 5/6] Training Multilayer Perceptron (MLP) Deep Learning Classifier...")
    model, history = train_mlp_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=True,
        model_save_path="results/model_evaluation/best_mlp_model.keras"
    )

    if not args.skip_experiments:
        print("\n[STEP 5b] Running Hyperparameter Tuning Experiments...")
        run_hyperparameter_experiments(X_train, y_train, X_val, y_val, output_dir="results/model_evaluation")

    # 6. Model Evaluation on Held-Out Test Split
    print("\n[STEP 6/6] Evaluating Model Performance on Held-Out Test Set...")
    evaluator = ModelEvaluator(output_dir="results/model_evaluation")
    evaluator.plot_training_history(history)

    class_names = [str(cls) for cls in prep.classes_]
    eval_results = evaluator.evaluate_model_performance(model, X_test, y_test, class_names=class_names)
    evaluator.plot_confusion_matrix(y_test, eval_results["y_pred"], class_names=class_names)
    evaluator.plot_roc_curves(y_test, eval_results["y_probs"], class_names=class_names)
    evaluator.plot_precision_recall_curves(y_test, eval_results["y_probs"], class_names=class_names)

    # Permutation feature importance
    try:
        imp_df = compute_permutation_feature_importance(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=prep.feature_columns,
            top_n=15
        )
        evaluator.plot_feature_importance(imp_df)
    except Exception as e:
        print(f"[Warning] Permutation importance skipped due to: {e}")

    # Re-generate docx report with embedded results
    print("\n[FINAL STEP] Updating Academic DOCX Report with Fresh Results & Figures...")
    create_academic_docx_report()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Output Artifacts:")
    print("  - EDA Figures: results/eda/")
    print("  - Statistical Summary Tables: results/tables/")
    print("  - Evaluation Figures & Metrics: results/model_evaluation/")
    print("  - Academic Word Report: results/report/project_report.docx")
    print("=" * 80)


if __name__ == "__main__":
    main()
