"""
evaluate_model.py
Evaluation and diagnostic visualization module for Deep Learning intrusion detection models.
Satisfies Course Outcome CO5 (Application of EDA and Visualization - 16 marks).

Generates 5+ evaluation visualizations:
1. Training and Validation Loss Curves vs Epoch
2. Training and Validation Accuracy Curves vs Epoch
3. Confusion Matrix with Normalized & Count Annotations
4. Receiver Operating Characteristic (ROC) & AUC Curves (Binary & Multiclass OvR)
5. Precision-Recall Curves (PR-AUC / Average Precision) for Imbalanced Assessment
6. Permutation Feature Importance Bar Chart

Computes comprehensive quantitative performance metrics strictly on held-out test sets:
- Accuracy, Precision, Recall, F1-Score (Macro and Weighted)
- Per-class classification reports
- ROC-AUC scores
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from tensorflow import keras

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "font.size": 10})


class ModelEvaluator:
    """
    Evaluator generating publication-grade diagnostics and metrics exclusively on test splits.
    """

    def __init__(self, output_dir: str = "results/model_evaluation"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_training_history(self, history: keras.callbacks.History) -> Tuple[str, str]:
        """
        Plots loss and accuracy trajectories across training epochs.
        """
        hist = history.history
        epochs = range(1, len(hist["loss"]) + 1)

        # 1. Loss Curve
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, hist["loss"], "b-o", label="Training Loss", linewidth=1.8, markersize=4)
        ax.plot(epochs, hist["val_loss"], "r--s", label="Validation Loss", linewidth=1.8, markersize=4)
        ax.set_title("Training vs Validation Loss Trajectory", fontsize=12, weight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss (Cross-Entropy)", fontsize=11)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)
        loss_path = os.path.join(self.output_dir, "01_training_val_loss.png")
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Accuracy Curve
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, hist["accuracy"], "g-o", label="Training Accuracy", linewidth=1.8, markersize=4)
        ax.plot(epochs, hist["val_accuracy"], "m--^", label="Validation Accuracy", linewidth=1.8, markersize=4)
        ax.set_title("Training vs Validation Accuracy Trajectory", fontsize=12, weight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)
        acc_path = os.path.join(self.output_dir, "02_training_val_accuracy.png")
        plt.savefig(acc_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[Evaluation] Saved loss and accuracy curves to: {self.output_dir}")
        return loss_path, acc_path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str]
    ) -> str:
        """
        Plots normalized and annotated confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            ax=ax,
            linewidths=0.5
        )
        ax.set_title("Normalized Confusion Matrix (Held-Out Test Set)", fontsize=13, weight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("Ground Truth Label", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)

        out_path = os.path.join(self.output_dir, "03_confusion_matrix.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Evaluation] Saved confusion matrix to: {out_path}")
        return out_path

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        class_names: List[str]
    ) -> str:
        """
        Plots ROC curves and computes Area Under the Curve (AUC).
        Supports binary and multiclass (One-vs-Rest).
        """
        fig, ax = plt.subplots(figsize=(7.5, 6))
        num_classes = len(class_names)

        if num_classes == 2:
            # Binary ROC
            probs = y_probs.ravel() if y_probs.ndim > 1 and y_probs.shape[1] == 1 else y_probs[:, 1]
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="#d95f02", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        else:
            # Multiclass OvR ROC
            y_true_dummies = pd.get_dummies(y_true).values
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_dummies[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
        ax.legend(loc="lower right", frameon=True, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)

        out_path = os.path.join(self.output_dir, "04_roc_auc_curves.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Evaluation] Saved ROC-AUC curves to: {out_path}")
        return out_path

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        class_names: List[str]
    ) -> str:
        """
        Plots Precision-Recall curve with Average Precision (AP).
        Crucial for imbalanced network intrusion evaluation where accuracy can be deceptively high.
        """
        fig, ax = plt.subplots(figsize=(7.5, 6))
        num_classes = len(class_names)

        if num_classes == 2:
            probs = y_probs.ravel() if y_probs.ndim > 1 and y_probs.shape[1] == 1 else y_probs[:, 1]
            precision, recall, _ = precision_recall_curve(y_true, probs)
            ap = average_precision_score(y_true, probs)
            ax.plot(recall, precision, color="#7570b3", lw=2, label=f"PR Curve (AP = {ap:.4f})")
        else:
            y_true_dummies = pd.get_dummies(y_true).values
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(y_true_dummies[:, i], y_probs[:, i])
                ap = average_precision_score(y_true_dummies[:, i], y_probs[:, i])
                ax.plot(recall, precision, lw=1.5, label=f"{class_names[i]} (AP = {ap:.3f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title("Precision-Recall Curves (Class-Imbalance Diagnostic)", fontsize=13, weight="bold")
        ax.set_xlabel("Recall (Coverage)", fontsize=11)
        ax.set_ylabel("Precision (Purity)", fontsize=11)
        ax.legend(loc="lower left", frameon=True, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)

        out_path = os.path.join(self.output_dir, "05_precision_recall_curve.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Evaluation] Saved PR curves to: {out_path}")
        return out_path

    def plot_feature_importance(self, df_importance: pd.DataFrame) -> str:
        """
        Plots Permutation Feature Importance bar chart.
        """
        fig, ax = plt.subplots(figsize=(9, 6))
        palette = sns.color_palette("mako", len(df_importance))
        bars = ax.barh(df_importance["Feature"], df_importance["Importance (F1 Drop)"], color=palette, edgecolor="black", linewidth=0.6)
        ax.invert_yaxis()
        ax.set_title("Permutation Feature Importance on Held-Out Test Set", fontsize=13, weight="bold", pad=12)
        ax.set_xlabel("Drop in Test Macro F1-Score When Feature is Shuffled", fontsize=11)
        ax.set_ylabel("Network Flow Feature", fontsize=11)

        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontsize=9)

        out_path = os.path.join(self.output_dir, "06_feature_importance.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Evaluation] Saved permutation importance to: {out_path}")
        return out_path

    def evaluate_model_performance(
        self,
        model: keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Computes complete evaluation metrics on the held-out test split.
        """
        num_classes = len(class_names)
        is_binary = (num_classes == 2)

        # Generate probabilities and discrete predictions
        y_probs = model.predict(X_test, verbose=0)
        if is_binary:
            y_pred = (y_probs > 0.5).astype(int).ravel()
            roc_auc = float(roc_auc_score(y_test, y_probs.ravel()))
        else:
            y_pred = np.argmax(y_probs, axis=1)
            try:
                roc_auc = float(roc_auc_score(pd.get_dummies(y_test), y_probs, multi_class="ovr", average="macro"))
            except Exception:
                roc_auc = 0.0

        # Classification metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

        metrics = {
            "test_accuracy": round(acc, 4),
            "test_precision_macro": round(prec_macro, 4),
            "test_recall_macro": round(rec_macro, 4),
            "test_f1_macro": round(f1_macro, 4),
            "test_precision_weighted": round(prec_weighted, 4),
            "test_recall_weighted": round(rec_weighted, 4),
            "test_f1_weighted": round(f1_weighted, 4),
            "test_roc_auc": round(roc_auc, 4),
            "classification_report": report_dict,
            "classification_report_text": report_text
        }

        # Save metrics JSON and Report Text
        json_path = os.path.join(self.output_dir, "test_metrics_summary.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        txt_path = os.path.join(self.output_dir, "classification_report.txt")
        with open(txt_path, "w") as f:
            f.write(report_text)

        print(f"[Evaluation] Saved test metrics to {json_path}")
        print("\n=== CLASSIFICATION REPORT (HELD-OUT TEST SET) ===")
        print(report_text)

        return {
            "metrics": metrics,
            "y_pred": y_pred,
            "y_probs": y_probs
        }
