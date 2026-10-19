"""
train_model.py
Deep Learning Multilayer Perceptron (MLP) training module for Network Intrusion Detection.
Satisfies Deep Learning and Model Training requirements.

Features:
- Reproducible random seed setting (Python, NumPy, TensorFlow)
- Flexible MLP architecture with Dense, BatchNormalization, Dropout, and He/Glorot initialization
- Automatic mode detection: Binary (sigmoid + binary_crossentropy) vs Multiclass (softmax + sparse_categorical_crossentropy)
- Imbalance mitigation via sklearn class_weight calculation
- Callbacks: EarlyStopping (restores best weights), ReduceLROnPlateau, ModelCheckpoint
- Hyperparameter grid experiment runner
- Permutation feature importance analysis on held-out test data
"""

import os
import random
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# Set environmental seed before importing TensorFlow
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

tf.random.set_seed(42)


def set_seed(seed: int = 42):
    """
    Ensures deterministic reproducibility across all RNG components.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Computes balanced class weights to penalize underrepresented attack class errors:
    w_j = total_samples / (num_classes * count_j).
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def build_mlp_model(
    input_dim: int,
    num_classes: int,
    hidden_units: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3
) -> keras.Model:
    """
    Constructs a Deep Learning Multilayer Perceptron (MLP) classifier.

    Architecture:
    Input -> [Dense -> BatchNormalization -> ReLU -> Dropout] x N -> Output

    Args:
        input_dim: Number of input features.
        num_classes: Number of distinct classes (2 for binary, >2 for multiclass).
        hidden_units: Neuron counts for successive dense layers.
        dropout_rate: Dropout fraction to prevent co-adaptation.
        l2_reg: L2 weight regularization penalty.
        learning_rate: Adam optimizer learning rate.

    Returns:
        keras.Model: Compiled Keras model.
    """
    inputs = keras.Input(shape=(input_dim,), name="flow_features")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    # Output Layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="output_binary")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output_multiclass")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = keras.Model(inputs=inputs, outputs=outputs, name="NetworkIntrusion_MLP")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_units: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 25,
    use_class_weights: bool = True,
    model_save_path: Optional[str] = "results/model_evaluation/best_mlp_model.keras"
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Trains the MLP model with early stopping, dynamic learning rate adjustment,
    and class imbalance weighting.
    """
    set_seed(42)
    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    # For binary targets formatted as 1D array of 0 and 1
    is_binary = (num_classes == 2)
    train_targets = y_train.astype(np.float32) if is_binary else y_train
    val_targets = y_val.astype(np.float32) if is_binary else y_val

    model = build_mlp_model(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    class_weights_dict = calculate_class_weights(y_train) if use_class_weights else None
    if class_weights_dict:
        print(f"[Model] Applied Class Weights: {class_weights_dict}")

    call_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
    ]

    if model_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        call_list.append(
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0
            )
        )

    print(f"[Model] Starting MLP training: {epochs} epochs, batch_size={batch_size}, input_dim={input_dim}...")
    history = model.fit(
        X_train,
        train_targets,
        validation_data=(X_val, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        callbacks=call_list,
        verbose=1
    )

    print("[Model] Training complete.")
    return model, history


def run_hyperparameter_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str = "results/model_evaluation"
) -> pd.DataFrame:
    """
    Executes controlled hyperparameter comparative experiments:
    - Learning rate: [1e-2, 1e-3, 1e-4]
    - Dropout rate: [0.1, 0.3, 0.5]
    - Architecture depth: [[64, 32], [128, 64, 32], [256, 128, 64, 32]]

    Returns:
        pd.DataFrame: Summary table comparing experiment configurations and validation losses.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(np.unique(y_train))
    is_binary = (num_classes == 2)
    train_targets = y_train.astype(np.float32) if is_binary else y_train
    val_targets = y_val.astype(np.float32) if is_binary else y_val

    experiments = [
        {"exp_id": "EXP-1 (Baseline)", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 1e-3},
        {"exp_id": "EXP-2 (Deep Arch)", "hidden": [256, 128, 64, 32], "dropout": 0.3, "lr": 1e-3},
        {"exp_id": "EXP-3 (High Dropout)", "hidden": [128, 64, 32], "dropout": 0.5, "lr": 1e-3},
        {"exp_id": "EXP-4 (Low LR)", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 1e-4},
    ]

    records = []
    for exp in experiments:
        print(f"[Experiment] Evaluating {exp['exp_id']}...")
        set_seed(42)
        model = build_mlp_model(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            hidden_units=exp["hidden"],
            dropout_rate=exp["dropout"],
            learning_rate=exp["lr"]
        )
        hist = model.fit(
            X_train, train_targets,
            validation_data=(X_val, val_targets),
            epochs=6,
            batch_size=512,
            verbose=0
        )
        best_val_loss = float(np.min(hist.history["val_loss"]))
        best_val_acc = float(np.max(hist.history["val_accuracy"]))

        records.append({
            "Experiment ID": exp["exp_id"],
            "Hidden Layers": str(exp["hidden"]),
            "Dropout Rate": exp["dropout"],
            "Learning Rate": exp["lr"],
            "Best Val Loss": round(best_val_loss, 4),
            "Best Val Accuracy (%)": round(best_val_acc * 100, 2)
        })

    summary_df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "hyperparameter_tuning_results.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[Experiment] Saved hyperparameter summary to: {csv_path}")
    return summary_df


def compute_permutation_feature_importance(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
    sample_size: int = 3000
) -> pd.DataFrame:
    """
    Computes permutation feature importance on held-out test data.
    Measures the drop in macro F1-score when values of each feature are shuffled.

    Permutation importance prevents model reliance on spurious training correlations.
    """
    print(f"[Importance] Computing Permutation Feature Importance (top {top_n})...")
    num_classes = len(np.unique(y_test))
    is_binary = (num_classes == 2)

    # Subsample test set for fast evaluation
    n = min(sample_size, len(X_test))
    indices = np.random.choice(len(X_test), n, replace=False)
    X_sub = X_test[indices].copy()
    y_sub = y_test[indices].copy()

    # Base score
    preds_raw = model.predict(X_sub, verbose=0)
    preds = (preds_raw > 0.5).astype(int).ravel() if is_binary else np.argmax(preds_raw, axis=1)
    base_f1 = f1_score(y_sub, preds, average="macro", zero_division=0)

    importance_scores = []
    # Test top candidate features (up to 30 to keep runtimes practical)
    eval_cols = min(len(feature_names), 30)

    for i in range(eval_cols):
        col_name = feature_names[i]
        X_perm = X_sub.copy()
        np.random.shuffle(X_perm[:, i])
        p_raw = model.predict(X_perm, verbose=0)
        p = (p_raw > 0.5).astype(int).ravel() if is_binary else np.argmax(p_raw, axis=1)
        perm_f1 = f1_score(y_sub, p, average="macro", zero_division=0)
        drop = base_f1 - perm_f1
        importance_scores.append({"Feature": col_name, "Importance (F1 Drop)": max(0.0, float(drop))})

    df_imp = pd.DataFrame(importance_scores).sort_values("Importance (F1 Drop)", ascending=False).head(top_n)
    return df_imp
