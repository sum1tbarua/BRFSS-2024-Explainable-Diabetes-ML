"""Held-out evaluation of eight BRFSS 2024 multiclass models.

Run from the repository root after preprocessing.py and train_models.py:
    python src/evaluate_models.py

Reads:
    data/processed/prepared_data.joblib
    models/{lr,dt,rf,xgb}_model{1,2}.joblib

Writes CSV tables to results/tables/:
    overall_model_performance.csv
    class_specific_performance.csv
    confusion_matrices.csv
    test_class_distribution.csv

Implements the primary, *unweighted* held-out assessment in the accepted
manuscript. Calibration, bootstrap intervals, and sampling-weight sensitivity
are intentionally handled by separate modules. Models and preprocessors are
never fitted here. No thresholds are adjusted using the held-out labels.

This file was syntax-checked but has NOT been run against the original data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASSES = np.array([0, 1, 2], dtype=int)
CLASS_NAMES = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}
ALGORITHMS = {"lr": "Logistic Regression", "dt": "Decision Tree",
              "rf": "Random Forest", "xgb": "XGBoost"}
FEATURE_SETS = {"model1": "X1_test", "model2": "X2_test"}
EXPECTED_TEST_N = 90_649
EXPECTED_TEST_COUNTS = {0: 75_226, 1: 2_261, 2: 13_162}


def aligned_probabilities(model, x_test) -> np.ndarray:
    """Align model predict_proba columns with the paper's labels 0, 1, 2."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} has no predict_proba method")
    raw = np.asarray(model.predict_proba(x_test), dtype=float)
    observed_classes = np.asarray(model.classes_).astype(int)
    if raw.ndim != 2 or raw.shape[1] != len(observed_classes):
        raise ValueError("Unexpected probability-array shape")
    if set(observed_classes.tolist()) != set(CLASSES.tolist()):
        raise ValueError(f"Model class labels mismatch: {observed_classes}")
    positions = [int(np.flatnonzero(observed_classes == c)[0]) for c in CLASSES]
    proba = raw[:, positions]
    if not np.all(np.isfinite(proba)):
        raise ValueError("Probability outputs contain non-finite values")
    if np.any(proba < 0) or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Invalid class probability distribution")
    return proba


def evaluate_one(y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: np.ndarray, algorithm: str,
                 feature_set: str) -> tuple[dict, list[dict], list[dict]]:
    """Compute the overall, class-specific and confusion-matrix results."""
    if len(y_true) != len(y_pred) or len(y_true) != len(y_proba):
        raise ValueError("Inconsistent lengths of labels, predictions and probabilities")
    if not set(np.unique(y_pred)).issubset(set(CLASSES)):
        raise ValueError("Predictions include an unexpected class")

    info = {"algorithm": algorithm, "specification": feature_set,
            "n_test": int(len(y_true))}
    overall = {
        **info,
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_precision": precision_score(y_true, y_pred, labels=CLASSES,
                                                average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, labels=CLASSES,
                                          average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=CLASSES,
                                 average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES,
                              average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        # One-vs-rest unweighted macro-area as reported in the study.
        "macro_roc_auc": roc_auc_score(y_true, y_proba, labels=CLASSES,
                                        multi_class="ovr", average="macro"),
        "log_loss": log_loss(y_true, y_proba, labels=CLASSES),
    }
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    class_rows = [
        {**info, "class_id": int(cls), "class_name": CLASS_NAMES[int(cls)],
         "precision": float(precisions[i]), "recall": float(recalls[i]),
         "f1": float(f1s[i]), "support": int(supports[i])}
        for i, cls in enumerate(CLASSES)
    ]
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    confusion_rows = [
        {**info, "actual_class": CLASS_NAMES[int(actual)],
         "predicted_class": CLASS_NAMES[int(predicted)], "count": int(cm[i, j])}
        for i, actual in enumerate(CLASSES)
        for j, predicted in enumerate(CLASSES)
    ]
    return overall, class_rows, confusion_rows


def run(prepared_path: Path, model_dir: Path, out_dir: Path,
        validate: bool = True) -> None:
    if not prepared_path.is_file():
        raise FileNotFoundError(f"Missing prepared data: {prepared_path}")
    # joblib/pickle content must come exclusively from a trusted local source.
    prepared = joblib.load(prepared_path)
    for key in ("y_test", *FEATURE_SETS.values()):
        if key not in prepared:
            raise KeyError(f"Missing key in prepared data: {key}")
    y_true = np.asarray(prepared["y_test"], dtype=int)
    class_counts = {int(c): int(n) for c, n in zip(*np.unique(y_true, return_counts=True))}
    if validate and (len(y_true) != EXPECTED_TEST_N or
                     class_counts != EXPECTED_TEST_COUNTS):
        raise ValueError(f"Test sample differs from study: n={len(y_true)}, "
                         f"class counts={class_counts}")

    overall_rows, class_rows, confusion_rows = [], [], []
    for feature_set, x_key in FEATURE_SETS.items():
        X_test = prepared[x_key]
        if X_test.shape[0] != len(y_true):
            raise ValueError(f"Test matrix row count mismatch: {x_key}")
        for prefix, display_name in ALGORITHMS.items():
            path = model_dir / f"{prefix}_{feature_set}.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"Missing fitted model: {path}")
            model = joblib.load(path)
            y_pred = np.asarray(model.predict(X_test), dtype=int)
            y_proba = aligned_probabilities(model, X_test)
            overall, by_class, confusion = evaluate_one(
                y_true, y_pred, y_proba, display_name,
                "Model-1" if feature_set == "model1" else "Model-2"
            )
            overall_rows.append(overall)
            class_rows.extend(by_class)
            confusion_rows.extend(confusion)
            print(f"{display_name:<20} {feature_set:<7} "
                  f"macro-F1={overall['macro_f1']:.4f} "
                  f"balanced-accuracy={overall['balanced_accuracy']:.4f} "
                  f"weighted-F1={overall['weighted_f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall_rows).to_csv(out_dir / "overall_model_performance.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out_dir / "class_specific_performance.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame([{"class_id": int(k), "class_name": CLASS_NAMES[k], "count": v}
                  for k, v in sorted(class_counts.items())]).to_csv(
        out_dir / "test_class_distribution.csv", index=False
    )
    print(f"Saved four evaluation tables to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=Path("data/processed/prepared_data.joblib"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/tables"))
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip exact final-study test-sample count checks")
    args = parser.parse_args()
    run(args.prepared, args.models, args.out, validate=not args.skip_validation)


if __name__ == "__main__":
    main()
