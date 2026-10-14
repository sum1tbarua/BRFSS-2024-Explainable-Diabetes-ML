"""Probability calibration for the final BRFSS 2024 Model-2 classifiers.

Run from the repository root after preprocessing and model training:
    python src/calibration.py

Inputs:
    data/processed/prepared_data.joblib
    models/lr_model2.joblib
    models/xgb_model2.joblib

Output:
    results/tables/calibration_metrics.csv

This module follows the final research analysis: unweighted multiclass Brier
score (sum of three squared probability errors, averaged over respondents),
multiclass log loss, and 10-bin top-label expected calibration error (ECE-10).
It also provides the optional class-specific Brier scores used in the analysis.

No model fitting, recalibration, or test-set threshold selection is performed.
Results are NOT design-based BRFSS population estimates.

This is an organized archival code file; syntax-checked only. It has not
been run against the original BRFSS data or verified against the paper's
numerical results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

CLASSES = np.array([0, 1, 2], dtype=int)
MODEL_NAMES = {"lr": "Logistic Regression Model-2", "xgb": "XGBoost Model-2"}
EXPECTED_TEST_N = 90_649
EXPECTED_TEST_COUNTS = {0: 75_226, 1: 2_261, 2: 13_162}


def aligned_probabilities(model, X_test) -> np.ndarray:
    """Return predict_proba in label order 0, 1, 2."""
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("Fitted model must provide predict_proba and classes_")
    raw = np.asarray(model.predict_proba(X_test), dtype=float)
    actual_labels = np.asarray(model.classes_, dtype=int)
    if raw.ndim != 2 or raw.shape[1] != len(actual_labels):
        raise ValueError("Invalid model probability dimensions")
    if set(actual_labels.tolist()) != set(CLASSES.tolist()):
        raise ValueError(f"Unexpected class labels: {actual_labels}")
    indices = [int(np.flatnonzero(actual_labels == c)[0]) for c in CLASSES]
    probabilities = raw[:, indices]
    if (not np.all(np.isfinite(probabilities)) or
            np.any(probabilities < 0) or
            not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid model probability outputs")
    return probabilities


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean sum of squared deviations from three-class one-hot labels."""
    one_hot = np.eye(len(CLASSES), dtype=float)[y_true]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def top_label_ece(y_true: np.ndarray, proba: np.ndarray,
                  n_bins: int = 10) -> float:
    """Original analysis ECE: accuracy versus top-class confidence in 10 bins.

    Uses equal-width confidence bins [0, 0.1), ..., [0.9, 1.0].
    This is a top-label ECE, not a class-wise calibration error.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    predictions = CLASSES[np.argmax(proba, axis=1)]
    confidences = np.max(proba, axis=1)
    correct = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_index in range(n_bins):
        low, high = edges[bin_index:bin_index + 2]
        if bin_index == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(
            float(np.mean(correct[mask])) - float(np.mean(confidences[mask]))
        )
    return float(ece)


def calculate_calibration(y_true: np.ndarray, proba: np.ndarray,
                          model_name: str) -> dict:
    """Calculate metrics using the same definitions as the final rerun."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if proba.shape != (len(y_true), 3):
        raise ValueError("Expected an n-by-3 probability matrix")
    if not np.all(np.isin(y_true, CLASSES)):
        raise ValueError("Unexpected labels: expected 0, 1, 2")
    result = {
        "model": model_name,
        "n_test": len(y_true),
        "log_loss": float(log_loss(y_true, proba, labels=CLASSES)),
        "multiclass_brier": multiclass_brier(y_true, proba),
        "ece_10": top_label_ece(y_true, proba, n_bins=10),
    }
    for cls, label in [(0, "no_diabetes"), (1, "prediabetes"), (2, "diabetes")]:
        result[f"brier_{label}"] = float(
            brier_score_loss((y_true == cls).astype(int), proba[:, cls])
        )
    return result


def run(prepared_path: Path, model_dir: Path, out_dir: Path,
        validate: bool = True) -> pd.DataFrame:
    if not prepared_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing output: {prepared_path}")
    # Only load locally trusted joblib/pickle artifacts.
    prepared = joblib.load(prepared_path)
    for field in ("y_test", "X2_test"):
        if field not in prepared:
            raise KeyError(f"Missing prepared-data field: {field}")
    y_true = np.asarray(prepared["y_test"], dtype=int)
    X_test = prepared["X2_test"]
    if X_test.shape[0] != len(y_true):
        raise ValueError("Test labels and Model-2 feature matrix differ in length")
    counts = {int(c): int(n) for c, n in zip(*np.unique(y_true, return_counts=True))}
    if validate and (len(y_true) != EXPECTED_TEST_N or counts != EXPECTED_TEST_COUNTS):
        raise ValueError(f"Held-out sample differs from final study: n={len(y_true)}, counts={counts}")

    rows = []
    for prefix, name in MODEL_NAMES.items():
        model_path = model_dir / f"{prefix}_model2.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing fitted Model-2 classifier: {model_path}")
        model = joblib.load(model_path)
        proba = aligned_probabilities(model, X_test)
        row = calculate_calibration(y_true, proba, name)
        rows.append(row)
        print(f"{name}: Brier={row['multiclass_brier']:.4f}, "
              f"log loss={row['log_loss']:.4f}, ECE-10={row['ece_10']:.4f}")

    results = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "calibration_metrics.csv"
    results.to_csv(destination, index=False)
    print(f"Saved: {destination}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=Path("data/processed/prepared_data.joblib"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/tables"))
    parser.add_argument("--skip-validation", action="store_true",
                        help="Disable checks for the final-study test count/distribution")
    args = parser.parse_args()
    run(args.prepared, args.models, args.out, validate=not args.skip_validation)


if __name__ == "__main__":
    main()
