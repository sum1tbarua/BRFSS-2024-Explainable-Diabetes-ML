"""Evaluation-only BRFSS sampling-weight sensitivity for Model-2 classifiers.

From repository root, after preprocessing.py and train_models.py:
    python src/survey_weight_analysis.py

Reads data/processed/prepared_data.joblib and models/{lr,xgb}_model2.joblib.
Writes results/tables/survey_weight_sensitivity.csv.

Compares unweighted held-out metrics against metrics computed with BRFSS
_LLCPWT as sample weights. No sample weights are used in training here, no
retraining is performed, and this is NOT a design-aware analysis: it does not
model strata, clusters or design-based variance. Values from these standalone
scripts have NOT been recomputed or confirmed against the accepted article.
Only load joblib files produced locally from trusted data/code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

CLASSES = np.array([0, 1, 2], dtype=int)
MODELS = {"lr": "Logistic Regression Model-2", "xgb": "XGBoost Model-2"}
EXPECTED_TEST_N = 90_649
EXPECTED_TEST_COUNTS = {0: 75_226, 1: 2_261, 2: 13_162}


def aligned_probabilities(model, X_test) -> np.ndarray:
    """Arrange predict_proba columns in class order [0, 1, 2]."""
    if not hasattr(model, "classes_") or not hasattr(model, "predict_proba"):
        raise TypeError("Expected a fitted classifier with classes_ and predict_proba")
    actual_classes = np.asarray(model.classes_, dtype=int)
    if set(actual_classes.tolist()) != set(CLASSES.tolist()):
        raise ValueError(f"Unexpected fitted classes: {actual_classes}")
    raw = np.asarray(model.predict_proba(X_test), dtype=float)
    if raw.shape != (X_test.shape[0], 3):
        raise ValueError(f"Unexpected probability shape: {raw.shape}")
    indices = [int(np.flatnonzero(actual_classes == label)[0]) for label in CLASSES]
    p = raw[:, indices]
    if (not np.all(np.isfinite(p)) or np.any(p < 0) or
            not np.allclose(np.sum(p, axis=1), 1.0, atol=1e-5)):
        raise ValueError("Invalid class probability predictions")
    return p


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      proba: np.ndarray, weights: np.ndarray | None) -> dict:
    """Calculate the five metrics presented in sensitivity Table 5, panel A."""
    options = {"sample_weight": weights}
    return {
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0,
            **options)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=CLASSES, average="macro", zero_division=0,
            **options)),
        "balanced_accuracy": float(balanced_accuracy_score(
            y_true, y_pred, **options)),
        "macro_roc_auc": float(roc_auc_score(
            y_true, proba, labels=CLASSES, multi_class="ovr", average="macro",
            **options)),
        "prediabetes_recall": float(recall_score(
            y_true, y_pred, labels=[1], average=None, zero_division=0,
            **options)[0]),
    }


def run(prepared_path: Path, model_dir: Path, output_dir: Path,
        validate: bool = True) -> pd.DataFrame:
    if not prepared_path.is_file():
        raise FileNotFoundError(f"Missing prepared data: {prepared_path}")
    prepared = joblib.load(prepared_path)
    for key in ("y_test", "X2_test", "meta_test"):
        if key not in prepared:
            raise KeyError(f"Missing prepared-data key: {key}")
    y_test = np.asarray(prepared["y_test"], dtype=int)
    X_test = prepared["X2_test"]
    meta = prepared["meta_test"]
    if "_LLCPWT" not in meta.columns:
        raise KeyError("meta_test must contain BRFSS _LLCPWT values")
    if X_test.shape[0] != len(y_test) or len(meta) != len(y_test):
        raise ValueError("Misaligned test features, labels and BRFSS weights")
    class_counts = dict(zip(*[a.astype(int).tolist() for a in
                              np.unique(y_test, return_counts=True)]))
    if validate and (len(y_test) != EXPECTED_TEST_N or
                     class_counts != EXPECTED_TEST_COUNTS):
        raise ValueError(f"Test sample is not the final study split: {class_counts}")

    weights = pd.to_numeric(meta["_LLCPWT"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Missing, non-finite, zero or negative LLCPWT in test partition")

    rows = []
    for prefix, display_name in MODELS.items():
        model_path = model_dir / f"{prefix}_model2.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing fitted model: {model_path}")
        model = joblib.load(model_path)
        y_pred = np.asarray(model.predict(X_test), dtype=int)
        probabilities = aligned_probabilities(model, X_test)
        for method, sample_weight in (("unweighted", None), ("LLCPWT", weights)):
            metrics = calculate_metrics(y_test, y_pred, probabilities, sample_weight)
            rows.append({"model": display_name, "evaluation": method,
                         "n_test": len(y_test), **metrics})
            print(f"{display_name}, {method}: "
                  f"weighted-F1={metrics['weighted_f1']:.4f}, "
                  f"macro-F1={metrics['macro_f1']:.4f}, "
                  f"balanced-accuracy={metrics['balanced_accuracy']:.4f}")

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "survey_weight_sensitivity.csv"
    result.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=Path("data/processed/prepared_data.joblib"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/tables"))
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip final-paper held-out class-count validation")
    args = parser.parse_args()
    run(args.prepared, args.models, args.out, validate=not args.skip_validation)


if __name__ == "__main__":
    main()
