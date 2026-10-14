"""Bootstrap uncertainty for the two contrasting Model-2 classifiers.

Run from repository root, after preprocessing.py and train_models.py:
    python src/bootstrap_analysis.py

Reads:
    data/processed/prepared_data.joblib
    models/lr_model2.joblib
    models/xgb_model2.joblib

Writes:
    results/tables/bootstrap_confidence_intervals.csv
    results/tables/bootstrap_point_estimates.csv

Method described in the accepted paper:
  * Hold out the same test set; do not refit the models.
  * Take 1,000 nonparametric resamples of test observations, WITH replacement,
    at the original test-set size.
  * Preserve each observation's true label, predicted class and predicted
    probability vector together (sample the same row indices for each).
  * Report 2.5th and 97.5th percentiles of the metric distributions as 95% CIs.

No formal population/survey-design variance inference is implied. All results
are primary unweighted test-set results. Model probabilities are aligned to the
class convention: 0 = no diabetes, 1 = prediabetes, 2 = diabetes.

Organized from the final paper and recovered analysis; syntax-checked only.
Not executed against the BRFSS raw data or compared to publication results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CLASSES = np.array([0, 1, 2], dtype=int)
MODELS = {"lr": "Logistic Regression", "xgb": "XGBoost"}
METRICS = ("weighted_f1", "macro_f1", "balanced_accuracy", "macro_roc_auc")
DEFAULT_SEED = 42
EXPECTED_TEST_N = 90_649


def probability_matrix(model, X) -> np.ndarray:
    """Reorder probability columns into the study's (0, 1, 2) class order."""
    raw = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = np.asarray(model.classes_, dtype=int)
    if set(model_classes.tolist()) != set(CLASSES.tolist()):
        raise ValueError(f"Unexpected classes: {model_classes}")
    indices = [int(np.where(model_classes == cls)[0][0]) for cls in CLASSES]
    proba = raw[:, indices]
    if not np.all(np.isfinite(proba)) or np.any(proba < 0):
        raise ValueError("Model returned invalid probabilities")
    if not np.allclose(proba.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probability rows must sum to 1")
    return proba


def calculate_metrics(y: np.ndarray, pred: np.ndarray,
                      proba: np.ndarray) -> dict[str, float]:
    """Calculate the four metrics reported with bootstrap intervals in Results."""
    if len(np.unique(y)) != 3:
        raise ValueError("A bootstrap resample must contain all three classes")
    return {
        "weighted_f1": float(f1_score(y, pred, labels=CLASSES,
                                      average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES,
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_roc_auc": float(roc_auc_score(y, proba, labels=CLASSES,
                                              multi_class="ovr", average="macro")),
    }


def run(prepared_path: Path, model_dir: Path, out_dir: Path,
        resamples: int = 1000, seed: int = DEFAULT_SEED,
        validate: bool = True) -> None:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not prepared_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing output: {prepared_path}")
    prepared = joblib.load(prepared_path)  # Only load locally trusted joblib files.
    for key in ("y_test", "X2_test"):
        if key not in prepared:
            raise KeyError(f"Missing prepared data key: {key}")
    y = np.asarray(prepared["y_test"], dtype=int)
    X = prepared["X2_test"]
    n = len(y)
    if X.shape[0] != n:
        raise ValueError("Test feature/label row counts do not match")
    if validate and n != EXPECTED_TEST_N:
        raise ValueError(f"Test size {n} differs from the final study {EXPECTED_TEST_N}")
    if set(np.unique(y).tolist()) != set(CLASSES.tolist()):
        raise ValueError("Test set does not contain expected labels 0, 1, 2")

    # Draw identical respondent positions for both models (paired test samples).
    # Reusing resample positions is not itself a paired-difference significance test.
    rng = np.random.default_rng(seed)
    sampled_indices = [rng.integers(0, n, size=n) for _ in range(resamples)]
    point_rows: list[dict] = []
    ci_rows: list[dict] = []

    for model_key, model_name in MODELS.items():
        model_file = model_dir / f"{model_key}_model2.joblib"
        if not model_file.is_file():
            raise FileNotFoundError(f"Missing trained model: {model_file}")
        model = joblib.load(model_file)
        predictions = np.asarray(model.predict(X), dtype=int)
        probabilities = probability_matrix(model, X)
        if len(predictions) != n or probabilities.shape != (n, 3):
            raise ValueError(f"Unexpected outputs for {model_name}")

        point = calculate_metrics(y, predictions, probabilities)
        for metric in METRICS:
            point_rows.append({"model": model_name, "specification": "Model-2",
                               "metric": metric, "estimate": point[metric],
                               "n_test": n})

        distributions = {metric: np.empty(resamples, dtype=float)
                         for metric in METRICS}
        for iteration, positions in enumerate(sampled_indices):
            sample_metrics = calculate_metrics(
                y[positions], predictions[positions], probabilities[positions]
            )
            for metric in METRICS:
                distributions[metric][iteration] = sample_metrics[metric]
            if (iteration + 1) % 100 == 0:
                print(f"{model_name}: {iteration + 1}/{resamples} resamples")

        for metric in METRICS:
            lower, upper = np.percentile(distributions[metric], [2.5, 97.5])
            ci_rows.append({
                "model": model_name,
                "specification": "Model-2",
                "metric": metric,
                "estimate": point[metric],
                "ci_2_5": float(lower),
                "ci_97_5": float(upper),
                "confidence_level": 0.95,
                "resamples": resamples,
                "seed": seed,
                "n_test": n,
                "resampling_unit": "test observation",
                "model_refitted": False,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point_rows).to_csv(
        out_dir / "bootstrap_point_estimates.csv", index=False
    )
    pd.DataFrame(ci_rows).to_csv(
        out_dir / "bootstrap_confidence_intervals.csv", index=False
    )
    print(f"Saved bootstrap result tables to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=Path("data/processed/prepared_data.joblib"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/tables"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    run(args.prepared, args.models, args.out, args.resamples, args.seed,
        validate=not args.skip_validation)


if __name__ == "__main__":
    main()
