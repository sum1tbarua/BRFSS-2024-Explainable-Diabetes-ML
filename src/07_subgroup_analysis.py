"""Descriptive subgroup analysis for the four BRFSS 2024 Model-2 classifiers.

From repository root, after preprocessing.py and train_models.py:
    python src/subgroup_analysis.py

Inputs:
    data/processed/prepared_data.joblib
    models/{lr,dt,rf,xgb}_model2.joblib

Outputs:
    results/tables/subgroup_performance.csv
    results/tables/subgroup_performance_gaps.csv

The max-minus-min gaps are descriptive, not inferential fairness estimates.
This script has not been run against the full original BRFSS dataset;
published numbers have not been independently reproduced.
Only load joblib artifacts from trusted sources.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

SEED = 42
CLASSES = [0, 1, 2]
EXPECTED_TEST_N = 90_649
EXPECTED_TEST_COUNTS = {0: 75_226, 1: 2_261, 2: 13_162}
MODELS = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}
DIMENSIONS = ("Sex", "Race/Ethnicity", "Education", "Age")
METRICS = (
    "accuracy", "weighted_f1", "macro_f1", "balanced_accuracy",
    "prediabetes_recall", "diabetes_recall",
)


def age_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    if 18 <= value <= 34:
        return "18-34"
    if 35 <= value <= 49:
        return "35-49"
    if 50 <= value <= 64:
        return "50-64"
    if value >= 65:
        return "65+"
    return np.nan


def label_subgroups(source: pd.DataFrame) -> pd.DataFrame:
    """Map held-out, original BRFSS codes to final notebook subgroup labels."""
    expected = {"SEXVAR", "_RACEGR3", "EDUCA", "_AGE80"}
    if not expected.issubset(source.columns):
        raise KeyError(f"Missing demographic fields: {sorted(expected - set(source.columns))}")
    return pd.DataFrame({
        "Sex": source["SEXVAR"].map({1: "Male", 2: "Female"}),
        "Race/Ethnicity": source["_RACEGR3"].map({
            1: "White", 2: "Black", 3: "Other", 4: "Multiracial",
            5: "Hispanic",
        }),
        "Education": source["EDUCA"].map({
            1: "Never attended/kindergarten",
            2: "Grades 1-8",
            3: "Grades 9-11",
            4: "High school graduate",
            5: "Some college/technical school",
            6: "College/technical graduate",
        }),
        "Age": source["_AGE80"].apply(age_group),
    })


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = recall_score(y_true, y_pred, labels=CLASSES,
                           average=None, zero_division=0)
    # balanced_accuracy_score follows the original notebook's convention:
    # mean recall over classes present in this particular subgroup.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASSES,
                                      average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES,
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "prediabetes_recall": float(recalls[1]),
        "diabetes_recall": float(recalls[2]),
    }


def run(prepared_path: Path, model_dir: Path, output_dir: Path,
        validate: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not prepared_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing output: {prepared_path}")
    data = joblib.load(prepared_path)
    for key in ("y_test", "X2_test", "subgroup_test"):
        if key not in data:
            raise KeyError(f"Missing prepared-data key: {key}")

    y_test = np.asarray(data["y_test"], dtype=int)
    features = data["X2_test"]
    source = data["subgroup_test"].reset_index(drop=True)
    if features.shape[0] != len(y_test) or len(source) != len(y_test):
        raise ValueError("Test features, labels and subgroup metadata are not aligned")
    observed_counts = {int(label): int(count) for label, count in
                       zip(*np.unique(y_test, return_counts=True))}
    if validate and (len(y_test) != EXPECTED_TEST_N or
                     observed_counts != EXPECTED_TEST_COUNTS):
        raise ValueError(f"Held-out partition differs from final study: {observed_counts}")

    groups = label_subgroups(source)
    rows = []
    for prefix, model_name in MODELS.items():
        model_path = model_dir / f"{prefix}_model2.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing trained model: {model_path}")
        fitted = joblib.load(model_path)
        predicted = np.asarray(fitted.predict(features), dtype=int)
        if len(predicted) != len(y_test) or not np.isin(predicted, CLASSES).all():
            raise ValueError(f"Invalid Model-2 predictions from {model_path}")
        for dimension in DIMENSIONS:
            for label, members in groups.groupby(dimension, dropna=True).groups.items():
                position = np.asarray(list(members), dtype=int)
                scores = group_metrics(y_test[position], predicted[position])
                rows.append({
                    "model": model_name,
                    "dimension": dimension,
                    "group": label,
                    "n": int(len(position)),
                    **scores,
                })

    detail = pd.DataFrame(rows)
    gaps = (detail.groupby(["model", "dimension"], sort=False)[list(METRICS)]
            .agg(lambda s: float(s.max() - s.min())).reset_index())
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_file = output_dir / "subgroup_performance.csv"
    gaps_file = output_dir / "subgroup_performance_gaps.csv"
    detail.to_csv(detail_file, index=False)
    gaps.to_csv(gaps_file, index=False)
    print(f"Saved subgroup metrics: {detail_file}")
    print(f"Saved descriptive max-min gaps: {gaps_file}")
    return detail, gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=Path("data/processed/prepared_data.joblib"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/tables"))
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip expected test split size and class counts check")
    args = parser.parse_args()
    run(args.prepared, args.models, args.out, validate=not args.skip_validation)


if __name__ == "__main__":
    main()
