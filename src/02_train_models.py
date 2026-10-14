"""Train the eight fixed classifiers reported in the BRFSS 2024 study.

Run from the repository root, after preprocessing.py:
    python src/02_train_models.py

Inputs (created by preprocessing.py):
    data/processed/prepared_data.joblib

Outputs:
    models/{lr,dt,rf,xgb}_model{1,2}.joblib
    models/training_manifest.json

No test-set information is used to fit the models. This script intentionally
performs no hyperparameter search, SMOTE, or post-hoc retuning. Final published
metrics are produced by the separate evaluation scripts.

The implementation was organized from the final research notebook; it has been
syntax-checked but has NOT been executed against the original BRFSS dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SEED = 42
CLASSES = (0, 1, 2)  # No diabetes, prediabetes, diabetes
EXPECTED_TRAIN_N = 362_592
FEATURE_SETS = {"model1": "X1_train", "model2": "X2_train"}


def classifiers() -> dict[str, object]:
    """Fixed configurations used for each of the two feature specifications."""
    return {
        "lr": LogisticRegression(
            max_iter=2000,
            multi_class="ovr",  # Preserves the original scikit-learn 1.6.1 specification.
            class_weight="balanced",
            n_jobs=-1,
        ),
        "dt": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=100,
            class_weight="balanced",
            random_state=SEED,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=50,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
            # No class_weight or sample_weight in primary XGBoost training.
        ),
    }


def run(input_path: Path, model_dir: Path, validate: bool = True) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing {input_path}; the preprocessing output is required."
        )
    data = joblib.load(input_path)  # Load only trusted, locally generated artifacts.
    required = {"y_train", "X1_train", "X2_train"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Missing prepared-data keys: {sorted(missing)}")

    y_train = np.asarray(data["y_train"])
    counts = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    if validate:
        if len(y_train) != EXPECTED_TRAIN_N:
            raise ValueError(f"Unexpected training size: {len(y_train)}")
        if set(counts) != set(CLASSES):
            raise ValueError(f"Unexpected labels: {sorted(counts)}")

    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Fixed comparative BRFSS 2024 model configurations",
        "training_set_n": len(y_train),
        "training_class_counts": counts,
        "seed": SEED,
        "no_test_set_fit": True,
        "no_hyperparameter_optimization": True,
        "versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "models": [],
    }

    for feature_set, matrix_key in FEATURE_SETS.items():
        X_train = data[matrix_key]
        if X_train.shape[0] != len(y_train):
            raise ValueError(f"Row mismatch for {matrix_key}")

        for short_name, estimator in classifiers().items():
            file_name = f"{short_name}_{feature_set}.joblib"
            print(f"Training {short_name.upper()} / {feature_set}: {X_train.shape}")
            estimator.fit(X_train, y_train)
            joblib.dump(estimator, model_dir / file_name)
            manifest["models"].append({
                "name": short_name,
                "feature_set": feature_set,
                "training_matrix": matrix_key,
                "n_training_features": int(X_train.shape[1]),
                "file": file_name,
                "configuration": estimator.get_params(deep=False),
            })

    (model_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(manifest['models'])} models to {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", type=Path, default=Path("data/processed/prepared_data.joblib")
    )
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Permit a different training sample size (not final-study reproduction).",
    )
    args = parser.parse_args()
    run(args.prepared, args.out, validate=not args.skip_validation)


if __name__ == "__main__":
    main()
