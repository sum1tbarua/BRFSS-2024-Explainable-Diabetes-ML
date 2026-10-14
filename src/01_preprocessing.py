"""Reproduce the leakage-safe BRFSS 2024 preprocessing used in the final study.

Run from repository root:
    python src/01_preprocessing.py --raw data/raw/CDC_BRFSS2024.XPT

The official raw XPT file is not included. This script does not train a classifier.
It preserves the final analysis' ordering: deterministic recoding, a single
stratified split, then train-fitted imputation/standardization/one-hot encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "DIABETE4"
CLASS_COLUMN = "target_3class"
BEHAVIORAL = [
    "EXERANY2", "_PHYS14D", "SMOKE100", "SMOKDAY2", "_SMOKER3",
    "_RFSMOK3", "ALCDAY4", "AVEDRNK4", "DRNK3GE5", "DRNKANY6",
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
]
ANTHROPOMETRIC = ["_BMI5", "_BMI5CAT", "_RFBMI5", "WEIGHT2"]
DEMOGRAPHIC = ["SEXVAR", "_AGE80", "_RACEGR3", "EDUCA", "INCOME3"]
MODEL1_FEATURES = BEHAVIORAL
MODEL2_FEATURES = BEHAVIORAL + ANTHROPOMETRIC + DEMOGRAPHIC
META_COLUMNS = ["SEQNO", "_STATE", "_LLCPWT"]
NUMERIC_FEATURES = [
    "_BMI5", "WEIGHT2", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "_PHYS14D", "ALCDAY4", "AVEDRNK4", "DRNK3GE5",
]
SPECIAL_MISSING = {
    "EXERANY2": [7, 9], "_PHYS14D": [9], "SMOKE100": [7, 9],
    "SMOKDAY2": [7, 9], "_SMOKER3": [9], "_RFSMOK3": [9],
    "DRNKANY6": [7, 9], "GENHLTH": [7, 9], "_RFBMI5": [9],
    "_RACEGR3": [9], "EDUCA": [9], "INCOME3": [77, 99],
}
EXPECTED_COUNTS = {0: 376125, 1: 11307, 2: 65809}
EXPECTED_TRAIN = 362592
EXPECTED_TEST = 90649
EXPECTED_MODEL2_ENCODED = 122


def load_and_recode(raw_path: Path) -> pd.DataFrame:
    """Apply the final study's deterministic eligibility and recoding rules."""
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw CDC SAS XPT file not found: {raw_path}")
    raw = pd.read_sas(raw_path, format="xport")
    required = list(dict.fromkeys([TARGET] + MODEL2_FEATURES + META_COLUMNS))
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required BRFSS columns: {missing}")

    df = raw[required].copy()
    df = df[df[TARGET].isin([1, 3, 4])].copy()
    # Preserve the original final notebook's duplicate-record treatment.
    df = df.drop_duplicates()
    df[CLASS_COLUMN] = df[TARGET].map({3: 0, 4: 1, 1: 2}).astype(int)

    for col in MODEL2_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, codes in SPECIAL_MISSING.items():
        df.loc[df[col].isin(codes), col] = np.nan

    for col in ("PHYSHLTH", "MENTHLTH", "POORHLTH"):
        df.loc[df[col] == 88, col] = 0
        df.loc[df[col].isin([77, 99]), col] = np.nan

    structural_zero = (
        df["POORHLTH"].isna()
        & (df["PHYSHLTH"] == 0)
        & (df["MENTHLTH"] == 0)
    )
    df.loc[structural_zero, "POORHLTH"] = 0

    df.loc[df["ALCDAY4"] == 888, "ALCDAY4"] = 0
    df.loc[df["ALCDAY4"].isin([777, 999]), "ALCDAY4"] = np.nan
    df.loc[df["AVEDRNK4"] == 88, "AVEDRNK4"] = 0
    df.loc[df["AVEDRNK4"].isin([77, 99]), "AVEDRNK4"] = np.nan
    df.loc[df["DRNK3GE5"] == 88, "DRNK3GE5"] = 0
    df.loc[df["DRNK3GE5"].isin([77, 99]), "DRNK3GE5"] = np.nan
    no_drink = df["DRNKANY6"] == 2
    for col in ("ALCDAY4", "AVEDRNK4", "DRNK3GE5"):
        df.loc[no_drink, col] = 0

    df.loc[df["_BMI5"] == 9999, "_BMI5"] = np.nan
    df.loc[df["WEIGHT2"].isin([7777, 9999]), "WEIGHT2"] = np.nan
    # Final notebook: valid metric-coded entries 9023..9352; convert kg to lb.
    metric_weight = df["WEIGHT2"].between(9023, 9352)
    df.loc[metric_weight, "WEIGHT2"] = (
        (df.loc[metric_weight, "WEIGHT2"] - 9000) * 2.2046226218
    )
    return df


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    """Build an unfitted preprocessor; fit only on the training partition."""
    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c not in numeric]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def run(raw_path: Path, out_dir: Path, validate: bool = True) -> None:
    df = load_and_recode(raw_path)
    counts = {int(k): int(v) for k, v in df[CLASS_COLUMN].value_counts().sort_index().items()}
    if validate and counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Class counts differ from final study: observed {counts}; "
            f"expected {EXPECTED_COUNTS}. Verify the raw CDC data/version first."
        )

    # Matches final notebook: split row positions, then reset partition indices.
    train_pos, test_pos = train_test_split(
        np.arange(len(df)), test_size=0.20,
        stratify=df[CLASS_COLUMN], random_state=SEED,
    )
    train_df = df.iloc[train_pos].copy().reset_index(drop=True)
    test_df = df.iloc[test_pos].copy().reset_index(drop=True)
    if validate and (len(train_df), len(test_df)) != (EXPECTED_TRAIN, EXPECTED_TEST):
        raise RuntimeError("Train/test partition sizes differ from final study")

    prep1 = build_preprocessor(MODEL1_FEATURES)
    prep2 = build_preprocessor(MODEL2_FEATURES)
    X1_train = prep1.fit_transform(train_df[MODEL1_FEATURES])
    X1_test = prep1.transform(test_df[MODEL1_FEATURES])
    X2_train = prep2.fit_transform(train_df[MODEL2_FEATURES])
    X2_test = prep2.transform(test_df[MODEL2_FEATURES])
    if validate and X2_train.shape[1] != EXPECTED_MODEL2_ENCODED:
        raise RuntimeError(
            f"Model-2 encoded {X2_train.shape[1]} features; expected "
            f"{EXPECTED_MODEL2_ENCODED}. Verify input and categorical mappings."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Single artifact keeps row alignment, metadata and feature names intact.
    joblib.dump({
        "X1_train": X1_train, "X1_test": X1_test,
        "X2_train": X2_train, "X2_test": X2_test,
        "y_train": train_df[CLASS_COLUMN].to_numpy(),
        "y_test": test_df[CLASS_COLUMN].to_numpy(),
        "meta_train": train_df[META_COLUMNS].reset_index(drop=True),
        "meta_test": test_df[META_COLUMNS].reset_index(drop=True),
        "subgroup_train": train_df[DEMOGRAPHIC].reset_index(drop=True),
        "subgroup_test": test_df[DEMOGRAPHIC].reset_index(drop=True),
        "train_positions": train_pos, "test_positions": test_pos,
        "feature_names_1": prep1.get_feature_names_out().tolist(),
        "feature_names_2": prep2.get_feature_names_out().tolist(),
    }, out_dir / "prepared_data.joblib", compress=3)
    joblib.dump(prep1, out_dir / "preprocessor_model1.joblib")
    joblib.dump(prep2, out_dir / "preprocessor_model2.joblib")
    report = {
        "analytic_n": len(df), "class_counts": counts,
        "training_n": len(train_df), "testing_n": len(test_df),
        "model1_original_features": len(MODEL1_FEATURES),
        "model2_original_features": len(MODEL2_FEATURES),
        "model1_encoded_features": X1_train.shape[1],
        "model2_encoded_features": X2_train.shape[1],
        "seed": SEED, "training_only_preprocessing": True,
    }
    (out_dir / "preprocessing_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Preprocessed artifacts saved in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/CDC_BRFSS2024.XPT"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Allow data other than the study's original release (not for reproduction).",
    )
    args = parser.parse_args()
    run(args.raw, args.out, validate=not args.skip_validation)


if __name__ == "__main__":
    main()
