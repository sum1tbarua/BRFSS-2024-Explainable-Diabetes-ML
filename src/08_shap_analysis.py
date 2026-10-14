"""Reconstruct the final diabetes-class SHAP analyses for BRFSS 2024.

Run from repository root after preprocessing.py and train_models.py:
    python src/shap_analysis.py

This script follows the recovered final SHAP notebook (2500 test observations,
seed 42; background sample up to 800, seed 123; XGBoost Models 1 and 2).
It is archival source code: syntax-checked, not run on the original BRFSS data.
SHAP percentages describe model attribution, not causal or population risk.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

SEED = 42
BACKGROUND_SEED = 123
SHAP_N = 2500
BACKGROUND_N = 800
DIABETES_CLASS = 2
BEHAVIORAL = ["EXERANY2", "_PHYS14D", "SMOKE100", "SMOKDAY2", "_SMOKER3", "_RFSMOK3", "ALCDAY4", "AVEDRNK4", "DRNK3GE5", "DRNKANY6", "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH"]
DEMOGRAPHIC = ["SEXVAR", "_AGE80", "_RACEGR3", "EDUCA", "INCOME3"]
ANTHROPOMETRIC = ["_BMI5", "_BMI5CAT", "_RFBMI5", "WEIGHT2"]
GEO_NAMES = {
    1:"Alabama",2:"Alaska",4:"Arizona",5:"Arkansas",6:"California",8:"Colorado",
    9:"Connecticut",10:"Delaware",11:"District of Columbia",12:"Florida",13:"Georgia",
    15:"Hawaii",16:"Idaho",17:"Illinois",18:"Indiana",19:"Iowa",20:"Kansas",
    21:"Kentucky",22:"Louisiana",23:"Maine",24:"Maryland",25:"Massachusetts",
    26:"Michigan",27:"Minnesota",28:"Mississippi",29:"Missouri",30:"Montana",
    31:"Nebraska",32:"Nevada",33:"New Hampshire",34:"New Jersey",35:"New Mexico",
    36:"New York",37:"North Carolina",38:"North Dakota",39:"Ohio",40:"Oklahoma",
    41:"Oregon",42:"Pennsylvania",44:"Rhode Island",45:"South Carolina",
    46:"South Dakota",47:"Tennessee",48:"Texas",49:"Utah",50:"Vermont",
    51:"Virginia",53:"Washington",54:"West Virginia",55:"Wisconsin",56:"Wyoming",
    66:"Guam",72:"Puerto Rico",78:"U.S. Virgin Islands",
}


def feature_domain(name: str) -> str:
    """Map exact ColumnTransformer encoded names to their original domains."""
    for domain, codes in (("Behavioral/health-status", BEHAVIORAL),
                          ("Demographic", DEMOGRAPHIC),
                          ("Anthropometric", ANTHROPOMETRIC)):
        for code in codes:
            if name == f"num__{code}" or name.startswith(f"cat__{code}_"):
                return domain
    raise ValueError(f"Encoded feature cannot be assigned to a domain: {name}")


def class_shap(model, X: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
    """Use final notebook's background-based TreeExplainer (without additivity check)."""
    explainer = shap.TreeExplainer(model, data=background)
    values = np.asarray(explainer(X, check_additivity=False).values)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"Expected multiclass SHAP [N, features, 3], got {values.shape}")
    return values[:, :, DIABETES_CLASS]


def run(prepared: Path, models: Path, out: Path) -> None:
    if not prepared.is_file():
        raise FileNotFoundError(f"Missing prepared data: {prepared}")
    data = joblib.load(prepared)  # Only load trusted artifacts produced locally.
    required = ["y_test", "X1_test", "X2_test", "feature_names_1", "feature_names_2", "meta_test"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing prepared-data key: {key}")
    y_test = np.asarray(data["y_test"])
    rng = np.random.default_rng(SEED)
    positions = rng.choice(len(y_test), size=min(SHAP_N, len(y_test)), replace=False)
    matrices = {}
    for idx in (1, 2):
        names = list(data[f"feature_names_{idx}"])
        x_arr = data[f"X{idx}_test"][positions]
        if hasattr(x_arr, "toarray"):
            x_arr = x_arr.toarray()
        matrices[idx] = pd.DataFrame(np.asarray(x_arr), columns=names)
    if len(matrices[2].columns) != 122:
        raise ValueError(f"Expected 122 encoded Model-2 features, found {len(matrices[2].columns)}")
    values = {}
    importance = {}
    for idx in (1, 2):
        path = models / f"xgb_model{idx}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fitted XGBoost model: {path}")
        model = joblib.load(path)
        evaluation = matrices[idx]
        background = evaluation.sample(n=min(BACKGROUND_N, len(evaluation)), random_state=BACKGROUND_SEED)
        values[idx] = class_shap(model, evaluation, background)
        importance[idx] = pd.Series(
            np.abs(values[idx]).mean(axis=0), index=evaluation.columns, name="MeanAbsSHAP"
        ).sort_values(ascending=False)
    out.mkdir(parents=True, exist_ok=True)
    for idx in (1, 2):
        importance[idx].rename_axis("Feature").reset_index().to_csv(
            out / f"SHAP_global_model{idx}_diabetes.csv", index=False
        )
    mapping = pd.DataFrame({"Feature": list(matrices[2].columns),
                            "MeanAbsSHAP": np.abs(values[2]).mean(axis=0)})
    mapping["Domain"] = mapping["Feature"].map(feature_domain)
    summary = mapping.groupby("Domain", as_index=False).agg(
        SummedMeanAbsSHAP=("MeanAbsSHAP", "sum"), EncodedFeatures=("Feature", "size")
    )
    summary["Percent"] = 100 * summary["SummedMeanAbsSHAP"] / summary["SummedMeanAbsSHAP"].sum()
    summary.sort_values("Percent", ascending=False, inplace=True)
    mapping.to_csv(out / "SHAP_feature_domain_mapping.csv", index=False)
    summary.to_csv(out / "SHAP_domain_summary.csv", index=False)
    topk = []
    for k in (5, 10, 20, 30):
        overlap = len(set(importance[1].head(k).index) & set(importance[2].head(k).index))
        topk.append({"k": k, "OverlapCount": overlap, "OverlapPercent": 100 * overlap / k})
    pd.DataFrame(topk).to_csv(out / "SHAP_topk_overlap.csv", index=False)
    geo_codes = pd.to_numeric(data["meta_test"]["_STATE"], errors="coerce").iloc[positions].reset_index(drop=True)
    geography = pd.DataFrame({"Jurisdiction": geo_codes.map(GEO_NAMES),
                              "TotalAbsSHAP": np.abs(values[2]).sum(axis=1)})
    geo_summary = (geography.dropna(subset=["Jurisdiction"])
                   .groupby("Jurisdiction", as_index=False)
                   .agg(N=("TotalAbsSHAP", "size"), MeanTotalAbsSHAP=("TotalAbsSHAP", "mean"))
                   .sort_values("MeanTotalAbsSHAP", ascending=False))
    geo_summary.to_csv(out / "SHAP_geographic_summary.csv", index=False)
    np.savez_compressed(out / "final_SHAP_diabetes_values.npz",
                        shap_model1=values[1], shap_model2=values[2],
                        shap_positions=positions,
                        feature_names_1=np.asarray(matrices[1].columns, dtype=str),
                        feature_names_2=np.asarray(matrices[2].columns, dtype=str))
    print("SHAP observations:", len(positions))
    print("Domain attribution (%):\n", summary.to_string(index=False))
    print("Top 5 jurisdictions:\n", geo_summary.head().to_string(index=False))
    print("Outputs saved to", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/prepared_data.joblib"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    run(args.prepared, args.models, args.out)


if __name__ == "__main__":
    main()
