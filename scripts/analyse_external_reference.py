"""Compare out-of-fold latent estimates with the external IS2SITMOGR4 product."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE = ROOT / "data" / "processed" / "external_reference_March2020_transect.csv"
COMBINED = OUT / "external_reference_model_comparison.csv"
METRICS_CSV = OUT / "external_reference_consistency_metrics.csv"
SUMMARY_JSON = OUT / "external_reference_consistency_summary.json"


def load(path: Path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


reference = load(REFERENCE)
nested = load(OUT / "March_8_nested_gp_predictions.csv")
positive_all = load(OUT / "positive_gp_noise_sensitivity_predictions.csv")
positive = positive_all[
    (positive_all["window"] == "March_8")
    & (positive_all["noise_config"] == "is2_heavy")
]
drf = load(OUT / "blocked_validation_drf_predictions.csv")

if not (
    np.array_equal(nested["row_id"], positive["row_id"])
    and np.array_equal(nested["row_id"], drf["row_id"])
):
    raise RuntimeError("Model row identifiers do not align")

model_by_row = {
    int(row_id): index for index, row_id in enumerate(nested["row_id"])
}

methods = {
    "Point + linear": (nested, "point_Hi_m", "point_Hs_m", None, None),
    "Nested linear GP": (
        nested,
        "nested_gp_Hi_m",
        "nested_gp_Hs_m",
        "nested_gp_Hi_std_m",
        "nested_gp_Hs_std_m",
    ),
    "Positive GP": (
        positive,
        "positive_gp_Hi_m",
        "positive_gp_Hs_m",
        "positive_gp_Hi_std_m",
        "positive_gp_Hs_std_m",
    ),
    "Hard-positive DRF": (
        drf,
        "drf_Hi_m",
        "drf_Hs_m",
        "drf_Hi_ensemble_std_m",
        "drf_Hs_ensemble_std_m",
    ),
}


combined_rows = []
for ref in reference:
    row_id = int(ref["row_id"])
    index = model_by_row[row_id]
    row = {
        name: ref[name].item() if hasattr(ref[name], "item") else ref[name]
        for name in reference.dtype.names
    }
    for method, (array, hi_name, hs_name, hi_std_name, hs_std_name) in methods.items():
        prefix = {
            "Point + linear": "point",
            "Nested linear GP": "nested_gp",
            "Positive GP": "positive_gp",
            "Hard-positive DRF": "drf",
        }[method]
        row[f"{prefix}_Hi_m"] = float(array[hi_name][index])
        row[f"{prefix}_Hs_m"] = float(array[hs_name][index])
        row[f"{prefix}_Hi_std_m"] = (
            float(array[hi_std_name][index]) if hi_std_name else ""
        )
        row[f"{prefix}_Hs_std_m"] = (
            float(array[hs_std_name][index]) if hs_std_name else ""
        )
    combined_rows.append(row)

with COMBINED.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
    writer.writeheader()
    writer.writerows(combined_rows)


def metrics(actual, predicted, weights=None):
    error = predicted - actual
    if weights is None:
        rmse = np.sqrt(np.mean(error**2))
    else:
        rmse = np.sqrt(np.sum(weights * error**2) / np.sum(weights))
    correlation = np.corrcoef(actual, predicted)[0, 1]
    return {
        "RMSE_m": float(rmse),
        "MAE_m": float(np.mean(np.abs(error))),
        "bias_m": float(np.mean(error)),
        "correlation": float(correlation),
    }


reference_hi = np.asarray(reference["reference_Hi_m"], dtype=float)
reference_hs = np.asarray(reference["reference_Hs_m"], dtype=float)
reference_hi_unc = np.asarray(reference["reference_Hi_unc_m"], dtype=float)
weights_hi = 1.0 / np.maximum(reference_hi_unc, 1e-6) ** 2
metric_rows = []
summary_methods = {}

for method, (array, hi_name, hs_name, hi_std_name, hs_std_name) in methods.items():
    indices = np.array([model_by_row[int(row_id)] for row_id in reference["row_id"]])
    pred_hi = np.asarray(array[hi_name][indices], dtype=float)
    pred_hs = np.asarray(array[hs_name][indices], dtype=float)
    hi_result = metrics(reference_hi, pred_hi)
    hs_result = metrics(reference_hs, pred_hs)
    hi_weighted = metrics(reference_hi, pred_hi, weights=weights_hi)["RMSE_m"]
    hi_result["uncertainty_weighted_RMSE_m"] = hi_weighted

    coverage = None
    if hi_std_name:
        model_std = np.asarray(array[hi_std_name][indices], dtype=float)
        combined_std = np.sqrt(model_std**2 + reference_hi_unc**2)
        coverage = float(np.mean(np.abs(pred_hi - reference_hi) <= 1.96 * combined_std))
        hi_result["combined_95_interval_coverage"] = coverage

    summary_methods[method] = {"Hi": hi_result, "Hs": hs_result}
    for latent, result in [("Hi", hi_result), ("Hs", hs_result)]:
        metric_rows.append(
            {
                "method": method,
                "latent": latent,
                "n": len(reference),
                "RMSE_m": result["RMSE_m"],
                "MAE_m": result["MAE_m"],
                "bias_m": result["bias_m"],
                "correlation": result["correlation"],
                "uncertainty_weighted_RMSE_m": result.get("uncertainty_weighted_RMSE_m", ""),
                "combined_95_interval_coverage": result.get("combined_95_interval_coverage", ""),
            }
        )

with METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
    writer.writeheader()
    writer.writerows(metric_rows)

summary = {
    "reference": {
        "product": "IS2SITMOGR4 version 003",
        "month": "March 2020 monthly",
        "matched_rows": int(len(reference)),
        "maximum_collocation_distance_m": 12500.0,
        "independence_status": "external comparison product, not independent truth",
        "dependence_reason": (
            "ice thickness uses ICESat-2 ATL10 freeboard and NESOSIM snow loading; "
            "snow depth is redistributed NESOSIM v1.1"
        ),
        "time_mismatch": "monthly reference versus March 8-labelled 15-day retrieval window",
        "coordinate_correction": (
            "source EPSG:3411 was correctly transformed to EASE2 North; the existing "
            "server preprocessing used an incorrect lon_0=0 source definition"
        ),
    },
    "methods": summary_methods,
}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(COMBINED)
print(METRICS_CSV)
print(SUMMARY_JSON)
