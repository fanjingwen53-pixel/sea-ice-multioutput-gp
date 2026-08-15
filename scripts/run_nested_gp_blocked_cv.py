"""Nested spatial blocked validation for the sea-ice multi-output GP.

Outer folds estimate spatial generalisation.  For every outer fold, latent
length scales are selected using only three contiguous inner folds.  This
prevents the held-out outer block from influencing hyperparameter selection
and removes the unconstrained near-zero length-scale failure mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03_real_data_march8_1d.ipynb"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "real_data_March8_1D_transect.csv")
parser.add_argument("--window", default=None)
parser.add_argument("--output", type=Path, default=RESULTS / "nested_blocked_validation_gp_predictions.csv")
parser.add_argument("--selection-output", type=Path, default=RESULTS / "nested_gp_lengthscale_selection.json")
args = parser.parse_args()


with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

exec("".join(notebook["cells"][1]["source"]), globals())
exec("".join(notebook["cells"][2]["source"]), globals())

data = np.genfromtxt(args.data, delimiter=",", names=True, dtype=None, encoding="utf-8")
if args.window is not None:
    if "window" not in data.dtype.names:
        raise ValueError("--window requires a data file containing a window column")
    data = data[data["window"] == args.window]
    if len(data) == 0:
        raise ValueError(f"No rows found for window {args.window!r}")
distance_km = np.asarray(data["distance_km"], dtype=np.float64)
X = (distance_km / 1000.0)[:, None]
Y = np.column_stack(
    [
        np.asarray(data["freeboard_cs2_m"], dtype=np.float64),
        np.asarray(data["freeboard_is2_m"], dtype=np.float64),
    ]
)

rho_w, rho_i, rho_s = 1024.0, 915.0, 300.0
c, c_s = 299792458.0, 229792458.0
A = (rho_w - rho_i) / rho_w
B_cs = 1.0 - c / c_s - rho_s / rho_w
B_is = 1.0 - rho_s / rho_w
H = np.array([[A, B_cs], [A, B_is]], dtype=np.float64)
R = np.diag([0.03**2, 0.02**2]).astype(np.float64)
latent_baseline = np.array([1.5, 0.15], dtype=np.float64)
observation_baseline = H @ latent_baseline
W_PRIOR = np.array([[0.45, 0.0], [0.0, 0.10]], dtype=np.float64)

# Kilometres. These cover the short scales found in the full-data fit while
# excluding the sub-grid collapse observed in the original outer fold zero.
CANDIDATE_LENGTHS_KM = [
    (25.0, 25.0),
    (50.0, 25.0),
    (75.0, 50.0),
    (100.0, 50.0),
    (150.0, 75.0),
    (200.0, 100.0),
    (300.0, 150.0),
    (400.0, 200.0),
]


def build_model(train_idx: np.ndarray, lengths_km: tuple[float, float]):
    """Build a model with pre-specified, non-trainable latent length scales."""
    hi_km, hs_km = lengths_km
    kern_hi = gpflow.kernels.Matern52(lengthscales=hi_km / 1000.0, variance=1.0)
    kern_hs = gpflow.kernels.Matern52(lengthscales=hs_km / 1000.0, variance=1.0)
    kernel = gpflow.kernels.LinearCoregionalization([kern_hi, kern_hs], W=W_PRIOR)
    likelihood = LinearModelLikelihood(input_dim=1, variance=R, H=H)
    model = MultioutputGPR(
        data=(X[train_idx], Y[train_idx] - observation_baseline),
        kernel=kernel,
        likelihood=likelihood,
        num_latent_gps=2,
    )
    gpflow.utilities.set_trainable(model.kernel.W, False)
    gpflow.utilities.set_trainable(model.likelihood.variance, False)
    gpflow.utilities.set_trainable(model.kernel.kernels[0].lengthscales, False)
    gpflow.utilities.set_trainable(model.kernel.kernels[1].lengthscales, False)
    return model


def fit_model(train_idx: np.ndarray, lengths_km: tuple[float, float]):
    model = build_model(train_idx, lengths_km)
    result = gpflow.optimizers.Scipy().minimize(
        model.training_loss,
        model.trainable_variables,
        options=dict(maxiter=500),
    )
    if not result.success:
        raise RuntimeError(f"GP optimisation failed: {result.message}")
    return model


def predict_observations(model, test_idx: np.ndarray):
    latent_dev, latent_var = model.predict_f(X[test_idx])
    latent = latent_dev.numpy() + latent_baseline
    latent_std = np.sqrt(np.maximum(latent_var.numpy(), 0.0))
    observations = latent @ H.T
    return latent, latent_std, observations


def standardised_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Give the two sensors equal weight after training-only scale normalisation."""
    scales = np.maximum(np.std(actual, axis=0, ddof=1), 0.02)
    return float(np.mean(((predicted - actual) / scales) ** 2))


def choose_lengths(outer_train_idx: np.ndarray):
    ordered = outer_train_idx[np.argsort(X[outer_train_idx, 0])]
    inner_folds = np.array_split(ordered, 3)
    candidate_records = []
    for lengths in CANDIDATE_LENGTHS_KM:
        predictions = []
        actuals = []
        fold_scores = []
        for inner_test_idx in inner_folds:
            inner_train_idx = np.setdiff1d(outer_train_idx, inner_test_idx)
            model = fit_model(inner_train_idx, lengths)
            _, _, pred = predict_observations(model, inner_test_idx)
            score = standardised_mse(Y[inner_test_idx], pred)
            fold_scores.append(score)
            predictions.append(pred)
            actuals.append(Y[inner_test_idx])
        pooled_score = standardised_mse(np.vstack(actuals), np.vstack(predictions))
        candidate_records.append(
            {
                "Hi_km": lengths[0],
                "Hs_km": lengths[1],
                "pooled_standardised_mse": pooled_score,
                "inner_fold_scores": fold_scores,
            }
        )
    best = min(candidate_records, key=lambda item: item["pooled_standardised_mse"])
    return (best["Hi_km"], best["Hs_km"]), candidate_records


n = len(X)
outer_folds = np.array_split(np.arange(n), 5)
fold_id = np.empty(n, dtype=int)
point_f = np.full((n, 2), np.nan)
point_y = np.full((n, 2), np.nan)
gp_f = np.full((n, 2), np.nan)
gp_std = np.full((n, 2), np.nan)
gp_y = np.full((n, 2), np.nan)
selection_records = []

for outer_fold, test_idx in enumerate(outer_folds):
    fold_id[test_idx] = outer_fold
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    train_order = train_idx[np.argsort(X[train_idx, 0])]

    train_f_point = np.linalg.solve(H, Y[train_order].T).T
    for output in range(2):
        point_f[test_idx, output] = np.interp(
            X[test_idx, 0], X[train_order, 0], train_f_point[:, output]
        )
    point_y[test_idx] = point_f[test_idx] @ H.T

    selected, candidates = choose_lengths(train_idx)
    model = fit_model(train_idx, selected)
    gp_f[test_idx], gp_std[test_idx], gp_y[test_idx] = predict_observations(
        model, test_idx
    )
    record = {
        "outer_fold": outer_fold,
        "selected_Hi_km": selected[0],
        "selected_Hs_km": selected[1],
        "candidates": candidates,
    }
    selection_records.append(record)
    print(json.dumps({key: value for key, value in record.items() if key != "candidates"}), flush=True)

if not all(np.isfinite(array).all() for array in [point_f, point_y, gp_f, gp_std, gp_y]):
    raise RuntimeError("Non-finite out-of-fold predictions were produced")

fieldnames = [
    "row_id", "fold", "distance_km", "actual_cs2_m", "actual_is2_m",
    "point_Hi_m", "point_Hs_m", "point_pred_cs2_m", "point_pred_is2_m",
    "nested_gp_Hi_m", "nested_gp_Hs_m", "nested_gp_Hi_std_m",
    "nested_gp_Hs_std_m", "nested_gp_pred_cs2_m", "nested_gp_pred_is2_m",
]
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for index in range(n):
        writer.writerow(
            {
                "row_id": index,
                "fold": int(fold_id[index]),
                "distance_km": distance_km[index],
                "actual_cs2_m": Y[index, 0],
                "actual_is2_m": Y[index, 1],
                "point_Hi_m": point_f[index, 0],
                "point_Hs_m": point_f[index, 1],
                "point_pred_cs2_m": point_y[index, 0],
                "point_pred_is2_m": point_y[index, 1],
                "nested_gp_Hi_m": gp_f[index, 0],
                "nested_gp_Hs_m": gp_f[index, 1],
                "nested_gp_Hi_std_m": gp_std[index, 0],
                "nested_gp_Hs_std_m": gp_std[index, 1],
                "nested_gp_pred_cs2_m": gp_y[index, 0],
                "nested_gp_pred_is2_m": gp_y[index, 1],
            }
        )

args.selection_output.parent.mkdir(parents=True, exist_ok=True)
args.selection_output.write_text(json.dumps(selection_records, indent=2), encoding="utf-8")


def metrics(actual, predicted):
    error = predicted - actual
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


summary = {
    "point": {
        "CS2": metrics(Y[:, 0], point_y[:, 0]),
        "IS2": metrics(Y[:, 1], point_y[:, 1]),
    },
    "nested_gp": {
        "CS2": metrics(Y[:, 0], gp_y[:, 0]),
        "IS2": metrics(Y[:, 1], gp_y[:, 1]),
    },
    "selected_lengths": [
        {
            "outer_fold": row["outer_fold"],
            "Hi_km": row["selected_Hi_km"],
            "Hs_km": row["selected_Hs_km"],
        }
        for row in selection_records
    ],
    "negative_latents": {
        "Hi": int((gp_f[:, 0] < 0).sum()),
        "Hs": int((gp_f[:, 1] < 0).sum()),
    },
}
print(json.dumps(summary, indent=2))
print(args.output)
print(args.selection_output)
