"""Run five-fold contiguous blocked validation for the pointwise and MOGP models."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03_real_data_march8_1d.ipynb"
DATA = ROOT / "data" / "processed" / "real_data_March8_1D_transect.csv"
OUTPUT = ROOT / "results" / "blocked_validation_gp_predictions.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

# Reuse the exact imports and classes already executed in the real-data notebook.
exec("".join(notebook["cells"][1]["source"]), globals())
exec("".join(notebook["cells"][2]["source"]), globals())

data = np.genfromtxt(DATA, delimiter=",", names=True, dtype=None, encoding="utf-8")
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

n = len(X)
folds = np.array_split(np.arange(n), 5)
fold_id = np.empty(n, dtype=int)

point_f = np.full((n, 2), np.nan)
point_y = np.full((n, 2), np.nan)
gp_f = np.full((n, 2), np.nan)
gp_std = np.full((n, 2), np.nan)
gp_y = np.full((n, 2), np.nan)
lengthscales = []

for fold, test_idx in enumerate(folds):
    fold_id[test_idx] = fold
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    order = train_idx[np.argsort(X[train_idx, 0])]

    # Baseline: pointwise inversion at training sites followed by 1D linear interpolation.
    train_f_point = np.linalg.solve(H, Y[order].T).T
    for output in range(2):
        point_f[test_idx, output] = np.interp(
            X[test_idx, 0], X[order, 0], train_f_point[:, output]
        )
    point_y[test_idx] = point_f[test_idx] @ H.T

    # Multi-output GP trained only on the non-held-out spatial blocks.
    kern_hi = gpflow.kernels.Matern52(lengthscales=0.4, variance=1.0)
    kern_hs = gpflow.kernels.Matern52(lengthscales=0.3, variance=1.0)
    W_prior = np.array([[0.45, 0.0], [0.0, 0.10]], dtype=np.float64)
    kernel = gpflow.kernels.LinearCoregionalization([kern_hi, kern_hs], W=W_prior)
    likelihood = LinearModelLikelihood(input_dim=1, variance=R, H=H)
    model = MultioutputGPR(
        data=(X[train_idx], Y[train_idx] - observation_baseline),
        kernel=kernel,
        likelihood=likelihood,
        num_latent_gps=2,
    )
    gpflow.utilities.set_trainable(model.kernel.W, False)
    gpflow.utilities.set_trainable(model.likelihood.variance, False)
    optimizer = gpflow.optimizers.Scipy()
    result = optimizer.minimize(
        model.training_loss,
        model.trainable_variables,
        options=dict(maxiter=1000),
    )
    if not result.success:
        raise RuntimeError(f"GP fold {fold} did not converge: {result.message}")

    mean_dev, variance = model.predict_f(X[test_idx])
    gp_f[test_idx] = mean_dev.numpy() + latent_baseline
    gp_std[test_idx] = np.sqrt(np.maximum(variance.numpy(), 0.0))
    gp_y[test_idx] = gp_f[test_idx] @ H.T
    lengthscales.append(
        {
            "fold": fold,
            "Hi_km": float(model.kernel.kernels[0].lengthscales.numpy() * 1000.0),
            "Hs_km": float(model.kernel.kernels[1].lengthscales.numpy() * 1000.0),
        }
    )

if not all(np.isfinite(array).all() for array in [point_f, point_y, gp_f, gp_std, gp_y]):
    raise RuntimeError("Non-finite out-of-fold predictions were produced")

fieldnames = [
    "row_id",
    "fold",
    "distance_km",
    "actual_cs2_m",
    "actual_is2_m",
    "point_Hi_m",
    "point_Hs_m",
    "point_pred_cs2_m",
    "point_pred_is2_m",
    "gp_Hi_m",
    "gp_Hs_m",
    "gp_Hi_std_m",
    "gp_Hs_std_m",
    "gp_pred_cs2_m",
    "gp_pred_is2_m",
]

with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
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
                "gp_Hi_m": gp_f[index, 0],
                "gp_Hs_m": gp_f[index, 1],
                "gp_Hi_std_m": gp_std[index, 0],
                "gp_Hs_std_m": gp_std[index, 1],
                "gp_pred_cs2_m": gp_y[index, 0],
                "gp_pred_is2_m": gp_y[index, 1],
            }
        )


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
    "gp": {
        "CS2": metrics(Y[:, 0], gp_y[:, 0]),
        "IS2": metrics(Y[:, 1], gp_y[:, 1]),
    },
    "lengthscales": lengthscales,
}
print(json.dumps(summary, indent=2))
print(OUTPUT)
