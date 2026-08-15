"""Positive-latent SVGP noise sensitivity with a frozen temporal replication.

March 8 is the development window. Four pre-specified sensor-noise settings
are compared using the same five outer spatial folds. The configuration with
the lowest mean RMSE relative to the point-interpolation baseline is frozen and
then evaluated on the non-overlapping March 29 window.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.likelihoods import QuadratureLikelihood
from gpflow.quadrature import NDiagGHQuadrature


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "processed" / "real_data_March_all_fixed_transects.csv"
PREDICTIONS = OUT / "positive_gp_noise_sensitivity_predictions.csv"
SUMMARY = OUT / "positive_gp_noise_sensitivity_summary.json"

NOISE_CONFIGS = {
    "cs2_heavy": (0.02, 0.04),
    "balanced": (0.03, 0.03),
    "nominal": (0.03, 0.02),
    "is2_heavy": (0.05, 0.02),
}
DEVELOPMENT_WINDOW = "March_8"
REPLICATION_WINDOW = "March_29"
STEPS = 200

gpflow.config.set_default_float(np.float64)


class PositivePhysicsLikelihood(QuadratureLikelihood):
    """Independent Gaussian sensor errors after a softplus physical layer."""

    def __init__(self, H: np.ndarray, sigma: np.ndarray):
        super().__init__(
            input_dim=1,
            latent_dim=2,
            observation_dim=2,
            quadrature=NDiagGHQuadrature(2, 8),
        )
        self.H = tf.convert_to_tensor(H, dtype=tf.float64)
        self.sigma = tf.convert_to_tensor(sigma, dtype=tf.float64)

    def _conditional_mean(self, X, F):
        return tf.linalg.matvec(self.H, tf.nn.softplus(F))

    def _conditional_variance(self, X, F):
        return tf.broadcast_to(tf.square(self.sigma), tf.shape(F))

    def _log_prob(self, X, F, Y):
        mean = self._conditional_mean(X, F)
        z = (Y - mean) / self.sigma
        return tf.reduce_sum(
            -0.5 * tf.square(z)
            - tf.math.log(self.sigma)
            - 0.5 * np.log(2.0 * np.pi),
            axis=-1,
        )


rho_w, rho_i, rho_s = 1024.0, 915.0, 300.0
c, c_s = 299792458.0, 229792458.0
H = np.array(
    [
        [(rho_w - rho_i) / rho_w, 1.0 - c / c_s - rho_s / rho_w],
        [(rho_w - rho_i) / rho_w, 1.0 - rho_s / rho_w],
    ],
    dtype=np.float64,
)
RAW_BASELINE = np.log(np.expm1(np.array([1.5, 0.15], dtype=np.float64)))


all_data = np.genfromtxt(DATA, delimiter=",", names=True, dtype=None, encoding="utf-8")


def load_window(window: str):
    subset = all_data[all_data["window"] == window]
    order = np.argsort(subset["distance_km"])
    subset = subset[order]
    X = (np.asarray(subset["distance_km"], dtype=np.float64) / 1000.0)[:, None]
    Y = np.column_stack(
        [subset["freeboard_cs2_m"], subset["freeboard_is2_m"]]
    ).astype(np.float64)
    return subset, X, Y


def point_baseline(X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray]):
    latent = np.full_like(Y, np.nan)
    observations = np.full_like(Y, np.nan)
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
        order = train_idx[np.argsort(X[train_idx, 0])]
        train_latent = np.linalg.solve(H, Y[order].T).T
        for output in range(2):
            latent[test_idx, output] = np.interp(
                X[test_idx, 0], X[order, 0], train_latent[:, output]
            )
        observations[test_idx] = latent[test_idx] @ H.T
    return latent, observations


def positive_moments(mu: np.ndarray, variance: np.ndarray):
    """Gauss-Hermite moments of softplus(N(mu, variance))."""
    nodes, weights = np.polynomial.hermite.hermgauss(30)
    draws = mu[..., None] + np.sqrt(2.0 * np.maximum(variance, 0.0))[..., None] * nodes
    values = np.logaddexp(0.0, draws)
    mean = np.sum(values * weights, axis=-1) / np.sqrt(np.pi)
    second = np.sum(values**2 * weights, axis=-1) / np.sqrt(np.pi)
    std = np.sqrt(np.maximum(second - mean**2, 0.0))
    return mean, std


def fit_predict(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    lengths_km: tuple[float, float],
    sigma: tuple[float, float],
):
    tf.keras.backend.clear_session()
    tf.random.set_seed(20260808)
    kernels = [
        gpflow.kernels.Matern52(lengthscales=lengths_km[0] / 1000.0, variance=0.3),
        gpflow.kernels.Matern52(lengthscales=lengths_km[1] / 1000.0, variance=0.1),
    ]
    kernel = gpflow.kernels.SeparateIndependent(kernels)
    likelihood = PositivePhysicsLikelihood(H, np.asarray(sigma, dtype=np.float64))
    inducing_locations = np.linspace(
        X[train_idx].min(), X[train_idx].max(), min(30, len(train_idx))
    )[:, None]
    inducing = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(inducing_locations)
    )
    model = gpflow.models.SVGP(
        kernel,
        likelihood,
        inducing,
        mean_function=gpflow.mean_functions.Constant(RAW_BASELINE),
        num_latent_gps=2,
        q_diag=False,
        num_data=len(train_idx),
    )
    for latent_kernel in model.kernel.kernels:
        gpflow.utilities.set_trainable(latent_kernel.lengthscales, False)
    gpflow.utilities.set_trainable(model.inducing_variable, False)
    gpflow.utilities.set_trainable(model.mean_function, False)

    loss = model.training_loss_closure((X[train_idx], Y[train_idx]), compile=True)
    natural_gradient = gpflow.optimizers.NaturalGradient(gamma=0.05)
    gpflow.utilities.set_trainable(model.q_mu, False)
    gpflow.utilities.set_trainable(model.q_sqrt, False)
    adam = tf.optimizers.Adam(learning_rate=0.01)
    for _ in range(STEPS):
        natural_gradient.minimize(loss, var_list=[(model.q_mu, model.q_sqrt)])
        with tf.GradientTape() as tape:
            loss_value = loss()
        gradients = tape.gradient(loss_value, model.trainable_variables)
        adam.apply_gradients(zip(gradients, model.trainable_variables))

    raw_mean, raw_variance = model.predict_f(X[test_idx])
    latent_mean, latent_std = positive_moments(
        raw_mean.numpy(), raw_variance.numpy()
    )
    observations = latent_mean @ H.T
    return latent_mean, latent_std, observations, float(loss())


def rmse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((predicted - actual) ** 2, axis=0))


def run_window(window: str, configurations: dict[str, tuple[float, float]]):
    subset, X, Y = load_window(window)
    folds = list(np.array_split(np.arange(len(X)), 5))
    fold_id = np.empty(len(X), dtype=int)
    for fold, idx in enumerate(folds):
        fold_id[idx] = fold
    point_latent, point_observations = point_baseline(X, Y, folds)
    selection = json.loads((OUT / f"{window}_nested_gp_selection.json").read_text())

    records = []
    metrics = {}
    for config_name, sigma in configurations.items():
        latent = np.full_like(Y, np.nan)
        latent_std = np.full_like(Y, np.nan)
        observations = np.full_like(Y, np.nan)
        losses = []
        for fold, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
            lengths = (
                float(selection[fold]["selected_Hi_km"]),
                float(selection[fold]["selected_Hs_km"]),
            )
            latent[test_idx], latent_std[test_idx], observations[test_idx], loss = fit_predict(
                X, Y, train_idx, test_idx, lengths, sigma
            )
            losses.append(loss)
            print(
                json.dumps(
                    {
                        "window": window,
                        "noise": config_name,
                        "fold": fold,
                        "lengths_km": lengths,
                        "loss": loss,
                    }
                ),
                flush=True,
            )
        if not np.isfinite(observations).all() or not np.isfinite(latent).all():
            raise RuntimeError(f"Non-finite positive-GP prediction for {window}/{config_name}")
        config_rmse = rmse(Y, observations)
        metrics[config_name] = {
            "sigma_cs2_m": sigma[0],
            "sigma_is2_m": sigma[1],
            "CS2_RMSE_m": float(config_rmse[0]),
            "IS2_RMSE_m": float(config_rmse[1]),
            "negative_Hi": int((latent[:, 0] < 0).sum()),
            "negative_Hs": int((latent[:, 1] < 0).sum()),
            "mean_final_loss": float(np.mean(losses)),
        }
        for row in range(len(X)):
            records.append(
                {
                    "window": window,
                    "noise_config": config_name,
                    "sigma_cs2_m": sigma[0],
                    "sigma_is2_m": sigma[1],
                    "row_id": row,
                    "fold": int(fold_id[row]),
                    "distance_km": float(subset["distance_km"][row]),
                    "actual_cs2_m": Y[row, 0],
                    "actual_is2_m": Y[row, 1],
                    "point_Hi_m": point_latent[row, 0],
                    "point_Hs_m": point_latent[row, 1],
                    "point_pred_cs2_m": point_observations[row, 0],
                    "point_pred_is2_m": point_observations[row, 1],
                    "positive_gp_Hi_m": latent[row, 0],
                    "positive_gp_Hs_m": latent[row, 1],
                    "positive_gp_Hi_std_m": latent_std[row, 0],
                    "positive_gp_Hs_std_m": latent_std[row, 1],
                    "positive_gp_pred_cs2_m": observations[row, 0],
                    "positive_gp_pred_is2_m": observations[row, 1],
                }
            )
    baseline_rmse = rmse(Y, point_observations)
    return records, metrics, {
        "CS2_RMSE_m": float(baseline_rmse[0]),
        "IS2_RMSE_m": float(baseline_rmse[1]),
    }


development_records, development_metrics, development_baseline = run_window(
    DEVELOPMENT_WINDOW, NOISE_CONFIGS
)
for values in development_metrics.values():
    values["relative_score"] = 0.5 * (
        values["CS2_RMSE_m"] / development_baseline["CS2_RMSE_m"]
        + values["IS2_RMSE_m"] / development_baseline["IS2_RMSE_m"]
    )
selected_name = min(development_metrics, key=lambda name: development_metrics[name]["relative_score"])
selected_noise = NOISE_CONFIGS[selected_name]
print(json.dumps({"selected_noise": selected_name, "sigma": selected_noise}), flush=True)

replication_records, replication_metrics, replication_baseline = run_window(
    REPLICATION_WINDOW, {selected_name: selected_noise}
)
all_records = development_records + replication_records
with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
    writer.writeheader()
    writer.writerows(all_records)

summary = {
    "design": {
        "development_window": DEVELOPMENT_WINDOW,
        "replication_window": REPLICATION_WINDOW,
        "replication_window_does_not_overlap_development": True,
        "training_steps": STEPS,
        "selection_score": "mean of sensor RMSE divided by its point-baseline RMSE",
    },
    "development_baseline": development_baseline,
    "development_noise_sensitivity": development_metrics,
    "selected_noise_config": selected_name,
    "selected_sigma_m": {"CS2": selected_noise[0], "IS2": selected_noise[1]},
    "replication_baseline": replication_baseline,
    "replication_positive_gp": replication_metrics[selected_name],
}
SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(PREDICTIONS)
print(SUMMARY)
