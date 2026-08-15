"""Synthetic recovery sensitivity for the positive multi-output GP."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.likelihoods import QuadratureLikelihood
from gpflow.quadrature import NDiagGHQuadrature


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "synthetic_sensitivity_metrics.csv"
PREDICTIONS = OUT / "synthetic_sensitivity_predictions.csv"
SUMMARY = OUT / "synthetic_sensitivity_summary.json"

logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
gpflow.config.set_default_float(np.float64)

rho_w, rho_i, rho_s = 1024.0, 915.0, 300.0
c, c_s = 299792458.0, 229792458.0
A = (rho_w - rho_i) / rho_w


def physics_matrix(alpha_cs: float, alpha_is: float) -> np.ndarray:
    return np.array(
        [
            [A, 1.0 - alpha_cs * c / c_s - rho_s / rho_w],
            [A, 1.0 - alpha_is * c / c_s - rho_s / rho_w],
        ],
        dtype=np.float64,
    )


H_ASSUMED = physics_matrix(alpha_cs=1.0, alpha_is=0.0)
RAW_BASELINE = np.log(np.expm1(np.array([1.8, 0.18], dtype=np.float64)))
STEPS = 120
REPLICATES = [11, 29, 47, 83]

SCENARIOS = [
    {
        "scenario": "noise_low",
        "factor": "noise",
        "level": "low",
        "n_observations": 60,
        "sigma_cs": 0.015,
        "sigma_is": 0.010,
        "alpha_cs_true": 1.0,
        "alpha_is_true": 0.0,
    },
    {
        "scenario": "noise_nominal",
        "factor": "noise",
        "level": "nominal",
        "n_observations": 60,
        "sigma_cs": 0.030,
        "sigma_is": 0.020,
        "alpha_cs_true": 1.0,
        "alpha_is_true": 0.0,
    },
    {
        "scenario": "noise_high",
        "factor": "noise",
        "level": "high",
        "n_observations": 60,
        "sigma_cs": 0.060,
        "sigma_is": 0.040,
        "alpha_cs_true": 1.0,
        "alpha_is_true": 0.0,
    },
    {
        "scenario": "density_sparse",
        "factor": "density",
        "level": "25 observations",
        "n_observations": 25,
        "sigma_cs": 0.030,
        "sigma_is": 0.020,
        "alpha_cs_true": 1.0,
        "alpha_is_true": 0.0,
    },
    {
        "scenario": "density_dense",
        "factor": "density",
        "level": "120 observations",
        "n_observations": 120,
        "sigma_cs": 0.030,
        "sigma_is": 0.020,
        "alpha_cs_true": 1.0,
        "alpha_is_true": 0.0,
    },
    {
        "scenario": "penetration_radar_partial",
        "factor": "penetration",
        "level": "true alpha_cs=0.8",
        "n_observations": 60,
        "sigma_cs": 0.030,
        "sigma_is": 0.020,
        "alpha_cs_true": 0.8,
        "alpha_is_true": 0.0,
    },
    {
        "scenario": "penetration_laser_partial",
        "factor": "penetration",
        "level": "true alpha_is=0.1",
        "n_observations": 60,
        "sigma_cs": 0.030,
        "sigma_is": 0.020,
        "alpha_cs_true": 1.0,
        "alpha_is_true": 0.1,
    },
    {
        "scenario": "penetration_combined",
        "factor": "penetration",
        "level": "true alpha_cs=0.8 and alpha_is=0.1",
        "n_observations": 60,
        "sigma_cs": 0.030,
        "sigma_is": 0.020,
        "alpha_cs_true": 0.8,
        "alpha_is_true": 0.1,
    },
]


def truth(x_km: np.ndarray) -> np.ndarray:
    hi = (
        1.8
        + 0.34 * np.sin(2.0 * np.pi * x_km / 650.0)
        + 0.18 * np.cos(2.0 * np.pi * x_km / 220.0)
        + 0.35 * np.exp(-0.5 * ((x_km - 710.0) / 85.0) ** 2)
    )
    hs = (
        0.18
        + 0.045 * np.cos(2.0 * np.pi * x_km / 450.0)
        + 0.030 * np.sin(2.0 * np.pi * x_km / 160.0)
        + 0.055 * np.exp(-0.5 * ((x_km - 320.0) / 70.0) ** 2)
    )
    return np.column_stack([hi, hs])


class PositivePhysicsLikelihood(QuadratureLikelihood):
    def __init__(self, sigma: np.ndarray):
        super().__init__(
            input_dim=1,
            latent_dim=2,
            observation_dim=2,
            quadrature=NDiagGHQuadrature(2, 8),
        )
        self.H = tf.convert_to_tensor(H_ASSUMED, dtype=tf.float64)
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


def positive_moments(mu: np.ndarray, variance: np.ndarray):
    nodes, weights = np.polynomial.hermite.hermgauss(30)
    draws = mu[..., None] + np.sqrt(2.0 * np.maximum(variance, 0.0))[..., None] * nodes
    values = np.logaddexp(0.0, draws)
    mean = np.sum(values * weights, axis=-1) / np.sqrt(np.pi)
    second = np.sum(values**2 * weights, axis=-1) / np.sqrt(np.pi)
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def fit_positive_gp(X_train, Y_train, X_test, sigma, seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    kernel = gpflow.kernels.SeparateIndependent(
        [
            gpflow.kernels.Matern52(lengthscales=0.150, variance=0.25),
            gpflow.kernels.Matern52(lengthscales=0.075, variance=0.05),
        ]
    )
    likelihood = PositivePhysicsLikelihood(np.asarray(sigma, dtype=np.float64))
    Z = np.linspace(X_train.min(), X_train.max(), min(30, len(X_train)))[:, None]
    inducing = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(Z)
    )
    model = gpflow.models.SVGP(
        kernel,
        likelihood,
        inducing,
        mean_function=gpflow.mean_functions.Constant(RAW_BASELINE),
        num_latent_gps=2,
        q_diag=False,
        num_data=len(X_train),
    )
    for latent_kernel in model.kernel.kernels:
        gpflow.utilities.set_trainable(latent_kernel.lengthscales, False)
    gpflow.utilities.set_trainable(model.inducing_variable, False)
    gpflow.utilities.set_trainable(model.mean_function, False)
    loss = model.training_loss_closure((X_train, Y_train), compile=True)
    natural_gradient = gpflow.optimizers.NaturalGradient(gamma=0.05)
    gpflow.utilities.set_trainable(model.q_mu, False)
    gpflow.utilities.set_trainable(model.q_sqrt, False)
    adam = tf.optimizers.Adam(learning_rate=0.01)
    for _ in range(STEPS):
        natural_gradient.minimize(loss, var_list=[(model.q_mu, model.q_sqrt)])
        with tf.GradientTape() as tape:
            value = loss()
        gradients = tape.gradient(value, model.trainable_variables)
        adam.apply_gradients(zip(gradients, model.trainable_variables))
    raw_mean, raw_variance = model.predict_f(X_test)
    latent_mean, latent_std = positive_moments(raw_mean.numpy(), raw_variance.numpy())
    return latent_mean, latent_std, float(loss())


def metric_values(actual, predicted):
    error = predicted - actual
    return {
        "RMSE_m": float(np.sqrt(np.mean(error**2))),
        "MAE_m": float(np.mean(np.abs(error))),
        "bias_m": float(np.mean(error)),
    }


x_dense_km = np.linspace(0.0, 1000.0, 401)
X_dense = (x_dense_km / 1000.0)[:, None]
truth_dense = truth(x_dense_km)
metric_rows = []
prediction_rows = []

for scenario in SCENARIOS:
    H_true = physics_matrix(scenario["alpha_cs_true"], scenario["alpha_is_true"])
    for replicate, seed in enumerate(REPLICATES):
        rng = np.random.default_rng(seed + 1000 * SCENARIOS.index(scenario))
        n = scenario["n_observations"]
        x_random = np.sort(rng.uniform(0.0, 1000.0, size=n - 2))
        x_train_km = np.concatenate([[0.0], x_random, [1000.0]])
        X_train = (x_train_km / 1000.0)[:, None]
        latent_train = truth(x_train_km)
        clean_observations = latent_train @ H_true.T
        sigma = np.array([scenario["sigma_cs"], scenario["sigma_is"]], dtype=float)
        observations = clean_observations + rng.normal(scale=sigma, size=clean_observations.shape)

        point_train = np.linalg.solve(H_ASSUMED, observations.T).T
        point_dense = np.column_stack(
            [np.interp(x_dense_km, x_train_km, point_train[:, j]) for j in range(2)]
        )
        positive_dense, positive_std, final_loss = fit_positive_gp(
            X_train, observations, X_dense, sigma, seed
        )

        for method, estimate in [("Point + linear", point_dense), ("Positive GP", positive_dense)]:
            for latent_index, latent_name in enumerate(["Hi", "Hs"]):
                values = metric_values(truth_dense[:, latent_index], estimate[:, latent_index])
                metric_rows.append(
                    {
                        **scenario,
                        "replicate": replicate,
                        "seed": seed,
                        "method": method,
                        "latent": latent_name,
                        **values,
                        "negative_dense_predictions": int((estimate[:, latent_index] < 0).sum()),
                        "final_training_loss": final_loss if method == "Positive GP" else "",
                    }
                )

        for row, x_km in enumerate(x_dense_km):
            prediction_rows.append(
                {
                    "scenario": scenario["scenario"],
                    "factor": scenario["factor"],
                    "level": scenario["level"],
                    "replicate": replicate,
                    "seed": seed,
                    "x_km": x_km,
                    "true_Hi_m": truth_dense[row, 0],
                    "true_Hs_m": truth_dense[row, 1],
                    "point_Hi_m": point_dense[row, 0],
                    "point_Hs_m": point_dense[row, 1],
                    "positive_gp_Hi_m": positive_dense[row, 0],
                    "positive_gp_Hs_m": positive_dense[row, 1],
                    "positive_gp_Hi_std_m": positive_std[row, 0],
                    "positive_gp_Hs_std_m": positive_std[row, 1],
                }
            )
        print(json.dumps({"scenario": scenario["scenario"], "replicate": replicate, "loss": final_loss}), flush=True)

with METRICS.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
    writer.writeheader()
    writer.writerows(metric_rows)
with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
    writer.writeheader()
    writer.writerows(prediction_rows)

aggregates = []
for scenario in SCENARIOS:
    for method in ["Point + linear", "Positive GP"]:
        for latent in ["Hi", "Hs"]:
            selected = [
                row for row in metric_rows
                if row["scenario"] == scenario["scenario"]
                and row["method"] == method
                and row["latent"] == latent
            ]
            rmses = np.array([row["RMSE_m"] for row in selected])
            aggregates.append(
                {
                    "scenario": scenario["scenario"],
                    "factor": scenario["factor"],
                    "level": scenario["level"],
                    "method": method,
                    "latent": latent,
                    "mean_RMSE_m": float(rmses.mean()),
                    "std_RMSE_m": float(rmses.std(ddof=1)),
                    "replicates": len(rmses),
                    "total_negative_predictions": int(
                        sum(row["negative_dense_predictions"] for row in selected)
                    ),
                }
            )

summary = {
    "design": {
        "domain_km": [0.0, 1000.0],
        "dense_truth_points": len(x_dense_km),
        "replicate_seeds": REPLICATES,
        "model_assumed_alpha": {"CS2": 1.0, "IS2": 0.0},
        "fixed_lengthscales_km": {"Hi": 150.0, "Hs": 75.0},
        "training_steps": STEPS,
    },
    "aggregates": aggregates,
}
SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(METRICS)
print(PREDICTIONS)
print(SUMMARY)
