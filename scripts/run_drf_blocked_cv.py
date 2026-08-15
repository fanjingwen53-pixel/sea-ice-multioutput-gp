"""Physics-informed DRF blocked validation on the real March 2020 transect."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


drf_source = os.environ.get("DRF_SRC")
if drf_source:
    sys.path.insert(0, drf_source)
try:
    from DRF.models import initialize_model
except ImportError as exc:
    raise ImportError(
        "DeepRandomFeatures is not bundled with this repository. Clone the "
        "official repository and install it with `pip install -e`, or set "
        "DRF_SRC to the cloned repository's src directory."
    ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--ensemble", type=int, default=3)
args = parser.parse_args()

frame = pd.read_csv(args.input).sort_values("distance_km").reset_index(drop=True)
distance = frame["distance_km"].to_numpy(dtype=np.float32)
Y = frame[["freeboard_cs2_m", "freeboard_is2_m"]].to_numpy(dtype=np.float32)

rho_w, rho_i, rho_s = 1024.0, 915.0, 300.0
c, c_s = 299792458.0, 229792458.0
A = (rho_w - rho_i) / rho_w
B_cs = 1.0 - c / c_s - rho_s / rho_w
B_is = 1.0 - rho_s / rho_w
H = torch.tensor([[A, B_cs], [A, B_is]], dtype=torch.float32)
latent_baseline = torch.tensor([1.5, 0.15], dtype=torch.float32)
latent_offset = torch.log(torch.expm1(latent_baseline))
sigma = torch.tensor([0.03, 0.02], dtype=torch.float32)


def positive_latent(raw_output: torch.Tensor) -> torch.Tensor:
    """Guarantee positive Hi/Hs while centring a zero raw output on the baseline."""
    return F.softplus(raw_output + latent_offset)

n = len(frame)
folds = np.array_split(np.arange(n), 5)
fold_id = np.empty(n, dtype=int)
latent_mean = np.full((n, 2), np.nan, dtype=np.float32)
latent_std = np.full((n, 2), np.nan, dtype=np.float32)
Y_pred = np.full((n, 2), np.nan, dtype=np.float32)
fold_losses: list[dict[str, float]] = []

for fold, test_idx in enumerate(folds):
    fold_id[test_idx] = fold
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    x_mean = float(distance[train_idx].mean())
    x_std = float(distance[train_idx].std())

    x_train_1d = (distance[train_idx] - x_mean) / x_std
    x_test_1d = (distance[test_idx] - x_mean) / x_std
    spatial_train = torch.tensor(
        np.column_stack([x_train_1d, np.zeros_like(x_train_1d)]), dtype=torch.float32
    )
    spatial_test = torch.tensor(
        np.column_stack([x_test_1d, np.zeros_like(x_test_1d)]), dtype=torch.float32
    )
    temporal_train = torch.zeros((len(train_idx), 1), dtype=torch.float32)
    temporal_test = torch.zeros((len(test_idx), 1), dtype=torch.float32)
    y_train = torch.tensor(Y[train_idx], dtype=torch.float32)

    member_latents = []
    member_losses = []
    for member in range(args.ensemble):
        seed = 20260808 + fold * 100 + member
        set_seed(seed)
        model = initialize_model(
            model_name="DeepSpatiotemporalGPNN",
            num_layers=2,
            spatial_input_dim=2,
            temporal_input_dim=1,
            hidden_dim=32,
            bottleneck_dim=16,
            output_dim=2,
            spatial_lengthscale=1.0,
            temporal_lengthscale=1.0,
            amplitude=1.0,
            device=torch.device("cpu"),
            spatial_layer_type="Matern",
            temporal_layer_type="Matern",
            model_kwargs={},
        ).to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

        best_loss = float("inf")
        best_state = None
        for _ in range(args.epochs):
            optimizer.zero_grad()
            latent = positive_latent(model(spatial_train, temporal_train))
            observation = latent @ H.T
            data_loss = torch.mean(((observation - y_train) / sigma) ** 2)
            loss = data_loss
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if value < best_loss:
                best_loss = value
                best_state = {
                    key: tensor.detach().clone() for key, tensor in model.state_dict().items()
                }

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            member_latents.append(
                positive_latent(model(spatial_test, temporal_test)).cpu().numpy()
            )
        member_losses.append(best_loss)

    ensemble_latent = np.stack(member_latents, axis=0)
    latent_mean[test_idx] = ensemble_latent.mean(axis=0)
    latent_std[test_idx] = ensemble_latent.std(axis=0, ddof=1)
    Y_pred[test_idx] = latent_mean[test_idx] @ H.numpy().T
    fold_losses.append(
        {
            "fold": fold,
            "mean_best_training_loss": float(np.mean(member_losses)),
            "min_best_training_loss": float(np.min(member_losses)),
        }
    )
    print(json.dumps(fold_losses[-1]), flush=True)

if not np.isfinite(latent_mean).all() or not np.isfinite(Y_pred).all():
    raise RuntimeError("DRF produced non-finite out-of-fold predictions")

result = pd.DataFrame(
    {
        "row_id": np.arange(n),
        "fold": fold_id,
        "distance_km": distance,
        "actual_cs2_m": Y[:, 0],
        "actual_is2_m": Y[:, 1],
        "drf_Hi_m": latent_mean[:, 0],
        "drf_Hs_m": latent_mean[:, 1],
        "drf_Hi_ensemble_std_m": latent_std[:, 0],
        "drf_Hs_ensemble_std_m": latent_std[:, 1],
        "drf_pred_cs2_m": Y_pred[:, 0],
        "drf_pred_is2_m": Y_pred[:, 1],
    }
)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
result.to_csv(args.output, index=False)

summary = {
    "DRF": {
        "CS2": metrics(Y[:, 0], Y_pred[:, 0]),
        "IS2": metrics(Y[:, 1], Y_pred[:, 1]),
    },
    "fold_training_losses": fold_losses,
    "epochs": args.epochs,
    "ensemble_members": args.ensemble,
}
print(json.dumps(summary, indent=2))
print(args.output)
