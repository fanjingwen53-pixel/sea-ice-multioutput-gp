"""Generate publication-ready figures from the archived dissertation results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#2166AC"
ORANGE = "#D95F02"
TEAL = "#1B9E77"
PURPLE = "#7570B3"
GREY = "#666666"


def rows(name: str):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update(
    {
        "font.size": 12.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 10.5,
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
    }
)


# Synthetic sensitivity: mean and standard deviation across four replicates.
synthetic = rows("synthetic_sensitivity_metrics.csv")
scenario_order = [
    "noise_low",
    "noise_nominal",
    "noise_high",
    "density_sparse",
    "density_dense",
    "penetration_radar_partial",
    "penetration_laser_partial",
    "penetration_combined",
]
scenario_labels = [
    "Low\nnoise",
    "Nominal\nnoise",
    "High\nnoise",
    "25\nobs.",
    "120\nobs.",
    r"$\alpha_{CS}=0.8$",
    r"$\alpha_{IS}=0.1$",
    "Both\npartial",
]
fig, axes = plt.subplots(2, 1, figsize=(7.15, 6.0), sharex=True)
for ax, latent, ylabel in zip(axes, ["Hi", "Hs"], ["SIT RMSE (m)", "Snow-depth RMSE (m)"]):
    x = np.arange(len(scenario_order))
    width = 0.36
    for offset, method, colour in [
        (-width / 2, "Point + linear", GREY),
        (width / 2, "Positive GP", BLUE),
    ]:
        means, stds = [], []
        for scenario in scenario_order:
            values = [
                float(r["RMSE_m"])
                for r in synthetic
                if r["scenario"] == scenario
                and r["method"] == method
                and r["latent"] == latent
            ]
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1))
        ax.bar(x + offset, means, width, yerr=stds, capsize=2.5, color=colour, label=method)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, ncol=2, loc="upper left")
axes[1].set_xticks(np.arange(len(scenario_order)), scenario_labels)
axes[1].set_xlabel("Sensitivity scenario")
fig.text(0.21, 0.015, "Noise", ha="center", color=GREY)
fig.text(0.50, 0.015, "Density", ha="center", color=GREY)
fig.text(0.79, 0.015, "Penetration mismatch", ha="center", color=GREY)
fig.suptitle(r"Synthetic recovery sensitivity (mean $\pm$ one SD, four replicates)", y=0.995)
fig.tight_layout(rect=(0, 0.035, 1, 0.98))
save(fig, "synthetic_sensitivity")


# Five-window one-dimensional blocked validation.
window_rows = rows("nested_gp_window_metrics.csv")
windows = ["March_1", "March_8", "March_15", "March_22", "March_29"]
labels = ["1 Mar", "8 Mar", "15 Mar", "22 Mar", "29 Mar"]
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.6), sharey=True)
for ax, sensor in zip(axes, ["CS2", "IS2"]):
    x = np.arange(len(windows))
    width = 0.36
    for offset, method, colour in [
        (-width / 2, "Point + linear", GREY),
        (width / 2, "Nested multi-output GP", BLUE),
    ]:
        values = [
            float(next(r for r in window_rows if r["window"] == window and r["method"] == method and r["sensor"] == sensor)["RMSE_m"])
            for window in windows
        ]
        ax.bar(x + offset, values, width, color=colour, label=method)
    ax.set_title(sensor)
    ax.set_xticks(x, labels, rotation=25)
    ax.set_xlabel("Retrieval window")
    ax.set_axisbelow(True)
axes[0].set_ylabel("Held-out freeboard RMSE (m)")
axes[0].legend(frameon=False, fontsize=10)
fig.suptitle("One-dimensional spatial blocked validation across five March windows")
fig.tight_layout()
save(fig, "one_dimensional_window_validation")


# Method comparison on development and non-overlapping replication windows.
method_rows = rows("positive_gp_method_comparison.csv")
methods = ["Point + linear", "Nested linear GP", "Positive GP", "Hard-positive DRF"]
method_colours = [GREY, PURPLE, BLUE, ORANGE]
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.7), sharey=True)
for ax, window, title in zip(axes, ["March_8", "March_29"], ["8 March (development)", "29 March (replication)"]):
    x = np.arange(len(methods))
    width = 0.36
    for j, (sensor, hatch) in enumerate([("CS2", ""), ("IS2", "///")]):
        vals = []
        for method in methods:
            match = [r for r in method_rows if r["window"] == window and r["method"] == method and r["sensor"] == sensor]
            vals.append(float(match[0]["RMSE_m"]) if match else np.nan)
        bars = ax.bar(x + (j - 0.5) * width, vals, width, color=method_colours, hatch=hatch, edgecolor="white", linewidth=0.4)
        if sensor == "IS2":
            for bar in bars:
                bar.set_edgecolor("#333333")
                bar.set_linewidth(0.4)
    ax.set_title(title)
    ax.set_xticks(x, ["Point", "Nested GP", "Positive GP", "DRF"], rotation=22)
    ax.set_xlabel("Method")
    ax.set_axisbelow(True)
axes[0].set_ylabel("Held-out freeboard RMSE (m)")
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(facecolor="#AAAAAA", label="CS2"), Patch(facecolor="#AAAAAA", hatch="///", edgecolor="#333333", label="IS2")], frameon=False)
fig.suptitle("One-dimensional method comparison and temporal replication")
fig.tight_layout()
save(fig, "one_dimensional_method_comparison")


# Two-dimensional latent fields and their uncertainty.
def grid_data(name: str):
    data = rows(name)
    x = np.array([float(r["local_x_km"]) for r in data])
    y = np.array([float(r["local_y_km"]) for r in data])
    ux, uy = np.unique(x), np.unique(y)
    order = np.lexsort((x, y))
    return ux, uy, {key: np.array([float(data[i][key]) for i in order]).reshape(len(uy), len(ux)) for key in ["positive_gp_Hi_m", "positive_gp_Hs_m", "positive_gp_Hi_std_m", "positive_gp_Hs_std_m"]}


grids = [
    ("8 March", grid_data("positive_gp_2d_full_grid.csv")),
    ("29 March", grid_data("positive_gp_2d_March29_full_grid.csv")),
]
hi_values = np.concatenate([g[1][2]["positive_gp_Hi_m"].ravel() for g in grids])
hs_values = np.concatenate([g[1][2]["positive_gp_Hs_m"].ravel() for g in grids])
fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.8), constrained_layout=True)
for col, (window, (ux, uy, fields)) in enumerate(grids):
    for row, (key, title, values, cmap) in enumerate([
        ("positive_gp_Hi_m", "Sea-ice thickness", hi_values, "viridis"),
        ("positive_gp_Hs_m", "Snow depth", hs_values, "magma"),
    ]):
        im = axes[row, col].imshow(fields[key], origin="lower", extent=[ux.min(), ux.max(), uy.min(), uy.max()], aspect="equal", cmap=cmap, vmin=np.percentile(values, 1), vmax=np.percentile(values, 99))
        axes[row, col].set_title(f"{window}: {title}")
        axes[row, col].set_xlabel("Local easting (km)")
        axes[row, col].set_ylabel("Local northing (km)")
        cb = fig.colorbar(im, ax=axes[row, col], shrink=0.82)
        cb.set_label("m")
fig.suptitle("Positive multi-output GP latent fields in the fixed 475 km region")
save(fig, "two_dimensional_latent_fields")


fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.8), constrained_layout=True)
for col, (window, (ux, uy, fields)) in enumerate(grids):
    for row, (key, title, cmap) in enumerate([
        ("positive_gp_Hi_std_m", "SIT posterior SD", "Blues"),
        ("positive_gp_Hs_std_m", "Snow-depth posterior SD", "Oranges"),
    ]):
        im = axes[row, col].imshow(fields[key], origin="lower", extent=[ux.min(), ux.max(), uy.min(), uy.max()], aspect="equal", cmap=cmap)
        axes[row, col].set_title(f"{window}: {title}")
        axes[row, col].set_xlabel("Local easting (km)")
        axes[row, col].set_ylabel("Local northing (km)")
        cb = fig.colorbar(im, ax=axes[row, col], shrink=0.82)
        cb.set_label("m")
fig.suptitle("Posterior uncertainty from the full-data visualisation fits")
save(fig, "two_dimensional_uncertainty")


# Two-dimensional held-out validation.
twod = rows("positive_gp_2d_temporal_comparison.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.6), sharey=True)
for ax, window, title in zip(axes, ["March_8", "March_29"], ["8 March", "29 March"]):
    x = np.arange(2)
    width = 0.36
    for offset, method, colour in [(-width / 2, "Point + nearest", GREY), (width / 2, "Positive GP", BLUE)]:
        vals = [float(next(r for r in twod if r["window"] == window and r["method"] == method and r["sensor"] == sensor)["RMSE_m"]) for sensor in ["CS2", "IS2"]]
        ax.bar(x + offset, vals, width, color=colour, label=method)
    ax.set_title(title)
    ax.set_xticks(x, ["CS2", "IS2"])
    ax.set_xlabel("Held-out observation")
    ax.set_axisbelow(True)
axes[0].set_ylabel("Quadrant-blocked RMSE (m)")
axes[0].legend(frameon=False)
fig.suptitle("Two-dimensional spatial validation in a fixed region")
fig.tight_layout()
save(fig, "two_dimensional_validation")


print(f"Wrote result figures to {FIGURES}")
