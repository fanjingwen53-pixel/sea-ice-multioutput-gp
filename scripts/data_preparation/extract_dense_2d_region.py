"""Select the densest pre-specified 500 km square from March 8 paired data."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import convolve2d


parser = argparse.ArgumentParser()
parser.add_argument(
    "--server-root",
    type=Path,
    required=True,
    help="Authorised CPOM directory containing CS2_1920 and IS2_1920.",
)
args = parser.parse_args()
ROOT = args.server_root
CS_PATH = ROOT / "CS2_1920/March_8.pkl"
IS_PATH = ROOT / "IS2_1920/March_8.pkl"
CELL_M = 25000
WINDOW_CELLS = 20


def aggregate(path: Path, value: str, count: str):
    frame = pd.read_pickle(path).copy()
    frame["x_m"] = np.rint(frame["x"]).astype("int64")
    frame["y_m"] = np.rint(frame["y"]).astype("int64")
    return frame.groupby(["x_m", "y_m"], as_index=False).agg(
        **{value: ("z", "mean"), count: ("z", "size")}
    )


cs = aggregate(CS_PATH, "freeboard_cs2_m", "n_cs2")
is2 = aggregate(IS_PATH, "freeboard_is2_m", "n_is2")
paired = cs.merge(is2, on=["x_m", "y_m"], validate="one_to_one")
paired = paired[(paired["freeboard_cs2_m"] >= 0) & (paired["freeboard_is2_m"] > 0)].copy()
paired["ix"] = np.rint(paired["x_m"] / CELL_M).astype(int)
paired["iy"] = np.rint(paired["y_m"] / CELL_M).astype(int)

ix_min, ix_max = int(paired["ix"].min()), int(paired["ix"].max())
iy_min, iy_max = int(paired["iy"].min()), int(paired["iy"].max())
occupancy = np.zeros((iy_max - iy_min + 1, ix_max - ix_min + 1), dtype=int)
occupancy[paired["iy"] - iy_min, paired["ix"] - ix_min] = 1
counts = convolve2d(occupancy, np.ones((WINDOW_CELLS, WINDOW_CELLS), dtype=int), mode="valid")
best_y, best_x = np.unravel_index(np.argmax(counts), counts.shape)
left_ix = ix_min + int(best_x)
bottom_iy = iy_min + int(best_y)

region = paired[
    paired["ix"].between(left_ix, left_ix + WINDOW_CELLS - 1)
    & paired["iy"].between(bottom_iy, bottom_iy + WINDOW_CELLS - 1)
].sort_values(["y_m", "x_m"]).reset_index(drop=True)
region.insert(0, "row_id", np.arange(len(region)))
region["local_x_km"] = (region["x_m"] - left_ix * CELL_M) / 1000.0
region["local_y_km"] = (region["y_m"] - bottom_iy * CELL_M) / 1000.0

diagnostics = {
    "selection_rule": "maximum paired-cell count in any pre-specified 20x20 EASE-cell square",
    "cell_size_km": CELL_M / 1000,
    "window_cells": WINDOW_CELLS,
    "nominal_extent_km": WINDOW_CELLS * CELL_M / 1000,
    "paired_rows_all_arctic": int(len(paired)),
    "selected_rows": int(len(region)),
    "bounds_m": {
        "x_min": int(left_ix * CELL_M),
        "x_max": int((left_ix + WINDOW_CELLS - 1) * CELL_M),
        "y_min": int(bottom_iy * CELL_M),
        "y_max": int((bottom_iy + WINDOW_CELLS - 1) * CELL_M),
    },
    "occupied_fraction": float(len(region) / (WINDOW_CELLS**2)),
}
print(json.dumps(diagnostics), file=sys.stderr)
columns = [
    "row_id", "x_m", "y_m", "local_x_km", "local_y_km",
    "freeboard_cs2_m", "freeboard_is2_m", "n_cs2", "n_is2",
]
region.to_csv(sys.stdout, index=False, columns=columns)
