"""Extract the same fixed-y transect from all five March 2020 windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--server-root", type=Path, required=True,
    help="Authorised CPOM directory containing CS2_1920 and IS2_1920.",
)
args = parser.parse_args()
ROOT = args.server_root
WINDOWS = ["March_1", "March_8", "March_15", "March_22", "March_29"]
FIXED_Y_M = -487500


def aggregate(path: Path, value_name: str, count_name: str) -> pd.DataFrame:
    frame = pd.read_pickle(path).copy()
    frame["x_m"] = np.rint(frame["x"]).astype("int64")
    frame["y_m"] = np.rint(frame["y"]).astype("int64")
    return frame.groupby(["x_m", "y_m"], as_index=False).agg(
        **{value_name: ("z", "mean"), count_name: ("z", "size")}
    )


parts = []
for window in WINDOWS:
    cs = aggregate(ROOT / "CS2_1920" / f"{window}.pkl", "freeboard_cs2_m", "n_cs2")
    is2 = aggregate(ROOT / "IS2_1920" / f"{window}.pkl", "freeboard_is2_m", "n_is2")
    paired = cs.merge(is2, on=["x_m", "y_m"], validate="one_to_one")
    paired = paired[
        (paired["freeboard_cs2_m"] >= 0)
        & (paired["freeboard_is2_m"] > 0)
        & (paired["y_m"] == FIXED_Y_M)
    ].sort_values("x_m").copy()
    paired.insert(0, "window", window)
    paired["distance_km"] = (paired["x_m"] - paired["x_m"].min()) / 1000.0
    parts.append(paired)
    print(window, len(paired), file=sys.stderr)

combined = pd.concat(parts, ignore_index=True)
combined.to_csv(sys.stdout, index=False)
