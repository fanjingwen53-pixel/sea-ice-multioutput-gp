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
CS2_PATH = args.server_root / "CS2_1920" / "March_8.pkl"
IS2_PATH = args.server_root / "IS2_1920" / "March_8.pkl"


def weekly_grid(path: Path, output_column: str, count_column: str) -> pd.DataFrame:
    frame = pd.read_pickle(path).copy()
    frame["x_m"] = np.rint(frame["x"]).astype("int64")
    frame["y_m"] = np.rint(frame["y"]).astype("int64")
    return frame.groupby(["x_m", "y_m"], as_index=False).agg(
        **{
            output_column: ("z", "mean"),
            count_column: ("z", "size"),
        }
    )


cs2 = weekly_grid(CS2_PATH, "freeboard_cs2_m", "n_cs2")
is2 = weekly_grid(IS2_PATH, "freeboard_is2_m", "n_is2")
paired = cs2.merge(is2, on=["x_m", "y_m"], how="inner", validate="one_to_one")
paired = paired[(paired["freeboard_cs2_m"] >= 0) & (paired["freeboard_is2_m"] > 0)].copy()

row_counts = paired.groupby("y_m").size().sort_values(ascending=False)
selected_y = int(row_counts.index[0])
transect = paired[paired["y_m"] == selected_y].sort_values("x_m").copy()
transect["distance_km"] = (transect["x_m"] - transect["x_m"].min()) / 1000.0
transect.insert(0, "source_window", "March_8_2020_15day")

columns = [
    "source_window",
    "distance_km",
    "x_m",
    "y_m",
    "freeboard_cs2_m",
    "freeboard_is2_m",
    "n_cs2",
    "n_is2",
]

print("selected_y_m", selected_y, file=sys.stderr)
print("transect_rows", len(transect), file=sys.stderr)
print("top_row_counts", row_counts.head(10).to_dict(), file=sys.stderr)
print(
    "x_range_m",
    (int(transect["x_m"].min()), int(transect["x_m"].max())),
    file=sys.stderr,
)
transect.to_csv(sys.stdout, index=False, columns=columns)
