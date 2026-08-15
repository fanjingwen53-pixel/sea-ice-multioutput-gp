"""Extract correctly reprojected March IS2SITMOGR4 reference values."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--server-root",
    type=Path,
    required=True,
    help="Authorised CPOM project root containing processed_pkl and validation.",
)
args = parser.parse_args()
ROOT = args.server_root
VALIDATION = ROOT / "validation"
MAX_DISTANCE_M = 12500.0


def aggregate(path: Path, value: str):
    frame = pd.read_pickle(path).copy()
    frame["x_m"] = np.rint(frame["x"]).astype("int64")
    frame["y_m"] = np.rint(frame["y"]).astype("int64")
    return frame.groupby(["x_m", "y_m"], as_index=False)["z"].mean().rename(columns={"z": value})


cs = aggregate(ROOT / "processed_pkl/7_days/CS2_1920/March_8.pkl", "freeboard_cs2_m")
is2 = aggregate(ROOT / "processed_pkl/7_days/IS2_1920/March_8.pkl", "freeboard_is2_m")
transect = cs.merge(is2, on=["x_m", "y_m"])
transect = transect[
    (transect["freeboard_cs2_m"] >= 0)
    & (transect["freeboard_is2_m"] > 0)
    & (transect["y_m"] == -487500)
].sort_values("x_m").reset_index(drop=True)
transect.insert(0, "row_id", np.arange(len(transect)))

ice = pd.read_pickle(VALIDATION / "ice" / "mar20.pkl")
snow = pd.read_pickle(VALIDATION / "snow" / "mar20.pkl")
unc = pd.read_pickle(VALIDATION / "ice_unc" / "mar20.pkl")

ice = ice.rename(columns={"z": "reference_Hi_m"})
snow = snow.rename(columns={"z": "reference_Hs_m"})
unc = unc.rename(columns={"z": "reference_Hi_unc_m"})
reference = ice[["og_x", "og_y", "reference_Hi_m", "x", "y"]].merge(
    snow[["og_x", "og_y", "reference_Hs_m"]], on=["og_x", "og_y"]
).merge(
    unc[["og_x", "og_y", "reference_Hi_unc_m"]], on=["og_x", "og_y"]
)

# The server preprocessing used lon_0=0 and WGS84 as the source CRS. The
# NetCDF metadata instead specify EPSG:3411 (lon_0=-45, Hughes 1980). Reproject
# from the documented CRS to the EASE2-North LAEA used by the freeboard grid.
transformer = Transformer.from_crs(
    "EPSG:3411",
    "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
    always_xy=True,
)
correct_x, correct_y = transformer.transform(
    reference["og_x"].to_numpy(dtype=float),
    reference["og_y"].to_numpy(dtype=float),
)
reference["correct_x_m"] = correct_x
reference["correct_y_m"] = correct_y
reference["stored_transform_error_m"] = np.sqrt(
    (reference["x"].to_numpy(dtype=float) - correct_x) ** 2
    + (reference["y"].to_numpy(dtype=float) - correct_y) ** 2
)

ref_xy = reference[["correct_x_m", "correct_y_m"]].to_numpy(dtype=float)
rows = []
nearest_all = []
for _, point in transect.iterrows():
    distances = np.sqrt(
        (ref_xy[:, 0] - float(point["x_m"])) ** 2
        + (ref_xy[:, 1] - float(point["y_m"])) ** 2
    )
    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])
    nearest_all.append(nearest_distance)
    if nearest_distance > MAX_DISTANCE_M:
        continue
    match = reference.iloc[nearest_index]
    rows.append(
        {
            "row_id": int(point["row_id"]),
            "x_m": int(point["x_m"]),
            "y_m": int(point["y_m"]),
            "freeboard_cs2_m": float(point["freeboard_cs2_m"]),
            "freeboard_is2_m": float(point["freeboard_is2_m"]),
            "reference_distance_m": nearest_distance,
            "reference_Hi_m": float(match["reference_Hi_m"]),
            "reference_Hs_m": float(match["reference_Hs_m"]),
            "reference_Hi_unc_m": float(match["reference_Hi_unc_m"]),
            "reference_source_x_m": float(match["og_x"]),
            "reference_source_y_m": float(match["og_y"]),
            "reference_ease_x_m": float(match["correct_x_m"]),
            "reference_ease_y_m": float(match["correct_y_m"]),
        }
    )

nearest_all = np.asarray(nearest_all)
diagnostics = {
    "transect_rows": int(len(transect)),
    "reference_common_Hi_Hs_rows": int(len(reference)),
    "matches_within_12_5km": int(len(rows)),
    "nearest_distance_all_m": {
        "min": float(nearest_all.min()),
        "median": float(np.median(nearest_all)),
        "max": float(nearest_all.max()),
    },
    "stored_transform_error_m": {
        "min": float(reference["stored_transform_error_m"].min()),
        "median": float(reference["stored_transform_error_m"].median()),
        "max": float(reference["stored_transform_error_m"].max()),
    },
    "unique_reference_cells_matched": int(
        len({(row["reference_source_x_m"], row["reference_source_y_m"]) for row in rows})
    ),
}
print(json.dumps(diagnostics), file=sys.stderr)
pd.DataFrame(rows).to_csv(sys.stdout, index=False)
