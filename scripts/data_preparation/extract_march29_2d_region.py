"""Extract March 29 pairs from the exact March 8 2D geographic window."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


parser=argparse.ArgumentParser()
parser.add_argument(
    "--server-root",type=Path,required=True,
    help="Authorised CPOM directory containing CS2_1920 and IS2_1920.",
)
args=parser.parse_args()
ROOT=args.server_root
CELL_M=25000
LEFT_IX=7
BOTTOM_IY=-21
WINDOW_CELLS=20


def aggregate(path,value,count):
    frame=pd.read_pickle(path).copy()
    frame["x_m"]=np.rint(frame["x"]).astype("int64")
    frame["y_m"]=np.rint(frame["y"]).astype("int64")
    return frame.groupby(["x_m","y_m"],as_index=False).agg(**{value:("z","mean"),count:("z","size")})


cs=aggregate(ROOT/"CS2_1920/March_29.pkl","freeboard_cs2_m","n_cs2")
is2=aggregate(ROOT/"IS2_1920/March_29.pkl","freeboard_is2_m","n_is2")
paired=cs.merge(is2,on=["x_m","y_m"],validate="one_to_one")
paired=paired[(paired["freeboard_cs2_m"]>=0)&(paired["freeboard_is2_m"]>0)].copy()
paired["ix"]=np.rint(paired["x_m"]/CELL_M).astype(int)
paired["iy"]=np.rint(paired["y_m"]/CELL_M).astype(int)
region=paired[
    paired["ix"].between(LEFT_IX,LEFT_IX+WINDOW_CELLS-1)
    & paired["iy"].between(BOTTOM_IY,BOTTOM_IY+WINDOW_CELLS-1)
].sort_values(["y_m","x_m"]).reset_index(drop=True)
region.insert(0,"row_id",np.arange(len(region)))
region["local_x_km"]=(region["x_m"]-LEFT_IX*CELL_M)/1000.0
region["local_y_km"]=(region["y_m"]-BOTTOM_IY*CELL_M)/1000.0

diagnostics={
    "window":"March_29",
    "selection":"fixed March 8 geographic cell-index bounds",
    "left_ix":LEFT_IX,"bottom_iy":BOTTOM_IY,"window_cells":WINDOW_CELLS,
    "paired_rows_all_arctic":int(len(paired)),"selected_rows":int(len(region)),
    "occupied_fraction":float(len(region)/(WINDOW_CELLS**2)),
    "actual_center_bounds_m":{
        "x_min":int(region["x_m"].min()),"x_max":int(region["x_m"].max()),
        "y_min":int(region["y_m"].min()),"y_max":int(region["y_m"].max()),
    },
}
print(json.dumps(diagnostics),file=sys.stderr)
columns=["row_id","x_m","y_m","local_x_km","local_y_km","freeboard_cs2_m","freeboard_is2_m","n_cs2","n_is2"]
region.to_csv(sys.stdout,index=False,columns=columns)
