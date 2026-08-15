# Data availability

Observation-level data are intentionally excluded from version control until
their redistribution status has been confirmed with the supervisor. The
real-data scripts expect authorised, locally generated CSV files in
`data/processed/`:

- `real_data_March8_1D_transect.csv`
- `real_data_March_all_fixed_transects.csv`
- `real_data_March8_dense_2D_region.csv`
- `real_data_March29_same_2D_region.csv`
- `external_reference_March2020_transect.csv`

The scripts in `scripts/data_preparation/` reproduce selected extracts from an
authorised CPOM project directory. They deliberately require an explicit
`--server-root` argument and contain no user-specific server path.

Do not commit raw satellite products, CPOM server files, access credentials or
locally extracted observation tables unless the data owner has approved public
redistribution. Compact derived metrics and prediction-only grids used to make
the dissertation figures are stored in `results/`.
