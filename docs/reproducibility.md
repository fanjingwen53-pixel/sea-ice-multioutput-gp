# Reproducibility guide

Run commands from the repository root. The numbered notebooks follow the same
order as the analysis below.

1. Run `scripts/run_synthetic_sensitivity.py` to test recovery under changes
   in sensor noise, observation density and penetration coefficients.
2. Use the authorised extraction scripts under `scripts/data_preparation/` or
   place approved March transects in `data/processed/`. Run
   `scripts/run_linear_gp_blocked_cv.py`, followed by
   `scripts/run_nested_gp_blocked_cv.py`, for spatially blocked GP validation.
3. Clone and install the official
   [DeepRandomFeatures](https://github.com/totony4real/DeepRandomFeatures)
   repository separately (or set `DRF_SRC` to its `src` directory), then run
   `scripts/run_drf_blocked_cv.py --input ... --output ...` for the DRF
   comparison. Its published example datasets are not inputs to this sea-ice
   retrieval experiment.
4. Run `scripts/run_positive_gp_noise_sensitivity.py` to select the sensor-noise
   configuration on March 8 and freeze it for the March 29 replication.
5. Run `scripts/run_positive_gp_2d.py` first with the March 8 data and then with
   the March 29 fixed-region data, changing the output filenames accordingly.
6. Run `scripts/analyse_external_reference.py` after generating the required
   out-of-fold prediction tables and authorised external-reference extract.
7. Run `scripts/make_result_figures.py` to regenerate the dissertation figures
   under `figures/`.

The notebooks are executed records of these stages. For their relative paths
to resolve, start Jupyter from the `notebooks/` directory. Stochastic scripts
use fixed seeds recorded in the source and result summaries. TensorFlow/GPflow
optimisation can nevertheless show small platform-dependent numerical
differences.
