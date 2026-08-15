# Code provenance

The repository separates dissertation-specific work from reference and
third-party material.

- The supervisor-provided `MultioutputGPPlayground.ipynb` informed the initial
  multi-output GP construction. The original notebook is excluded from this
  repository.
- The numbered notebooks implement the sea-ice forward equations, synthetic
  and real-data experiments, validation and visualisation used in the MSc
  dissertation.
- `scripts/run_drf_blocked_cv.py` is a dissertation-specific, physics-informed
  experiment wrapper around the official external
  [DeepRandomFeatures](https://github.com/totony4real/DeepRandomFeatures)
  package. That package remains a separate dependency; its source and example
  datasets are not vendored here.
- Data-extraction scripts encode the project-specific selection and coordinate
  processing steps but do not include server credentials or restricted data.

The dissertation and repository should cite the relevant methodological and
data-product publications. Before making the repository public, confirm the
preferred citation and licence for the supervisor notebook and the DRF package.

The upstream DRF repository identifies the associated work as Weibin Chen,
Azhir Mahmood, Michel Tsamados and So Takao (2024), *Deep Random Features for
Scalable Interpolation of Spatiotemporal Data*, arXiv:2412.11350. The upstream
repository did not display a release or licence file when this project was
archived, which is an additional reason not to copy its implementation into
this coursework repository.
