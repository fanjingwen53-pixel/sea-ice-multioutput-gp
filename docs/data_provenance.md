# Data provenance and access

The real-data experiments use collocated CryoSat-2 radar-freeboard and
ICESat-2 laser-freeboard tables prepared in the UCL Centre for Polar
Observation and Modelling project environment. The external consistency check
uses a project-held sea-ice thickness and snow-depth product after correcting
the documented projection mismatch described in the dissertation.

This repository does not establish a right to redistribute those source or
processed data. It therefore contains no raw products or observation-level
extracts. Users with independent authorisation can provide the server project
root to the extraction scripts and write the resulting CSV files to the local,
git-ignored `data/processed/` directory.

The `results/` directory contains only compact summary statistics and
prediction-only grids needed to reproduce the reported figures. If the
supervisor requires a completely private submission repository, the omitted
tables can be supplied through the institutionally approved channel.
