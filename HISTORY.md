# Project History Log

- Added custom agent configuration and environment rules under `.agents/` including `environment.md` (requiring `uv` for python environments) and `ledger.md` (requiring `HISTORY.md` updates before commits).
- Verified initial workspace structure and GolfDB inputs.
- Extracted and modularized joint rendering coordinates and drawing helper to `src/visualizer.py`, cleaning up duplicate code in scripts and notebooks.
- Implemented GolfDB parsing, video frame landmarks extraction, and labeling in `notebooks/data_pipeline_exploration.ipynb`.
- Compiled the reusable sliding window feature engineering logic into `src/feature_engineer.py` to be shared between training and inference pipelines.
