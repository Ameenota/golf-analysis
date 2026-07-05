# Project History Log

- Added custom agent configuration and environment rules under `.agents/` including `environment.md` (requiring `uv` for python environments) and `ledger.md` (requiring `HISTORY.md` updates before commits).
- Verified initial workspace structure and GolfDB inputs.
- Extracted and modularized joint rendering coordinates and drawing helper to `src/visualizer.py`, cleaning up duplicate code in scripts and notebooks.
- Implemented GolfDB parsing, video frame landmarks extraction, and labeling in `notebooks/data_pipeline_exploration.ipynb`.
- Compiled the reusable sliding window feature engineering logic into `src/feature_engineer.py` to be shared between training and inference pipelines.
- Integrated a programmatic label validation and filtering step in `notebooks/data_pipeline_exploration.ipynb` to discard chronologically corrupted metadata.
- Updated the horizontal visualizer slider cell to read directly from the processed master CSV and added a raw video frame inspector at the end of the notebook.
- Created `TODO.md` to track future feature engineering and model tuning experiments.
- Added CSV checkpointing/caching to the video processing loop in `notebooks/data_pipeline_exploration.ipynb` for robust resume capability.
- Integrated `tqdm` progress bar and added it to project dependencies in `pyproject.toml` using `uv` to keep notebook output logs clean.


