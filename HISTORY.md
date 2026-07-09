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
- Restored standard video processing print statements using `tqdm.write()` to preserve output history logs above the progress bar.
- Removed tqdm dependency from pyproject.toml and uv.lock, switching to plain prints in data_pipeline_exploration.ipynb to avoid dependency bloat.
- Generated and saved the complete master_training_dataset.csv with all 1399 cached video features and sliding window feature engineering.
- Created notebooks/xgb_model.ipynb and implemented a randomized group-split of the dataset by video_id into Train (80%), Val (10%), and Test (10%) sets.
- Added scikit-learn dependency and updated the .gitignore to ignore scratch/ directory.
- Configured sample weights for class imbalance and trained an XGBClassifier, evaluating predictions using a chronological post-processing peak-finding logic which reduced max sequence error from 724 frames to 84 frames.
- Exported the finalized training and inference functions to src/train_classifier.py, and updated the notebook to import from it to avoid code duplication.
- Created `notebooks/isolation_forest_gatekeeper.ipynb` documenting and demonstrating the unsupervised Isolation Forest gatekeeper model, threshold tuning, and video-level classification rules.
- Robustified `src/data_processor.py` to handle empty or undecodable video files gracefully, returning a clean ValueError and preventing internal Savitzky-Golay filtering crashes on NaN arrays.
- Created `.agents/rules/session_management.md` defining the workspace session context tracking protocol.
- Updated `session_context.md` to reflect current model components and map the UCF precomputation/evaluation next steps.
- Checked off the completed Isolation Forest gatekeeper prototyping task in `TODO.md`.
- Updated `src/data_processor.py` to calculate `num_people`, `torso_angle_deg`, and `is_upright` columns on the fly.
- Created `notebooks/xgb_gatekeeper_evaluation.ipynb` implementing and evaluating the unified XGBoost Milestone Confidence + Physical Rules gatekeeper on 1,642 UCF videos.
- Optimized the gatekeeper evaluation script to skip precomputed CSVs and copied false negatives/positives to a diagnostic folder.
- Added flexible posture window validation and golfer centroid tracking tasks to TODO.md.
- Added the dedicated binary XGBoost detector task to TODO.md specifying the baseline to beat (77% Recall, 47% Precision).



