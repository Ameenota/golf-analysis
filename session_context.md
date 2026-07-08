# Golf Swing Analyzer - Session Context Log

This log summarizes the progress made so far and the current state of the workspace. It is designed to help any coding agent quickly resume the project with full context.

## Current Project State

We are building a biomechanical golf swing analysis pipeline. 

### 1. Codebase & Components
- **[data_processor.py](file:///Users/sagar/Documents/ML/golf-analysis/src/data_processor.py)**: Contains the `GolfVideoProcessor` class, which uses MediaPipe Pose Landmarker to extract, smooth, center, and normalize coordinate landmarks. Robustified to handle corrupted/empty videos gracefully.
- **[feature_engineer.py](file:///Users/sagar/Documents/ML/golf-analysis/src/feature_engineer.py)**: Handles the sliding window of $\pm 5$ frames, appending joint coordinates of wrists, elbows, shoulders, and hips.
- **[train_classifier.py](file:///Users/sagar/Documents/ML/golf-analysis/src/train_classifier.py)**: Contains functions for training the XGBoost milestone classifier and evaluating sequence predictions using a chronological post-processing peak-finding logic.
- **[xgb_model.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/xgb_model.ipynb)**: Trains the XGBClassifier using a randomized group-split of the dataset by `video_id` into Train (80%), Val (10%), and Test (10%).
- **[isolation_forest_gatekeeper.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/isolation_forest_gatekeeper.ipynb)**: Explores training an unsupervised `sklearn.ensemble.IsolationForest` on coordinate features to detect golf swings and reject non-golf videos.
- **[precompute_ucf_features.py](file:///Users/sagar/Documents/ML/golf-analysis/scratch/precompute_ucf_features.py)**: Precomputes features on the UCF50 dataset (all GolfSwing videos + 1500 sampled non-golf videos) to evaluate the Isolation Forest model.

### 2. Git Configuration
- Active rule `auto-ledger` in `.agents/rules/ledger.md` requires summarizing changes in `HISTORY.md` before committing.
- Active rule `session-management` in `.agents/rules/session_management.md` requires reading and updating `session_context.md` and `TODO.md`.

---

## Technical Details Reference

### Coordinate Prefixes in Output DataFrames
- `raw_`: Raw pixel coordinates directly from MediaPipe.
- `smooth_`: Coordinates after Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
- `norm_`: Coordinates centered relative to `mid_hip` and scaled by `torso_scale`.

---

## Action Plan for Next Session

The next step is evaluating and tuning the **Isolation Forest Gatekeeper** on the precomputed UCF features:
1. **Precompute Features**: Complete running the [precompute_ucf_features.py](file:///Users/sagar/Documents/ML/golf-analysis/scratch/precompute_ucf_features.py) script on the UCF dataset to output `<video_id>.csv` files into `data/processed/ucf_test_features/`.
2. **Evaluate Gatekeeper**: Run the Isolation Forest model on the generated UCF features to classify videos as Golf vs. Non-Golf.
3. **Calculate Metrics**: Compute precision, recall, F1-score, and confusion matrix to measure detection accuracy.
4. **Hyperparameter Tuning**: Experiment with Isolation Forest hyperparameters (`contamination`, `n_estimators`, `max_features`) to optimize the precision/recall trade-off.
