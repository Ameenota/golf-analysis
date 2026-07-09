# Golf Swing Analyzer - Session Context Log

This log summarizes the progress made so far and the current state of the workspace. It is designed to help any coding agent quickly resume the project with full context.

## Current Project State

We are building a biomechanical golf swing analysis pipeline. We evaluated the unified **XGBoost Milestone Confidence + Physical Rules** gatekeeper on the 1,642 UCF video dataset, showing significantly higher performance (AUC PR of 0.7904, Max F1 of 0.8115) than the unsupervised Isolation Forest baseline (AUC PR 0.6802, F1 0.6568). We also optimized the evaluation notebook to skip precomputed CSVs.

### 1. Codebase & Components
- **[data_processor.py](file:///Users/sagar/Documents/ML/golf-analysis/src/data_processor.py)**: Contains the `GolfVideoProcessor` class, which uses MediaPipe Pose Landmarker to extract, smooth, center, and normalize coordinate landmarks. Robustified to handle corrupted/empty videos gracefully.
- **[feature_engineer.py](file:///Users/sagar/Documents/ML/golf-analysis/src/feature_engineer.py)**: Handles the sliding window of $\pm 5$ frames, appending joint coordinates of wrists, elbows, shoulders, and hips.
- **[train_classifier.py](file:///Users/sagar/Documents/ML/golf-analysis/src/train_classifier.py)**: Contains functions for training the XGBoost milestone classifier and evaluating sequence predictions using a chronological post-processing peak-finding logic.
- **[xgb_model.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/xgb_model.ipynb)**: Trains the XGBClassifier using a randomized group-split of the dataset by `video_id` into Train (80%), Val (10%), and Test (10%).
- **[xgb_gatekeeper_evaluation.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/xgb_gatekeeper_evaluation.ipynb)**: Implements and evaluates the gatekeeper validation checks (human count, verticality, downswing timing, milestone confidence) on 1,642 UCF videos. Skip logic added for existing CSVs.

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

1. **Unified Pipeline Integration**: Integrate the `check_golf_swing_video` validation function directly into the main video prediction pipeline wrapper.
2. **Biomechanical Rules Engine**: Begin implementing calculations for additional swing metrics (e.g. Spine Tilt angle at address, Knee Flex tracking during downswing, and Club Shaft angle).
3. **Occlusion & NaN Handling**: Enhance preprocessing to handle sequences with partial occlusions or high NaN density by testing spline interpolation.

