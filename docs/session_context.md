# Golf Swing Analyzer - Session Context Log

This log summarizes the progress made so far and the current state of the workspace. It is designed to help any coding agent quickly resume the project with full context.

## Current Project State

We reorganized the workspace documentation, creating a unified `docs/` folder, adding [Agents.md](file:///Users/sagar/Documents/ML/golf-analysis/Agents.md), [docs/architecture.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/architecture.md), and [docs/product.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/product.md), and migrating older text files to [docs/backlog.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/backlog.md), [docs/roadmap.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/roadmap.md), and [docs/history.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/history.md).

### 1. Codebase & Components
- **[data_preprocessing.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/data_preprocessing.ipynb)**: A unified notebook to consolidate GolfDB and UCF non-golf single video CSVs, calculate sliding window features, assign milestones, and output `master_dataset.csv`.
- **[xgb_binary_detector.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/xgb_binary_detector.ipynb)**: Our official gatekeeper binary model trained using XGBoost (99.0% test accuracy).
- **[lstm_keras.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/lstm_keras.ipynb)**: Our main multiclass milestone locator trained using a Bidirectional LSTM in TensorFlow/Keras, operating directly on base coordinates (no windowing required) and achieving an overall MAE of 4.65 frames.
- **[xgb_model.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/xgb_model.ipynb)**: Baseline multiclass phase locator using XGBoost (8.41 frames MAE).
- **[data_processor.py](file:///Users/sagar/Documents/ML/golf-analysis/src/data_processor.py)**: Core landmark extractor.

### 2. Workspace Cleanups
- Installed `seaborn` plotting library.
- Removed deprecated `.pt` PyTorch models from `models/`.
- Created [src/alignment.py](file:///Users/sagar/Documents/ML/golf-analysis/src/alignment.py) for Dynamic Programming monotonic milestone alignment.
- Implemented and validated the end-to-end inference script [analyze_swing.py](file:///Users/sagar/Documents/ML/golf-analysis/analyze_swing.py) with plot saving and skeletal video overlay.
- Added a batch validation utility [scratch/verify_pipeline.py](file:///Users/sagar/Documents/ML/golf-analysis/scratch/verify_pipeline.py) achieving 2.95 frames overall MAE.

---

## Technical Details Reference

### Coordinate Prefixes in Output DataFrames
- `raw_`: Raw pixel coordinates directly from MediaPipe.
- `smooth_`: Coordinates after Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
- `norm_`: Coordinates centered relative to `mid_hip` and scaled by `torso_scale`.

---

## Action Plan for Next Session

1. **Biomechanical Analysis**: Implement calculation of key angles (such as Spine Tilt, Knee Flex, and Lead Arm Flex) at setup and milestone frames.
2. **Coaching Drills Integration**: Map calculated biomechanical issues to coaching advice (e.g., if Lead Arm Flex is bent, suggest relevant training drills).
3. **Pro Matchmaker & Sync Video**: Plan the side-by-side video synchronization and warping module to match the user's swing tempo to a professional swing.

