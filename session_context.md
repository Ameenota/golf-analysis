# Golf Swing Analyzer - Session Context Log

This log summarizes the progress made so far and the current state of the workspace. It is designed to help any coding agent quickly resume the project with full context.

## Current Project State

We unified the dataset preprocessing pipeline, simplified the training notebooks to load from a single consolidated master dataset, and prepared the workspace for cleanup.

### 1. Codebase & Components
- **[data_preprocessing.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/data_preprocessing.ipynb)**: A unified notebook to consolidate GolfDB and UCF non-golf single video CSVs, calculate sliding window features, assign milestones, and output `master_dataset.csv`.
- **[xgb_binary_detector.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/xgb_binary_detector.ipynb)**: Our official gatekeeper binary model trained using XGBoost (99.0% test accuracy).
- **[lstm_model.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/lstm_model.ipynb)**: Our main multiclass milestone locator trained using a Bidirectional LSTM in Fastai, operating directly on base coordinates (no windowing required) and achieving an overall MAE of 3.52 frames.
- **[xgb_model.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/xgb_model.ipynb)**: Baseline multiclass phase locator using XGBoost (8.41 frames MAE).
- **[data_processor.py](file:///Users/sagar/Documents/ML/golf-analysis/src/data_processor.py)**: Core landmark extractor.

### 2. Workspace Cleanups
- Verified modifications to both training notebooks.
- Documented manual folder creation, dataset migration, and obsolete file deletion commands in `walkthrough.md`.

---

## Technical Details Reference

### Coordinate Prefixes in Output DataFrames
- `raw_`: Raw pixel coordinates directly from MediaPipe.
- `smooth_`: Coordinates after Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
- `norm_`: Coordinates centered relative to `mid_hip` and scaled by `torso_scale`.

---

## Action Plan for Next Session

1. **Inference Pipeline Integration**: Update the swing analyzer inference wrapper to run the high-accuracy XGBoost binary detector first, and if validated, invoke the Fastai LSTM milestone locator to extract swing milestones.
2. **Post-processing Validation**: Ensure predicted milestones from the LSTM conform to chronological order, integrating Viterbi-like or confidence threshold rules.
3. **Pipeline Verification**: Test the complete end-to-end pipeline on new test videos.
