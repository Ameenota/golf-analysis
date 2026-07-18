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
- Standardized project documentation under `docs/` and defined agent boundaries in [Agents.md](file:///Users/sagar/Documents/ML/golf-analysis/Agents.md).
- Switched python target to 3.13 for TensorFlow compatibility and set up Keras pipeline.
- Verified Keras LSTM model training and peak-finding evaluation.

---

## Technical Details Reference

### Coordinate Prefixes in Output DataFrames
- `raw_`: Raw pixel coordinates directly from MediaPipe.
- `smooth_`: Coordinates after Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
- `norm_`: Coordinates centered relative to `mid_hip` and scaled by `torso_scale`.

---

## Action Plan for Next Session

1. **Clean up Technical Debt**: Remove PyTorch/Fastai dependencies from `pyproject.toml` and delete the old PyTorch notebook `lstm_model.ipynb`.
2. **Inference Pipeline Integration**: Update the swing analyzer inference wrapper to run the high-accuracy XGBoost binary detector first, and if validated, invoke the TensorFlow/Keras LSTM milestone locator to extract swing milestones.
3. **Post-processing Validation**: Ensure predicted milestones from the LSTM conform to chronological order, integrating Viterbi-like or confidence threshold rules.
4. **Pipeline Verification**: Test the complete end-to-end pipeline on new test videos.

