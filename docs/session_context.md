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
- Implemented and validated the end-to-end inference script [analyze_swing.py](file:///Users/sagar/Documents/ML/golf-analysis/analyze_swing.py) with plot saving, skeletal video overlay, and file JSON caching.
- Added a batch validation utility [scratch/verify_pipeline.py](file:///Users/sagar/Documents/ML/golf-analysis/scratch/verify_pipeline.py) achieving 2.95 frames overall MAE.
- Processed 6 unseen QuickTime MOV videos in `data/r-videos/`, producing local JSON outputs and skeletal overlay `.mov` annotated clips using `scratch/process_r_videos.py`.
- Finalized design specifications for the Biomechanical Rules Engine and Pro Matchmaker during a `/grill-me` session.
- Created scratch script `/Users/sagar/.gemini/antigravity/brain/98a4998a-1fd4-4851-9ffc-b4d396267909/scratch/calculate_pro_ratios.py` to extract actual arm-to-torso ratios for 5 iconic professional players (Tiger Woods: 0.862, Sandra Gal: 1.050, Steve Stricker: 1.092, Greg Norman: 1.193, Cristie Kerr: 1.198).
- Drafted [implementation_plan.md](file:///Users/sagar/.gemini/antigravity/brain/98a4998a-1fd4-4851-9ffc-b4d396267909/implementation_plan.md) mapping out development paths for `src/coaching_engine.py` and `analyze_swing.py`.
- Conducted geometric coordinate analysis on 1,399 preprocessed GolfDB videos, formulating a 98.1% accurate DTL vs. FO camera view auto-detection logic (using normalized shoulder width and Z-depth difference at Address).
- Changed the default camera view parameter in `analyze_swing.py` to `down-the-line` and logged the auto-detection task to the prioritized project backlog (`docs/backlog.md`).

---

## Technical Details Reference

### Coordinate Prefixes in Output DataFrames
- `raw_`: Raw pixel coordinates directly from MediaPipe.
- `smooth_`: Coordinates after Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
- `norm_`: Coordinates centered relative to `mid_hip` and scaled by `torso_scale`.

---

## Action Plan for Next Session

1. **Synchronized Video Stitcher (Day 5)**: Develop `src/visual_stitcher.py` to warp the timelines of the user video and the matched pro video frame-by-frame based on the 8 detected milestones.
2. **Biomechanical Overlays (Day 5)**: Render the user's measured metrics (e.g. lead arm angle, hip sway) and skeletal lines dynamically on both frames of the side-by-side video.
3. **Visual Overlay Refinement**: Implement the persistent milestone log in the corner of the output video to track milestones as they are hit during slow-motion playback.


