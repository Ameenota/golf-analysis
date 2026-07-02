# Golf Swing Analyzer - Session Context Log

This log summarizes the progress made so far and the current state of the workspace. It is designed to help any coding agent quickly resume the project with full context.

## Current Project State (Day 1 Complete)
We are building a biomechanical golf swing analysis pipeline based on the [roadmap.md](file:///Users/sagar/Documents/ML/golf-analysis/roadmap.md).

### 1. Codebase & Components
- **[data_processor.py](file:///Users/sagar/Documents/ML/golf-analysis/src/data_processor.py)**: Contains the `GolfVideoProcessor` class, which uses MediaPipe Tasks Pose Landmarker (downloading `pose_landmarker_heavy.task` if missing) to extract coordinate landmarks.
  - It applies a Savitzky-Golay filter to smooth coordinate signals (removing MediaPipe pose detection jitter).
  - It centers coordinates relative to the mid-hip point.
  - It scales coordinates by the player's torso length (average of the first 5 frames, representing the Address setup) to achieve resolution-independence.
- **[data_exploration.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/data_exploration.ipynb)**: Active prototype notebook. Contains the visual overlay script that reads processed DataFrame rows, draws bones (`POSE_CONNECTIONS`) and joints (`POSE_JOINTS`) using OpenCV (`cv2`), and outputs:
  - `skeleton_overlay.mp4`: Skeletal wireframe overlaid on raw video.
  - `skeleton_blank.mp4`: White stick figure on a black background.

### 2. Git Configuration
- Repository initialized.
- `.gitignore` configured to ignore large assets and system clutter:
  - `data/` (raw/processed videos and datasets)
  - `models/` (large MediaPipe and XGBoost model artifacts)
  - `.DS_Store` and python/virtual environment files.

---

## Technical Details Reference

### Coordinate Prefixes in Output DataFrames
- `raw_`: Raw pixel coordinates directly from MediaPipe.
- `smooth_`: Coordinates after Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
- `norm_`: Coordinates centered relative to `mid_hip` and scaled by `torso_scale`.

---

## Action Plan for Next Session (Day 2)
The next step is **Day 2: GolfDB Parsing & Sliding Window Engineering** as detailed in [roadmap.md](file:///Users/sagar/Documents/ML/golf-analysis/roadmap.md#L102-L123):
1. Create `src/dataset_builder.py`.
2. Parse the GolfDB annotations file (`GolfDB.mat`) to extract the 8 milestone frame indices for each golf video.
3. Implement a sliding window of $\pm 5$ frames (appending coords of wrists, elbows, shoulders, hips from frames $T-5$ and $T+5$ to frame $T$'s row).
4. Save the combined dataset as a master `training_dataset.csv` in `data/processed/`.
