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
- Enhanced the inference video renderer with automatic crop clipping around swing boundaries (Address - 5 to Finish + 5), adjustable slow-motion playback (`--speed`), a top-right biomechanical scorecard metrics panel, and a bottom debug info bar showing view, handedness, pro matchup ratios, file info, and frames.
- Created [src/visual_stitcher.py](file:///Users/sagar/Documents/ML/golf-analysis/src/visual_stitcher.py) and integrated it into `analyze_swing.py` to compile a side-by-side synchronized dashboard video (Alternative B layout) containing the user, the matched pro, a dynamic milestone tracker log, and metrics scoreboard.
- Implemented a follow-through spine angle posture check (DTL only, minimum 20 degrees) in `src/coaching_engine.py` to detect early extension, and added it to the visual scorecard.
- Improved layout contrast in `src/visual_stitcher.py` by boosting the color of inactive metric rows and status badges to a highly legible light gray (170, 170, 170).
- Decoupled detection from analysis by refactoring the binary validation logic into a new module [detector.py](file:///Users/sagar/Documents/ML/golf-analysis/src/detector.py).
- Implemented a 2.0-second rolling window validation score aggregation, resolving the dilution issue on long videos.
- Configured a video duration validation check (defaulting to a maximum of 60.0 seconds) that runs immediately at the beginning of the pipeline using OpenCV metadata to reject long files before extracting landmarks.
- Updated the default gatekeeper validation threshold in [analyze_swing.py](file:///Users/sagar/Documents/ML/golf-analysis/analyze_swing.py) to `0.60` based on a mathematical Precision-Recall sweep after fixing the `min_periods=1` early-frame bug (yielding 97.0% accuracy, 99.3% precision, 94.3% recall, and successfully validating `IMG_0018.MOV` which scored `0.829`).
- Created and executed a large-scale evaluation test [evaluate_gatekeeper_rolling.py](file:///Users/sagar/Documents/ML/golf-analysis/scratch/evaluate_gatekeeper_rolling.py) to verify rolling window accuracy on the official 305-video train/test split.
- Ran regression testing [verify_pipeline.py](file:///Users/sagar/Documents/ML/golf-analysis/scratch/verify_pipeline.py) achieving an overall MAE of 5.08 frames (below the 6.0 limit) with 100% video validation success.
- Validated and batch-processed all 7 manual holdout videos in `data/r-videos/` (including the newly added `kin-1.mp4`), achieving a **100% pass rate** and successfully compiling comparison overlay videos and biomechanical markdown reports for each.
- Resolved a critical milestone naming mismatch between model classes and application display labels:
  - Corrected `MILESTONE_NAMES` to: Address (Class 1), Toe-Up (Class 2), Mid-Backswing (Class 3), Top of Backswing (Class 4), Mid-Downswing (Class 5), Impact (Class 6), Mid-Follow-Through (Class 7), Finish (Class 8).
  - Fixed physical heuristics in `analyze_swing.py` to search correct chronological windows (Address to Top for Top; Top to Impact for Impact).
  - Updated `src/coaching_engine.py` to query correct aligned keys (Class 4 for Top metrics, Class 6 for Impact metrics, Class 7 for Follow-Through metrics).
  - Confirmed the fix by running `scratch/verify_pipeline.py` which evaluated with an overall MAE of **3.26 frames**, and regenerated all 7 holdout reports/videos.
  - Logged the DTL handedness detection view-inversion bug to `docs/backlog.md`.


---

## Technical Details Reference

### Coordinate Prefixes in Output DataFrames
- `raw_`: Raw pixel coordinates directly from MediaPipe.
- `smooth_`: Coordinates after Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
- `norm_`: Coordinates centered relative to `mid_hip` and scaled by `torso_scale`.

---

## Action Plan for Next Session

1. **Integration Test Suite Setup**:
   - Write pytest unit tests using the Golden Coordinates (Option A) to verify the logic pipeline (XGBoost validation, Keras LSTM predictions, DP alignment, coaching metrics, and pro matchmaking).
   - Add temporary local videos to verify OpenCV decoding and MediaPipe landmarks extraction.
2. **Firebase Storage Asset Downloader**:
   - Determine optimal ffmpeg compression parameters to keep test videos under 200 KB.
   - Collect both golf and non-golf test videos.
   - Setup a Firebase Storage bucket to host custom ML models and media assets.
   - Develop `src/utils/downloader.py` to automatically fetch and cache missing models and media on demand.




