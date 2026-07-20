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
- Logged the view-dependent DTL handedness check task to `docs/backlog.md`.
- Implemented `src/kinematic_features.py` adding centered velocity differences and summary motion energy metrics across 12 key MediaPipe landmarks.
- Developed and executed `scratch/train_kinematic_ablations.py`, evaluating 7 model configurations (A through G) across 140 held-out test videos.
- Experiment E (108 features: Coords + 12 Landmark Velocities + Motion Summaries) achieved **2.66 frames MAE** (a **34.52% improvement** over the 4.07 baseline) and confirmed its status as the winning production standard.
- Confirmed Experiment E superiority over feature subset combinations: Exp F (Upper-body vels + Summaries [90 features]) achieved 3.06 frames MAE (losing to E due to missing lower-body posture signals), while Exp G (Wrist vels + Summaries [78 features]) regressed to 5.75 frames MAE.
- Promoted Experiment E weights to `models/lstm_phase_model.keras` and saved winning schema artifacts `models/kinematic_schema.json` and `models/kinematic_config.json`.
- Integrated dynamic kinematic feature extraction into `analyze_swing.py` for single-video CLI inference.
- Created benchmark dataset script `scripts/curate_benchmark_dataset.py` assembling 16 curated GolfDB pro videos and 7 user test videos into `data/benchmark/` with `manifest.json`.
- Implemented 98.1% accurate hybrid 2D/3D camera view auto-detection (`detect_camera_view()`) in [analyze_swing.py](file:///Users/sagar/Documents/ML/golf-analysis/analyze_swing.py) making `--view auto` the new default CLI parameter.
- Corrected Down-The-Line (DTL) handedness detection geometry in [coaching_engine.py](file:///Users/sagar/Documents/ML/golf-analysis/src/coaching_engine.py), eliminating false left-handed misclassifications and false bent lead arm warnings on DTL swings (e.g. `IMG_6826.MOV`).
- Successfully verified end-to-end pipeline execution on DTL test swing (`IMG_6826.MOV`), correctly auto-detecting `DOWN-THE-LINE` view, `right` handedness, measuring 170.97° lead arm flex (passing), and matching with DTL pro Sandra Gal.
- Fixed pro video feature extraction in [analyze_swing.py](file:///Users/sagar/Documents/ML/golf-analysis/analyze_swing.py) to dynamically extract kinematic features (Experiment E, 108 features) and corrected pro heuristic milestone references.
- Executed end-to-end analysis pipeline across user videos in `data/r-videos/`, verifying 100% gatekeeper pass rate and compiling synchronized dashboard videos into `output/`.
- Implemented foot weight transfer & heel lift biomechanical rules in `src/coaching_engine.py`.
- Logged DTL Handedness Misclassification Bug (e.g. `IMG_1103.mov`) into `docs/backlog.md`.
- Built Hugging Face asset management system & executed 100% upload:
  - Created [scripts/upload_assets_to_hf.py](file:///Users/sagar/Documents/ML/golf-analysis/scripts/upload_assets_to_hf.py) with `HF_HUB_DISABLE_XET=1` and pushed ML models (`sagsan/golf-swing-analyzer-models`) and pro/sample dataset assets (`sagsan/golf-swing-analyzer-dataset`) to Hugging Face Hub.
  - Created [src/utils/hf_downloader.py](file:///Users/sagar/Documents/ML/golf-analysis/src/utils/hf_downloader.py) for local and cloud downloading of models into `models/` and benchmark/preset assets into `data/`.
  - Packaged 3 pre-computed demo presets (`IMG_0018.MOV`, `IMG_6826.MOV`, `kin-1.mp4`) into `data/sample_presets/` with `scripts/prepare_sample_presets.py`.
- Built and refined segregated Streamlit dashboard application in `streamlit_app/`:
  - Configured [.streamlit/config.toml](file:///Users/sagar/Documents/ML/golf-analysis/.streamlit/config.toml) with 50MB client upload size limit and dark theme styling.
  - Set application-level upload validation to accept videos from 0.1 MB through 50 MB, accommodating compact GolfDB clips.
  - Fixed uploaded-video analysis output-path construction by importing `pathlib.Path` in `analyze_swing.py`.
  - Ignored the generated Matplotlib `.mpl_config/` runtime cache directory.
  - Renamed demo preset selectors in `streamlit_app/app.py` to Preset A (`IMG_0018`), Preset B (`IMG_6826`), and Preset C (`kin-1`), with realistic ~4.6s status progress animation.
  - Encoded dashboard output videos using ImageIO/FFmpeg `libx264` with the H.264 `avc1` tag and `yuv420p` pixel format for native HTML5 playback.
  - Restructured Tab 1 layout to place the 4-column Biomechanical Scorecard directly below the full-width side-by-side video clip.
  - Enhanced Tab 3 Plotly kinematic charts in [streamlit_app/charts.py](file:///Users/sagar/Documents/ML/golf-analysis/streamlit_app/charts.py) with auto-zoomed $X$-axis around the active swing range (`[Address - 15, Finish + 20]`), staggered milestone vertical text annotations, and a **Matched Pro Trajectory Overlay (Dashed Gold Line `🟡 Matched Pro (Name)`)** compared against the **User Swing (Solid Green Line `🟢 User Swing`)** with interactive legend keys.

- Deployed the public application to Streamlit Community Cloud from the GitHub `main` branch:
  - The application entry point is `streamlit_app/app.py`; Python and system dependencies are declared in `requirements.txt` and `packages.txt`.
  - Hugging Face Hub remains the external asset store at `sagsan/golf-swing-analyzer-models` and `sagsan/golf-swing-analyzer-dataset`.
  - Hugging Face Spaces hosting was attempted and abandoned after the deployment failed. The obsolete Spaces deployment script and README metadata were removed.
  - Presets now synchronize against Hugging Face Hub on each cold start, while retaining existing local copies if refresh fails.
- Fixed the custom-upload false rejection reproduced with `data/r-videos/kin-1.mp4`:
  - Confirmed the XGBoost gatekeeper accepts the upload-style temporary copy with a score of `0.9868`.
  - Loaded the BiLSTM with `compile=False` so clean deployments do not need the notebook-only custom training loss.
  - Restored the production Experiment E 108-feature inference builder using the saved schema/config artifacts.
  - Changed Streamlit error handling so internal analysis failures display their actual error instead of being mislabeled as `Gatekeeper Score: 0.00`.
  - Preserved `.mov`/`.mp4` upload suffixes and made generated output paths extension-independent.
  - Verified focused end-to-end inference shapes on `kin-1`: `(92, 108)` input and `(1, 92, 9)` output; all 8 kinematic unit tests pass.
- Added repository-wide, case-insensitive `.gitignore` rules for common video formats (`mp4`, `mov`, `avi`, `mkv`, `webm`, `m4v`, `mpeg`, `mpg`, `wmv`, and `flv`) and confirmed Git currently tracks no video files.

---

## Action Plan for Next Session

1. **Custom Upload Redeployment Verification**:
   - Redeploy the Streamlit app and upload `kin-1.mp4`; confirm analysis completes and the synchronized dashboard is playable/downloadable.
   - Profile the silent dashboard stitching stage if cloud rendering remains unusually slow.
2. **DTL Handedness Orientation Bugfix**:
   - Resolve the shoulder/hip X-coordinate inversion issue for Down-The-Line right-handed swings (`IMG_1103.mov`).
3. **Golfer Centroid Tracking & Occlusion Handling**:
   - Implement centroid lock-on to distinguish primary golfer from background bystanders.
