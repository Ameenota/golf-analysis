# Golf Swing Analyzer Project - Future Enhancements & Experiments

This file tracks the project's prioritized backlog, including upcoming tasks, low-priority enhancements, completed milestones, and dropped/archived experiments.

---

## 🔥 High Priority (Next Actions)

### 🌐 Streamlit Community Cloud Deployment
- [x] **Install the MediaPipe OpenGL ES runtime on Streamlit Cloud**
  - **Accomplishment**: Added Debian's `libgles2` system package, which supplies the `libGLESv2.so.2` shared library required by the Linux MediaPipe wheel during custom-upload landmark extraction.
- [x] **Fix custom-upload analysis hanging indefinitely during BiLSTM inference**
  - **Accomplishment**: Stack sampling identified a deadlock in Keras `model.predict()`'s `tf.data` prefetch path caused by TensorFlow/PyArrow Abseil symbol interaction. Replaced it with direct forward inference, applied softmax to the model logits, fixed the 9-class alignment input, and isolated MediaPipe timestamp state per video. Verified the 231 KB `kin-1` fixture completes through H.264 dashboard generation in 7.63 seconds.
- [x] **Preprocess deterministic professional reference videos**
  - **Accomplishment**: Added a manifest-driven, repeatable cache builder producing stem-preserving CSV/JSON artifacts for all 16 pro videos. Artifacts are hash/version validated, atomically written, skipped when current, and loaded at runtime with a live fallback. Cached end-to-end `kin-1` analysis completes in 4.06 seconds versus 7.63 seconds without the cache.
- [x] **Fix valid custom uploads being reported as Gatekeeper Score 0.00**
  - **Accomplishment**: Reproduced with `kin-1.mp4` (actual gatekeeper score `0.9868`) and fixed inference-only Keras loading, restored the production 108-feature kinematic builder, preserved upload extensions/output paths, and separated pipeline errors from genuine gatekeeper rejections in the UI.
- [x] **Deploy the public application to Streamlit Community Cloud**
  - **Accomplishment**: Configured `streamlit_app/app.py` as the public entry point, `requirements.txt` for Python runtime dependencies, and `packages.txt` for FFmpeg/system dependencies. Hugging Face Hub supplies models, benchmark data, and presets but does not host the application.
- [x] **Retire Hugging Face Spaces application hosting**
  - **Outcome**: The Spaces deployment was attempted and failed. Its deployment script and README metadata were removed; historical context remains in `docs/history.md`.
- [x] **Require browser-compatible generated videos**
  - **Accomplishment**: Removed the silent OpenCV `mp4v` fallback, made H.264/yuv420p output mandatory, and added an inline MP4 download fallback beneath the Streamlit video heading.

### 📁 Curated Benchmark Dataset (Pro & Test Videos)
- [x] **Curate 10–20 Pro Videos & 10–20 Sanitized Test Videos**
  - **Idea**: Assemble a standardized, high-quality benchmark test set containing:
    - **16 Pro Videos**: Selected from GolfDB covering DTL & FO views, male & female pros (Tiger Woods, Phil Mickelson, Sandra Gal, Steve Stricker, Greg Norman, etc.), right/left handedness, and varied clubs.
    - **7 Test Videos**: Sanitized user test clips in `data/benchmark/user_videos/`.
  - **Goal**: Created `scripts/curate_benchmark_dataset.py` and output `data/benchmark/manifest.json`.

### 🧹 Technical Debt & Dependency Cleanup
- [x] **Prevent video assets from being committed to Git**
  - **Accomplishment**: Added repository-wide, case-insensitive ignore rules for common video formats and confirmed no video files are currently tracked.
- [x] **Remove PyTorch and Fastai Dependencies**
  - **Idea**: Clean up the codebase to reduce dependency bloat and environment size.
  - **Action**: Deleted obsolete `.pt` PyTorch models from `models/` directory and verified no PyTorch dependencies are present in `pyproject.toml`.

### 🚀 Inference Pipeline Integration & Verification
- [x] **Integrate Inference Pipeline Wrapper**
  - **Idea**: Create the final end-to-end inference wrapper script ([analyze_swing.py](file:///Users/sagar/Documents/ML/golf-analysis/analyze_swing.py) at the root).
  - **Goal**: Successfully integrated landmark smoothing/normalization, XGBoost binary gatekeeper validation, and Keras LSTM milestone locator.
- [x] **Implement Chronological Post-processing Validation**
  - **Idea**: Ensure predicted milestone frames from the LSTM strictly satisfy chronological order ($T_1 < T_2 < \dots < T_8$).
  - **Goal**: Coded a Dynamic Programming monotonic alignment algorithm in [alignment.py](file:///Users/sagar/Documents/ML/golf-analysis/src/alignment.py).
- [x] **End-to-End Pipeline Verification**
  - **Idea**: Test the complete integrated pipeline on unseen and sample videos to ensure correct rejection of non-golf and accurate milestone extraction.
  - **Goal**: Verified on valid/invalid videos and created `scratch/verify_pipeline.py` which passes with a 2.95 frames overall MAE.
- [x] **Bidirectional LSTM Masking Layer & Dynamic Padding Invariance**
  - **Idea**: Add `layers.Masking(mask_value=0.0)` and dynamic input shape `(None, 66)` to prevent backward LSTM state corruption over zero-padded frames.
  - **Goal**: Retrained model, verified 100% mathematical padding invariance (0.00000000 max probability diff across sequence lengths), and improved pipeline MAE from 3.26 to 2.71 frames.

---

## ⚡ Medium Priority

### 📐 Biomechanical Rules Engine (Completed)
- [x] **Additional 2D Swing Metrics**
  - **Idea**: Calculate biomechanical feedback indicators:
    - **Spine Tilt**: Measured lateral tilt at Address/Top and loss of posture (tilt change).
    - **Knee Flex**: Soft knees flex check at Address.
    - **Lead Arm Flex**: Checked straightness (>= 160 degrees) at Top of Backswing.
    - **Lateral Hip Sway**: Slide displacement limit check (Face-On only).
    - **Vertical Head Stability**: Bobbing/dipping vertical displacement check (Face-On only).
    - **Foot Weight Transfer & Heel Lift**: Measured trail heel lift (hanging back check) and lead heel lift (lead foot stability check) at Finish.

### 🧹 Data Quality & Sanitization
- [ ] **Programmatic Label Validation**
  - **Idea**: Implement a utility validation script to scan metadata for:
    - Chronological consistency ($T_1 < T_2 < \dots < T_8$).
    - Relative frame range boundary checks.
    - Physical downswing duration validation (must be between 6 and 12 frames at 30 FPS).
- [ ] **Fix Down-The-Line (DTL) Handedness Detection Bug**
  - **Issue**: Videos like `IMG_1103.mov` are right-handed DTL swings but `detect_handedness()` in `src/coaching_engine.py` evaluates them as `left`-handed due to DTL perspective shoulder/hip X-coordinate inversion.
  - **Action**: Refine 3D Z-depth shoulder orientation heuristic or add explicit golfer body facing direction check for DTL perspective.
- [x] **Auto View Detection (DTL vs. FO)**
  - **Idea**: Dynamically detect if an incoming video is Down-The-Line or Face-On.
  - **Accomplishment**: Implemented 98.1% accurate hybrid 2D/3D heuristic (`detect_camera_view()`) in [analyze_swing.py](file:///Users/sagar/Documents/ML/golf-analysis/analyze_swing.py) with default CLI `--view auto` option.
- [x] **Down-The-Line Handedness Detection Correction**
  - **Issue**: Standard DTL video coordinate space is inverted compared to Face-On (FO).
  - **Accomplishment**: Updated `detect_handedness()` in [coaching_engine.py](file:///Users/sagar/Documents/ML/golf-analysis/src/coaching_engine.py) to inspect camera view perspective and correctly evaluate $x$-coordinate orientation for DTL swings.


### ☁️ Cloud Infrastructure & Integration Testing
- [ ] **On-Demand Test Asset & Model Downloader**
  - **Idea**: Implement an auto-downloader for test videos, coordinate CSVs, and trained custom ML models using Firebase Cloud Storage (GCS) to prepare for Firebase deployment.
  - **Action Items**:
    - Figure out how to compress test videos (using ffmpeg) to their lowest functional resolution and bitrate (e.g., 320x240, 15 FPS) to keep footprint minimal.
    - Gather a representative set of test videos, including valid **golf swing** videos (DTL and FO perspectives) and invalid **non-golf** videos to verify XGBoost gatekeeper rejection logic.
    - Upload trained models (`lstm_phase_model.keras`, `golf_binary_detector.json`) and test videos to a Firebase Cloud Storage bucket.
    - Write a helper script (`src/utils/downloader.py`) to auto-fetch and cache missing models and media assets locally on demand during testing or startup.

---


## 💤 Low Priority / Future Enhancements

### 🏌️ Biomechanical Rules
- [ ] **Add tolerance bands for biomechanical thresholds**
  - **Idea**: Replace binary pass/warning behavior near rule boundaries with a three-level result (pass, borderline, warning) that accounts for 2D pose-estimation and frame-selection noise.
  - **Spine Tilt Candidate**: For DTL address tilt, keep 24° to 42° as the pass range, consider a 2° borderline band on either side, and reserve warnings for values below 22° or above 44°.

### 🧹 Data Quality & Sanitization
- [ ] **Golfer Tracking & Lock-on (Centroid Tracking)**
  - **Idea**: Implement Centroid/Proximity tracking to lock onto the primary golfer in the frame, resolving multi-person scenarios (e.g. coach/bystanders in frame) instead of using a naive multi-person rejection.
- [ ] **Occlusion Handling**
  - **Idea**: Detect frames with high NaN density (MediaPipe tracking failures due to high motion blur or rotation) and interpolate using spline interpolation instead of linear.
- [ ] **Landmark Swap & Leg Crossing Detection (Finish Inversion Handling)**
  - **Idea**: Detect MediaPipe left/right leg label swaps during full rotational follow-through by checking 3D Z-depth ordering and tracking trajectory sign flips ($X_{\text{left}} - X_{\text{right}}$) from Address to Finish.

### 🎨 Visual Overlay Improvements
- [x] **Display source frame numbers on the dashboard video labels**
  - **Accomplishment**: The synchronized output video now labels each frame as `USER SWING (FXXX)` and `PRO (REFERENCE FXXX)` using each video's zero-padded source frame index.
- [ ] **Persistent Milestone Log Overlay (Option B)**
  - **Idea**: Add a permanent list in the corner of the annotated video that updates chronologically as milestones are hit, preventing confusion in slow-motion play.

### ⚡ Performance & Scaling
- [ ] **Parallel Video Processing**
  - **Idea**: Use Python's `ProcessPoolExecutor` to process multiple videos in parallel to utilize the full CPU cores of the M4 chip.

---

## 🛑 Dropped / Archived Experiments

- **Multi-Scale Sliding Window Context**
  - *Reason for dropping*: Obsolete. Our XGBoost binary detector achieves 99% accuracy on frame-level features, and our Keras LSTM milestone locator operates directly on temporal sequences of base coordinates without requiring sliding windows.
- **XGBoost Hyperparameter Tuning**
  - *Reason for dropping*: Unnecessary. Current binary classification accuracy is already at 99.0% on the test set.
- **Class Weight Optimization**
  - *Reason for dropping*: Obsolete. Transitioned from the multiclass XGBoost model to the Keras LSTM network for sequence boundary localization.
- **Flexible Posture Window Validation**
  - *Reason for dropping*: Deprioritized/dropped because the binary gatekeeper handles frame-level detection with high accuracy (99%) without needing strict manual address alignment windows.

---

## ✅ Completed Milestones

- [x] **Dedicated Binary XGBoost Detector (Golf vs. Non-Golf)**
  - **Accomplishment**: Trained the final binary gatekeeper model in [xgb_binary_detector.ipynb](file:///Users/sagar/Documents/ML/golf-analysis/notebooks/xgb_binary_detector.ipynb) using 98 normalized coordinates. Achieved **99.0% test accuracy** on the frame-level classification split.
- [x] **Unified XGBoost + Rules Gatekeeper**
  - **Accomplishment**: Reached AUC PR of 0.7904, Max F1-score of 0.8115, and 91.6% rejection rate of non-golf at target recall threshold of 0.20.
- [x] **Unsupervised Golf Swing Gatekeeper (Isolation Forest)**
  - **Accomplishment**: Tested unsupervised outlier detection on 1,642 UCF videos. Achieved AUC PR of 0.6802 and optimal F1-score of 0.6568.
- [x] **Biomechanical Coaching Rules Engine & Pro Matchmaker**
  - **Accomplishment**: Built a view-dependent biomechanical evaluation engine and integrated it into the inference pipeline. Added auto-handedness detection, a pro matchmaker against precalculated profiles, and automated Markdown coaching reports with comparison tables and drills.
- [x] **Clipped Slow-Motion Swing Annotator & Video Scorecard**
- [x] **Synthetic Kinematic Features Ablation Study & Model Promotion**
  - **Accomplishment**: Created `src/kinematic_features.py` (centered velocity differences & motion energy metrics) and ran a 5-experiment ablation study (`scratch/train_kinematic_ablations.py`). Experiment E (108 features) achieved **2.61 frames MAE** (**14.13% improvement** over 3.04 baseline), passed all promotion criteria (7/8 milestones improved), and was promoted to `models/lstm_phase_model.keras`. Integrated into `analyze_swing.py`, achieving **2.45 frames overall MAE** on batch pipeline verification.
- [x] **Decoupled Rolling Window Gatekeeper & Video Duration Validation**
  - **Accomplishment**: Decoupled the XGBoost binary validator from [analyze_swing.py](file:///Users/sagar/Documents/ML/golf-analysis/analyze_swing.py) into the modular class `GolfSwingDetector` in [detector.py](file:///Users/sagar/Documents/ML/golf-analysis/src/detector.py). Implemented 2.0-second rolling window validation (eliminating long-video dilution bug) and configured a 1-minute early duration gatekeeper check. Verified with a new 305-video comparative evaluation script achieving 97.4% overall accuracy and 98.4% recall.
