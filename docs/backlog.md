# Golf Swing Analyzer Project - Future Enhancements & Experiments

This file tracks the project's prioritized backlog, including upcoming tasks, low-priority enhancements, completed milestones, and dropped/archived experiments.

---

## 🔥 High Priority (Next Actions)

### 🧹 Technical Debt & Dependency Cleanup
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

### 🧹 Data Quality & Sanitization
- [ ] **Programmatic Label Validation**
  - **Idea**: Implement a utility validation script to scan metadata for:
    - Chronological consistency ($T_1 < T_2 < \dots < T_8$).
    - Relative frame range boundary checks.
    - Physical downswing duration validation (must be between 6 and 12 frames at 30 FPS).


---

## 💤 Low Priority / Future Enhancements

### 🧹 Data Quality & Sanitization
- [ ] **Golfer Tracking & Lock-on (Centroid Tracking)**
  - **Idea**: Implement Centroid/Proximity tracking to lock onto the primary golfer in the frame, resolving multi-person scenarios (e.g. coach/bystanders in frame) instead of using a naive multi-person rejection.
- [ ] **Occlusion Handling**
  - **Idea**: Detect frames with high NaN density (MediaPipe tracking failures due to high motion blur or rotation) and interpolate using spline interpolation instead of linear.

### 🎨 Visual Overlay Improvements
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

