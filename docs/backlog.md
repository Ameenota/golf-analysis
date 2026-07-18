# Golf Swing Analyzer Project - Future Enhancements & Experiments

This file tracks technical debt, future feature ideas, and machine learning experiments to run once the baseline model is fully implemented.

## 🧪 Machine Learning Experiments

- [ ] **Multi-Scale Sliding Window Context**
  - **Idea**: Expand the sliding window in [feature_engineer.py](file:///Users/sagar/Documents/ML/golf-analysis/src/feature_engineer.py) from a single $\pm 5$ shift to a multi-scale shift (e.g., `shift_steps = [5, 10]`).
  - **Goal**: Compare validation accuracy on transition phases (like Top of Backswing and Address) to see if the extra 32 columns improve sequence boundary prediction without overloading XGBoost.
  
- [ ] **XGBoost Hyperparameter Tuning**
  - **Idea**: Tune `max_depth`, `min_child_weight`, and `gamma` to prevent overfitting on specific golfers in the training set.
  
- [ ] **Class Weight Optimization**
  - **Idea**: Experiment with different scale ratios for the transitional class `0` to optimize the precision/recall trade-off for the milestone classes `1-8`.

- [x] **Unsupervised Golf Swing Gatekeeper (Isolation Forest)**
  - **Idea**: Train an `sklearn.ensemble.IsolationForest` on the exact same coordinate features in `master_training_dataset.csv` (ignoring labels).
  - **Goal**: Create a simple Yes/No swing detector to automatically reject non-golf videos (like dancing or empty rooms) during inference without needing any negative training data.
  - **Status**: Completed full evaluation on the 1,642 UCF dataset videos. Achieved AUC PR of 0.6802 and optimal F1-score of 0.6568 (precision 68.99%, recall 62.68% at threshold 0.1429).

- [x] **Unified XGBoost + Rules Gatekeeper**
  - **Idea**: Use a combination of physical rules (human count, verticality, downswing duration) and milestone confidence probability.
  - **Goal**: Build a highly selective, recall-first gatekeeper that filters out irrelevant content.
  - **Status**: Completed evaluation on 1,642 UCF videos. Reached AUC PR of 0.7904, Max F1-score of 0.8115, and 91.6% rejection rate of non-golf at target recall threshold of 0.20.

- [ ] **Dedicated Binary XGBoost Detector (Golf vs. Non-Golf)**
  - **Idea**: Train a separate, dedicated binary XGBoost model (using `objective="binary:logistic"`) on GolfDB (positives) and UCF50 non-golf (negatives).
  - **Goal**: Cleanly separate detection from milestone segmentation to achieve a more robust and less complex gatekeeper.
  - **Experiments to Try**:
    - [ ] **Postural Features**: Test if adding `is_upright` and `torso_angle_deg` as training features directly to the XGBoost binary detector improves the rejection of horizontal/non-upright movements.
  - **Target to Beat**: Golf Recall = **77%**, Golf Precision = **47%**, F1 = **58%** (Confusion Matrix: 110 TP, 32 FN, 126 FP, 1374 TN at 0.20 threshold).







## 🧹 Data Quality & Sanitization

- [ ] **Programmatic Label Validation**
  - **Idea**: Implement the annotation validation script to scan `df_meta` for:
    - Chronological milestone consistency ($T_1 < T_2 < \dots < T_8$).
    - Relative frame range boundary checks.
    - Physical downswing duration validation (downswing must be between 6 and 12 frames at 30 FPS).

- [ ] **Occlusion Handling**
  - **Idea**: Detect frames with high NaN density (MediaPipe tracking failures due to high motion blur or body rotation) and interpolate them using spline interpolation instead of linear interpolation.

- [ ] **Flexible Posture Window Validation**
  - **Idea**: Experiment with different validation windows for posture/verticality checks (e.g., middle frames vs. dynamically identifying the stationary address frame) to handle cases where the swing setup does not start at frame 0.

- [ ] **Golfer Tracking & Lock-on (Centroid Tracking)**
  - **Idea**: Implement Centroid or Proximity tracking to lock onto the primary golfer in the frame.
  - **Goal**: Support multi-person frames (e.g., coach in frame, bystanders) and ignore background people, replacing the naive multi-person rejection check.


## 📐 Biomechanical Rules Engine

- [ ] **Additional 2D Swing Metrics**
  - **Idea**: Add calculations for:
    - **Spine Tilt**: Measured relative to the vertical axis at Address.
    - **Knee Flex**: To detect hip sway or lunging during the downswing.
    - **Club Shaft Angle**: To identify casting or early release before Impact.

## ⚡ Performance & Scaling

- [ ] **Parallel Video Processing**
  - **Idea**: Use Python's `concurrent.futures.ProcessPoolExecutor` to process multiple videos in parallel.
  - **Goal**: Utilize all CPU cores of the M4 chip to achieve a 5x-8x speedup on batch dataset generation.

## 🧹 Technical Debt & Dependency Cleanup

- [ ] **Remove PyTorch and Fastai Dependencies**
  - **Idea**: Once the Keras LSTM model is fully validated and integrated into the inference pipeline, clean up the codebase.
  - **Goal**: Delete the old PyTorch notebook `notebooks/lstm_model.ipynb` and remove the `fastai` and `torch` dependencies using `uv remove` to reduce the environment size.

## 🚀 Inference Pipeline Integration & Verification

- [ ] **Integrate Inference Pipeline Wrapper**
  - **Idea**: Create the final end-to-end inference wrapper script (`analyze_swing.py` at the root). 
  - **Goal**: Read frames, run pose landmarker smoothing and normalization using [GolfVideoProcessor](file:///Users/sagar/Documents/ML/golf-analysis/src/data_processor.py#L11), filter through the binary XGBoost gatekeeper, and route to the Keras LSTM milestone locator to detect the 8 key frames.

- [ ] **Implement Chronological Post-processing Validation**
  - **Idea**: Predicted milestone frames from the LSTM must satisfy chronological order ($T_1 < T_2 < \dots < T_8$).
  - **Goal**: Integrate Viterbi-like or confidence threshold heuristics to resolve any out-of-order predictions.

- [ ] **End-to-End Pipeline Verification**
  - **Idea**: Test the complete integrated pipeline on unseen or sample videos.
  - **Goal**: Ensure the binary gatekeeper correctly rejects non-golf and accepts golf swings, and the LSTM extracts milestones reliably.



