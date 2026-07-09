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

