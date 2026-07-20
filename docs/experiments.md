# Golf Swing Analyzer - ML Experiments & Decision Log

This document serves as the official register for all machine learning feature engineering, model architecture, loss function, and post-processing experiments. Every ablation study or benchmark run must be logged here with its hypothesis, dataset split, metric results, and promotion decision.

---

## Experiment Protocol

When conducting a model experiment:
1. **Define Objective & Hypothesis**: Clearly state what problem is being solved and why the proposed feature/change should improve metrics.
2. **Isolate Baseline**: Keep train/val/test splits, model capacity, loss functions, and post-processing identical to isolate the effect of the experiment.
3. **Record Full Metrics**: Log overall Aligned MAE, Raw MAE, error distribution percentiles ($\le 1$ frame, $\le 3$ frames), per-milestone MAE, and slice breakdowns (Camera View, Video Speed).
4. **Evaluate Promotion Criteria**: Strictly verify whether the winning candidate meets required promotion thresholds before updating production artifacts.

---

## Experiment Log Index

| Exp ID | Date | Topic | Winning Candidate | Key Metric Impact | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | 2026-07-20 | Synthetic Kinematic Features Ablation (A $\rightarrow$ E) | **Experiment E** (108 Features) | **-14.13% Aligned MAE** (3.04 $\rightarrow$ 2.61 frames) | **PROMOTED TO PROD** |

---

## EXP-001: Synthetic Kinematic Features Ablation Study

* **Date**: July 20, 2026
* **Script**: `scratch/train_kinematic_ablations.py`
* **Dataset**: 1,399 GolfDB video sequences split 80% Train (1,119 vids), 10% Val (140 vids), 10% Test (140 vids) with seed `42`.
* **Model Architecture**: Bidirectional LSTM (256 units), Masking Layer (`mask_value=0.0`), Dynamic Unpadded Input `(1, N, num_features)`.
* **Alignment**: Monotonic Dynamic Programming alignment (`alignment.py`).

### 1. Objective & Hypotheses
Give the sequence model explicit velocity and kinetic energy signals rather than requiring it to infer all motion from static coordinate positions.

* **Exp A (Baseline)**: Base 66 normalized MediaPipe coordinates.
* **Exp B (Wrist Velocities)**: 66 Coords + 6 Wrist Velocities ($v_x, v_y, \text{speed}$). *Hypothesis*: Hand direction reversal pinpoints Top of Backswing and Impact.
* **Exp C (Upper-Body Velocities)**: 66 Coords + 18 Upper-Body Velocities (Wrists, Elbows, Shoulders). *Hypothesis*: Rapid downward arm movement clarifies Mid-Downswing.
* **Exp D (All MVP Velocities)**: 66 Coords + 36 Velocities for 12 key landmarks. *Hypothesis*: Full body kinematics provide holistic motion context.
* **Exp E (All Velocities + Motion Summaries)**: 66 Coords + 36 Velocities + 6 Group Summary Energy metrics (`mean_wrist_speed`, `mean_arm_speed`, `mean_hip_speed`, `mean_lower_body_speed`, `upper_body_motion_energy`, `whole_body_motion_energy`). *Hypothesis*: Aggregating energy across joint groups smooths out tracking jitter and provides clear kinetic energy peaks at Impact and setup/finish boundaries.

---

### 2. Primary Results Summary (140 Held-Out Test Videos)

| Exp | Description | Input Dim | Aligned MAE | Raw MAE | Median | P90 | $\le 1$ frame | $\le 3$ frames | MAE vs Base |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Baseline (Base 66 Coords) | 66 | 3.04 frames | 4.34 | 1.0 | 8.0 | 56.9% | 77.1% | Baseline |
| **B** | Coords + Wrist Velocities | 72 | 2.78 frames | 4.59 | 1.0 | 7.0 | 60.0% | 78.7% | **+8.69%** |
| **C** | Coords + Upper-Body Velocities | 84 | 2.95 frames | 3.38 | 1.0 | 7.0 | 60.4% | 78.7% | **+2.85%** |
| **D** | Coords + All 12 MVP Velocities | 102 | 3.19 frames | 5.15 | 1.0 | 7.0 | 62.9% | 80.1% | **-4.79%** |
| **E** | **Coords + All Vels + Summaries** | **108** | **2.61 frames** | **3.04** | **1.0** | **7.0** | **63.0%** | **80.3%** | **+14.13% (WINNER)** |

---

### 3. Per-Milestone Aligned MAE Breakdown (in Frames)

| Exp | Address | Toe-Up | Mid-Back | Top | Mid-Down | Impact | Follow-T | Finish |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 8.31 | 2.70 | 1.99 | **1.47** | 1.08 | 0.96 | 1.24 | 6.57 |
| **B** | 8.01 | 2.46 | 1.87 | 1.49 | 0.85 | 0.80 | 1.25 | 5.47 |
| **C** | 7.33 | 2.76 | 2.09 | 1.63 | 0.88 | 1.19 | 1.34 | 6.41 |
| **D** | 7.84 | 2.61 | 1.96 | 2.26 | 1.62 | 1.33 | 1.61 | 6.24 |
| **E** | **6.66** | **2.64** | **1.81** | **1.62** | **0.94** | **0.72** | **1.00** | **5.51** |

---

### 4. Slice Analysis (Experiment E vs Baseline A)

* **Camera View Slice**:
  * **Face-On (FO)**: Exp A = 2.82 frames $\rightarrow$ **Exp E = 2.41 frames** (**+14.5% improvement**)
  * **Down-The-Line (DTL)**: Exp A = 3.21 frames $\rightarrow$ **Exp E = 2.76 frames** (**+14.0% improvement**)
* **Video Playback Speed Slice**:
  * **Real-Time Video**: Exp A = 2.91 frames $\rightarrow$ **Exp E = 2.48 frames** (**+14.8% improvement**)
  * **Slow-Motion Video**: Exp A = 3.32 frames $\rightarrow$ **Exp E = 2.89 frames** (**+13.0% improvement**)

---

### 5. Promotion Criteria Verification & Decision

**Winning Candidate: Experiment E (108 Features)**

1. **Overall MAE Reduction $\ge 5.0\%$**: **14.13%** (3.04 $\rightarrow$ 2.61 frames) $\rightarrow$ **PASS**
2. **Milestones Improved $\ge 5/8$**: **7 / 8 milestones improved** (Address, Toe-Up, Mid-Backswing, Mid-Downswing, Impact, Follow-Through, Finish) $\rightarrow$ **PASS**
3. **Critical Top MAE Regression $\le +1.0$ frames**: **+0.15 frames** (1.47 $\rightarrow$ 1.62) $\rightarrow$ **PASS**
4. **Critical Impact MAE Regression $\le +1.0$ frames**: **-0.24 frames improvement** (0.96 $\rightarrow$ 0.72) $\rightarrow$ **PASS**

**Action Taken**:
* Updated primary model checkpoint at `models/lstm_phase_model.keras`.
* Created schema definition artifact at `models/kinematic_schema.json`.
* Created fitted standardization statistics artifact at `models/kinematic_config.json`.
* Integrated feature extraction into `analyze_swing.py`. Batch verification test achieved **2.45 frames overall MAE**.
