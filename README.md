---
title: Golf Swing Analyzer & Coaching Dashboard
emoji: ⛳
colorFrom: green
colorTo: green
sdk: streamlit
sdk_version: 1.42.0
app_file: streamlit_app/app.py
pinned: false
license: mit
---

# ⛳ Golf Swing Analyzer & Coaching Dashboard

An end-to-end Computer Vision and Machine Learning application that validates golf swing videos, pinpoints 8 key swing milestones using a Bidirectional LSTM, performs 2D biomechanical analysis, and provides synchronized pro comparison overlays and interactive kinematic charts.

## 🚀 Key Features

- **🤖 Gatekeeper Validation**: XGBoost binary classifier (99.0% accuracy) rejecting non-golf videos and long clips (>60s).
- **🧠 BiLSTM Milestone Locator**: Predicts 8 chronological swing events (*Address, Toe-Up, Mid-Backswing, Top, Mid-Downswing, Impact, Mid-Follow-Through, Finish*) with an overall MAE of **2.66 frames**.
- **📐 Biomechanical Scorecard**: Measures Spine Tilt loss, Lead Arm Straightness, Hip Sway Ratio, and Foot Weight Transfer.
- **🎥 Side-by-Side Pro Comparison**: Dynamic pro matchmaker against 16 pro reference models with synchronized speed control.
- **📈 Kinematic Charts**: Interactive Plotly plots comparing user swing kinematics against matched pro trajectories.

## 📦 Assets & Models
- **ML Models**: [sagsan/golf-swing-analyzer-models](https://huggingface.co/sagsan/golf-swing-analyzer-models)
- **Dataset & Presets**: [sagsan/golf-swing-analyzer-dataset](https://huggingface.co/datasets/sagsan/golf-swing-analyzer-dataset)

---
*Built with OpenCV, MediaPipe, TensorFlow/Keras, XGBoost, and Streamlit.*
