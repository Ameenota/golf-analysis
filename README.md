# ⛳ Golf Swing Analyzer & Coaching Dashboard

An end-to-end computer-vision application that validates golf swing videos, identifies eight swing milestones with a bidirectional LSTM, performs biomechanical analysis, and generates synchronized professional comparison videos.

## Features

- XGBoost gatekeeper validation for non-golf and overlong videos.
- Bidirectional LSTM detection of eight chronological swing milestones.
- Biomechanical scorecard and coaching feedback.
- H.264 side-by-side professional comparison videos.
- Interactive Plotly kinematic charts.
- Three precomputed demonstration presets.

## Run locally

This project uses `uv` for Python dependency management and execution.

```bash
uv sync
uv run streamlit run streamlit_app/app.py
```

The Streamlit configuration in `.streamlit/config.toml` limits uploads to 50 MB.

## Public deployment

The public application is deployed with Streamlit Community Cloud from the `main` branch of [Ameenota/golf-analysis](https://github.com/Ameenota/golf-analysis). The entry point is:

```text
streamlit_app/app.py
```

Streamlit Community Cloud installs Python packages from `requirements.txt` and operating-system packages from `packages.txt`. After pushing application or dependency changes, confirm the app redeploys successfully from **Manage app**. Use **Reboot app** when a clean runtime filesystem is needed.

Generated comparison videos require FFmpeg/ImageIO and are encoded as H.264 (`libx264`, `yuv420p`). The application deliberately fails with an encoding error instead of producing browser-incompatible `mp4v` video.

## Hugging Face Hub assets

Hugging Face is used only for application assets, not application hosting:

- [ML models](https://huggingface.co/sagsan/golf-swing-analyzer-models)
- [Benchmark dataset and presets](https://huggingface.co/datasets/sagsan/golf-swing-analyzer-dataset)

On each application cold start, missing models and benchmark data are downloaded. Presets are synchronized against the Hub so updated H.264 files replace stale runtime copies. If the Hub is temporarily unavailable, already-downloaded presets remain usable.

To publish updated local models, benchmark assets, or presets:

```bash
uv run python scripts/upload_assets_to_hf.py
```

Hugging Face Spaces was evaluated as an application host and abandoned; that outcome is retained in `docs/history.md`.
