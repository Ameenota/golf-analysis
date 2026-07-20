import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODELS_REPO = os.environ.get("HF_MODELS_REPO", "Ameenota/golf-swing-analyzer-models")
DEFAULT_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "Ameenota/golf-swing-analyzer-dataset")

def ensure_models_downloaded(repo_id: str = None):
    """Ensures ML model files exist locally in models/, downloading from HF Hub if missing."""
    repo_id = repo_id or DEFAULT_MODELS_REPO
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    required_files = [
        "lstm_phase_model.keras",
        "golf_binary_detector.json",
        "kinematic_schema.json",
        "kinematic_config.json"
    ]
    
    for fname in required_files:
        fpath = models_dir / fname
        if not fpath.exists():
            print(f"📥 Downloading missing model asset: {fname} from Hugging Face ({repo_id})...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    repo_type="model",
                    local_dir=str(models_dir)
                )
                print(f"  └─ Downloaded: {downloaded_path}")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not download {fname} from {repo_id}: {e}")

def ensure_pro_dataset_downloaded(repo_id: str = None):
    """Ensures pro benchmark dataset assets exist locally in data/benchmark/, downloading from HF Hub if missing."""
    repo_id = repo_id or DEFAULT_DATASET_REPO
    benchmark_dir = PROJECT_ROOT / "data" / "benchmark"
    manifest_file = benchmark_dir / "manifest.json"
    
    if not manifest_file.exists():
        print(f"📥 Downloading pro benchmark dataset from Hugging Face ({repo_id})...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns="benchmark/*",
                local_dir=str(PROJECT_ROOT / "data")
            )
            print("  └─ Pro dataset download complete!")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not download pro dataset from {repo_id}: {e}")

def ensure_sample_presets_downloaded(repo_id: str = None):
    """Ensures pre-computed sample dashboard presets exist locally in data/sample_presets/."""
    repo_id = repo_id or DEFAULT_DATASET_REPO
    presets_dir = PROJECT_ROOT / "data" / "sample_presets"
    manifest_file = presets_dir / "manifest.json"
    
    if not manifest_file.exists():
        print(f"📥 Downloading sample presets from Hugging Face ({repo_id})...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns="sample_presets/*",
                local_dir=str(PROJECT_ROOT / "data")
            )
            print("  └─ Sample presets download complete!")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not download sample presets from {repo_id}: {e}")

def ensure_all_assets(models_repo: str = None, dataset_repo: str = None):
    """Convenience wrapper to download all missing models, benchmark dataset, and sample presets."""
    ensure_models_downloaded(models_repo)
    ensure_pro_dataset_downloaded(dataset_repo)
    ensure_sample_presets_downloaded(dataset_repo)

if __name__ == "__main__":
    ensure_all_assets()
