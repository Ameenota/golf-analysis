import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def upload_models(api: HfApi, repo_id: str):
    """Uploads ML model artifacts to Hugging Face Model Hub."""
    print(f"📦 Uploading ML models to Hugging Face Model Hub: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    models_dir = PROJECT_ROOT / "models"
    files_to_upload = [
        "lstm_phase_model.keras",
        "golf_binary_detector.json",
        "kinematic_schema.json",
        "kinematic_config.json"
    ]
    
    for fname in files_to_upload:
        fpath = models_dir / fname
        if fpath.exists():
            print(f"  └─ Uploading {fname} ({fpath.stat().st_size / 1e6:.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model"
            )
        else:
            print(f"  ⚠️ Warning: Model file {fpath} not found. Skipping.")
            
    print("✅ ML models upload complete!\n")

def upload_dataset(api: HfApi, repo_id: str):
    """Uploads pro reference videos, manifest, and sample presets to Hugging Face Dataset Hub."""
    print(f"📊 Uploading dataset assets to Hugging Face Dataset Hub: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    # 1. Upload Pro Videos & Manifest
    benchmark_dir = PROJECT_ROOT / "data" / "benchmark"
    if benchmark_dir.exists():
        print("  └─ Uploading data/benchmark directory...")
        api.upload_folder(
            folder_path=str(benchmark_dir),
            path_in_repo="benchmark",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
    # 2. Upload Pre-computed Sample Presets if available
    presets_dir = PROJECT_ROOT / "data" / "sample_presets"
    if presets_dir.exists():
        print("  └─ Uploading data/sample_presets directory...")
        api.upload_folder(
            folder_path=str(presets_dir),
            path_in_repo="sample_presets",
            repo_id=repo_id,
            repo_type="dataset"
        )
    else:
        print("  ℹ️ Notice: data/sample_presets directory not found yet. Run preset generation first.")

    print("✅ Dataset upload complete!\n")

def main():
    parser = argparse.ArgumentParser(description="Upload ML models and dataset assets to Hugging Face Hub.")
    parser.add_argument("--owner", type=str, help="Hugging Face username or organization (defaults to logged-in user)")
    args = parser.parse_args()
    
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"🔑 Logged in as Hugging Face user: {username}")
    except Exception as e:
        print(f"❌ Error: Not logged in to Hugging Face. Please run 'uv run huggingface-cli login' first. ({e})")
        sys.exit(1)
        
    owner = args.owner or username
    models_repo_id = f"{owner}/golf-swing-analyzer-models"
    dataset_repo_id = f"{owner}/golf-swing-analyzer-dataset"
    
    upload_models(api, models_repo_id)
    upload_dataset(api, dataset_repo_id)
    
    print("🎉 All assets successfully uploaded to Hugging Face Hub!")
    print(f"   Model Repo  : https://huggingface.co/{models_repo_id}")
    print(f"   Dataset Repo: https://huggingface.co/datasets/{dataset_repo_id}")

if __name__ == "__main__":
    main()
