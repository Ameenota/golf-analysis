import os
import sys
import argparse
from pathlib import Path

# Disable XET middleware before importing huggingface_hub
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def deploy_space(space_repo_id: str):
    """Deploys the Streamlit web app and static assets to Hugging Face Spaces."""
    api = HfApi()
    
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"🔑 Logged in as Hugging Face user: {username}")
    except Exception as e:
        print(f"❌ Error: Not logged in to Hugging Face. ({e})")
        sys.exit(1)
        
    print(f"🚀 Ensuring Space repository exists: {space_repo_id}...", flush=True)
    try:
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True
        )
    except Exception as e:
        print(f"  └─ Repo creation info: {e}")
    
    print(f"📦 Uploading application assets to Hugging Face Space ({space_repo_id})...", flush=True)
    
    # Files to upload at root of space
    root_files = [
        "index.html",
        "README.md",
        "requirements.txt",
        "packages.txt",
        "pyproject.toml",
        "analyze_swing.py"
    ]
    
    for rfile in root_files:
        fpath = PROJECT_ROOT / rfile
        if fpath.exists():
            print(f"  └─ Uploading {rfile}...", flush=True)
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=rfile,
                repo_id=space_repo_id,
                repo_type="space"
            )
            print(f"     ✅ Uploaded {rfile}", flush=True)
            
    # Directories to upload
    directories = [
        ("streamlit_app", "streamlit_app"),
        ("src", "src"),
        (".streamlit", ".streamlit")
    ]
    
    for local_dir, repo_dir in directories:
        dpath = PROJECT_ROOT / local_dir
        if dpath.exists():
            print(f"  └─ Uploading directory: {local_dir}/...", flush=True)
            api.upload_folder(
                folder_path=str(dpath),
                path_in_repo=repo_dir,
                repo_id=space_repo_id,
                repo_type="space"
            )
            print(f"     ✅ Uploaded directory {local_dir}/", flush=True)
            
    print("\n🎉 Deployment completed successfully!")
    print(f"🌐 Access your Space at: https://huggingface.co/spaces/{space_repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Deploy Golf Swing Analyzer app to Hugging Face Spaces.")
    parser.add_argument("--repo-id", type=str, default=None, help="Full Space repo ID (e.g. sagsan/golf-swing-analyzer)")
    args = parser.parse_args()
    
    api = HfApi()
    username = api.whoami()["name"]
    space_repo_id = args.repo_id or f"{username}/golf-swing-analyzer"
    
    deploy_space(space_repo_id)

if __name__ == "__main__":
    main()
