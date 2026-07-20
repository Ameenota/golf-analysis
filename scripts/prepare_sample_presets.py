import os
import shutil
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
PRESETS_DIR = PROJECT_ROOT / "data" / "sample_presets"

PRESET_MAP = {
    "preset_1": {
        "source_prefix": "IMG_0018",
        "title": "Sample 1: Face-On Right Handed (IMG_0018)",
        "view": "face-on",
        "handedness": "right",
        "pro": "Greg Norman"
    },
    "preset_2": {
        "source_prefix": "IMG_6826",
        "title": "Sample 2: Down-The-Line Right Handed (IMG_6826)",
        "view": "down-the-line",
        "handedness": "right",
        "pro": "Sandra Gal"
    },
    "preset_3": {
        "source_prefix": "kin-1",
        "title": "Sample 3: Face-On Short Clip (kin-1)",
        "view": "face-on",
        "handedness": "right",
        "pro": "Greg Norman"
    }
}

def prepare_presets():
    print("🧹 Packaging sample preset dashboard artifacts into data/sample_presets/...")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest_presets = {}

    for preset_key, meta in PRESET_MAP.items():
        p_dir = PRESETS_DIR / preset_key
        p_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = meta["source_prefix"]
        dash_mp4 = OUTPUT_DIR / f"{prefix}_dashboard.mp4"
        report_md = OUTPUT_DIR / f"{prefix}_report.md"
        results_json = OUTPUT_DIR / f"{prefix}_results.json"
        
        if not (dash_mp4.exists() and report_md.exists() and results_json.exists()):
            raise FileNotFoundError(f"Missing required output artifacts for {prefix} in output/")
            
        # Copy files into preset dir
        shutil.copy(dash_mp4, p_dir / "dashboard.mp4")
        shutil.copy(report_md, p_dir / "report.md")
        shutil.copy(results_json, p_dir / "results.json")
        
        manifest_presets[preset_key] = {
            "title": meta["title"],
            "view": meta["view"],
            "handedness": meta["handedness"],
            "pro": meta["pro"],
            "dashboard_video": f"sample_presets/{preset_key}/dashboard.mp4",
            "report": f"sample_presets/{preset_key}/report.md",
            "results": f"sample_presets/{preset_key}/results.json"
        }
        print(f"  └─ Created {preset_key} ({meta['title']})")
        
    with open(PRESETS_DIR / "manifest.json", "w") as f:
        json.dump(manifest_presets, f, indent=2)
        
    print("✅ Sample presets packaging complete!")

if __name__ == "__main__":
    prepare_presets()
