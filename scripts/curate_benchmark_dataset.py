import os
import json
import shutil
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLFDB_CSV = os.path.join(PROJECT_ROOT, "data/GolfDB.csv")
PRO_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "data/videos_160/videos_160")
R_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "data/r-videos")

BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data/benchmark")
BENCHMARK_PRO_DIR = os.path.join(BENCHMARK_DIR, "pro_videos")
BENCHMARK_USER_DIR = os.path.join(BENCHMARK_DIR, "user_videos")

def curate_dataset():
    os.makedirs(BENCHMARK_PRO_DIR, exist_ok=True)
    os.makedirs(BENCHMARK_USER_DIR, exist_ok=True)

    df = pd.read_csv(GOLFDB_CSV)

    # 1. Handpick 16 iconic & diverse pro swings
    target_pro_ids = [
        0,     # Sandra Gal - Female DTL Driver
        4,     # Brooke Henderson - Female DTL Driver
        6,     # Nick Watney - Male DTL Driver
        12,    # Kyle Stanley - Male DTL Driver
        47,    # Tiger Woods - Male DTL Iron
        8,     # Cristie Kerr - Female FO Driver
        10,    # Steve Stricker - Male FO Driver
        14,    # Greg Norman - Male FO Driver
        19,    # Hyo Joo Kim - Female FO Driver
        21,    # Tiger Woods - Male FO Iron
        1050,  # Phil Mickelson - Male FO Fairway (Left-Handed)
        1052,  # Phil Mickelson - Male DTL Wedge (Left-Handed)
        2,     # Chris DiMarco - Male DTL Driver
        23,    # Graeme McDowell - Male DTL Iron
        9,     # Cristie Kerr - Female FO Driver
        11,    # Steve Stricker - Male FO Driver
    ]

    pro_manifest = []
    for pro_id in target_pro_ids:
        row = df[df["id"] == pro_id].iloc[0]
        src_video = os.path.join(PRO_VIDEOS_DIR, f"{pro_id}.mp4")
        dst_filename = f"pro_{pro_id}_{row['player'].replace(' ', '_').lower()}_{row['view']}.mp4"
        dst_video = os.path.join(BENCHMARK_PRO_DIR, dst_filename)
        
        if os.path.exists(src_video):
            shutil.copy2(src_video, dst_video)
            pro_manifest.append({
                "id": int(pro_id),
                "filename": dst_filename,
                "player": str(row['player']),
                "sex": str(row['sex']),
                "club": str(row['club']),
                "view": str(row['view']),
                "path": os.path.relpath(dst_video, PROJECT_ROOT)
            })

    # 2. Collect user test videos from data/r-videos
    user_manifest = []
    if os.path.exists(R_VIDEOS_DIR):
        user_files = [f for f in os.listdir(R_VIDEOS_DIR) if f.endswith(('.MOV', '.mov', '.mp4'))]
        for u_file in sorted(user_files):
            src_user = os.path.join(R_VIDEOS_DIR, u_file)
            dst_user = os.path.join(BENCHMARK_USER_DIR, u_file)
            shutil.copy2(src_user, dst_user)
            
            # Ground truth views for r-videos
            if u_file in ["IMG_0018.MOV", "kin-1.mp4"]:
                view_expected = "face-on"
            else:
                view_expected = "down-the-line"
                
            handedness_expected = "right"
                
            user_manifest.append({
                "filename": u_file,
                "path": os.path.relpath(dst_user, PROJECT_ROOT),
                "expected_view": view_expected,
                "expected_handedness": handedness_expected
            })

    manifest = {
        "pro_videos": pro_manifest,
        "user_videos": user_manifest,
        "total_pro": len(pro_manifest),
        "total_user": len(user_manifest)
    }

    manifest_path = os.path.join(BENCHMARK_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Curated benchmark dataset created at: {BENCHMARK_DIR}")
    print(f"Pro Videos: {len(pro_manifest)} | User Videos: {len(user_manifest)}")

if __name__ == "__main__":
    curate_dataset()
