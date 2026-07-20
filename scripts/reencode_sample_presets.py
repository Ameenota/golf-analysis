import os
from pathlib import Path
import imageio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = PROJECT_ROOT / "data" / "sample_presets"

def reencode_presets():
    print("🎬 Re-encoding all sample preset videos to web-compatible H.264 yuv420p...")
    presets = ["preset_1", "preset_2", "preset_3"]
    
    for p_key in presets:
        vpath = PRESETS_DIR / p_key / "dashboard.mp4"
        if vpath.exists():
            print(f"  └─ Re-encoding {p_key} ({vpath})...", flush=True)
            tmp_path = PRESETS_DIR / p_key / "dashboard_h264.mp4"
            reader = imageio.get_reader(str(vpath))
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer(str(tmp_path), fps=fps, codec='libx264', pixelformat='yuv420p')
            for frame in reader:
                writer.append_data(frame)
            writer.close()
            reader.close()
            os.replace(tmp_path, vpath)
            print(f"     ✅ Re-encoded {p_key}/dashboard.mp4 ({os.path.getsize(vpath)/1e6:.2f} MB)", flush=True)
            
    print("🎉 All sample preset videos successfully re-encoded to web H.264!")

if __name__ == "__main__":
    reencode_presets()
