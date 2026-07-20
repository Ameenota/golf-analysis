import os
import sys
import time
import json
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hf_downloader import ensure_all_assets
from analyze_swing import run_analysis, GolfSwingDetector
from streamlit_app.charts import create_spine_angle_chart, create_wrist_velocity_chart

# Page configuration
st.set_page_config(
    page_title="Golf Swing Analyzer",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #00E676; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #B0BEC5; margin-bottom: 1.5rem; }
    .metric-card { background-color: #1E222B; border-radius: 8px; padding: 12px; border-left: 4px solid #00E676; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system_assets():
    """Syncs missing ML models and pro dataset assets on cold start."""
    ensure_all_assets()

# Initialize assets on startup
initialize_system_assets()

PRESET_MANIFEST_PATH = PROJECT_ROOT / "data" / "sample_presets" / "manifest.json"

def load_preset_data(preset_key: str):
    """Loads pre-computed dashboard results for a preset sample video."""
    if not PRESET_MANIFEST_PATH.exists():
        st.error("Preset manifest not found. Downloading assets...")
        ensure_all_assets()
        
    with open(PRESET_MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
        
    meta = manifest.get(preset_key)
    if not meta:
        raise ValueError(f"Preset {preset_key} not found in manifest.")
        
    presets_dir = PROJECT_ROOT / "data" / "sample_presets" / preset_key
    
    with open(presets_dir / "results.json", "r") as f:
        results = json.load(f)
        
    with open(presets_dir / "report.md", "r") as f:
        report_md = f.read()
        
    video_path = presets_dir / "dashboard.mp4"
    if not video_path.exists():
        ensure_sample_presets_downloaded()
        
    return {
        "title": meta["title"],
        "video_path": str(video_path),
        "results": results,
        "report_md": report_md
    }

def run_simulated_processing():
    """Runs a simulated step-by-step UX progress animation (~4.6s) for preset quick selection."""
    with st.status("Analyzing Golf Swing Video...", expanded=True) as status:
        st.write("🔍 Stage 0 & 1: Verifying video resolution and duration...")
        time.sleep(1.2)
        st.write("🤖 Stage 2: Running XGBoost Binary Gatekeeper Validation...")
        time.sleep(1.2)
        st.write("🧠 Stage 3: Predicting 8 Swing Milestones via BiLSTM...")
        time.sleep(1.2)
        st.write("📐 Stage 4: Calculating Biomechanical Scorecard & Pro Matchup...")
        time.sleep(1.0)
        status.update(label="Analysis Complete!", state="complete", expanded=False)

def main():
    st.markdown('<div class="main-header">⛳ Golf Swing Analyzer & Coaching Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your swing video to get instant biomechanical feedback, pro matchup, and customized drills.</div>', unsafe_allow_html=True)

    # Sidebar: Preset Selectors & Settings
    st.sidebar.markdown("## 🎯 Preset Demo Swings")
    st.sidebar.caption("Click a preset to view pre-computed analysis:")

    col_p1, col_p2, col_p3 = st.sidebar.columns(3)
    p1_clicked = col_p1.button("Preset A\n(IMG_0018)")
    p2_clicked = col_p2.button("Preset B\n(IMG_6826)")
    p3_clicked = col_p3.button("Preset C\n(kin-1)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Settings")
    auto_view = st.sidebar.selectbox("Camera View Mode", options=["auto", "down-the-line", "face-on"], index=0)
    slow_speed = st.sidebar.slider("Slow-Motion Video Speed", min_value=0.25, max_value=1.0, value=0.5, step=0.25)

    # Active Analysis State
    selected_preset = None
    if p1_clicked: selected_preset = "preset_1"
    elif p2_clicked: selected_preset = "preset_2"
    elif p3_clicked: selected_preset = "preset_3"

    st.markdown("### 📤 Upload Your Swing Video")
    uploaded_file = st.file_uploader("Upload `.mp4` or `.mov` video (Max size: 50MB)", type=["mp4", "mov"])

    analysis_data = None

    if selected_preset:
        run_simulated_processing()
        preset_info = load_preset_data(selected_preset)
        analysis_data = {
            "source": f"Preset ({preset_info['title']})",
            "video_path": preset_info["video_path"],
            "results": preset_info["results"],
            "report_md": preset_info["report_md"]
        }
    elif uploaded_file is not None:
        # Stage 0: Client/Header Byte Size Rejection
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb < 1.0 or file_size_mb > 50.0:
            st.error(f"⚠️ Video file size invalid ({file_size_mb:.1f} MB). Must be between 1 MB and 50 MB.")
            st.stop()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            temp_video_path = tmp.name

        with st.status("Analyzing Uploaded Video...", expanded=True) as status:
            st.write("🔍 Stage 1: Validating video metadata and duration...")
            detector = GolfSwingDetector()
            valid_dur, duration = detector.check_duration(temp_video_path, max_duration=60.0)
            if not valid_dur:
                status.update(label="Rejected!", state="error")
                st.error(f"⚠️ Video exceeds maximum allowed duration (60.0s). Measured: {duration:.1f}s.")
                st.stop()

            st.write("🤖 Stage 2 & 3: Extracting landmarks & running XGBoost gatekeeper + BiLSTM inference...")
            out_video_path = temp_video_path.replace(".mp4", "_dashboard.mp4")
            out_json_path = temp_video_path.replace(".mp4", "_results.json")
            out_report_path = temp_video_path.replace(".mp4", "_report.md")

            # Execute pipeline
            res = run_analysis(
                video_path=temp_video_path,
                view_mode=auto_view,
                speed=slow_speed,
                output_video_path=out_video_path,
                save_json_path=out_json_path
            )

            if not res or not res.get("validated", False):
                status.update(label="Validation Failed!", state="error")
                score = res.get("gatekeeper_score", 0.0) if res else 0.0
                st.error(f"⚠️ Invalid Video: No golf swing detected (Gatekeeper Score: {score:.2f}). Please upload a valid swing video.")
                st.stop()

            status.update(label="Analysis Complete!", state="complete", expanded=False)

            report_md = ""
            if os.path.exists(out_report_path):
                with open(out_report_path, "r") as f:
                    report_md = f.read()

            analysis_data = {
                "source": uploaded_file.name,
                "video_path": out_video_path if os.path.exists(out_video_path) else temp_video_path,
                "results": res,
                "report_md": report_md
            }

    # Render Dashboard Tabs if analysis data is ready
    if analysis_data:
        st.markdown("---")
        results = analysis_data["results"]
        
        tab_video, tab_report, tab_charts = st.tabs([
            "🎥 Synchronized Video & Scorecard",
            "📝 Coaching Report & Drills",
            "📈 Kinematic Charts"
        ])

        # Tab 1: Video & Scorecard Badges Below
        with tab_video:
            st.markdown(f"#### 🎥 Synchronized Pro Comparison (`{analysis_data['source']}`)")
            if os.path.exists(analysis_data["video_path"]):
                with open(analysis_data["video_path"], "rb") as vf:
                    st.video(vf.read(), format="video/mp4")
            else:
                st.warning("Side-by-side video clip unavailable.")

            st.markdown("---")
            st.markdown("#### 📊 Biomechanical Scorecard")
            
            # Overview Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Camera View", results.get("view", "Unknown").upper())
            c2.metric("Handedness", results.get("handedness", "Unknown").capitalize())
            c3.metric("Matched Pro", results.get("matched_pro", "N/A"))
            c4.metric("Gatekeeper Score", f"{results.get('gatekeeper_score', 0.0):.2f}")

            st.markdown("##### Key Biomechanical Checks")
            metrics = results.get("biomechanical_metrics", {})
            k1, k2, k3, k4 = st.columns(4)

            # Spine tilt loss
            spine_loss = metrics.get("spine_tilt_loss", 0.0)
            k1.metric("Spine Tilt Change", f"{spine_loss:.1f}°", delta="PASS" if abs(spine_loss) <= 3.0 else "WARN Posture Shift", delta_color="inverse" if abs(spine_loss) > 3.0 else "normal")

            # Lead arm flex
            arm_flex = metrics.get("lead_arm_flex_at_top", 180.0)
            k2.metric("Lead Arm Flex at Top", f"{arm_flex:.1f}°", delta="PASS Straight Arm" if arm_flex >= 160.0 else "WARN Bent Elbow", delta_color="normal" if arm_flex >= 160.0 else "inverse")

            # Sway ratio
            sway = metrics.get("hip_sway_ratio", 0.0)
            k3.metric("Hip Sway Ratio", f"{sway*100:.1f}%", delta="PASS Stable Hips" if sway <= 0.15 else "WARN Excessive Slide", delta_color="inverse" if sway > 0.15 else "normal")

            # Trail heel lift
            heel_lift = metrics.get("trail_heel_lift_ratio", 0.0)
            k4.metric("Trail Heel Lift", f"{heel_lift*100:.1f}%", delta="PASS Weight Transfer" if heel_lift >= 0.10 else "WARN Hanging Back", delta_color="normal" if heel_lift >= 0.10 else "inverse")

        # Tab 2: Coaching Report & Drills
        with tab_report:
            st.markdown("### 📝 Personal Coaching & Practice Drills")
            if analysis_data["report_md"]:
                st.markdown(analysis_data["report_md"])
            else:
                st.info("Coaching report text unavailable.")

            col_d1, col_d2 = st.columns(2)
            if os.path.exists(analysis_data["video_path"]):
                with open(analysis_data["video_path"], "rb") as vf:
                    col_d1.download_button("📥 Download Stitched MP4 Video", data=vf, file_name="golf_swing_comparison.mp4", mime="video/mp4")
            if analysis_data["report_md"]:
                col_d2.download_button("📄 Download Coaching Report (.md)", data=analysis_data["report_md"], file_name="golf_coaching_report.md", mime="text/markdown")

        # Tab 3: Interactive Kinematic Charts
        with tab_charts:
            st.markdown("### 📈 Interactive Swing Kinematics & Timing")
            milestones = results.get("milestones", {})
            metrics = results.get("biomechanical_metrics", {})
            pro_name = results.get("matched_pro", "Matched Pro")
            pro_metrics = results.get("matched_pro_metrics", {})
            
            fig_spine = create_spine_angle_chart(
                df=pd.DataFrame(),
                milestones=milestones,
                metrics=metrics,
                pro_name=pro_name,
                pro_metrics=pro_metrics
            )
            st.plotly_chart(fig_spine, use_container_width=True)
            
            fig_vel = create_wrist_velocity_chart(
                df=pd.DataFrame(),
                milestones=milestones,
                metrics=metrics,
                pro_name=pro_name,
                pro_metrics=pro_metrics
            )
            st.plotly_chart(fig_vel, use_container_width=True)

if __name__ == "__main__":
    main()
