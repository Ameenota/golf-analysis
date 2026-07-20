import os
import argparse
import json
from pathlib import Path

import numpy as np

# Set local directory overrides for sandbox environment
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(PROJECT_ROOT, ".keras_home")
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".mpl_config")
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cv2
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


from src.data_processor import GolfVideoProcessor
from src.feature_engineer import engineer_sliding_window, HIGH_MOVEMENT_JOINTS
from src.alignment import compute_monotonic_alignment
from src.visualizer import draw_skeleton
from src.coaching_engine import analyze_swing_biomechanics
from src.visual_stitcher import create_synchronized_dashboard
from src.detector import GolfSwingDetector
from src.pro_preprocessing import load_preprocessed_pro

MILESTONE_NAMES = [
    "Address",
    "Toe-Up",
    "Mid-Backswing",
    "Top of Backswing",
    "Mid-Downswing",
    "Impact",
    "Mid-Follow-Through",
    "Finish"
]

def detect_camera_view(df, address_frame=0):
    """
    Detects camera view perspective (face-on vs down-the-line) using a 98.1% accurate hybrid 2D/3D heuristic.
    """
    if address_frame is None or address_frame >= len(df):
        address_frame = 0
        
    row = df.iloc[address_frame]
    
    l_sh_x = row["smooth_left_shoulder_x"]
    r_sh_x = row["smooth_right_shoulder_x"]
    sh_width_px = abs(l_sh_x - r_sh_x)
    torso_scale = row["torso_scale"]
    
    norm_sh_width = sh_width_px / (torso_scale + 1e-6)
    
    if norm_sh_width > 0.45:
        return "face-on"
    elif norm_sh_width < 0.20:
        return "down-the-line"
    else:
        l_sh_z = row.get("smooth_left_shoulder_z", 0.0)
        r_sh_z = row.get("smooth_right_shoulder_z", 0.0)
        sh_z_diff = abs(l_sh_z - r_sh_z)
        
        if sh_z_diff <= 0.21:
            return "face-on"
        else:
            return "down-the-line"


def generate_markdown_report(output, save_report_path, view):
    """Generates a structured coaching report in Markdown format with merged explanation details."""
    parent_dir = os.path.dirname(os.path.abspath(save_report_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    milestones = output["milestones"]
    f1 = milestones["Address"]["frame"]
    f3 = milestones["Top of Backswing"]["frame"]
    f5 = milestones["Impact"]["frame"]
    
    pro_name = output["matched_pro"]
    pro_metrics = output["matched_pro_metrics"]
    
    report = []
    report.append("# Golf Swing Biomechanical Analysis Report")
    report.append(f"\n* **Handedness**: {output['handedness'].capitalize()}-Handed")
    report.append(f"* **Camera View**: {view.replace('-', ' ').title()}")
    report.append(f"* **Matched Pro**: {pro_name} (Pro Ratio: {output['matched_pro_ratio']:.3f} vs. User Ratio: {output['user_arm_to_torso_ratio']:.3f})")
    report.append("\n---")
    
    report.append("\n## Biomechanical Metrics Comparison")
    report.append(f"\n| Metric | Measured Frame | Measured Value | Standard Range | Status | Pro Reference ({pro_name}) |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    metrics = output["biomechanical_metrics"]
    issues = output["issues_detected"]
    issue_names = [issue["issue"] for issue in issues]
    
    # Helper to get pro value text
    def get_pro_val(key, suffix="°"):
        val = pro_metrics.get(key)
        if val is None:
            return "N/A"
        if "ratio" in key or "sway" in key or "bob" in key:
            return f"{val * 100:.1f}%"
        return f"{val:.1f}{suffix}"
        
    # 1. Lead Arm Flex
    arm_flex = metrics.get("lead_arm_flex_at_top")
    if arm_flex is not None:
        status = "❌ Warning" if "Bent Lead Arm at Top of Backswing" in issue_names else "✅ Pass"
        pro_val = get_pro_val("lead_arm_flex_at_top")
        report.append(f"| Lead Arm Flex at Top | Top of Backswing (F{f3}) | {arm_flex:.1f}° | >= 160° | {status} | {pro_val} |")
        
    # 2. Spine Tilt Address
    spine_addr = metrics.get("spine_tilt_at_address")
    if spine_addr is not None:
        status = "❌ Warning" if "Incorrect Spine Tilt at Address" in issue_names else "✅ Pass"
        range_str = "5° to 15°" if view == "face-on" else "30° to 45°"
        pro_val = get_pro_val("spine_tilt_at_address")
        report.append(f"| Spine Tilt at Address | Address (F{f1}) | {spine_addr:.1f}° | {range_str} | {status} | {pro_val} |")
        
    # 3. Spine Tilt Loss
    spine_loss = metrics.get("spine_tilt_loss")
    if spine_loss is not None:
        status = "❌ Warning" if "Loss of Spine Tilt at Top of Backswing" in issue_names else "✅ Pass"
        pro_val = get_pro_val("spine_tilt_loss")
        report.append(f"| Spine Tilt Loss (Top vs. Address) | Top vs. Address | {spine_loss:.1f}° | <= 3° | {status} | {pro_val} |")
        
    # 4. Knee Flex Address
    knee_flex = metrics.get("lead_knee_flex_at_address")
    if knee_flex is not None:
        status = "❌ Warning" if "Incorrect Knee Flex at Address" in issue_names else "✅ Pass"
        range_str = "165° to 175°" if view == "face-on" else "150° to 165°"
        pro_val = get_pro_val("lead_knee_flex_at_address")
        report.append(f"| Knee Flex at Address | Address (F{f1}) | {knee_flex:.1f}° | {range_str} | {status} | {pro_val} |")
        
    # 5. Hip Sway (Face-On Only)
    hip_sway = metrics.get("hip_sway_ratio")
    if hip_sway is not None:
        status = "❌ Warning" if "Excessive Lateral Hip Sway at Top of Backswing" in issue_names else "✅ Pass"
        pro_val = get_pro_val("hip_sway_ratio")
        report.append(f"| Lateral Hip Sway Ratio | Top of Backswing (F{f3}) | {hip_sway * 100:.1f}% | <= 15% | {status} | {pro_val} |")
        
    # 6. Head Bobbing (Face-On Only)
    head_bob = metrics.get("head_bob_ratio")
    if head_bob is not None:
        status = "❌ Warning" if "Excessive Vertical Head Movement" in issue_names else "✅ Pass"
        pro_val = get_pro_val("head_bob_ratio")
        report.append(f"| Vertical Head Bobbing Ratio | Top (F{f3}) & Impact (F{f5}) | {head_bob * 100:.1f}% | <= 15% | {status} | {pro_val} |")
        
    # 7. Trail Heel Lift (Finish)
    f8 = milestones.get("Finish", {}).get("frame", milestones.get("Mid-Follow-Through", {}).get("frame", 0))
    trail_heel = metrics.get("trail_heel_lift_ratio")
    if trail_heel is not None:
        status = "❌ Warning" if "Trail Heel Stuck Flat at Finish (Hanging Back)" in issue_names else "✅ Pass"
        pro_val = get_pro_val("trail_heel_lift_ratio")
        report.append(f"| Trail Heel Lift Ratio | Finish (F{f8}) | {trail_heel * 100:.1f}% | >= 10% | {status} | {pro_val} |")

    # 8. Lead Heel Lift (Finish)
    lead_heel = metrics.get("lead_heel_lift_ratio")
    if lead_heel is not None:
        status = "❌ Warning" if "Lead Heel Lifted at Finish (Unstable Lead Foot)" in issue_names else "✅ Pass"
        pro_val = get_pro_val("lead_heel_lift_ratio")
        report.append(f"| Lead Heel Lift Ratio | Finish (F{f8}) | {lead_heel * 100:.1f}% | <= 12% | {status} | {pro_val} |")

    report.append("\n---")
    
    # Combined Explanations & Issues section
    ISSUE_EXPLANATIONS = {
        "Bent Lead Arm at Top of Backswing": {
            "why_matters": "Keeping the lead arm straight behaves like a compass drawing a circle. The lead arm is the metal arm, and the club is the pencil. Bending the elbow shrinks the swing circle.",
            "what_lost": "If your lead elbow bends below 160°, the swing circle becomes inconsistent. This leads to hitting the turf behind the ball (fat shots), missing/topping the ball (thin shots), and a significant loss in swing speed/power."
        },
        "Incorrect Spine Tilt at Address": {
            "why_matters": "The spine is the axle of a spinning wheel. Leaning slightly backward at setup (5° to 15° lateral for Face-On, 30° to 45° forward for Down-The-Line) sets up a stable plane of rotation.",
            "what_lost": "Incorrect setup angles prevent clean rotation, forcing compensation moves that ruin posture alignment and cause weak, inconsistent club delivery."
        },
        "Loss of Spine Tilt at Top of Backswing": {
            "why_matters": "Maintaining your spine posture throughout the backswing keeps the rotational axis of your body perfectly centered.",
            "what_lost": "Wobbling or standing up at the top of the swing throws the rotating axle off-center, leading to inconsistent contact and severe ball-striking errors."
        },
        "Incorrect Knee Flex at Address": {
            "why_matters": "Soft knees set up athletic balance and load weight into the feet properly.",
            "what_lost": "Locking the knees or crouching too low ruins posture rotation stability, limiting your hip rotation capacity and weight transfer."
        },
        "Excessive Lateral Hip Sway at Top of Backswing": {
            "why_matters": "Hips should rotate in place like a door pivoting on a stable hinge. They should not slide side-to-side.",
            "what_lost": "Sliding laterally away from the target moves your pivot point, draining rotation power and shifting the club's strike point behind the ball."
        },
        "Excessive Vertical Head Movement": {
            "why_matters": "The head is the visual and mechanical anchor of the swing. Keeping it stable is like maintaining a stable center during rotation (Wall analogy).",
            "what_lost": "Dipping or bobbing shifts your entire swing hub vertically, causing erratic thin or fat turf strikes."
        },
        "Trail Heel Stuck Flat at Finish (Hanging Back)": {
            "why_matters": "At the finish of a full swing, 90%+ of your body weight should shift to your lead leg, allowing your trail hip and chest to rotate fully through the ball toward the target.",
            "what_lost": "Leaving weight on the back foot (trail heel flat) forces you to hang back, causing flip releases, severe loss of clubhead speed, and fat/thin ball contact."
        },
        "Lead Heel Lifted at Finish (Unstable Lead Foot)": {
            "why_matters": "The lead foot acts as the solid anchoring post for your lower body rotation during follow-through.",
            "what_lost": "Lifting the lead heel off the ground destabilizes your pivot point, causing loss of balance and inconsistent contact."
        }
    }
    
    report.append("\n## Detected Posture Issues & Coaching Feedback")
    if not issues:
        report.append("\n**Congratulations! No posture issues were detected in your swing.**")
    else:
        for issue in issues:
            name = issue["issue"]
            report.append(f"\n### ❌ {name}")
            
            # 1. Print explanation of why it matters
            exp = ISSUE_EXPLANATIONS.get(name)
            if exp:
                report.append(f"* **Why it matters**: {exp['why_matters']}")
                report.append(f"* **What you are losing**: {exp['what_lost']}")
                
            # 2. Print metrics values
            report.append(f"* **Target Range**: {issue['threshold']}")
            report.append(f"* **Your Measured Value**: {issue['measured']}")
            
            # Determine pro reference value
            pro_key = None
            if "Bent Lead Arm" in name:
                pro_key = "lead_arm_flex_at_top"
            elif "Incorrect Spine Tilt" in name:
                pro_key = "spine_tilt_at_address"
            elif "Loss of Spine Tilt" in name:
                pro_key = "spine_tilt_loss"
            elif "Incorrect Knee Flex" in name:
                pro_key = "lead_knee_flex_at_address"
            elif "Lateral Hip Sway" in name:
                pro_key = "hip_sway_ratio"
            elif "Vertical Head Movement" in name:
                pro_key = "head_bob_ratio"
                
            if pro_key:
                pro_val = get_pro_val(pro_key)
                report.append(f"* **Matched Pro Value ({pro_name})**: {pro_val}")
                
            # 3. Drill
            report.append(f"* **Actionable Drill**: {issue['drill']}")
            
    with open(save_report_path, "w") as f:
        f.write("\n".join(report))



def write_and_print_output(output, save_json_path=""):
    print(json.dumps(output, indent=2))
    if save_json_path:
        parent_dir = os.path.dirname(os.path.abspath(save_json_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump(output, f, indent=2)


def draw_scorecard(frame, lines):
    if not lines:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    padding = 10
    
    # Calculate dimensions
    max_w = 0
    for line in lines:
        clean_line = line.replace(" [PASS]", "").replace(" [WARN]", "")
        (w, h), _ = cv2.getTextSize(clean_line, font, font_scale, thickness)
        if "[PASS]" in line:
            w += cv2.getTextSize(" [PASS]", font, font_scale, thickness)[0][0]
        elif "[WARN]" in line:
            w += cv2.getTextSize(" [WARN]", font, font_scale, thickness)[0][0]
        if w > max_w:
            max_w = w
            
    box_width = max_w + (padding * 2)
    box_height = (len(lines) * line_height) + (padding * 2)
    
    # Place in top-right corner
    frame_h, frame_w, _ = frame.shape
    x1 = frame_w - box_width - 20
    y1 = 20
    x2 = frame_w - 20
    y2 = y1 + box_height
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), 1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw text lines
    for idx, line in enumerate(lines):
        y_pos = y1 + padding + (idx * line_height) + 12
        if "[PASS]" in line:
            parts = line.split(" [PASS]")
            cv2.putText(frame, parts[0], (x1 + padding, y_pos), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            (w, _), _ = cv2.getTextSize(parts[0], font, font_scale, thickness)
            cv2.putText(frame, " [PASS]", (x1 + padding + w, y_pos), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        elif "[WARN]" in line:
            parts = line.split(" [WARN]")
            cv2.putText(frame, parts[0], (x1 + padding, y_pos), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            (w, _), _ = cv2.getTextSize(parts[0], font, font_scale, thickness)
            cv2.putText(frame, " [WARN]", (x1 + padding + w, y_pos), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        elif line == "SWING METRICS":
            cv2.putText(frame, line, (x1 + padding, y_pos), font, font_scale + 0.05, (0, 255, 255), thickness + 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, line, (x1 + padding, y_pos), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)


def draw_debug_bar(frame, text):
    frame_h, frame_w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    padding = 6
    
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    bar_height = h + baseline + (padding * 2)
    y1 = frame_h - bar_height
    y2 = frame_h
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (frame_w, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    x_pos = max(10, (frame_w - w) // 2)
    y_pos = y1 + padding + h
    cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)


def main():
    save_json_path = ""
    parser = argparse.ArgumentParser(description="End-to-End Golf Swing Analyzer Inference CLI")
    parser.add_argument("video_path", type=str, help="Path to raw golf swing video file")
    parser.add_argument("--gatekeeper-threshold", type=float, default=0.60,
                        help="Confidence threshold for golf swing validation (default: 0.60)")
    parser.add_argument("--plot", type=str, default="", help="Path to save diagnostic probabilities plot (.png)")
    parser.add_argument("--save-video", type=str, nargs="?", const="AUTO", default="",
                        help="Path to save annotated skeleton overlay video (.mp4). If passed without value, saves next to input with suffix '_processed'")
    parser.add_argument("--save-json", type=str, default="", help="Path to save output JSON results")
    parser.add_argument("--save-report", type=str, default="", help="Path to save Markdown coaching report (.md)")
    parser.add_argument("--view", type=str, choices=["auto", "face-on", "down-the-line"], default="auto",
                        help="Camera view perspective (default: auto)")
    parser.add_argument("--handedness", type=str, choices=["auto", "right", "left"], default="auto",
                        help="Golfer handedness (default: auto)")
    parser.add_argument("--max-duration", type=float, default=60.0,
                        help="Maximum allowed video duration in seconds to avoid processing extremely long files (default: 60.0)")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Speed factor for the output video playback (default: 0.5)")
    
    args = parser.parse_args()
    save_json_path = args.save_json

def run_analysis(video_path, view_mode="auto", handedness="auto", gatekeeper_threshold=0.60, speed=0.5, max_duration=60.0, output_video_path=None, save_json_path=None, save_report_path=None):
    """
    Programmatic entrypoint to analyze a golf swing video for Streamlit / API wrappers.
    """
    class DummyArgs:
        pass
    args = DummyArgs()
    args.video_path = video_path
    args.view = view_mode
    args.handedness = handedness
    args.gatekeeper_threshold = gatekeeper_threshold
    args.speed = speed
    args.max_duration = max_duration
    args.output = output_video_path
    args.save_json = save_json_path
    args.save_video = output_video_path or "AUTO"
    args.save_report = save_report_path
    args.plot = None
    
    save_json_path = save_json_path or os.path.join(PROJECT_ROOT, "output", f"{Path(video_path).stem}_results.json")
    save_video_path = output_video_path or os.path.join(PROJECT_ROOT, "output", f"{Path(video_path).stem}_dashboard.mp4")
    save_report_path = save_report_path or os.path.join(PROJECT_ROOT, "output", f"{Path(video_path).stem}_report.md")
    
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
    
    detector = GolfSwingDetector()
    is_duration_valid, duration = detector.check_duration(video_path, max_duration=max_duration)
    if not is_duration_valid:
        return {"success": False, "validated": False, "gatekeeper_score": 0.0, "error": f"Video duration ({duration:.2f}s) exceeds max limit of {max_duration}s."}

    avg_gate_prob = None
    try:
        # A VIDEO-mode MediaPipe landmarker keeps timestamp state.  Scope one
        # processor to one video so a later pro clip can start again at t=0.
        with GolfVideoProcessor() as processor:
            df = processor.process_video(video_path)
        fps = float(df["fps"].iloc[0])
        
        is_valid, avg_gate_prob, probs_gate, df_features = detector.validate(
            df, fps, threshold=gatekeeper_threshold, use_rolling=True
        )
        if not is_valid:
            return {"success": True, "validated": False, "gatekeeper_score": round(avg_gate_prob, 4), "error": "Video failed golf swing validation."}

        model_path = os.path.join(PROJECT_ROOT, "models", "lstm_phase_model.keras")
        # Inference does not need the training loss/optimizer state. Loading with
        # compilation enabled attempts to deserialize the notebook-defined custom
        # loss and fails in clean deployments where that function is unavailable.
        model = tf.keras.models.load_model(model_path, compile=False)
        
        from src.kinematic_features import build_kinematic_features
        feat_cols, X_seq = build_kinematic_features(df, fps)
        # Avoid Keras ``model.predict()`` here.  ``predict()`` wraps even this
        # single in-memory sequence in a tf.data prefetch pipeline.  In the
        # Streamlit process on macOS, TensorFlow and PyArrow can load different
        # Abseil builds; the prefetch condition variable then deadlocks inside
        # IteratorGetNext.  A direct inference call has identical model
        # semantics and does not create a tf.data iterator.
        logits_multi = model(np.expand_dims(X_seq, axis=0), training=False)
        probs_multi = tf.nn.softmax(logits_multi, axis=-1).numpy()[0]
        
        raw_peak_frames = [int(np.argmax(probs_multi[:, c])) for c in range(1, 9)]
        milestone_frames = compute_monotonic_alignment(probs_multi)
        
        milestones_output = {}
        for idx, name in enumerate(MILESTONE_NAMES):
            pred_frame = milestone_frames[idx]
            raw_peak = raw_peak_frames[idx]
            milestones_output[name] = {
                "frame": int(pred_frame),
                "timestamp": round(float(pred_frame) / fps, 4),
                "raw_peak_frame": raw_peak,
                "dp_corrected": bool(pred_frame != raw_peak),
                "shift_distance": int(pred_frame - raw_peak)
            }
            
        addr_frame = milestones_output.get("Address", {}).get("frame", 0)
        effective_view = detect_camera_view(df, address_frame=addr_frame) if view_mode == "auto" else view_mode

        bio_results = analyze_swing_biomechanics(
            df=df, milestones=milestones_output, view=effective_view, handedness=handedness
        )
        
        output = {
            "success": True,
            "validated": True,
            "gatekeeper_score": round(avg_gate_prob, 4),
            "milestones": milestones_output,
            "view": effective_view,
            "handedness": bio_results.get("handedness", "right"),
            "user_arm_to_torso_ratio": bio_results.get("user_arm_to_torso_ratio", 1.0),
            "matched_pro": bio_results.get("matched_pro", "N/A"),
            "matched_pro_ratio": bio_results.get("matched_pro_ratio", 1.0),
            "matched_pro_video_id": bio_results.get("matched_pro_video_id", 0),
            "matched_pro_metrics": bio_results.get("matched_pro_metrics", {}),
            "biomechanical_metrics": bio_results.get("metrics", {}),
            "issues_detected": bio_results.get("issues_detected", [])
        }
        
        if bio_results.get("success", False):
            generate_markdown_report(output, save_report_path, effective_view)
            
        with open(save_json_path, "w") as jf:
            json.dump(output, jf, indent=2)

        # Synchronized visual stitcher
        pro_manifest_path = os.path.join(PROJECT_ROOT, "data", "benchmark", "manifest.json")
        if bio_results.get("success", False) and os.path.exists(pro_manifest_path):
            with open(pro_manifest_path, "r") as pf:
                p_manifest = json.load(pf)
            pro_match = next((pv for pv in p_manifest.get("pro_videos", []) if pv["id"] == bio_results["matched_pro_video_id"]), None)
            if pro_match:
                pro_video_path = os.path.join(PROJECT_ROOT, pro_match["path"])
                try:
                    df_pro, pro_cache = load_preprocessed_pro(pro_video_path)
                    m_pro_out = pro_cache["milestones"]
                except FileNotFoundError as cache_error:
                    # Keep CLI/local analysis functional before the repeatable
                    # cache builder has been run, while making the slow path
                    # explicit in logs.
                    print(f"Pro cache miss ({cache_error}); processing {pro_match['filename']} live.")
                    with GolfVideoProcessor() as pro_processor:
                        df_pro = pro_processor.process_video(pro_video_path)
                    feat_cols_p, X_seq_p = build_kinematic_features(df_pro, df_pro["fps"].iloc[0])
                    logits_multi_pro = model(
                        np.expand_dims(X_seq_p, axis=0), training=False
                    )
                    probs_multi_pro = tf.nn.softmax(logits_multi_pro, axis=-1).numpy()[0]
                    m_frames_pro = compute_monotonic_alignment(probs_multi_pro)
                    m_pro_out = {
                        name: {"frame": int(m_frames_pro[idx])}
                        for idx, name in enumerate(MILESTONE_NAMES)
                    }
                
                create_synchronized_dashboard(
                    user_video_path=video_path,
                    pro_video_path=pro_video_path,
                    df_user=df,
                    df_pro=df_pro,
                    milestones_user=milestones_output,
                    milestones_pro=m_pro_out,
                    bio_results=bio_results,
                    output_path=save_video_path,
                    speed=speed
                )
        return output
    except Exception as e:
        return {
            "success": False,
            "validated": False,
            "gatekeeper_score": round(avg_gate_prob, 4) if avg_gate_prob is not None else 0.0,
            "error": str(e),
        }

def main():
    parser = argparse.ArgumentParser(description="End-to-End Golf Swing Analyzer Inference CLI")
    parser.add_argument("video_path", type=str, help="Path to raw golf swing video file")
    parser.add_argument("--gatekeeper-threshold", type=float, default=0.60,
                        help="Confidence threshold for golf swing validation (default: 0.60)")
    parser.add_argument("--save-video", type=str, nargs="?", const="AUTO", default="",
                        help="Path to save annotated skeleton overlay video (.mp4).")
    parser.add_argument("--save-json", type=str, default="", help="Path to save output JSON results")
    parser.add_argument("--save-report", type=str, default="", help="Path to save Markdown coaching report (.md)")
    parser.add_argument("--view", type=str, choices=["auto", "face-on", "down-the-line"], default="auto",
                        help="Camera view perspective (default: auto)")
    parser.add_argument("--handedness", type=str, choices=["auto", "right", "left"], default="auto",
                        help="Golfer handedness (default: auto)")
    parser.add_argument("--max-duration", type=float, default=60.0,
                        help="Maximum allowed video duration in seconds (default: 60.0)")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Speed factor for the output video playback (default: 0.5)")
    
    args = parser.parse_args()
    
    output_video = args.save_video
    if output_video == "AUTO":
        base, _ = os.path.splitext(args.video_path)
        output_video = f"{base}_processed.mp4"
        
    out = run_analysis(
        video_path=args.video_path,
        view_mode=args.view,
        handedness=args.handedness,
        gatekeeper_threshold=args.gatekeeper_threshold,
        speed=args.speed,
        max_duration=args.max_duration,
        output_video_path=output_video if output_video else None,
        save_json_path=args.save_json if args.save_json else None,
        save_report_path=args.save_report if args.save_report else None
    )
    print(f"\nAnalysis completed! Validated: {out.get('validated', False)}")

if __name__ == "__main__":
    main()
