import os
import argparse
import json
import numpy as np
import cv2
import pandas as pd
import xgboost as xgb
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


from src.data_processor import GolfVideoProcessor
from src.feature_engineer import engineer_sliding_window, HIGH_MOVEMENT_JOINTS
from src.alignment import compute_monotonic_alignment
from src.visualizer import draw_skeleton
from src.coaching_engine import analyze_swing_biomechanics
from src.visual_stitcher import create_synchronized_dashboard

MILESTONE_NAMES = [
    "Address",
    "Toe-Up",
    "Top of Backswing",
    "Downswing",
    "Impact",
    "Release",
    "Follow-Through",
    "Finish"
]

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
    parser.add_argument("--gatekeeper-threshold", type=float, default=0.7346,
                        help="Confidence threshold for golf swing validation (default: 0.7346)")
    parser.add_argument("--plot", type=str, default="", help="Path to save diagnostic probabilities plot (.png)")
    parser.add_argument("--save-video", type=str, nargs="?", const="AUTO", default="",
                        help="Path to save annotated skeleton overlay video (.mp4). If passed without value, saves next to input with suffix '_processed'")
    parser.add_argument("--save-json", type=str, default="", help="Path to save output JSON results")
    parser.add_argument("--save-report", type=str, default="", help="Path to save Markdown coaching report (.md)")
    parser.add_argument("--view", type=str, choices=["face-on", "down-the-line"], default="down-the-line",
                        help="Camera view perspective (default: down-the-line)")
    parser.add_argument("--handedness", type=str, choices=["auto", "right", "left"], default="auto",
                        help="Golfer handedness (default: auto)")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Speed factor for the output video playback (default: 0.5)")
    
    args = parser.parse_args()
    save_json_path = args.save_json

    # Resolve AUTO name for save-video
    save_video_path = args.save_video
    if save_video_path == "AUTO":
        base, ext = os.path.splitext(args.video_path)
        save_video_path = f"{base}_processed.mp4"


    
    if not os.path.exists(args.video_path):
        output = {
            "success": False,
            "validated": False,
            "gatekeeper_score": 0.0,
            "error": f"Video file not found: {args.video_path}",
            "milestones": None
        }
        write_and_print_output(output, save_json_path)
        return
        
    try:
        # 2. Run MediaPipe Landmark Extraction & Smoothing
        processor = GolfVideoProcessor()
        df = processor.process_video(args.video_path)
        
        width = int(df["width"].iloc[0])
        height = int(df["height"].iloc[0])
        fps = float(df["fps"].iloc[0])
        N = len(df)
        
        # 3. Sliding Window Feature Engineering for Gatekeeper
        df_features = engineer_sliding_window(df, joints=HIGH_MOVEMENT_JOINTS)
        feature_cols = sorted([c for c in df_features.columns if c.startswith("norm_")])
        
        if len(feature_cols) != 98:
            raise ValueError(f"Feature extraction yielded {len(feature_cols)} normalized features. Expected exactly 98.")
            
        # 4. XGBoost Binary Gatekeeper Validation
        gate_model = xgb.XGBClassifier()
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        gate_model_path = os.path.join(model_dir, "golf_binary_detector.json")
        
        if not os.path.exists(gate_model_path):
            raise FileNotFoundError(f"XGBoost gatekeeper model not found at {gate_model_path}")
            
        gate_model.load_model(gate_model_path)
        
        X_gate = df_features[feature_cols].values
        probs_gate = gate_model.predict_proba(X_gate)[:, 1]
        avg_gate_prob = float(np.mean(probs_gate))
        
        # Check validation threshold
        if avg_gate_prob < args.gatekeeper_threshold:
            output = {
                "success": False,
                "validated": False,
                "gatekeeper_score": round(avg_gate_prob, 4),
                "error": f"Video failed golf swing validation. Gatekeeper score of {avg_gate_prob:.4f} is below the threshold of {args.gatekeeper_threshold:.4f}.",
                "milestones": None
            }
            write_and_print_output(output, save_json_path)
            return
            
        # 5. Keras LSTM Milestone Inference
        lstm_model_path = os.path.join(model_dir, "lstm_phase_model.keras")
        if not os.path.exists(lstm_model_path):
            raise FileNotFoundError(f"Keras LSTM model not found at {lstm_model_path}")
            
        lstm_model = tf.keras.models.load_model(lstm_model_path, compile=False)
        
        base_features = sorted([c for c in feature_cols if not (c.endswith("t-5") or c.endswith("t+5"))])
        if len(base_features) != 66:
            raise ValueError(f"Base landmark feature extraction yielded {len(base_features)} features. Expected exactly 66.")
            
        x_seq = df_features[base_features].values.astype(np.float32)
        
        # Pad with 0.0 to 1280
        max_len = 1280
        if N < max_len:
            pad_len = max_len - N
            x_padded = np.pad(x_seq, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
        else:
            x_padded = x_seq[:max_len]
            
        x_batch = np.expand_dims(x_padded, axis=0) # (1, 1280, 66)
        
        # Run prediction
        logits = lstm_model(x_batch, training=False)
        probs_lstm = tf.nn.softmax(logits, axis=-1)
        probs_lstm = tf.squeeze(probs_lstm, axis=0).numpy() # (1280, 9)
        v_probs = probs_lstm[:N, :] # (N, 9)
        
        # 6. DP Monotonic Chronological Alignment
        milestone_frames = compute_monotonic_alignment(v_probs)
        
        # Prepare milestone results and check DP adjustments
        milestones_output = {}
        for idx, name in enumerate(MILESTONE_NAMES):
            c = idx + 1
            pred_frame = milestone_frames[idx]
            raw_peak_frame = int(np.argmax(v_probs[:, c]))
            dp_corrected = (pred_frame != raw_peak_frame)
            shift_dist = pred_frame - raw_peak_frame
            
            milestones_output[name] = {
                "frame": int(pred_frame),
                "timestamp": round(float(pred_frame) / fps, 4),
                "raw_peak_frame": raw_peak_frame,
                "dp_corrected": bool(dp_corrected),
                "shift_distance": int(shift_dist),
                "heuristic_corrected": False
            }
            
        # 6b. Apply Physical Heuristic Adjustments
        try:
            # 1. Adjust Impact to lowest hand height between Downswing & Release
            downswing_frame = milestones_output["Downswing"]["frame"]
            release_frame = milestones_output["Release"]["frame"]
            
            if release_frame - downswing_frame > 1:
                start_search = downswing_frame + 1
                end_search = release_frame - 1
                
                downswing_segment = df.iloc[start_search:end_search+1]
                wrist_y = (downswing_segment["smooth_left_wrist_y"] + downswing_segment["smooth_right_wrist_y"]) / 2.0
                lowest_hand_frame = int(wrist_y.idxmax())
                
                old_impact = milestones_output["Impact"]["frame"]
                if old_impact != lowest_hand_frame:
                    milestones_output["Impact"]["frame"] = lowest_hand_frame
                    milestones_output["Impact"]["timestamp"] = round(float(lowest_hand_frame) / fps, 4)
                    milestones_output["Impact"]["heuristic_corrected"] = True
                    milestones_output["Impact"]["shift_distance"] = lowest_hand_frame - milestones_output["Impact"]["raw_peak_frame"]
            
            # 2. Adjust Top of Backswing to highest hand height between Address & Downswing
            address_frame = milestones_output["Address"]["frame"]
            downswing_frame = milestones_output["Downswing"]["frame"]
            
            if downswing_frame - address_frame > 1:
                start_search = address_frame + 1
                end_search = downswing_frame - 1
                
                backswing_segment = df.iloc[start_search:end_search+1]
                wrist_y = (backswing_segment["smooth_left_wrist_y"] + backswing_segment["smooth_right_wrist_y"]) / 2.0
                highest_hand_frame = int(wrist_y.idxmin())
                
                old_top = milestones_output["Top of Backswing"]["frame"]
                if old_top != highest_hand_frame:
                    milestones_output["Top of Backswing"]["frame"] = highest_hand_frame
                    milestones_output["Top of Backswing"]["timestamp"] = round(float(highest_hand_frame) / fps, 4)
                    milestones_output["Top of Backswing"]["heuristic_corrected"] = True
                    milestones_output["Top of Backswing"]["shift_distance"] = highest_hand_frame - milestones_output["Top of Backswing"]["raw_peak_frame"]
                    
            # Sync back milestone_frames list for correct plotting
            milestone_frames = [milestones_output[m_name]["frame"] for m_name in MILESTONE_NAMES]
        except Exception as he:
            pass
            
        # 7. Run Biomechanical Analysis and Pro Matchmaker
        bio_results = analyze_swing_biomechanics(
            df=df,
            milestones=milestones_output,
            view=args.view,
            handedness=args.handedness
        )
        
        output = {
            "success": True,
            "validated": True,
            "gatekeeper_score": round(avg_gate_prob, 4),
            "milestones": milestones_output
        }
        
        if bio_results.get("success", False):
            output.update({
                "handedness": bio_results["handedness"],
                "user_arm_to_torso_ratio": bio_results["user_arm_to_torso_ratio"],
                "matched_pro": bio_results["matched_pro"],
                "matched_pro_ratio": bio_results["matched_pro_ratio"],
                "matched_pro_video_id": bio_results["matched_pro_video_id"],
                "matched_pro_metrics": bio_results["matched_pro_metrics"],
                "biomechanical_metrics": bio_results["metrics"],
                "issues_detected": bio_results["issues_detected"]
            })
            
            # Save Markdown coaching report if requested
            if args.save_report:
                generate_markdown_report(output, args.save_report, args.view)
        else:
            output["biomechanical_error"] = bio_results.get("error", "Unknown biomechanical error")
            
        write_and_print_output(output, save_json_path)

        
        # 8. Save diagnostic probabilities plot if requested
        if args.plot:
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(14, 6))
            
            # Plot probability curves for classes 1-8
            for c in range(1, 9):
                plt.plot(v_probs[:, c], label=MILESTONE_NAMES[c-1], linewidth=2)
                
            # Draw vertical markers at DP chosen frames
            colors = sns.color_palette("husl", 8)
            for idx, frame in enumerate(milestone_frames):
                plt.axvline(x=frame, color=colors[idx], linestyle="--", alpha=0.7, 
                            label=f"DP {MILESTONE_NAMES[idx]}: F{frame}")
                
            plt.title("Milestone Probabilities and Chronological DP Alignment", fontsize=14, fontweight="bold")
            plt.xlabel("Frame Index", fontsize=12)
            plt.ylabel("Probability", fontsize=12)
            plt.xlim(0, N - 1)
            plt.ylim(-0.05, 1.05)
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
            plt.savefig(args.plot, dpi=150)
            plt.close()
            
        # 9. Save         # 9. Save annotated skeleton overlay video if requested
        if save_video_path:
            parent_dir = os.path.dirname(os.path.abspath(save_video_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
                
            # If we successfully ran coaching analysis and matched a pro, run the stitcher!
            if bio_results.get("success", False):
                pro_video_id = bio_results["matched_pro_video_id"]
                project_root = os.path.dirname(os.path.abspath(__file__))
                pro_video_path = os.path.join(project_root, "data/videos_160/videos_160", f"{pro_video_id}.mp4")
                
                if not os.path.exists(pro_video_path):
                    raise FileNotFoundError(f"Matched professional swing video not found at: {pro_video_path}")
                
                print(f"Matched professional: {bio_results['matched_pro']} (ID: {pro_video_id})")
                print("Running landmark extraction and pipeline on pro video...")
                
                # Run landmarks on pro video using a fresh processor instance to reset MediaPipe timestamps
                pro_processor = GolfVideoProcessor()
                df_pro = pro_processor.process_video(pro_video_path)
                pro_processor.close()
                df_features_pro = engineer_sliding_window(df_pro, joints=HIGH_MOVEMENT_JOINTS)
                
                # LSTM sequence mapping on pro
                x_seq_pro = df_features_pro[base_features].values.astype(np.float32)
                N_pro = len(df_pro)
                if N_pro < max_len:
                    pad_len_pro = max_len - N_pro
                    x_padded_pro = np.pad(x_seq_pro, ((0, pad_len_pro), (0, 0)), mode="constant", constant_values=0.0)
                else:
                    x_padded_pro = x_seq_pro[:max_len]
                
                x_batch_pro = np.expand_dims(x_padded_pro, axis=0)
                
                logits_pro = lstm_model(x_batch_pro, training=False)
                probs_lstm_pro = tf.nn.softmax(logits_pro, axis=-1)
                probs_lstm_pro = tf.squeeze(probs_lstm_pro, axis=0).numpy()
                v_probs_pro = probs_lstm_pro[:N_pro, :]
                
                # DP alignment for pro
                milestone_frames_pro = compute_monotonic_alignment(v_probs_pro)
                
                milestones_output_pro = {}
                for idx, name in enumerate(MILESTONE_NAMES):
                    c = idx + 1
                    pred_frame_pro = milestone_frames_pro[idx]
                    raw_peak_pro = int(np.argmax(v_probs_pro[:, c]))
                    
                    milestones_output_pro[name] = {
                        "frame": int(pred_frame_pro),
                        "raw_peak_frame": raw_peak_pro
                    }
                    
                # Apply heuristics to pro
                try:
                    down_p = milestones_output_pro["Downswing"]["frame"]
                    rel_p = milestones_output_pro["Release"]["frame"]
                    if rel_p - down_p > 1:
                        wrist_y_p = (df_pro.iloc[down_p+1:rel_p-1]["smooth_left_wrist_y"] + df_pro.iloc[down_p+1:rel_p-1]["smooth_right_wrist_y"]) / 2.0
                        milestones_output_pro["Impact"]["frame"] = int(wrist_y_p.idxmax())
                        
                    addr_p = milestones_output_pro["Address"]["frame"]
                    down_p = milestones_output_pro["Downswing"]["frame"]
                    if down_p - addr_p > 1:
                        wrist_y_p = (df_pro.iloc[addr_p+1:down_p-1]["smooth_left_wrist_y"] + df_pro.iloc[addr_p+1:down_p-1]["smooth_right_wrist_y"]) / 2.0
                        milestones_output_pro["Top of Backswing"]["frame"] = int(wrist_y_p.idxmin())
                except Exception:
                    pass
                
                # Call synchronized visual stitcher
                create_synchronized_dashboard(
                    user_video_path=args.video_path,
                    pro_video_path=pro_video_path,
                    df_user=df,
                    df_pro=df_pro,
                    milestones_user=milestones_output,
                    milestones_pro=milestones_output_pro,
                    bio_results=bio_results,
                    output_path=save_video_path,
                    speed=args.speed
                )
            else:
                # Simple fallback to user-only rendering if pro matching failed
                print("Pro matching failed. Rendering user skeleton video only.")
                cap = cv2.VideoCapture(args.video_path)
                first_milestone = "Address"
                last_milestone = "Finish"
                if output.get("success", False) and milestones_output and first_milestone in milestones_output and last_milestone in milestones_output:
                    start_frame = max(0, milestones_output[first_milestone]["frame"] - 5)
                    end_frame = min(N - 1, milestones_output[last_milestone]["frame"] + 5)
                else:
                    start_frame = 0
                    end_frame = N - 1
                    
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_fps = fps * args.speed
                out_writer = cv2.VideoWriter(save_video_path, fourcc, out_fps, (width, height))
                
                if start_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_idx = start_frame
                
                while frame_idx <= end_frame:
                    ret, frame = cap.read()
                    if not ret: break
                    draw_skeleton(frame, df.iloc[frame_idx], prefix="smooth_")
                    out_writer.write(frame)
                    frame_idx += 1
                    
                cap.release()
                out_writer.release()
            
    except Exception as e:
        output = {
            "success": False,
            "validated": False,
            "gatekeeper_score": 0.0,
            "error": f"An error occurred during analysis: {str(e)}",
            "milestones": None
        }
        write_and_print_output(output, save_json_path)
        return

if __name__ == "__main__":
    main()
