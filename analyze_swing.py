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

def main():
    save_json_path = ""
    parser = argparse.ArgumentParser(description="End-to-End Golf Swing Analyzer Inference CLI")
    parser.add_argument("video_path", type=str, help="Path to raw golf swing video file")
    parser.add_argument("--gatekeeper-threshold", type=float, default=0.7346,
                        help="Confidence threshold for golf swing validation (default: 0.7346)")
    parser.add_argument("--plot", type=str, default="", help="Path to save diagnostic probabilities plot (.png)")
    parser.add_argument("--save-video", type=str, default="", help="Path to save annotated skeleton overlay video (.mp4)")
    parser.add_argument("--save-json", type=str, default="", help="Path to save output JSON results")
    parser.add_argument("--save-report", type=str, default="", help="Path to save Markdown coaching report (.md)")
    parser.add_argument("--view", type=str, choices=["face-on", "down-the-line"], default="face-on",
                        help="Camera view perspective (default: face-on)")
    parser.add_argument("--handedness", type=str, choices=["auto", "right", "left"], default="auto",
                        help="Golfer handedness (default: auto)")
    
    args = parser.parse_args()
    save_json_path = args.save_json

    
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
                "shift_distance": int(shift_dist)
            }
            
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
            
        # 9. Save annotated skeleton overlay video if requested
        if args.save_video:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_video)), exist_ok=True)
            cap = cv2.VideoCapture(args.video_path)
            
            # Use 'mp4v' for writing mp4
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
            
            # Create a reverse mapping of frame index to milestone name for fast display check
            frame_to_milestone = {meta["frame"]: name for name, meta in milestones_output.items()}
            
            frame_idx = 0
            active_label = ""
            label_cooldown = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 1. Draw Skeleton Overlay
                row = df.iloc[frame_idx]
                draw_skeleton(frame, row, prefix="smooth_")
                
                # 2. Check if a milestone was reached on this frame
                if frame_idx in frame_to_milestone:
                    active_label = frame_to_milestone[frame_idx].upper()
                    label_cooldown = 10 # Display label for 10 frames
                    
                # 3. Draw transient milestone label overlay if active
                if label_cooldown > 0 and active_label:
                    text = f"MILESTONE: {active_label}"
                    # Font settings
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 1.0
                    thickness = 2
                    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Draw a nice semi-transparent black background box for readability
                    bx = 20
                    by = height - 100
                    cv2.rectangle(frame, (bx - 10, by - th - 10), (bx + tw + 10, by + baseline + 10), (0, 0, 0), -1)
                    # Text overlay
                    cv2.putText(frame, text, (bx, by), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    
                    label_cooldown -= 1
                    
                # 4. Status Bar Overlay at top
                status_text = f"Golf Validated (Score: {avg_gate_prob:.3f}) | Frame: {frame_idx} | Time: {frame_idx/fps:.2f}s"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (stw, sth), sbaseline = cv2.getTextSize(status_text, font, font_scale, thickness)
                cv2.rectangle(frame, (10, 10), (20 + stw, 20 + sth + sbaseline), (0, 0, 0), -1)
                cv2.putText(frame, status_text, (15, 20 + sth), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
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
