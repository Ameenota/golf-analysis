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

def main():
    parser = argparse.ArgumentParser(description="End-to-End Golf Swing Analyzer Inference CLI")
    parser.add_argument("video_path", type=str, help="Path to raw golf swing video file")
    parser.add_argument("--gatekeeper-threshold", type=float, default=0.7346,
                        help="Confidence threshold for golf swing validation (default: 0.7346)")
    parser.add_argument("--plot", type=str, default="", help="Path to save diagnostic probabilities plot (.png)")
    parser.add_argument("--save-video", type=str, default="", help="Path to save annotated skeleton overlay video (.mp4)")
    
    args = parser.parse_args()
    
    # 1. Verification of inputs
    if not os.path.exists(args.video_path):
        output = {
            "success": False,
            "validated": false,
            "gatekeeper_score": 0.0,
            "error": f"Video file not found: {args.video_path}",
            "milestones": None
        }
        print(json.dumps(output, indent=2))
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
            print(json.dumps(output, indent=2))
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
            
        # 7. Print output JSON to stdout
        output = {
            "success": True,
            "validated": True,
            "gatekeeper_score": round(avg_gate_prob, 4),
            "milestones": milestones_output
        }
        print(json.dumps(output, indent=2))
        
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
        print(json.dumps(output, indent=2))
        return

if __name__ == "__main__":
    main()
