import sys
import os
import cv2
import numpy as np
import pandas as pd

# Add the project directory to python path
sys.path.append("/Users/sagar/Documents/ML/golf-analysis")

from src.visualizer import draw_skeleton, POSE_CONNECTIONS, POSE_JOINTS

def main():
    csv_path = "/Users/sagar/Documents/ML/golf-analysis/data/processed/video_0_processed.csv"
    video_path = "/Users/sagar/Documents/ML/golf-analysis/data/videos_160/videos_160/0.mp4"
    processed_dir = "/Users/sagar/Documents/ML/golf-analysis/data/processed"
    
    if not os.path.exists(csv_path):
        print(f"Error: Processed CSV not found at {csv_path}. Run test_pipeline.py first.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Open the original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 1. Output Path for Skeleton Overlaid on Original Video
    overlay_out_path = os.path.join(processed_dir, "skeleton_overlay.mp4")
    # 2. Output Path for Blank Stick Figure Animation
    blank_out_path = os.path.join(processed_dir, "skeleton_blank.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_overlay = cv2.VideoWriter(overlay_out_path, fourcc, fps, (width, height))
    out_blank = cv2.VideoWriter(blank_out_path, fourcc, fps, (width, height))
    
    print("Generating skeleton demonstrations...")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= len(df):
            break
            
        row = df.iloc[frame_idx]
        
        # Frame A: Overlay skeleton on the original frame
        overlay_frame = frame.copy()
        draw_skeleton(overlay_frame, row, prefix="smooth_", line_color=(0, 255, 0), joint_color=(0, 0, 255))
        
        # Frame B: Draw stick figure on a blank black frame
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_skeleton(blank_frame, row, prefix="smooth_", line_color=(255, 255, 255), joint_color=(0, 255, 0))
        
        out_overlay.write(overlay_frame)
        out_blank.write(blank_frame)
        
        frame_idx += 1
        
    cap.release()
    out_overlay.release()
    out_blank.release()
    
    print(f"Success! Generated:")
    print(f"  - Skeleton Overlay video: {overlay_out_path}")
    print(f"  - Blank Stick Figure video: {blank_out_path}")

if __name__ == "__main__":
    main()
