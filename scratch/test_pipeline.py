import sys
import os
import cv2
import numpy as np
import pandas as pd

# Add the project directory to python path so we can import src.data_processor
sys.path.append("/Users/sagar/Documents/ML/golf-analysis")
from src.data_processor import GolfVideoProcessor

def main():
    # Define paths
    video_path = "/Users/sagar/Documents/ML/golf-analysis/data/videos_160/videos_160/0.mp4"
    processed_dir = "/Users/sagar/Documents/ML/golf-analysis/data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Step 1: Running the landmark extraction, smoothing, and normalization pipeline...")
    # Initialize the processor (it will download the model file on first run)
    processor = GolfVideoProcessor()
    
    # Process the video
    df = processor.process_video(video_path)
    
    print(f"Step 2: Verification:")
    print(f"  - Extracted shape: {df.shape}")
    print(f"  - Calculated Torso Scale: {df['torso_scale'].iloc[0]:.2f} pixels")
    
    # Save the processed DataFrame to CSV
    csv_path = os.path.join(processed_dir, "video_0_processed.csv")
    df.to_csv(csv_path, index=False)
    print(f"  - Saved master processed dataset to {csv_path}")
    
    # Generate the visual demo video
    width = int(df["width"].iloc[0])
    height = int(df["height"].iloc[0])
    fps = df["fps"].iloc[0]
    
    demo_video_path = os.path.join(processed_dir, "wrist_demo.mp4")
    # Open VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(demo_video_path, fourcc, fps, (width, height))
    
    print("Step 3: Creating visual demo video (raw vs. smoothed wrist trajectory)...")
    raw_history = []
    smooth_history = []
    
    for idx, row in df.iterrows():
        # Create a blank black canvas of the same dimensions as the original video
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get right wrist coordinates
        rx = row["raw_right_wrist_x"]
        ry = row["raw_right_wrist_y"]
        sx = row["smooth_right_wrist_x"]
        sy = row["smooth_right_wrist_y"]
        
        if not pd.isna(rx) and not pd.isna(ry):
            raw_history.append((int(rx), int(ry)))
        if not pd.isna(sx) and not pd.isna(sy):
            smooth_history.append((int(sx), int(sy)))
            
        # Draw historical trajectories
        # Raw is red (thin lines)
        for i in range(1, len(raw_history)):
            cv2.line(frame, raw_history[i-1], raw_history[i], (0, 0, 150), 1)
        # Smoothed is green (thicker lines)
        for i in range(1, len(smooth_history)):
            cv2.line(frame, smooth_history[i-1], smooth_history[i], (0, 200, 0), 2)
            
        # Draw current points
        if not pd.isna(rx) and not pd.isna(ry):
            cv2.circle(frame, (int(rx), int(ry)), 6, (0, 0, 255), -1)  # Red dot
        if not pd.isna(sx) and not pd.isna(sy):
            cv2.circle(frame, (int(sx), int(sy)), 6, (0, 255, 0), -1)  # Green dot
            
        out.write(frame)
        
    out.release()
    print(f"Step 4: Demo video saved to {demo_video_path}")
    print("Processing complete!")

if __name__ == "__main__":
    main()
