import os
import cv2
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = '/Users/sagar/Documents/ML/golf-analysis'
sys.path.append(PROJECT_ROOT)

from src.visualizer import draw_skeleton

def main():
    processor = GolfVideoProcessor()
    df = processor.process_video('data/videos_160/videos_160/0.mp4', video_id=0)
    processor.close()
    
    video_path = 'data/videos_160/videos_160/0.mp4'
    cap = cv2.VideoCapture(video_path)
    
    frames_to_save = [65, 68, 73, 77]
    
    for f_idx in frames_to_save:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {f_idx}")
            continue
            
        row = df.iloc[f_idx]
        draw_skeleton(frame, row, prefix="smooth_", line_color=(0, 255, 0), joint_color=(0, 0, 255))
        
        # Add frame index label
        cv2.rectangle(frame, (10, 10), (150, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {f_idx}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out_path = f"data/processed/frame_{f_idx}.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")
        
    cap.release()

if __name__ == "__main__":
    main()
