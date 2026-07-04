import cv2
import pandas as pd

# Joint connection list defining the human skeletal structure
POSE_CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle")
]

# List of key joints to draw as dots
POSE_JOINTS = [
    "left_shoulder", "right_shoulder", 
    "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", 
    "left_hip", "right_hip", 
    "left_knee", "right_knee", 
    "left_ankle", "right_ankle"
]

def draw_skeleton(frame, row, prefix="smooth_", line_color=(0, 255, 0), joint_color=(0, 0, 255)):
    """Draws skeletal lines and joint dots on the frame."""
    # Draw connection lines
    for start_joint, end_joint in POSE_CONNECTIONS:
        start_x = row[f"{prefix}{start_joint}_x"]
        start_y = row[f"{prefix}{start_joint}_y"]
        end_x = row[f"{prefix}{end_joint}_x"]
        end_y = row[f"{prefix}{end_joint}_y"]
        
        if not pd.isna(start_x) and not pd.isna(start_y) and not pd.isna(end_x) and not pd.isna(end_y):
            cv2.line(frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), line_color, 2)
            
    # Draw joint dots
    for joint in POSE_JOINTS:
        x = row[f"{prefix}{joint}_x"]
        y = row[f"{prefix}{joint}_y"]
        
        if not pd.isna(x) and not pd.isna(y):
            cv2.circle(frame, (int(x), int(y)), 4, joint_color, -1)
