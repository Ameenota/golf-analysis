import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

class GolfVideoProcessor:
    """
    Biomechanical pipeline processor for raw golf swing videos.
    Extracts, smooths, and normalizes landmarks using the modern MediaPipe Tasks Pose Landmarker.
    """
    def __init__(self, model_type="heavy"):
        # Resolve model directory and download if not exists
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_filename = f"pose_landmarker_{model_type}.task"
        self.model_path = os.path.join(model_dir, model_filename)
        
        if not os.path.exists(self.model_path):
            url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{model_type}/float16/1/{model_filename}"
            print(f"Model file not found. Downloading from {url}...")
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")
            
        # Initialize Pose Landmarker detector
        self.base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=5,  # Enable multiple people detection
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
        
        self.landmark_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Closes the MediaPipe detector to release resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()

    def extract_raw_landmarks(self, video_path):
        """
        Opens a video file and extracts raw pixel coordinates for all 33 pose landmarks.
        Interpolates any frames where the pose detector fails to detect landmarks.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback in case metadata is missing
            
        frames_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Calculate timestamp in ms (required for MediaPipe Tasks VIDEO mode)
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            
            frame_row = {
                "frame_index": frame_idx,
            }
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                frame_row["num_people"] = len(result.pose_landmarks)
                # Use the first detected person
                landmarks = result.pose_landmarks[0]
                for i, name in enumerate(self.landmark_names):
                    lm = landmarks[i]
                    # Convert raw normalized coordinates [0, 1] to pixel space
                    frame_row[f"raw_{name}_x"] = lm.x * width
                    frame_row[f"raw_{name}_y"] = lm.y * height
                    frame_row[f"raw_{name}_z"] = lm.z  # Relative depth
                    frame_row[f"raw_{name}_vis"] = lm.visibility
            else:
                frame_row["num_people"] = 0
                # If pose detection fails, fill with NaNs to be interpolated
                for name in self.landmark_names:
                    frame_row[f"raw_{name}_x"] = np.nan
                    frame_row[f"raw_{name}_y"] = np.nan
                    frame_row[f"raw_{name}_z"] = np.nan
                    frame_row[f"raw_{name}_vis"] = np.nan
                    
            frames_data.append(frame_row)
            frame_idx += 1
            
        cap.release()
        
        df = pd.DataFrame(frames_data)
        
        # Interpolate NaNs linearly across frames to ensure continuous signal for filtering
        df = df.interpolate(method="linear", limit_direction="both")
        return df, width, height, fps

    def process_video(self, video_path, video_id=None, window=11, polyorder=3):
        """
        Executes the full pipeline:
        1. Landmark extraction in pixel space.
        2. Signal smoothing via Savitzky-Golay filtering.
        3. Mid-hip centering & static torso scale normalization.
        """
        if video_id is None:
            video_id = os.path.basename(video_path).split('.')[0]
            
        df, width, height, fps = self.extract_raw_landmarks(video_path)
        
        if df.empty or len(df) == 0:
            raise ValueError(f"No frames could be decoded from video: {video_path}")
        
        # Verify length is sufficient for Savitzky-Golay window
        n_frames = len(df)
        if n_frames < window:
            window = n_frames if n_frames % 2 != 0 else n_frames - 1
            if window < 3:
                window = 3
                
        # 1. Apply Savitzky-Golay filter to raw coordinates
        for name in self.landmark_names:
            for coord in ['x', 'y']:
                col_name = f"raw_{name}_{coord}"
                smooth_col_name = f"smooth_{name}_{coord}"
                if df[col_name].isna().all():
                    df[smooth_col_name] = np.nan
                else:
                    df[smooth_col_name] = savgol_filter(
                        df[col_name].values, 
                        window_length=window, 
                        polyorder=min(polyorder, window - 1)
                    )
                
        # 2. Compute mid-hip and mid-shoulder using smoothed pixel coordinates
        df["mid_hip_x"] = (df["smooth_left_hip_x"] + df["smooth_right_hip_x"]) / 2.0
        df["mid_hip_y"] = (df["smooth_left_hip_y"] + df["smooth_right_hip_y"]) / 2.0
        
        df["mid_shoulder_x"] = (df["smooth_left_shoulder_x"] + df["smooth_right_shoulder_x"]) / 2.0
        df["mid_shoulder_y"] = (df["smooth_left_shoulder_y"] + df["smooth_right_shoulder_y"]) / 2.0
        
        # Calculate Torso Length (Euclidean distance from mid-shoulder to mid-hip)
        df["torso_length"] = np.sqrt(
            (df["mid_shoulder_x"] - df["mid_hip_x"])**2 + 
            (df["mid_shoulder_y"] - df["mid_hip_y"])**2
        )
        
        # 3. Calculate TORSO_SCALE from the first 5 frames (representing Address setup)
        n_scale_frames = min(5, len(df))
        torso_scale = df["torso_length"].iloc[:n_scale_frames].mean()
        
        # If torso_scale is 0 or NaN, default to 1.0 to avoid division by zero
        if pd.isna(torso_scale) or torso_scale == 0:
            torso_scale = 1.0
            
        # Add metadata columns
        df["video_id"] = video_id
        df["torso_scale"] = torso_scale
        df["width"] = width
        df["height"] = height
        df["fps"] = fps
        
        # 4. Center relative to mid-hip and scale by torso_scale
        norm_cols = {}
        for name in self.landmark_names:
            for coord in ['x', 'y']:
                smooth_col = f"smooth_{name}_{coord}"
                norm_col = f"norm_{name}_{coord}"
                mid_col = f"mid_hip_{coord}"
                norm_cols[norm_col] = (df[smooth_col] - df[mid_col]) / torso_scale
                
        df = pd.concat([df, pd.DataFrame(norm_cols, index=df.index)], axis=1)
        
        # 5. Compute torso verticality and orientation relative to vertical y-axis
        dx = df["mid_shoulder_x"] - df["mid_hip_x"]
        dy = df["mid_shoulder_y"] - df["mid_hip_y"]
        df["torso_angle_deg"] = np.degrees(np.arctan2(np.abs(dx), np.abs(dy) + 1e-10))
        df["is_upright"] = (df["mid_shoulder_y"] < df["mid_hip_y"]).astype(int)
        
        return df
