import os
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from src.feature_engineer import engineer_sliding_window, HIGH_MOVEMENT_JOINTS

class GolfSwingDetector:
    """
    Gatekeeper classifier to validate whether a video contains a golf swing.
    Uses an XGBoost binary classifier trained on sliding window landmark features.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models", "golf_binary_detector.json")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost gatekeeper model not found at {model_path}")
            
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
    def check_duration(self, video_path, max_duration=60.0):
        """
        Loads the video metadata using OpenCV and validates if the duration exceeds the limit.
        Returns:
            is_valid (bool): True if duration <= max_duration, False otherwise.
            duration (float): Video duration in seconds.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")
            
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0:
            fps = 30.0  # Fallback
            
        duration = float(frames / fps)
        is_valid = duration <= max_duration
        return is_valid, duration
        
    def predict_probs(self, df):
        """
        Calculates sliding window features and returns frame-level golf swing probabilities.
        Returns:
            probs (np.ndarray): Array of float probabilities.
            df_features (pd.DataFrame): Engineered sliding window features DataFrame.
        """
        df_features = engineer_sliding_window(df, joints=HIGH_MOVEMENT_JOINTS)
        feature_cols = sorted([c for c in df_features.columns if c.startswith("norm_")])
        
        if len(feature_cols) != 98:
            raise ValueError(f"Feature extraction yielded {len(feature_cols)} features. Expected exactly 98.")
            
        X = df_features[feature_cols].values
        probs = self.model.predict_proba(X)[:, 1]
        return probs, df_features
        
    def validate(self, df, fps, threshold=0.7346, use_rolling=True):
        """
        Validates the video using a global mean or a rolling window max.
        Returns:
            is_valid (bool): True if the validation score meets the threshold.
            score (float): The computed validation score.
            probs (np.ndarray): Frame-level probabilities.
            df_features (pd.DataFrame): Sliding window features.
        """
        probs, df_features = self.predict_probs(df)
        N = len(df)
        
        if use_rolling:
            # 2.0 seconds rolling window size in frames
            window_size = min(N, int(fps * 2.0))
            if len(probs) >= window_size and window_size > 0:
                rolling_probs = pd.Series(probs).rolling(window=window_size).mean()
                score = float(rolling_probs.max())
                if pd.isna(score):
                    score = float(np.mean(probs))
            else:
                score = float(np.mean(probs))
        else:
            score = float(np.mean(probs))
            
        is_valid = score >= threshold
        return is_valid, score, probs, df_features
