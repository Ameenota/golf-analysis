import pandas as pd

# List of high-movement joints used for sequence classification
HIGH_MOVEMENT_JOINTS = [
    'left_wrist', 'right_wrist', 
    'left_elbow', 'right_elbow', 
    'left_shoulder', 'right_shoulder', 
    'left_hip', 'right_hip'
]

def engineer_sliding_window(df, joints=HIGH_MOVEMENT_JOINTS, shift_steps=5):
    """
    Appends coordinates from T-5 and T+5 as new columns to frame T's row.
    Groups by video_id to avoid bleeding across different videos.
    Pads boundaries (first and last 5 frames of a video) using backfill and forward-fill.
    """
    df_features = df.copy()
    
    for joint in joints:
        for coord in ['x', 'y']:
            col_name = f'norm_{joint}_{coord}'
            if col_name not in df_features.columns:
                continue
                
            # Shift backwards (T-5)
            df_features[f'{col_name}_t-5'] = df_features.groupby('video_id')[col_name].shift(shift_steps)
            df_features[f'{col_name}_t-5'] = df_features.groupby('video_id')[f'{col_name}_t-5'].bfill()
            
            # Shift forwards (T+5)
            df_features[f'{col_name}_t+5'] = df_features.groupby('video_id')[col_name].shift(-shift_steps)
            df_features[f'{col_name}_t+5'] = df_features.groupby('video_id')[f'{col_name}_t+5'].ffill()
            
    return df_features
