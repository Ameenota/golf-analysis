import os
import sys
import scipy.io
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = '/Users/sagar/Documents/ML/golf-analysis'
sys.path.append(PROJECT_ROOT)

from src.data_processor import GolfVideoProcessor

def parse_golfdb_mat(mat_path):
    mat_data = scipy.io.loadmat(mat_path)
    db = mat_data['golfDB'][0]
    
    metadata = []
    for i in range(len(db)):
        rec = db[i]
        vid_id = int(rec['id'][0][0])
        events = rec['events'][0]
        
        start_frame = events[0]
        rel_events = events - start_frame
        milestones = rel_events[1:9]
        
        metadata.append({
            'video_id': vid_id,
            'start_frame': start_frame,
            'end_frame': rel_events[9],
            'milestones': milestones.tolist()
        })
        
    return pd.DataFrame(metadata)

def engineer_sliding_window(df, joints, shift_steps=5):
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

def assign_labels(df, df_meta):
    df_labeled = df.copy()
    meta_dict = df_meta.set_index('video_id')['milestones'].to_dict()
    df_labeled['label'] = 0
    
    for vid_id, group in df_labeled.groupby('video_id'):
        if vid_id not in meta_dict:
            continue
        milestones = meta_dict[vid_id]
        
        for milestone_idx, frame_idx in enumerate(milestones):
            label_val = milestone_idx + 1
            mask = (df_labeled['video_id'] == vid_id) & (df_labeled['frame_index'] == frame_idx)
            df_labeled.loc[mask, 'label'] = label_val
            
    return df_labeled

def main():
    mat_path = os.path.join(PROJECT_ROOT, 'data/golfDB.mat')
    df_meta = parse_golfdb_mat(mat_path)
    
    NUM_VIDEOS_TO_PROCESS = 10
    VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data/videos_160/videos_160')
    
    processed_dfs = []
    
    for i in range(NUM_VIDEOS_TO_PROCESS):
        video_id = df_meta.loc[i, 'video_id']
        video_path = os.path.join(VIDEO_DIR, f'{video_id}.mp4')
        try:
            processor = GolfVideoProcessor()
            df_vid = processor.process_video(video_path, video_id=video_id)
            processed_dfs.append(df_vid)
            processor.close()
        except Exception as e:
            print(f"Error: {e}")
            
    df_raw_features = pd.concat(processed_dfs, ignore_index=True)
    
    HIGH_MOVEMENT_JOINTS = [
        'left_wrist', 'right_wrist', 
        'left_elbow', 'right_elbow', 
        'left_shoulder', 'right_shoulder', 
        'left_hip', 'right_hip'
    ]
    df_windowed = engineer_sliding_window(df_raw_features, HIGH_MOVEMENT_JOINTS)
    df_final = assign_labels(df_windowed, df_meta)
    
    # We want to check label 4 (Top of Backswing)
    top_rows = df_final[df_final['label'] == 4]
    
    print("\nVerifying 'Top of Backswing' (label = 4) right wrist Y coordinates:")
    print("Physical assumption: Y decreases upwards. Thus, at Top of Backswing (T),")
    print("the wrist is highest, so Y_t should be smaller than Y_t-5 (backswing) and Y_t+5 (downswing).")
    
    valid_count = 0
    total_count = 0
    
    for idx, row in top_rows.iterrows():
        vid_id = row['video_id']
        frame_idx = row['frame_index']
        y_t = row['norm_right_wrist_y']
        y_prev = row['norm_right_wrist_y_t-5']
        y_next = row['norm_right_wrist_y_t+5']
        
        # Valid if wrist is higher at T than at T-5 and T+5
        # Since smaller Y means higher, Y_t should be smaller than both Y_prev and Y_next.
        is_valid = (y_prev > y_t) and (y_next > y_t)
        total_count += 1
        if is_valid:
            valid_count += 1
            
        print(f"Video {int(vid_id)}, Frame {int(frame_idx)}:")
        print(f"  Right Wrist Y (T-5): {y_prev: .4f}")
        print(f"  Right Wrist Y (T):   {y_t: .4f}")
        print(f"  Right Wrist Y (T+5): {y_next: .4f}")
        print(f"  Valid physics (Y_t-5 > Y_t and Y_t+5 > Y_t)? {is_valid}")
        
    print(f"\nVerification Results (label 4): {valid_count}/{total_count} videos matched expectations.")

if __name__ == "__main__":
    main()
