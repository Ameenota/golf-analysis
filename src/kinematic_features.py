import numpy as np
import pandas as pd
import json
import os

# 12 selected landmarks for velocity feature extraction
SELECTED_LANDMARKS = [
    "left_wrist", "right_wrist",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "nose", "mid_hip"
]

UPPER_BODY_LANDMARKS = [
    "left_wrist", "right_wrist",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder"
]

WRIST_LANDMARKS = ["left_wrist", "right_wrist"]
ARM_LANDMARKS = ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]
HIP_LANDMARKS = ["left_hip", "right_hip", "mid_hip"]
LOWER_BODY_LANDMARKS = ["left_hip", "right_hip", "mid_hip", "left_knee", "right_knee"]

def derive_mid_hip_norm(df):
    """
    Ensures norm_mid_hip_x and norm_mid_hip_y exist in DataFrame.
    """
    df_copy = df.copy()
    if "norm_left_hip_x" in df_copy.columns and "norm_right_hip_x" in df_copy.columns:
        if "norm_mid_hip_x" not in df_copy.columns:
            df_copy["norm_mid_hip_x"] = (df_copy["norm_left_hip_x"] + df_copy["norm_right_hip_x"]) / 2.0
        if "norm_mid_hip_y" not in df_copy.columns:
            df_copy["norm_mid_hip_y"] = (df_copy["norm_left_hip_y"] + df_copy["norm_right_hip_y"]) / 2.0
    return df_copy

def compute_centered_velocity_1d(arr):
    """
    Computes centered finite difference velocity for a 1D numpy array representing frame sequence.
    v(t) = (x(t+1) - x(t-1)) / 2
    v(0) = x(1) - x(0)
    v(N-1) = x(N-1) - x(N-2)
    """
    arr = np.asarray(arr, dtype=np.float64)
    # Replace non-finite input values in array with 0.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(arr)
    vel = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return vel
    if n == 2:
        diff = arr[1] - arr[0]
        vel[0] = diff
        vel[1] = diff
        return vel

    # Interior points: centered difference
    vel[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    # Endpoints: one-sided differences
    vel[0] = arr[1] - arr[0]
    vel[-1] = arr[-1] - arr[-2]
    return vel

def compute_kinematic_features_for_video(df_vid, fps=None, include_fps_adjusted=False):
    """
    Computes velocity and motion summary features for a single video sequence.
    """
    df_vid = derive_mid_hip_norm(df_vid)
    feature_dict = {}

    # 1. Compute velocity and speed for each landmark
    for lm in SELECTED_LANDMARKS:
        x_col = f"norm_{lm}_x"
        y_col = f"norm_{lm}_y"

        if x_col in df_vid.columns:
            x_vals = df_vid[x_col].values
        else:
            x_vals = np.zeros(len(df_vid))

        if y_col in df_vid.columns:
            y_vals = df_vid[y_col].values
        else:
            y_vals = np.zeros(len(df_vid))

        vx = compute_centered_velocity_1d(x_vals)
        vy = compute_centered_velocity_1d(y_vals)
        spd = np.sqrt(vx**2 + vy**2)

        if include_fps_adjusted and fps is not None and fps > 0:
            vx_fps = vx * fps
            vy_fps = vy * fps
            spd_fps = spd * fps
            feature_dict[f"vel_{lm}_x_sec"] = vx_fps
            feature_dict[f"vel_{lm}_y_sec"] = vy_fps
            feature_dict[f"speed_{lm}_sec"] = spd_fps

        feature_dict[f"vel_{lm}_x"] = vx
        feature_dict[f"vel_{lm}_y"] = vy
        feature_dict[f"speed_{lm}"] = spd

    res_df = pd.DataFrame(feature_dict, index=df_vid.index)

    # 2. Compute summary motion features
    res_df["mean_wrist_speed"] = res_df[[f"speed_{lm}" for lm in WRIST_LANDMARKS]].mean(axis=1)
    res_df["mean_arm_speed"] = res_df[[f"speed_{lm}" for lm in ARM_LANDMARKS]].mean(axis=1)
    res_df["mean_hip_speed"] = res_df[[f"speed_{lm}" for lm in HIP_LANDMARKS]].mean(axis=1)
    res_df["mean_lower_body_speed"] = res_df[[f"speed_{lm}" for lm in LOWER_BODY_LANDMARKS]].mean(axis=1)

    res_df["upper_body_motion_energy"] = (res_df[[f"speed_{lm}" for lm in UPPER_BODY_LANDMARKS]]**2).sum(axis=1)
    res_df["whole_body_motion_energy"] = (res_df[[f"speed_{lm}" for lm in SELECTED_LANDMARKS]]**2).sum(axis=1)

    return res_df

def add_velocity_features(df, fps=None, include_fps_adjusted=False, config=None, fit_config=False):
    """
    Main interface function to compute velocity features across one or more videos in a DataFrame.
    Guarantees boundary isolation per video_id.
    Optionally clips extreme percentiles and standardizes features using config stats.
    """
    df_out = df.copy()
    df_out = derive_mid_hip_norm(df_out)

    # Replace any non-finite values in base features first
    base_cols = [c for c in df_out.columns if c.startswith("norm_")]
    if base_cols:
        df_out[base_cols] = df_out[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Calculate features grouping by video_id if present
    if "video_id" in df_out.columns:
        computed_dfs = []
        for vid, group in df_out.groupby("video_id", sort=False):
            # Sort by frame_index if present
            if "frame_index" in group.columns:
                group_sorted = group.sort_values("frame_index")
            else:
                group_sorted = group
            
            vid_fps = fps
            if vid_fps is None and "fps" in group_sorted.columns:
                vid_fps = group_sorted["fps"].iloc[0]

            feat_df = compute_kinematic_features_for_video(
                group_sorted, 
                fps=vid_fps, 
                include_fps_adjusted=include_fps_adjusted
            )
            computed_dfs.append(feat_df)
        all_kinematic_features = pd.concat(computed_dfs).reindex(df_out.index)
    else:
        all_kinematic_features = compute_kinematic_features_for_video(
            df_out, 
            fps=fps, 
            include_fps_adjusted=include_fps_adjusted
        )

    # Replace non-finite values (NaN / Inf) with 0.0
    all_kinematic_features = all_kinematic_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Concatenate computed kinematic features to avoid fragmentation
    new_cols = [col for col in all_kinematic_features.columns if col not in df_out.columns]
    if new_cols:
        df_out = pd.concat([df_out, all_kinematic_features[new_cols]], axis=1)

    kinematic_cols = list(all_kinematic_features.columns)

    if fit_config:
        config = {
            "percentiles": {},
            "mean": {},
            "std": {}
        }
        for col in kinematic_cols:
            vals = df_out[col].values
            p_low = float(np.percentile(vals, 0.5))
            p_high = float(np.percentile(vals, 99.5))
            config["percentiles"][col] = (p_low, p_high)

            clipped = np.clip(vals, p_low, p_high)
            m = float(np.mean(clipped))
            s = float(np.std(clipped))
            if s == 0 or np.isnan(s):
                s = 1.0
            config["mean"][col] = m
            config["std"][col] = s

    if config is not None:
        for col in kinematic_cols:
            if col in config.get("percentiles", {}):
                p_low, p_high = config["percentiles"][col]
                df_out[col] = np.clip(df_out[col], p_low, p_high)
            if col in config.get("mean", {}) and col in config.get("std", {}):
                m = config["mean"][col]
                s = config["std"][col]
                df_out[col] = (df_out[col] - m) / s

    return df_out, config

def get_milestone_feature_columns(df=None, feature_group="E"):
    """
    Returns explicit, deterministic, ordered feature column names for Experiments A through E.
    Exp A: 66 base norm coordinates
    Exp B: 66 coords + 6 wrist velocities (72)
    Exp C: 66 coords + 18 upper-body velocities (84)
    Exp D: 66 coords + 36 all MVP velocities (102)
    Exp E: 66 coords + 36 MVP velocities + 6 summary motion features (108)
    """
    # 66 base coordinate columns (alphabetical, matching existing lstm_keras notebook)
    base_coords = [
        "norm_left_ankle_x", "norm_left_ankle_y",
        "norm_left_ear_x", "norm_left_ear_y",
        "norm_left_elbow_x", "norm_left_elbow_y",
        "norm_left_eye_inner_x", "norm_left_eye_inner_y",
        "norm_left_eye_outer_x", "norm_left_eye_outer_y",
        "norm_left_eye_x", "norm_left_eye_y",
        "norm_left_foot_index_x", "norm_left_foot_index_y",
        "norm_left_heel_x", "norm_left_heel_y",
        "norm_left_hip_x", "norm_left_hip_y",
        "norm_left_index_x", "norm_left_index_y",
        "norm_left_knee_x", "norm_left_knee_y",
        "norm_left_pinky_x", "norm_left_pinky_y",
        "norm_left_shoulder_x", "norm_left_shoulder_y",
        "norm_left_thumb_x", "norm_left_thumb_y",
        "norm_left_wrist_x", "norm_left_wrist_y",
        "norm_mouth_left_x", "norm_mouth_left_y",
        "norm_mouth_right_x", "norm_mouth_right_y",
        "norm_nose_x", "norm_nose_y",
        "norm_right_ankle_x", "norm_right_ankle_y",
        "norm_right_ear_x", "norm_right_ear_y",
        "norm_right_elbow_x", "norm_right_elbow_y",
        "norm_right_eye_inner_x", "norm_right_eye_inner_y",
        "norm_right_eye_outer_x", "norm_right_eye_outer_y",
        "norm_right_eye_x", "norm_right_eye_y",
        "norm_right_foot_index_x", "norm_right_foot_index_y",
        "norm_right_heel_x", "norm_right_heel_y",
        "norm_right_hip_x", "norm_right_hip_y",
        "norm_right_index_x", "norm_right_index_y",
        "norm_right_knee_x", "norm_right_knee_y",
        "norm_right_pinky_x", "norm_right_pinky_y",
        "norm_right_shoulder_x", "norm_right_shoulder_y",
        "norm_right_thumb_x", "norm_right_thumb_y",
        "norm_right_wrist_x", "norm_right_wrist_y"
    ]

    # Wrist velocity columns (6 features)
    wrist_vels = []
    for lm in WRIST_LANDMARKS:
        wrist_vels.extend([f"vel_{lm}_x", f"vel_{lm}_y", f"speed_{lm}"])

    # Upper body velocity columns (18 features)
    upper_vels = []
    for lm in UPPER_BODY_LANDMARKS:
        upper_vels.extend([f"vel_{lm}_x", f"vel_{lm}_y", f"speed_{lm}"])

    # All MVP velocity columns (36 features)
    all_mvp_vels = []
    for lm in SELECTED_LANDMARKS:
        all_mvp_vels.extend([f"vel_{lm}_x", f"vel_{lm}_y", f"speed_{lm}"])

    # Summary motion columns (6 features)
    summary_features = [
        "mean_wrist_speed",
        "mean_arm_speed",
        "mean_hip_speed",
        "mean_lower_body_speed",
        "upper_body_motion_energy",
        "whole_body_motion_energy"
    ]

    group = str(feature_group).upper()
    if group == "A":
        cols = base_coords
    elif group == "B":
        cols = base_coords + wrist_vels
    elif group == "C":
        cols = base_coords + upper_vels
    elif group == "D":
        cols = base_coords + all_mvp_vels
    elif group == "E":
        cols = base_coords + all_mvp_vels + summary_features
    elif group == "F":
        cols = base_coords + upper_vels + summary_features
    elif group == "G":
        cols = base_coords + wrist_vels + summary_features
    else:
        raise ValueError(f"Unknown feature_group '{feature_group}'. Must be A, B, C, D, E, F, or G.")

    if df is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required feature columns in DataFrame for Group {group}: {missing}")

    return cols

def save_kinematic_config(filepath, config):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_kinematic_config(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def build_kinematic_features(df, fps=None, model_dir=None):
    """Build the exact ordered feature matrix expected by the production LSTM."""
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    schema_path = os.path.join(model_dir, "kinematic_schema.json")
    config_path = os.path.join(model_dir, "kinematic_config.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Kinematic feature schema not found at {schema_path}")

    schema = load_kinematic_config(schema_path)
    config = load_kinematic_config(config_path) if os.path.exists(config_path) else None
    feature_group = schema.get("feature_group", "E")
    df_kinematic, _ = add_velocity_features(df, fps=fps, config=config)
    feature_cols = get_milestone_feature_columns(df_kinematic, feature_group=feature_group)

    expected_dim = int(schema.get("input_dim", len(feature_cols)))
    if len(feature_cols) != expected_dim:
        raise ValueError(
            f"Kinematic feature extraction yielded {len(feature_cols)} features; "
            f"the model schema expects {expected_dim}."
        )

    return feature_cols, df_kinematic[feature_cols].values.astype(np.float32)
