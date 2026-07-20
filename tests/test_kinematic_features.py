import pytest
import numpy as np
import pandas as pd

from src.kinematic_features import (
    compute_centered_velocity_1d,
    add_velocity_features,
    get_milestone_feature_columns,
    SELECTED_LANDMARKS
)

def test_constant_position():
    # Position: [1.0, 1.0, 1.0, 1.0]
    pos = np.array([1.0, 1.0, 1.0, 1.0])
    vel = compute_centered_velocity_1d(pos)
    np.testing.assert_array_almost_equal(vel, np.zeros(4))

def test_constant_linear_movement():
    # Position: [0.0, 1.0, 2.0, 3.0]
    pos = np.array([0.0, 1.0, 2.0, 3.0])
    vel = compute_centered_velocity_1d(pos)
    # v(0) = 1 - 0 = 1
    # v(1) = (2 - 0)/2 = 1
    # v(2) = (3 - 1)/2 = 1
    # v(3) = 3 - 2 = 1
    np.testing.assert_array_almost_equal(vel, np.ones(4))

def test_direction_reversal():
    # Position: [0.0, 1.0, 2.0, 1.0, 0.0]
    pos = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    vel = compute_centered_velocity_1d(pos)
    # v(0) = 1 - 0 = 1.0 (positive)
    # v(1) = (2 - 0)/2 = 1.0 (positive)
    # v(2) = (1 - 1)/2 = 0.0 (peak/reversal)
    # v(3) = (0 - 2)/2 = -1.0 (negative)
    # v(4) = 0 - 1 = -1.0 (negative)
    assert vel[0] > 0
    assert vel[1] > 0
    assert vel[2] == 0.0
    assert vel[3] < 0
    assert vel[4] < 0

def test_video_boundary_isolation():
    # Create DataFrame with 2 videos
    df = pd.DataFrame({
        "video_id": ["vid1", "vid1", "vid1", "vid2", "vid2", "vid2"],
        "frame_index": [0, 1, 2, 0, 1, 2],
        "norm_left_wrist_x": [0.0, 1.0, 2.0, 10.0, 20.0, 30.0],
        "norm_left_wrist_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    # Add dummy remaining norm coords
    base_cols = get_milestone_feature_columns(feature_group="A")
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0.0

    df_out, _ = add_velocity_features(df)
    # Check that vid1 has velocity ~ 1.0 and vid2 has velocity ~ 10.0
    v1_vel = df_out[df_out["video_id"] == "vid1"]["vel_left_wrist_x"].values
    v2_vel = df_out[df_out["video_id"] == "vid2"]["vel_left_wrist_x"].values

    np.testing.assert_array_almost_equal(v1_vel, [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(v2_vel, [10.0, 10.0, 10.0])

def test_non_finite_handling():
    df = pd.DataFrame({
        "video_id": ["vid1", "vid1", "vid1"],
        "frame_index": [0, 1, 2],
        "norm_left_wrist_x": [0.0, np.nan, 2.0],
        "norm_left_wrist_y": [np.inf, 0.0, 0.0],
    })
    base_cols = get_milestone_feature_columns(feature_group="A")
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0.0

    df_out, _ = add_velocity_features(df)
    cols = get_milestone_feature_columns(df_out, feature_group="E")
    assert not df_out[cols].isna().any().any()
    assert not np.isinf(df_out[cols].values).any()

def test_feature_group_column_counts():
    assert len(get_milestone_feature_columns(feature_group="A")) == 66
    assert len(get_milestone_feature_columns(feature_group="B")) == 72
    assert len(get_milestone_feature_columns(feature_group="C")) == 84
    assert len(get_milestone_feature_columns(feature_group="D")) == 102
    assert len(get_milestone_feature_columns(feature_group="E")) == 108

def test_train_vs_inference_symmetry():
    # Batch processing vs single video processing must yield identical results
    df = pd.DataFrame({
        "video_id": ["v1", "v1", "v1", "v2", "v2", "v2"],
        "frame_index": [0, 1, 2, 0, 1, 2],
        "norm_left_wrist_x": [0.0, 1.0, 3.0, 0.0, 2.0, 4.0],
        "norm_left_wrist_y": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    })
    base_cols = get_milestone_feature_columns(feature_group="A")
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Fit config on batch
    df_batch, config = add_velocity_features(df, fit_config=True)

    # Process single video v1 with fitted config
    df_single = df[df["video_id"] == "v1"].copy()
    df_single_out, _ = add_velocity_features(df_single, config=config)

    cols_e = get_milestone_feature_columns(df_batch, feature_group="E")
    v1_batch_vals = df_batch[df_batch["video_id"] == "v1"][cols_e].values
    v1_single_vals = df_single_out[cols_e].values

    np.testing.assert_array_almost_equal(v1_batch_vals, v1_single_vals, decimal=6)

def test_left_right_flip_swapping():
    # When left and right x coordinates swap, their velocities should also swap
    df_orig = pd.DataFrame({
        "video_id": ["v1", "v1", "v1"],
        "frame_index": [0, 1, 2],
        "norm_left_wrist_x": [0.0, 1.0, 2.0],
        "norm_right_wrist_x": [10.0, 10.0, 10.0],
        "norm_left_wrist_y": [0.0, 0.0, 0.0],
        "norm_right_wrist_y": [0.0, 0.0, 0.0],
    })
    df_flipped = pd.DataFrame({
        "video_id": ["v1", "v1", "v1"],
        "frame_index": [0, 1, 2],
        "norm_left_wrist_x": [10.0, 10.0, 10.0],
        "norm_right_wrist_x": [0.0, 1.0, 2.0],
        "norm_left_wrist_y": [0.0, 0.0, 0.0],
        "norm_right_wrist_y": [0.0, 0.0, 0.0],
    })
    base_cols = get_milestone_feature_columns(feature_group="A")
    for c in base_cols:
        if c not in df_orig.columns:
            df_orig[c] = 0.0
            df_flipped[c] = 0.0

    res_orig, _ = add_velocity_features(df_orig)
    res_flipped, _ = add_velocity_features(df_flipped)

    np.testing.assert_array_almost_equal(
        res_orig["vel_left_wrist_x"].values,
        res_flipped["vel_right_wrist_x"].values
    )
    np.testing.assert_array_almost_equal(
        res_orig["vel_right_wrist_x"].values,
        res_flipped["vel_left_wrist_x"].values
    )
