import os
import numpy as np
import pandas as pd
import xgboost as xgb

def train_xgboost_model(train_csv_path, val_csv_path, model_output_path):
    """
    Trains the multiclass XGBoost milestone classifier using training and validation CSVs
    and saves the serialized model to the output path.
    """
    print(f"Loading training data from {train_csv_path}...")
    df_train = pd.read_csv(train_csv_path)
    print(f"Loading validation data from {val_csv_path}...")
    df_val = pd.read_csv(val_csv_path)
    
    # Identify feature columns
    # We select all 'norm_' columns and the 3 categorical metadata columns
    pose_features = sorted([c for c in df_train.columns if c.startswith("norm_")])
    categorical_cols = ["view", "sex", "club"]
    feature_cols = pose_features + categorical_cols
    
    # Cast categorical columns to category dtype
    for col in categorical_cols:
        df_train[col] = df_train[col].astype("category")
        df_val[col] = df_val[col].astype("category")
        
    X_train = df_train[feature_cols]
    y_train = df_train["label"]
    
    X_val = df_val[feature_cols]
    y_val = df_val["label"]
    
    # Compute inverse-frequency class weights to handle severe class imbalance
    class_counts = y_train.value_counts().sort_index()
    total_samples = len(y_train)
    n_classes = len(class_counts)
    
    class_weights = {}
    for cls, count in class_counts.items():
        class_weights[cls] = total_samples / (n_classes * count)
        
    sample_weights = y_train.map(class_weights)
    
    # Initialize XGBoost multiclass classifier
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=9,
        tree_method="hist",
        enable_categorical=True,
        random_state=42
    )
    
    print("Training XGBClassifier with early stopping...")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    # Save the serialized model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save_model(model_output_path)
    print(f"Model successfully saved to {model_output_path}")
    return model

def predict_swing_milestones(video_features, model):
    """
    Runs model inference on a video's features and identifies the peak frame index
    for each of the 8 swing milestone events in chronological order.
    
    Parameters:
    - video_features: pd.DataFrame containing the features for a single video.
    - model: Trained XGBClassifier.
    
    Returns:
    - list of 8 integers representing the frame indices where milestones 1 to 8 occur.
    """
    # Identify feature columns
    pose_features = sorted([c for c in video_features.columns if c.startswith("norm_")])
    categorical_cols = ["view", "sex", "club"]
    feature_cols = pose_features + categorical_cols
    
    # Extract features copy and cast categoricals
    df_feat = video_features[feature_cols].copy()
    for col in categorical_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("category")
            
    # Get class probabilities: shape (N, 9)
    probs = model.predict_proba(df_feat)
    
    # Find argmax index across frames in strict chronological sequence (T_1 < T_2 < ... < T_8)
    milestones = []
    start_frame = 0
    for c in range(1, 9):
        # Slice probability array to search only from start_frame onwards
        search_window = probs[start_frame:, c]
        if len(search_window) > 0:
            peak_idx = np.argmax(search_window) + start_frame
        else:
            peak_idx = start_frame
            
        milestones.append(int(peak_idx))
        # The next event must occur at least 1 frame after the current one
        start_frame = peak_idx + 1
        
    return milestones
