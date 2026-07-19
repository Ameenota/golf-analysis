import os
import cv2
import numpy as np
import pandas as pd
from src.visualizer import draw_skeleton

# Order of milestones in the swing
MILESTONE_NAMES = [
    "Address",
    "Toe-Up",
    "Mid-Backswing",
    "Top of Backswing",
    "Mid-Downswing",
    "Impact",
    "Mid-Follow-Through",
    "Finish"
]

def load_video_frames(video_path):
    """Reads all frames of a video into memory as a list of NumPy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def build_temporal_mapping(df_user, df_pro, milestones_user, milestones_pro):
    """
    Computes a frame-to-frame mapping from User to Pro timeline.
    Warps the pro video timeline frame-by-frame based on the 8 milestones.
    """
    u_milestones = [milestones_user[m]["frame"] for m in MILESTONE_NAMES]
    p_milestones = [milestones_pro[m]["frame"] for m in MILESTONE_NAMES]
    
    total_u = len(df_user)
    total_p = len(df_pro)
    
    # Boundary anchors
    u_points = [0] + u_milestones + [total_u - 1]
    p_points = [0] + p_milestones + [total_p - 1]
    
    mapping = {}
    for i in range(len(u_points) - 1):
        u_start, u_end = u_points[i], u_points[i+1]
        p_start, p_end = p_points[i], p_points[i+1]
        
        if u_end == u_start:
            for f in range(u_start, u_end + 1):
                mapping[f] = p_start
            continue
            
        for f in range(u_start, u_end + 1):
            progress = (f - u_start) / (u_end - u_start)
            p_frame = p_start + progress * (p_end - p_start)
            mapping[f] = int(round(p_frame))
            
    return mapping

def resize_and_pad_frame(frame, target_size=(500, 500)):
    """Resizes and pads a frame to fit exactly inside a target square without cropping."""
    h_orig, w_orig, _ = frame.shape
    tw, th = target_size
    
    scale = min(tw / w_orig, th / h_orig)
    w_new = int(w_orig * scale)
    h_new = int(h_orig * scale)
    
    resized = cv2.resize(frame, (w_new, h_new))
    
    pad_y = (th - h_new) // 2
    pad_x = (tw - w_new) // 2
    
    padded = cv2.copyMakeBorder(
        resized, 
        pad_y, th - h_new - pad_y, 
        pad_x, tw - w_new - pad_x, 
        cv2.BORDER_CONSTANT, 
        value=(15, 15, 15)  # Dark border
    )
    return padded

def get_current_milestone_index(user_frame, milestones_user):
    """Determines which milestone segment the current frame belongs to."""
    current_idx = 0
    for idx, name in enumerate(MILESTONE_NAMES):
        if user_frame >= milestones_user[name]["frame"]:
            current_idx = idx
        else:
            break
    return current_idx

def draw_coaching_metrics(canvas, bio_results, user_frame, milestones_user, y_offset=630):
    """Draws user and pro comparison metrics underneath the video windows, highlighting active ones."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.43
    thickness = 1
    
    # 1. Draw User metrics (left side: x = 70 to 570)
    user_metrics = bio_results.get("metrics", {})
    issues = [issue["issue"] for issue in bio_results.get("issues_detected", [])]
    
    metrics_to_show = [
        ("lead_arm_flex_at_top", "Lead Arm Flex at Top", ">= 160", "Bent Lead Arm at Top of Backswing"),
        ("spine_tilt_at_address", "Spine Tilt at Address", "5-15 FO / 30-45 DTL", "Incorrect Spine Tilt at Address"),
        ("lead_knee_flex_at_address", "Knee Flex at Address", "165-175 FO / 150-165 DTL", "Incorrect Knee Flex at Address"),
        ("spine_tilt_at_follow", "Spine Tilt at Follow-Through", ">= 20", "Loss of Posture at Follow-Through")
    ]
    
    METRIC_MILESTONES = {
        "lead_arm_flex_at_top": "Top of Backswing",
        "spine_tilt_at_address": "Address",
        "lead_knee_flex_at_address": "Address",
        "spine_tilt_at_follow": "Follow-Through"
    }
    
    # Dynamic chronological sorting by milestone order
    def get_milestone_order(item):
        key = item[0]
        m_name = METRIC_MILESTONES.get(key)
        try:
            return MILESTONE_NAMES.index(m_name)
        except ValueError:
            return 999  # Place at bottom if milestone not in standard list
            
    metrics_to_show.sort(key=get_milestone_order)
    
    # Left column user metrics
    for idx, (key, label, limit, issue_name) in enumerate(metrics_to_show):
        val = user_metrics.get(key)
        if val is None:
            continue
            
        # Determine if the current frame is near the milestone for this metric
        m_name = METRIC_MILESTONES.get(key)
        is_active = False
        if m_name in milestones_user:
            m_frame = milestones_user[m_name]["frame"]
            is_active = (abs(user_frame - m_frame) <= 5)
            
        y_pos = y_offset + (idx * 22) + 20
        status_text = "WARN" if issue_name in issues else "PASS"
        
        # Color codes: bright when active, muted when inactive
        if is_active:
            text_color = (0, 255, 255) # Bright yellow-cyan
            status_color = (0, 0, 255) if status_text == "WARN" else (0, 255, 0)
            bullet = "> "
            text_thickness = 2
        else:
            text_color = (170, 170, 170) # Brighter gray (changed from 110)
            status_color = (50, 50, 180) if status_text == "WARN" else (50, 180, 50) # Brighter status
            bullet = "  "
            text_thickness = 1
            
        cv2.putText(canvas, f"{bullet}{label}: {val:.1f} (Limit: {limit})", (50, y_pos), font, font_scale, text_color, text_thickness, cv2.LINE_AA)
        cv2.putText(canvas, f" [{status_text}]", (50 + 380, y_pos), font, font_scale, status_color, text_thickness, cv2.LINE_AA)

    # 2. Draw Pro metrics (right side: x = 710 to 1210)
    pro_metrics = bio_results.get("matched_pro_metrics", {})
    pro_name = bio_results.get("matched_pro", "Pro")
    
    for idx, (key, label, _, _) in enumerate(metrics_to_show):
        val = pro_metrics.get(key)
        if val is None:
            continue
            
        m_name = METRIC_MILESTONES.get(key)
        is_active = False
        if m_name in milestones_user:
            m_frame = milestones_user[m_name]["frame"]
            is_active = (abs(user_frame - m_frame) <= 5)
            
        y_pos = y_offset + (idx * 22) + 20
        text_color = (255, 255, 255) if is_active else (170, 170, 170) # Brighter gray (changed from 110)
        text_thickness = 2 if is_active else 1
        
        cv2.putText(canvas, f"{pro_name} {label}: {val:.1f}", (710, y_pos), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

def create_synchronized_dashboard(
    user_video_path,
    pro_video_path,
    df_user,
    df_pro,
    milestones_user,
    milestones_pro,
    bio_results,
    output_path,
    speed=0.5
):
    """
    Compiles a side-by-side synchronized dashboard video (1280x720) 
    showing the user and matched pro.
    """
    print(f"Pre-loading frames for user video: {user_video_path}")
    u_frames = load_video_frames(user_video_path)
    print(f"Pre-loading frames for pro video: {pro_video_path}")
    p_frames = load_video_frames(pro_video_path)
    
    # Establish temporal mapping
    mapping = build_temporal_mapping(df_user, df_pro, milestones_user, milestones_pro)
    
    # Crop boundaries around user swing (Address - 5 to Finish + 5)
    f_start = max(0, milestones_user["Address"]["frame"] - 5)
    f_end = min(len(u_frames) - 1, milestones_user["Finish"]["frame"] + 5)
    
    # Get metadata
    pro_name = bio_results.get("matched_pro", "Professional")
    user_ratio = bio_results.get("user_arm_to_torso_ratio", 0.0)
    pro_ratio = bio_results.get("matched_pro_ratio", 0.0)
    handedness = bio_results.get("handedness", "right").capitalize()
    
    # Setup Video Writer
    fps = float(df_user["fps"].iloc[0])
    out_fps = fps * speed
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (1280, 720))
    
    # Layout placements
    # Left Box: x=70 to 570, y=100 to 600 (500x500)
    # Right Box: x=710 to 1210, y=100 to 600 (500x500)
    # Center Divider: x=570 to 710
    
    print("Compiling video frames...")
    
    for f in range(f_start, f_end + 1):
        # 1. Prepare base background canvas
        canvas = np.full((720, 1280, 3), 30, dtype=np.uint8) # Dark gray charcoal
        
        # 2. Get user frame and draw its skeleton
        u_img = u_frames[f].copy()
        draw_skeleton(u_img, df_user.iloc[f], prefix="smooth_", line_color=(0, 255, 0), joint_color=(0, 0, 255))
        u_window = resize_and_pad_frame(u_img, (500, 500))
        
        # 3. Get matched warped pro frame and draw its skeleton
        p_idx = mapping.get(f, milestones_pro["Finish"]["frame"])
        p_idx = max(0, min(p_idx, len(p_frames) - 1))
        
        p_img = p_frames[p_idx].copy()
        draw_skeleton(p_img, df_pro.iloc[p_idx], prefix="smooth_", line_color=(255, 200, 0), joint_color=(0, 0, 255))
        p_window = resize_and_pad_frame(p_img, (500, 500))
        
        # Place windows on canvas
        canvas[100:600, 70:570] = u_window
        canvas[100:600, 710:1210] = p_window
        
        # 4. Draw Header
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "GOLF SWING ANALYSIS", (470, 40), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"User ({handedness}) vs. {pro_name} (Limb Ratio: User {user_ratio:.2f} / Pro {pro_ratio:.2f})", 
                    (390, 70), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Labels below videos
        cv2.putText(canvas, "USER SWING", (70, 93), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{pro_name.upper()} (REFERENCE)", (710, 93), font, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
        
        # 5. Draw Milestone Tracker (Center Divider: x = 570 to 710)
        current_m_idx = get_current_milestone_index(f, milestones_user)
        
        for m_idx, m_name in enumerate(MILESTONE_NAMES):
            y_pos = 120 + (m_idx * 60)
            
            # Draw highlight block for current segment milestone
            if m_idx == current_m_idx:
                color = (0, 255, 0) # Bright green
                bullet = "> "
                thickness_m = 2
            else:
                color = (120, 120, 120) # Gray
                bullet = "  "
                thickness_m = 1
                
            cv2.putText(canvas, f"{bullet}{m_name}", (580, y_pos), font, 0.38, color, thickness_m, cv2.LINE_AA)
            # Draw connect line in divider
            if m_idx < len(MILESTONE_NAMES) - 1:
                cv2.line(canvas, (640, y_pos + 12), (640, y_pos + 38), (60, 60, 60), 1)

        # 6. Draw Scorecard Metrics (Bottom Margin)
        draw_coaching_metrics(canvas, bio_results, f, milestones_user, y_offset=615)
        
        # Write frame
        writer.write(canvas)
        
    writer.release()
    print(f"Synchronized comparison video compiled successfully to: {output_path}")
