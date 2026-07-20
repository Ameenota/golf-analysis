import numpy as np
import pandas as pd

# Database of iconic professional golfers with precalculated ratios and milestone metrics
PRO_DATABASE = [
    {
        "name": "Sandra Gal",
        "video_id": 0,
        "gender": "f",
        "view": "down-the-line",
        "ratio": 1.0501,
        "metrics": {
            "lead_arm_flex_at_top": 172.5,
            "spine_tilt_at_address": 24.7,
            "spine_tilt_loss": 1.7,
            "spine_tilt_at_follow": 24.7,
            "lead_knee_flex_at_address": 157.8,
            "hip_sway_ratio": None,
            "head_bob_ratio": None,
            "trail_heel_lift_ratio": 0.284,
            "lead_heel_lift_ratio": 0.015
        }
    },
    {
        "name": "Cristie Kerr",
        "video_id": 8,
        "gender": "f",
        "view": "face-on",
        "ratio": 1.1978,
        "metrics": {
            "lead_arm_flex_at_top": 146.1,
            "spine_tilt_at_address": 0.7,
            "spine_tilt_loss": 1.7,
            "lead_knee_flex_at_address": 176.4,
            "hip_sway_ratio": 0.0639,
            "head_bob_ratio": 0.0384,
            "trail_heel_lift_ratio": 0.265,
            "lead_heel_lift_ratio": 0.021
        }
    },
    {
        "name": "Steve Stricker",
        "video_id": 10,
        "gender": "m",
        "view": "face-on",
        "ratio": 1.0924,
        "metrics": {
            "lead_arm_flex_at_top": 140.2,
            "spine_tilt_at_address": 8.9,
            "spine_tilt_loss": 1.7,
            "lead_knee_flex_at_address": 179.0,
            "hip_sway_ratio": 0.0491,
            "head_bob_ratio": 0.1362,
            "trail_heel_lift_ratio": 0.310,
            "lead_heel_lift_ratio": 0.018
        }
    },
    {
        "name": "Greg Norman",
        "video_id": 14,
        "gender": "m",
        "view": "face-on",
        "ratio": 1.1934,
        "metrics": {
            "lead_arm_flex_at_top": 161.7,
            "spine_tilt_at_address": 1.5,
            "spine_tilt_loss": 1.3,
            "lead_knee_flex_at_address": 177.1,
            "hip_sway_ratio": 0.1511,
            "head_bob_ratio": 0.1401,
            "trail_heel_lift_ratio": 0.292,
            "lead_heel_lift_ratio": 0.012
        }
    },
    {
        "name": "Tiger Woods",
        "video_id": 21,
        "gender": "m",
        "view": "face-on",
        "ratio": 0.8625,
        "metrics": {
            "lead_arm_flex_at_top": 177.2,
            "spine_tilt_at_address": 7.8,
            "spine_tilt_loss": 2.7,
            "lead_knee_flex_at_address": 158.1,
            "hip_sway_ratio": 0.0678,
            "head_bob_ratio": 0.0990,
            "trail_heel_lift_ratio": 0.345,
            "lead_heel_lift_ratio": 0.010
        }
    }
]

# Coaching drills and issues database
COACHING_DB = {
    "lead_arm_bend": {
        "issue": "Bent Lead Arm at Top of Backswing",
        "threshold": "Lead Arm Flex >= 160°",
        "drill": "Towel Drill: Place a rolled towel under your lead armpit (left armpit for righties) and keep it squeezed to the top of your swing to prevent the elbow from bending."
    },
    "spine_tilt_address": {
        "issue": "Incorrect Spine Tilt at Address",
        "threshold": "Spine Tilt: 5° to 15° lateral (FO) or 24° to 42° forward (DTL)",
        "drill": "Spine Alignment Drill: Tilt your upper body slightly away from the target so your lead shoulder is higher than your trail shoulder (Face-On), or ensure you bend from your hips, keeping your back straight (Down-The-Line)."
    },
    "spine_tilt_loss": {
        "issue": "Loss of Spine Tilt at Top of Backswing",
        "threshold": "Top tilt within 3° of Address tilt",
        "drill": "Spike Angle Drill: Keep your chest angled down towards the ball at the top of your turn instead of standing up or lunging forward."
    },
    "spine_tilt_follow": {
        "issue": "Loss of Posture at Follow-Through",
        "threshold": "Forward Spine Tilt >= 20° (DTL Only)",
        "drill": "Wall Butt Drill: Stand with your trail hip (right hip for righties) touching a wall at Address. Practice swinging through to the follow-through, keeping your hip against the wall to maintain your forward bend."
    },
    "knee_flex_address": {
        "issue": "Incorrect Knee Flex at Address",
        "threshold": "Knee Flex: 150° to 165° (DTL) or 165° to 175° (FO)",
        "drill": "Flex Stability Drill: Stand with your knees slightly soft, feeling the weight in the balls of your feet, not locked out or sitting too low."
    },
    "hip_sway": {
        "issue": "Excessive Lateral Hip Sway at Top of Backswing",
        "threshold": "Lateral hip slide <= 15% of torso length (FO Only)",
        "drill": "Stability Board Drill: Position a chair or rod just outside your trail hip at setup. Turn your hips without letting them slide laterally into the object."
    },
    "head_bobbing": {
        "issue": "Excessive Vertical Head Movement",
        "threshold": "Vertical head movement <= 15% of torso length (FO Only)",
        "drill": "Wall Head Drill: Practice your backswing with your forehead lightly touching a wall. Focus on rotating without bobbing up or dipping down."
    },
    "trail_heel_flat": {
        "issue": "Trail Heel Stuck Flat at Finish (Hanging Back)",
        "threshold": "Trail Heel Lift >= 10% of torso length at Finish",
        "drill": "Step-Through Drill: Practice taking a step forward with your trail foot after impact to ensure 100% weight transfer onto your lead leg."
    },
    "lead_heel_lifted": {
        "issue": "Lead Heel Lifted at Finish (Unstable Lead Foot)",
        "threshold": "Lead Heel Lift <= 12% of torso length at Finish",
        "drill": "Planted Lead Foot Drill: Drive through your lead heel into the ground during downswing and follow-through to build a solid pivot post."
    }
}

def calculate_joint_2d_angle(A, B, C):
    """Calculates the 2D angle (in degrees) at joint B given points A, B, and C."""
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    ba = A - B
    bc = C - B
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)

def detect_handedness(df, milestones, view, user_override=None):
    """
    Detects golfer handedness based on wrist positioning relative to torso at the Top of Backswing (F3).
    Returns: 'right' or 'left'
    """
    if user_override in ["right", "left"]:
        return user_override
        
    if "Top of Backswing" not in milestones:
        return "right"
        
    f3_idx = milestones["Top of Backswing"]["frame"]
    if f3_idx >= len(df):
        return "right"
        
    row = df.iloc[f3_idx]
    
    # Torso center
    torso_x = row["mid_hip_x"]
    
    # Wrists center
    left_wrist_x = row["smooth_left_wrist_x"]
    right_wrist_x = row["smooth_right_wrist_x"]
    wrists_x = (left_wrist_x + right_wrist_x) / 2.0
    
    if view == "down-the-line":
        # In Down-The-Line (DTL) view, a right-handed golfer faces right in 2D image coordinates.
        # At Top of Backswing (F3), hands coil back towards the right (larger X, wrists_x > torso_x).
        if wrists_x > torso_x:
            return "right"
        else:
            return "left"
    else:
        # In Face-On (FO) view, a right-handed golfer faces the camera.
        # At Top of Backswing (F3), hands coil back towards the viewer's left (smaller X, wrists_x < torso_x).
        if wrists_x < torso_x:
            return "right"
        else:
            return "left"

def match_pro_golfer(user_ratio, gender, view):
    """Matches the user to the closest pro golfer based on ratio, filtering by view and gender."""
    # 1. Filter by view first
    candidates = [p for p in PRO_DATABASE if p["view"] == view]
    if not candidates:
        candidates = PRO_DATABASE
        
    # 2. Filter by gender if matches exist in that view
    gender_candidates = [p for p in candidates if p["gender"] == gender]
    if gender_candidates:
        candidates = gender_candidates
        
    # 3. Find closest ratio
    best_match = min(candidates, key=lambda p: abs(p["ratio"] - user_ratio))
    return best_match

def analyze_swing_biomechanics(df, milestones, view, handedness="auto", custom_thresholds=None):
    """
    Runs all view-dependent biomechanical validations and matching.
    """
    if "Address" not in milestones or "Top of Backswing" not in milestones or "Impact" not in milestones:
        return {
            "success": False,
            "error": "Missing key milestones (Address, Top of Backswing, or Impact) required for analysis."
        }
        
    f1_idx = milestones["Address"]["frame"]
    f3_idx = milestones["Top of Backswing"]["frame"]
    f5_idx = milestones["Impact"]["frame"]
    
    if any(idx >= len(df) for idx in [f1_idx, f3_idx, f5_idx]):
        return {
            "success": False,
            "error": "Detected milestones frame indices exceed the length of processed video frames."
        }
        
    # Establish thresholds (allow overrides)
    thresholds = {
        "lead_arm_limit": 160.0,
        "spine_tilt_fo_min": 5.0,
        "spine_tilt_fo_max": 15.0,
        "spine_tilt_dtl_min": 24.0,
        "spine_tilt_dtl_max": 42.0,
        "spine_tilt_loss_limit": 3.0,
        "knee_flex_fo_min": 165.0,
        "knee_flex_fo_max": 175.0,
        "knee_flex_dtl_min": 150.0,
        "knee_flex_dtl_max": 165.0,
        "hip_sway_limit": 0.15,
        "head_bob_limit": 0.15,
        "spine_tilt_follow_min": 20.0
    }
    if custom_thresholds:
        thresholds.update(custom_thresholds)
        
    # 1. Handedness
    h = detect_handedness(df, milestones, view, handedness)
    lead_prefix = "left_" if h == "right" else "right_"
    trail_prefix = "right_" if h == "right" else "left_"
    
    # 2. Arm-to-Torso Ratio at Address
    row_address = df.iloc[f1_idx]
    
    # Left Arm
    l_shoulder = np.array([row_address["smooth_left_shoulder_x"], row_address["smooth_left_shoulder_y"]])
    l_elbow = np.array([row_address["smooth_left_elbow_x"], row_address["smooth_left_elbow_y"]])
    l_wrist = np.array([row_address["smooth_left_wrist_x"], row_address["smooth_left_wrist_y"]])
    l_arm = np.linalg.norm(l_shoulder - l_elbow) + np.linalg.norm(l_elbow - l_wrist)
    
    # Right Arm
    r_shoulder = np.array([row_address["smooth_right_shoulder_x"], row_address["smooth_right_shoulder_y"]])
    r_elbow = np.array([row_address["smooth_right_elbow_x"], row_address["smooth_right_elbow_y"]])
    r_wrist = np.array([row_address["smooth_right_wrist_x"], row_address["smooth_right_wrist_y"]])
    r_arm = np.linalg.norm(r_shoulder - r_elbow) + np.linalg.norm(r_elbow - r_wrist)
    
    user_arm_length = (l_arm + r_arm) / 2.0
    user_torso_length = row_address["torso_length"]
    user_ratio = user_arm_length / user_torso_length
    
    # 3. Match Pro
    # Deduce gender based on standard or average profile default
    gender = "m" # Default fallback
    pro_match = match_pro_golfer(user_ratio, gender, view)
    
    # 4. Biomechanical Rules Checks
    issues = []
    metrics = {}
    
    # Rule A: Lead Arm Flex at Top of Backswing (F3)
    row_top = df.iloc[f3_idx]
    top_lead_shoulder = [row_top[f"smooth_{lead_prefix}shoulder_x"], row_top[f"smooth_{lead_prefix}shoulder_y"]]
    top_lead_elbow = [row_top[f"smooth_{lead_prefix}elbow_x"], row_top[f"smooth_{lead_prefix}elbow_y"]]
    top_lead_wrist = [row_top[f"smooth_{lead_prefix}wrist_x"], row_top[f"smooth_{lead_prefix}wrist_y"]]
    lead_arm_flex = calculate_joint_2d_angle(top_lead_shoulder, top_lead_elbow, top_lead_wrist)
    metrics["lead_arm_flex_at_top"] = lead_arm_flex
    
    if lead_arm_flex < thresholds["lead_arm_limit"]:
        issue_info = COACHING_DB["lead_arm_bend"].copy()
        issue_info["measured"] = f"{lead_arm_flex:.1f}°"
        issues.append(issue_info)
        
    # Rule B: Spine Tilt (Address vs Top)
    spine_tilt_address = row_address["torso_angle_deg"]
    spine_tilt_top = row_top["torso_angle_deg"]
    metrics["spine_tilt_at_address"] = spine_tilt_address
    metrics["spine_tilt_at_top"] = spine_tilt_top
    
    if view == "face-on":
        if not (thresholds["spine_tilt_fo_min"] <= spine_tilt_address <= thresholds["spine_tilt_fo_max"]):
            issue_info = COACHING_DB["spine_tilt_address"].copy()
            issue_info["measured"] = f"{spine_tilt_address:.1f}°"
            issues.append(issue_info)
    else: # down-the-line
        if not (thresholds["spine_tilt_dtl_min"] <= spine_tilt_address <= thresholds["spine_tilt_dtl_max"]):
            issue_info = COACHING_DB["spine_tilt_address"].copy()
            issue_info["measured"] = f"{spine_tilt_address:.1f}°"
            issues.append(issue_info)
            
    # Loss of posture tilt at Top of swing
    tilt_diff = abs(spine_tilt_top - spine_tilt_address)
    metrics["spine_tilt_loss"] = tilt_diff
    if tilt_diff > thresholds["spine_tilt_loss_limit"]:
        issue_info = COACHING_DB["spine_tilt_loss"].copy()
        issue_info["measured"] = f"Shift of {tilt_diff:.1f}°"
        issues.append(issue_info)
        
    # Rule B2: Spine Tilt at Follow-Through (F7)
    f7_idx = milestones.get("Mid-Follow-Through", {}).get("frame")
    if f7_idx is not None and f7_idx < len(df):
        row_follow = df.iloc[f7_idx]
        spine_tilt_follow = row_follow["torso_angle_deg"]
        metrics["spine_tilt_at_follow"] = spine_tilt_follow
        
        if view == "down-the-line":
            if spine_tilt_follow < thresholds["spine_tilt_follow_min"]:
                issue_info = COACHING_DB["spine_tilt_follow"].copy()
                issue_info["measured"] = f"{spine_tilt_follow:.1f}°"
                issues.append(issue_info)
        
    # Rule C: Knee Flex at Address (F1)
    addr_lead_hip = [row_address[f"smooth_{lead_prefix}hip_x"], row_address[f"smooth_{lead_prefix}hip_y"]]
    addr_lead_knee = [row_address[f"smooth_{lead_prefix}knee_x"], row_address[f"smooth_{lead_prefix}knee_y"]]
    addr_lead_ankle = [row_address[f"smooth_{lead_prefix}ankle_x"], row_address[f"smooth_{lead_prefix}ankle_y"]]
    lead_knee_flex = calculate_joint_2d_angle(addr_lead_hip, addr_lead_knee, addr_lead_ankle)
    metrics["lead_knee_flex_at_address"] = lead_knee_flex
    
    knee_min = thresholds["knee_flex_dtl_min"] if view == "down-the-line" else thresholds["knee_flex_fo_min"]
    knee_max = thresholds["knee_flex_dtl_max"] if view == "down-the-line" else thresholds["knee_flex_fo_max"]
    
    if not (knee_min <= lead_knee_flex <= knee_max):
        issue_info = COACHING_DB["knee_flex_address"].copy()
        issue_info["measured"] = f"{lead_knee_flex:.1f}°"
        issues.append(issue_info)
        
    # Rule D: Lateral Hip Sway (Face-On Only)
    if view == "face-on":
        address_hip_x = row_address["mid_hip_x"]
        top_hip_x = row_top["mid_hip_x"]
        torso_scale = row_address["torso_scale"]
        
        hip_displacement = abs(top_hip_x - address_hip_x)
        hip_sway_ratio = hip_displacement / torso_scale
        metrics["hip_sway_ratio"] = hip_sway_ratio
        
        if hip_sway_ratio > thresholds["hip_sway_limit"]:
            issue_info = COACHING_DB["hip_sway"].copy()
            issue_info["measured"] = f"{hip_sway_ratio * 100:.1f}% of torso"
            issues.append(issue_info)
            
    # Rule E: Vertical Head Stability (Face-On Only)
    if view == "face-on":
        row_impact = df.iloc[f5_idx]
        address_nose_y = row_address["smooth_nose_y"]
        top_nose_y = row_top["smooth_nose_y"]
        impact_nose_y = row_impact["smooth_nose_y"]
        torso_scale = row_address["torso_scale"]
        
        max_head_shift = max(abs(top_nose_y - address_nose_y), abs(impact_nose_y - address_nose_y))
        head_bob_ratio = max_head_shift / torso_scale
        metrics["head_bob_ratio"] = head_bob_ratio
        
        if head_bob_ratio > thresholds["head_bob_limit"]:
            issue_info = COACHING_DB["head_bobbing"].copy()
            issue_info["measured"] = f"{head_bob_ratio * 100:.1f}% of torso"
            issues.append(issue_info)
            
    # Rule F: Foot Weight Transfer & Heel Lift at Finish
    f8_idx = milestones.get("Finish", {}).get("frame")
    if f8_idx is None or f8_idx >= len(df):
        f8_idx = milestones.get("Mid-Follow-Through", {}).get("frame")
        
    if f8_idx is not None and f8_idx < len(df):
        row_finish = df.iloc[f8_idx]
        torso_scale_finish = row_finish["torso_scale"]
        
        trail_heel_y = row_finish[f"smooth_{trail_prefix}heel_y"]
        trail_toe_y = row_finish[f"smooth_{trail_prefix}foot_index_y"]
        trail_heel_lift_ratio = (trail_toe_y - trail_heel_y) / (torso_scale_finish + 1e-6)
        
        lead_heel_y = row_finish[f"smooth_{lead_prefix}heel_y"]
        lead_toe_y = row_finish[f"smooth_{lead_prefix}foot_index_y"]
        lead_heel_lift_ratio = (lead_toe_y - lead_heel_y) / (torso_scale_finish + 1e-6)
        
        metrics["trail_heel_lift_ratio"] = float(trail_heel_lift_ratio)
        metrics["lead_heel_lift_ratio"] = float(lead_heel_lift_ratio)
        
        if trail_heel_lift_ratio < thresholds.get("trail_heel_lift_min", 0.10):
            issue_info = COACHING_DB["trail_heel_flat"].copy()
            issue_info["measured"] = f"{trail_heel_lift_ratio * 100:.1f}% of torso (flat)"
            issues.append(issue_info)
            
        if lead_heel_lift_ratio > thresholds.get("lead_heel_lift_max", 0.12):
            issue_info = COACHING_DB["lead_heel_lifted"].copy()
            issue_info["measured"] = f"{lead_heel_lift_ratio * 100:.1f}% of torso (lifted)"
            issues.append(issue_info)
            
    return {
        "success": True,
        "handedness": h,
        "user_arm_to_torso_ratio": round(float(user_ratio), 4),
        "matched_pro": pro_match["name"],
        "matched_pro_ratio": pro_match["ratio"],
        "matched_pro_video_id": pro_match["video_id"],
        "matched_pro_metrics": pro_match["metrics"],
        "metrics": metrics,
        "issues_detected": issues
    }
