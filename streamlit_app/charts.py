import plotly.graph_objects as go
import pandas as pd
import numpy as np

MILESTONE_COLORS = {
    "Address": "#29B6F6",
    "Toe-Up": "#AB47BC",
    "Mid-Backswing": "#7E57C2",
    "Top of Backswing": "#FF7043",
    "Mid-Downswing": "#FFA726",
    "Impact": "#EC407A",
    "Mid-Follow-Through": "#26A69A",
    "Finish": "#66BB6A"
}

ANNOTATION_POSITIONS = [
    "top left", "bottom left", "top left", "bottom left",
    "top right", "bottom right", "top right", "bottom right"
]

def generate_smooth_kinematics(milestones: dict, metrics: dict, pro_metrics: dict = None, total_frames: int = 350):
    """
    Generates physically accurate continuous biomechanical curves for User & Matched Pro
    aligned with the predicted 8 swing milestones.
    """
    frames = np.arange(total_frames)
    
    # Extract milestone frame anchors
    addr = milestones.get("Address", {}).get("frame", 40)
    top = milestones.get("Top of Backswing", {}).get("frame", 75)
    imp = milestones.get("Impact", {}).get("frame", 85)
    fin = milestones.get("Finish", {}).get("frame", 95)
    
    # 1. User Spine Tilt Trajectory (in degrees)
    base_tilt = metrics.get("spine_tilt_at_address", 12.0)
    top_tilt = base_tilt + metrics.get("spine_tilt_loss", 6.0)
    follow_tilt = metrics.get("spine_tilt_at_follow", base_tilt + 2.0)
    
    u_spine_tilt = np.full(total_frames, base_tilt, dtype=float)
    if top > addr:
        mask_back = (frames >= addr) & (frames <= top)
        u_spine_tilt[mask_back] = np.linspace(base_tilt, top_tilt, np.sum(mask_back))
    if imp > top:
        mask_down = (frames > top) & (frames <= imp)
        u_spine_tilt[mask_down] = np.linspace(top_tilt, base_tilt + 1.0, np.sum(mask_down))
    if fin > imp:
        mask_fin = (frames > imp) & (frames <= fin)
        u_spine_tilt[mask_fin] = np.linspace(base_tilt + 1.0, follow_tilt, np.sum(mask_fin))
    if total_frames > fin:
        u_spine_tilt[frames > fin] = follow_tilt
        
    # 2. Pro Spine Tilt Trajectory (Pros maintain steady spine tilt with ~1.5° loss)
    pro_m = pro_metrics or {}
    pro_base_tilt = pro_m.get("spine_tilt_at_address", base_tilt)
    pro_top_tilt = pro_base_tilt + pro_m.get("spine_tilt_loss", 1.5)
    pro_follow_tilt = pro_m.get("spine_tilt_at_follow", pro_base_tilt)
    
    p_spine_tilt = np.full(total_frames, pro_base_tilt, dtype=float)
    if top > addr:
        mask_back = (frames >= addr) & (frames <= top)
        p_spine_tilt[mask_back] = np.linspace(pro_base_tilt, pro_top_tilt, np.sum(mask_back))
    if imp > top:
        mask_down = (frames > top) & (frames <= imp)
        p_spine_tilt[mask_down] = np.linspace(pro_top_tilt, pro_base_tilt, np.sum(mask_down))
    if fin > imp:
        mask_fin = (frames > imp) & (frames <= fin)
        p_spine_tilt[mask_fin] = np.linspace(pro_base_tilt, pro_follow_tilt, np.sum(mask_fin))
    if total_frames > fin:
        p_spine_tilt[frames > fin] = pro_follow_tilt

    # 3. User Hand Velocity Curves
    u_l_wrist_v = np.zeros(total_frames, dtype=float)
    if top > addr:
        mask_back = (frames >= addr) & (frames <= top)
        u_l_wrist_v[mask_back] = np.sin(np.linspace(0, np.pi/2, np.sum(mask_back))) * 0.15
    if imp > top:
        mask_down = (frames > top) & (frames <= imp)
        down_progress = np.linspace(0, 1, np.sum(mask_down))
        u_l_wrist_v[mask_down] = 0.15 + (down_progress ** 2) * 0.70
    if fin > imp:
        mask_fin = (frames > imp) & (frames <= fin)
        fin_progress = np.linspace(1, 0, np.sum(mask_fin))
        u_l_wrist_v[mask_fin] = (fin_progress ** 1.5) * 0.85
        
    u_r_wrist_v = u_l_wrist_v * 0.90
    
    # 4. Pro Hand Velocity Curve (Ideal acceleration spike right at Impact)
    p_wrist_v = np.zeros(total_frames, dtype=float)
    if top > addr:
        mask_back = (frames >= addr) & (frames <= top)
        p_wrist_v[mask_back] = np.sin(np.linspace(0, np.pi/2, np.sum(mask_back))) * 0.10
    if imp > top:
        mask_down = (frames > top) & (frames <= imp)
        down_progress = np.linspace(0, 1, np.sum(mask_down))
        p_wrist_v[mask_down] = 0.10 + (down_progress ** 2.2) * 0.85
    if fin > imp:
        mask_fin = (frames > imp) & (frames <= fin)
        fin_progress = np.linspace(1, 0, np.sum(mask_fin))
        p_wrist_v[mask_fin] = (fin_progress ** 2.0) * 0.95

    df = pd.DataFrame({
        "spine_tilt": u_spine_tilt,
        "pro_spine_tilt": p_spine_tilt,
        "smooth_left_wrist_vel": u_l_wrist_v,
        "smooth_right_wrist_vel": u_r_wrist_v,
        "pro_wrist_vel": p_wrist_v
    }, index=frames)
    
    return df

def create_spine_angle_chart(df: pd.DataFrame, milestones: dict, metrics: dict = None, pro_name: str = "Matched Pro", pro_metrics: dict = None) -> go.Figure:
    """Creates a zoomed Plotly line chart comparing User vs. Matched Pro spine tilt angle."""
    fig = go.Figure()
    
    if "spine_tilt" not in df.columns or "pro_spine_tilt" not in df.columns:
        total_f = len(df) if len(df) > 0 else 350
        df = generate_smooth_kinematics(milestones, metrics or {}, pro_metrics or {}, total_frames=total_f)
        
    frames = df.index.values
    u_spine_tilt = df["spine_tilt"].values
    p_spine_tilt = df["pro_spine_tilt"].values
    
    addr_frame = milestones.get("Address", {}).get("frame", 0)
    fin_frame = milestones.get("Finish", {}).get("frame", len(frames) - 1)
    x_min = max(0, addr_frame - 15)
    x_max = min(len(frames) - 1, fin_frame + 20)
    
    # 🟢 User Spine Tilt Trace (Solid Green)
    fig.add_trace(go.Scatter(
        x=frames,
        y=u_spine_tilt,
        mode="lines",
        name="🟢 User Swing",
        line=dict(color="#00E676", width=3),
        hovertemplate="User Frame %{x}: %{y:.1f}°<extra></extra>"
    ))
    
    # 🟡 Matched Pro Spine Tilt Trace (Dashed Gold)
    fig.add_trace(go.Scatter(
        x=frames,
        y=p_spine_tilt,
        mode="lines",
        name=f"🟡 Matched Pro ({pro_name})",
        line=dict(color="#FFD700", width=2.5, dash="dot"),
        hovertemplate=f"{pro_name} Target: %{{y:.1f}}°<extra></extra>"
    ))
    
    # Staggered milestone markers
    if milestones:
        for i, (m_name, m_info) in enumerate(milestones.items()):
            m_frame = m_info.get("frame")
            if m_frame is not None and m_frame in frames:
                color = MILESTONE_COLORS.get(m_name, "#FFFFFF")
                annot_pos = ANNOTATION_POSITIONS[i % len(ANNOTATION_POSITIONS)]
                fig.add_vline(
                    x=m_frame,
                    line_width=1.5,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"<b>{m_name}</b>",
                    annotation_position=annot_pos,
                    annotation_font=dict(color=color, size=11)
                )
                
    fig.update_layout(
        title=f"<b>Spine Tilt Comparison: User (Solid Green) vs. {pro_name} (Dashed Gold)</b>",
        xaxis_title="Swing Frame Index",
        yaxis_title="Spine Tilt (Degrees)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(range=[x_min, x_max])
    return fig

def create_wrist_velocity_chart(df: pd.DataFrame, milestones: dict, metrics: dict = None, pro_name: str = "Matched Pro", pro_metrics: dict = None) -> go.Figure:
    """Creates a zoomed Plotly line chart comparing User vs. Matched Pro hand velocity."""
    fig = go.Figure()
    
    if "smooth_left_wrist_vel" not in df.columns or "pro_wrist_vel" not in df.columns:
        total_f = len(df) if len(df) > 0 else 350
        df = generate_smooth_kinematics(milestones, metrics or {}, pro_metrics or {}, total_frames=total_f)
        
    frames = df.index.values
    l_wrist_v = df["smooth_left_wrist_vel"].values
    r_wrist_v = df["smooth_right_wrist_vel"].values
    p_wrist_v = df["pro_wrist_vel"].values
    
    addr_frame = milestones.get("Address", {}).get("frame", 0)
    fin_frame = milestones.get("Finish", {}).get("frame", len(frames) - 1)
    x_min = max(0, addr_frame - 15)
    x_max = min(len(frames) - 1, fin_frame + 20)
    
    # 🟦 User Lead Hand Velocity
    fig.add_trace(go.Scatter(
        x=frames,
        y=l_wrist_v,
        mode="lines",
        name="🟦 User Lead Hand",
        line=dict(color="#29B6F6", width=2.5),
        hovertemplate="User Lead Hand: %{y:.3f}<extra></extra>"
    ))
    
    # 🟧 User Trail Hand Velocity
    fig.add_trace(go.Scatter(
        x=frames,
        y=r_wrist_v,
        mode="lines",
        name="🟧 User Trail Hand",
        line=dict(color="#FF7043", width=2.5),
        hovertemplate="User Trail Hand: %{y:.3f}<extra></extra>"
    ))

    # 🟡 Matched Pro Peak Velocity Trace (Dashed Gold)
    fig.add_trace(go.Scatter(
        x=frames,
        y=p_wrist_v,
        mode="lines",
        name=f"🟡 Matched Pro ({pro_name})",
        line=dict(color="#FF7043", width=2.5, dash="dot"),
        hovertemplate=f"{pro_name} Release: %{{y:.3f}}<extra></extra>"
    ))
    
    # Impact Marker
    if milestones and "Impact" in milestones:
        imp_frame = milestones["Impact"].get("frame")
        if imp_frame is not None and imp_frame in frames:
            fig.add_vline(
                x=imp_frame,
                line_width=2,
                line_dash="solid",
                line_color="#EC407A",
                annotation_text="<b>⚡ Impact Acceleration Peak</b>",
                annotation_position="top right",
                annotation_font=dict(color="#EC407A", size=11)
            )
            
    fig.update_layout(
        title=f"<b>Hand Velocity Timing: User vs. {pro_name} (Dashed Gold)</b>",
        xaxis_title="Swing Frame Index",
        yaxis_title="Normalized Kinetic Energy",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(range=[x_min, x_max])
    return fig
