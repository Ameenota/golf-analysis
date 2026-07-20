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

def create_spine_angle_chart(df: pd.DataFrame, milestones: dict) -> go.Figure:
    """Creates an interactive Plotly line chart of spine tilt angle over time with milestone markers."""
    fig = go.Figure()
    
    frames = df.index.values
    spine_tilt = df.get("spine_tilt", np.zeros(len(df)))
    
    # Line trace for spine tilt
    fig.add_trace(go.Scatter(
        x=frames,
        y=spine_tilt,
        mode="lines",
        name="Spine Tilt (°)",
        line=dict(color="#00E676", width=2.5),
        hovertemplate="Frame %{x}: %{y:.1f}°<extra></extra>"
    ))
    
    # Vertical lines for milestones
    if milestones:
        for m_name, m_info in milestones.items():
            m_frame = m_info.get("frame")
            if m_frame is not None and m_frame in frames:
                color = MILESTONE_COLORS.get(m_name, "#FFFFFF")
                fig.add_vline(
                    x=m_frame,
                    line_width=1.5,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=m_name,
                    annotation_position="top left",
                    annotation_font=dict(color=color, size=10)
                )
                
    fig.update_layout(
        title="<b>Spine Tilt Trajectory Across Swing Frames</b>",
        xaxis_title="Frame Index",
        yaxis_title="Spine Tilt (Degrees)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380
    )
    return fig

def create_wrist_velocity_chart(df: pd.DataFrame, milestones: dict) -> go.Figure:
    """Creates an interactive Plotly line chart of wrist velocities to visualize acceleration spikes."""
    fig = go.Figure()
    
    frames = df.index.values
    l_wrist_v = df.get("smooth_left_wrist_vel", np.zeros(len(df)))
    r_wrist_v = df.get("smooth_right_wrist_vel", np.zeros(len(df)))
    
    fig.add_trace(go.Scatter(
        x=frames,
        y=l_wrist_v,
        mode="lines",
        name="Lead Wrist Velocity",
        line=dict(color="#29B6F6", width=2),
        hovertemplate="Frame %{x}: %{y:.3f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=frames,
        y=r_wrist_v,
        mode="lines",
        name="Trail Wrist Velocity",
        line=dict(color="#FF7043", width=2),
        hovertemplate="Frame %{x}: %{y:.3f}<extra></extra>"
    ))
    
    # Vertical line for Impact frame
    if milestones and "Impact" in milestones:
        imp_frame = milestones["Impact"].get("frame")
        if imp_frame is not None:
            fig.add_vline(
                x=imp_frame,
                line_width=2,
                line_dash="solid",
                line_color="#EC407A",
                annotation_text="Impact Peak",
                annotation_position="top right"
            )
            
    fig.update_layout(
        title="<b>Hand Acceleration & Release Timing</b>",
        xaxis_title="Frame Index",
        yaxis_title="Normalized Velocity",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380
    )
    return fig
