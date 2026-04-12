"""情感时间线可视化（用于视频/长音频分析）"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import EMOTION_COLORS, EMOTION_LABELS, EMOTION_LABELS_CN
from visualization.styles import PLOTLY_TEMPLATE


def create_timeline(segments: list[dict], title: str = "情感时间线") -> go.Figure:
    """创建情感随时间变化的折线图。

    Args:
        segments: 列表，每项 {"start": float, "end": float, "scores": dict}
        title: 图表标题
    """
    if not segments:
        fig = go.Figure()
        fig.update_layout(title="暂无数据", template=PLOTLY_TEMPLATE, height=400)
        return fig

    # 时间点取每段中点
    times = [(s["start"] + s["end"]) / 2 for s in segments]

    fig = go.Figure()
    for emotion in EMOTION_LABELS:
        values = [s["scores"].get(emotion, 0) for s in segments]
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode="lines+markers",
            name=EMOTION_LABELS_CN[emotion],
            line=dict(color=EMOTION_COLORS[emotion], width=2),
            marker=dict(size=5),
            hovertemplate=(
                f"{EMOTION_LABELS_CN[emotion]}<br>"
                "时间: %{x:.1f}s<br>"
                "概率: %{y:.1%}<extra></extra>"
            ),
        ))

    # 添加主导情感背景色带
    for seg in segments:
        dominant = max(EMOTION_LABELS, key=lambda e: seg["scores"].get(e, 0))
        color = EMOTION_COLORS[dominant]
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor=color, opacity=0.08,
            layer="below", line_width=0,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title="时间 (秒)", ticksuffix="s"),
        yaxis=dict(title="概率", tickformat=".0%", range=[0, 1]),
        template=PLOTLY_TEMPLATE,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    return fig


def create_dual_timeline(face_segments: list[dict], speech_segments: list[dict],
                         title: str = "面部 vs 语音 情感时间线") -> go.Figure:
    """创建双轨时间线：上方面部表情，下方语音情感。"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("面部表情", "语音情感"),
    )

    for emotion in EMOTION_LABELS:
        # 面部
        if face_segments:
            times_f = [(s["start"] + s["end"]) / 2 for s in face_segments]
            vals_f = [s["scores"].get(emotion, 0) for s in face_segments]
            fig.add_trace(go.Scatter(
                x=times_f, y=vals_f,
                mode="lines",
                name=EMOTION_LABELS_CN[emotion],
                line=dict(color=EMOTION_COLORS[emotion], width=2),
                legendgroup=emotion,
                showlegend=True,
            ), row=1, col=1)

        # 语音
        if speech_segments:
            times_s = [(s["start"] + s["end"]) / 2 for s in speech_segments]
            vals_s = [s["scores"].get(emotion, 0) for s in speech_segments]
            fig.add_trace(go.Scatter(
                x=times_s, y=vals_s,
                mode="lines",
                name=EMOTION_LABELS_CN[emotion],
                line=dict(color=EMOTION_COLORS[emotion], width=2, dash="dot"),
                legendgroup=emotion,
                showlegend=False,
            ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        template=PLOTLY_TEMPLATE,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title="时间 (秒)", row=2, col=1)
    return fig
