"""情感置信度柱状图 (Plotly)"""

import plotly.graph_objects as go

from config import EMOTION_COLORS, EMOTION_LABELS, EMOTION_LABELS_CN
from visualization.styles import PLOTLY_TEMPLATE


def create_bar_chart(scores: dict[str, float], title: str = "情感置信度") -> go.Figure:
    """创建水平置信度柱状图，按概率降序排列。

    Args:
        scores: 7类情感概率字典
        title: 图表标题
    """
    # 按概率降序排列
    sorted_emotions = sorted(EMOTION_LABELS, key=lambda e: scores.get(e, 0), reverse=True)
    labels_cn = [EMOTION_LABELS_CN[e] for e in sorted_emotions]
    values = [scores.get(e, 0) for e in sorted_emotions]
    colors = [EMOTION_COLORS[e] for e in sorted_emotions]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels_cn,
        orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="rgba(255,255,255,0.3)")),
        text=[f"{v:.1%}" for v in values],
        textposition="auto",
        textfont=dict(size=13, color="white"),
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(range=[0, 1], tickformat=".0%", title="概率"),
        yaxis=dict(autorange="reversed"),
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(t=50, b=40, l=80, r=30),
        showlegend=False,
    )
    return fig


def create_comparison_bars(scores_dict: dict[str, dict[str, float]],
                           title: str = "多模态得分对比") -> go.Figure:
    """创建多模态并排对比柱状图。

    Args:
        scores_dict: {"文本": scores, "语音": scores, "面部": scores}
    """
    colors = {"文本": "#FF6B6B", "语音": "#4ECDC4", "面部": "#45B7D1", "融合": "#96CEB4"}

    fig = go.Figure()
    for name, scores in scores_dict.items():
        values = [scores.get(e, 0) for e in EMOTION_LABELS]
        fig.add_trace(go.Bar(
            name=name,
            x=[EMOTION_LABELS_CN[e] for e in EMOTION_LABELS],
            y=values,
            marker_color=colors.get(name, "#888"),
            text=[f"{v:.0%}" for v in values],
            textposition="auto",
            textfont=dict(size=10),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        barmode="group",
        yaxis=dict(tickformat=".0%", title="概率"),
        template=PLOTLY_TEMPLATE,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig
