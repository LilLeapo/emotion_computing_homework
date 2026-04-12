"""情感雷达图 (Plotly 交互式)"""

import plotly.graph_objects as go

from config import EMOTION_COLORS, EMOTION_LABELS, EMOTION_LABELS_CN
from visualization.styles import CN_LABELS, COLOR_SEQUENCE, PLOTLY_TEMPLATE


def create_radar_chart(scores: dict[str, float], title: str = "情感分布") -> go.Figure:
    """创建交互式情感雷达图。

    Args:
        scores: 7类情感概率字典
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    values = [scores.get(e, 0) for e in EMOTION_LABELS]
    # 闭合雷达图
    labels = CN_LABELS + [CN_LABELS[0]]
    vals = values + [values[0]]

    # 主导情感的颜色
    dominant_idx = values.index(max(values))
    dominant_color = COLOR_SEQUENCE[dominant_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals,
        theta=labels,
        fill="toself",
        fillcolor=f"rgba({_hex_to_rgb(dominant_color)}, 0.25)",
        line=dict(color=dominant_color, width=2.5),
        marker=dict(size=8, color=dominant_color),
        name="情感分布",
        hovertemplate="%{theta}: %{r:.1%}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                tickfont=dict(size=10),
            ),
            angularaxis=dict(tickfont=dict(size=13)),
        ),
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(t=60, b=30, l=60, r=60),
        showlegend=False,
    )
    return fig


def create_multi_radar(scores_list: list[dict], labels: list[str],
                       title: str = "多模态情感对比") -> go.Figure:
    """创建多模态对比雷达图（多条线叠加）。

    Args:
        scores_list: 多组情感概率字典
        labels: 每组对应的标签名（如 "文本", "语音", "面部"）
        title: 图表标题
    """
    fig = go.Figure()
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    for i, (scores, name) in enumerate(zip(scores_list, labels)):
        values = [scores.get(e, 0) for e in EMOTION_LABELS]
        vals = values + [values[0]]
        theta = CN_LABELS + [CN_LABELS[0]]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=theta,
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(color)}, 0.1)",
            line=dict(color=color, width=2),
            name=name,
            hovertemplate=f"{name}<br>%{{theta}}: %{{r:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%"),
            angularaxis=dict(tickfont=dict(size=13)),
        ),
        template=PLOTLY_TEMPLATE,
        height=450,
        margin=dict(t=60, b=30, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """将 #RRGGBB 转为 R, G, B 字符串"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r}, {g}, {b}"
