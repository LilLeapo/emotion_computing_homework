"""融合权重可视化"""

import plotly.graph_objects as go

from config import EMOTION_LABELS, EMOTION_LABELS_CN, EMOTION_COLORS
from visualization.styles import PLOTLY_TEMPLATE


def create_fusion_weight_pie(weights: dict[str, float],
                             title: str = "模态融合权重") -> go.Figure:
    """创建融合权重饼图。

    Args:
        weights: {"文本": 0.4, "语音": 0.3, "面部": 0.3}
    """
    colors = {"文本": "#FF6B6B", "语音": "#4ECDC4", "面部": "#45B7D1"}

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        marker=dict(colors=[colors.get(k, "#888") for k in weights]),
        hole=0.4,
        textinfo="label+percent",
        textfont=dict(size=14),
        hovertemplate="%{label}: %{value:.1%}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(t=50, b=30, l=30, r=30),
        annotations=[dict(text="融合<br>权重", x=0.5, y=0.5, font_size=14,
                          showarrow=False, font_color="white")],
    )
    return fig


def create_sankey_diagram(modality_scores: dict[str, dict[str, float]],
                          fusion_scores: dict[str, float],
                          weights: dict[str, float]) -> go.Figure:
    """创建桑基图展示多模态融合流程。

    展示 文本/语音/面部 的主导情感如何流入融合结果。
    """
    modality_names = list(modality_scores.keys())
    emotion_names = [EMOTION_LABELS_CN[e] for e in EMOTION_LABELS]

    # 节点：3个模态 + 7个融合情感
    node_labels = modality_names + [f"融合-{n}" for n in emotion_names]
    modality_colors = {"文本": "#FF6B6B", "语音": "#4ECDC4", "面部": "#45B7D1"}
    node_colors = [modality_colors.get(m, "#888") for m in modality_names]
    node_colors += [EMOTION_COLORS[e] for e in EMOTION_LABELS]

    # 边：每个模态到每种融合情感的贡献
    sources, targets, values, link_colors = [], [], [], []
    for m_idx, (m_name, scores) in enumerate(modality_scores.items()):
        w = weights.get(m_name, 1.0 / len(modality_names))
        for e_idx, emotion in enumerate(EMOTION_LABELS):
            val = scores.get(emotion, 0) * w
            if val > 0.01:  # 过滤太小的流
                sources.append(m_idx)
                targets.append(len(modality_names) + e_idx)
                values.append(val)
                link_colors.append(f"rgba({_hex_to_rgb(EMOTION_COLORS[emotion])}, 0.4)")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="rgba(255,255,255,0.3)", width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    )])

    fig.update_layout(
        title=dict(text="多模态融合流程", x=0.5, font=dict(size=16)),
        template=PLOTLY_TEMPLATE,
        height=450,
        font=dict(size=12, color="white"),
    )
    return fig


def create_result_card(emotion: str, confidence: float, emoji: str) -> str:
    """创建融合结果展示卡片 HTML"""
    color = EMOTION_COLORS.get(emotion, "#888")
    cn_name = EMOTION_LABELS_CN.get(emotion, emotion)
    return f"""
    <div style="
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, {color}22, {color}44);
        border: 2px solid {color};
        border-radius: 16px;
        margin: 10px 0;
    ">
        <div style="font-size: 64px; margin-bottom: 10px;">{emoji}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{cn_name}</div>
        <div style="font-size: 20px; color: #ccc; margin-top: 8px;">置信度: {confidence:.1%}</div>
    </div>
    """


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r}, {g}, {b}"
