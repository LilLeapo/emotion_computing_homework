"""文本注意力热力图可视化

将 Transformer 注意力权重渲染为 HTML，字符级别着色。
"""


def create_attention_html(char_weights: list[tuple[str, float]],
                          dominant_emotion: str = "neutral") -> str:
    """创建字符级注意力热力图 HTML。

    Args:
        char_weights: [(字符, 注意力权重), ...]  权重已归一化到 [0, 1]
        dominant_emotion: 主导情感（决定着色色调）

    Returns:
        HTML 字符串，可直接在 Gradio HTML 组件中渲染
    """
    if not char_weights:
        return "<p style='color:#aaa;'>无注意力数据</p>"

    # 情感对应的色调 (HSL hue)
    emotion_hues = {
        "happy": 45,      # 金色
        "sad": 220,        # 蓝色
        "angry": 348,      # 红色
        "fearful": 280,    # 紫色
        "disgusted": 140,  # 绿色
        "surprised": 30,   # 橙色
        "neutral": 0,      # 灰色
    }
    hue = emotion_hues.get(dominant_emotion, 0)

    # 找到最大权重用于归一化显示
    max_weight = max(w for _, w in char_weights) if char_weights else 1.0

    spans = []
    for char, weight in char_weights:
        # 归一化到 [0, 1]
        norm_w = weight / max_weight if max_weight > 0 else 0
        # 背景色：饱和度和亮度随权重变化
        if dominant_emotion == "neutral":
            bg = f"rgba(150, 150, 150, {norm_w * 0.7:.2f})"
        else:
            saturation = 70 + norm_w * 30  # 70-100%
            lightness = 85 - norm_w * 40   # 45-85%
            bg = f"hsla({hue}, {saturation:.0f}%, {lightness:.0f}%, {norm_w * 0.8:.2f})"

        spans.append(
            f'<span style="'
            f"background-color: {bg}; "
            f"padding: 4px 2px; "
            f"margin: 1px; "
            f"border-radius: 3px; "
            f"display: inline-block; "
            f"font-size: 18px; "
            f"line-height: 1.8; "
            f"color: white; "
            f"transition: all 0.2s;"
            f'" title="注意力权重: {weight:.4f}">{char}</span>'
        )

    html = f"""
    <div style="
        background: #1a1a2e;
        padding: 20px;
        border-radius: 12px;
        font-family: 'Microsoft YaHei', sans-serif;
        border: 1px solid #333;
    ">
        <div style="color: #aaa; font-size: 13px; margin-bottom: 12px;">
            📝 注意力热力图 — 颜色越深表示模型对该字词的关注度越高
        </div>
        <div style="line-height: 2.2; letter-spacing: 2px;">
            {''.join(spans)}
        </div>
    </div>
    """
    return html


def create_attention_legend() -> str:
    """创建注意力热力图的图例"""
    return """
    <div style="
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        font-size: 12px;
        color: #aaa;
    ">
        <span>低关注</span>
        <div style="
            width: 150px; height: 12px;
            background: linear-gradient(to right, rgba(255,255,255,0.05), rgba(255,165,0,0.8));
            border-radius: 6px;
        "></div>
        <span>高关注</span>
    </div>
    """
