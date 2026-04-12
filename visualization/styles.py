"""统一配色方案和可视化风格"""

from config import EMOTION_COLORS, EMOTION_LABELS, EMOTION_LABELS_CN

# Plotly 模板配置
PLOTLY_TEMPLATE = "plotly_dark"

# 情感颜色列表（与 EMOTION_LABELS 对齐）
COLOR_SEQUENCE = [EMOTION_COLORS[e] for e in EMOTION_LABELS]

# 中文标签列表（与 EMOTION_LABELS 对齐）
CN_LABELS = [EMOTION_LABELS_CN[e] for e in EMOTION_LABELS]

# 雷达图配置
RADAR_CONFIG = {
    "fill": "toself",
    "opacity": 0.6,
    "line_width": 2,
}

# 柱状图配置
BAR_CONFIG = {
    "orientation": "h",
    "text_template": "%{x:.1%}",
    "height": 350,
}

# 时间线配置
TIMELINE_CONFIG = {
    "height": 400,
    "line_width": 2.5,
}
