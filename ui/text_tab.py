"""文本情感分析标签页"""

import gradio as gr

from models.text_emotion import TextEmotionRecognizer
from ui.components import create_examples_text, format_emotion_result
from utils.label_mapper import get_dominant_emotion
from visualization.attention_viz import create_attention_html
from visualization.bar_chart import create_bar_chart
from visualization.radar_chart import create_radar_chart

# 全局模型实例（延迟加载）
_text_model = None


def _get_model() -> TextEmotionRecognizer:
    global _text_model
    if _text_model is None:
        _text_model = TextEmotionRecognizer()
    return _text_model


def analyze_text(text: str):
    """分析文本情感"""
    if not text or not text.strip():
        return None, None, "<p style='color:#aaa;'>请输入文本</p>", ""

    model = _get_model()
    scores, char_weights = model.predict_with_attention(text)
    dominant, _ = get_dominant_emotion(scores)

    radar = create_radar_chart(scores, "文本情感分布")
    bar = create_bar_chart(scores, "文本情感置信度")
    attention_html = create_attention_html(char_weights, dominant)
    result_md = format_emotion_result(scores)

    return radar, bar, attention_html, result_md


def create_text_tab() -> gr.Tab:
    """创建文本情感分析标签页"""
    with gr.Tab("📝 文本情感分析") as tab:
        gr.Markdown("### 输入中文文本，分析其情感倾向")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="请输入要分析的中文文本...",
                    lines=4,
                    max_lines=10,
                )
                analyze_btn = gr.Button("🔍 开始分析", variant="primary", size="lg")
                gr.Examples(
                    examples=create_examples_text(),
                    inputs=text_input,
                    label="示例文本",
                )

            with gr.Column(scale=2):
                result_md = gr.Markdown(label="分析结果")
                with gr.Row():
                    radar_plot = gr.Plot(label="情感雷达图")
                    bar_plot = gr.Plot(label="置信度柱状图")
                attention_output = gr.HTML(label="注意力热力图")

        analyze_btn.click(
            fn=analyze_text,
            inputs=[text_input],
            outputs=[radar_plot, bar_plot, attention_output, result_md],
        )
        # 回车也可触发
        text_input.submit(
            fn=analyze_text,
            inputs=[text_input],
            outputs=[radar_plot, bar_plot, attention_output, result_md],
        )

    return tab
