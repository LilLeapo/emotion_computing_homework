"""语音情感分析标签页"""

import gradio as gr

from models.speech_emotion import SpeechEmotionRecognizer
from processing.audio_processor import load_audio
from ui.components import format_emotion_result
from visualization.bar_chart import create_bar_chart
from visualization.radar_chart import create_radar_chart

_speech_model = None


def _get_model() -> SpeechEmotionRecognizer:
    global _speech_model
    if _speech_model is None:
        _speech_model = SpeechEmotionRecognizer()
    return _speech_model


def analyze_audio(audio_input):
    """分析语音情感"""
    if audio_input is None:
        return None, None, ""

    # Gradio 音频组件返回 (采样率, numpy数组) 或文件路径
    if isinstance(audio_input, tuple):
        sr, audio_data = audio_input
        import numpy as np
        # 转为 float32 并归一化
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
        # 如果是立体声，取平均
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        model = _get_model()
        scores = model.predict(audio_data, sr)
    else:
        model = _get_model()
        scores = model.predict(audio_input)

    radar = create_radar_chart(scores, "语音情感分布")
    bar = create_bar_chart(scores, "语音情感置信度")
    result_md = format_emotion_result(scores)

    return radar, bar, result_md


def create_audio_tab() -> gr.Tab:
    """创建语音情感分析标签页"""
    with gr.Tab("🎤 语音情感分析") as tab:
        gr.Markdown("### 上传音频或录制语音，分析情感特征")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="上传音频或录制",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                analyze_btn = gr.Button("🔍 开始分析", variant="primary", size="lg")

            with gr.Column(scale=2):
                result_md = gr.Markdown(label="分析结果")
                with gr.Row():
                    radar_plot = gr.Plot(label="情感雷达图")
                    bar_plot = gr.Plot(label="置信度柱状图")

        analyze_btn.click(
            fn=analyze_audio,
            inputs=[audio_input],
            outputs=[radar_plot, bar_plot, result_md],
        )

    return tab
