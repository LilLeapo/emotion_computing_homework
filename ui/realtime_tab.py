"""实时摄像头表情识别标签页"""

import cv2
import gradio as gr
import numpy as np

from config import EMOTION_COLORS, EMOTION_EMOJIS, EMOTION_LABELS, EMOTION_LABELS_CN
from models.face_emotion import FaceEmotionRecognizer
from utils.label_mapper import get_dominant_emotion

_face_model = None


def _get_model() -> FaceEmotionRecognizer:
    global _face_model
    if _face_model is None:
        _face_model = FaceEmotionRecognizer()
    return _face_model


def process_frame(frame):
    """处理单帧摄像头图像，返回标注图和情感柱状图。"""
    if frame is None:
        return None, _create_empty_bar_html()

    model = _get_model()
    # Gradio 传入 RGB numpy
    cv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    scores, annotated, faces = model.predict_with_annotation(cv_image)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    bar_html = _create_live_bar_html(scores)
    return annotated_rgb, bar_html


def _create_live_bar_html(scores: dict[str, float]) -> str:
    """创建实时情感柱状图 HTML"""
    dominant, confidence = get_dominant_emotion(scores)

    bars = []
    for emotion in sorted(EMOTION_LABELS, key=lambda e: scores.get(e, 0), reverse=True):
        value = scores.get(emotion, 0)
        color = EMOTION_COLORS[emotion]
        emoji = EMOTION_EMOJIS[emotion]
        cn_name = EMOTION_LABELS_CN[emotion]
        width_pct = value * 100

        bars.append(f"""
        <div style="display: flex; align-items: center; margin: 6px 0; gap: 8px;">
            <span style="width: 80px; text-align: right; font-size: 14px;">{emoji} {cn_name}</span>
            <div style="flex: 1; background: #2a2a3e; border-radius: 8px; height: 24px; overflow: hidden;">
                <div style="
                    width: {width_pct:.1f}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color}88, {color});
                    border-radius: 8px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <span style="width: 50px; font-size: 13px; color: #ccc;">{value:.0%}</span>
        </div>
        """)

    return f"""
    <div style="
        background: #1a1a2e;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
    ">
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="font-size: 36px;">{EMOTION_EMOJIS[dominant]}</span>
            <span style="font-size: 20px; color: {EMOTION_COLORS[dominant]}; font-weight: bold; margin-left: 10px;">
                {EMOTION_LABELS_CN[dominant]} ({confidence:.0%})
            </span>
        </div>
        {''.join(bars)}
    </div>
    """


def _create_empty_bar_html() -> str:
    return """
    <div style="background: #1a1a2e; padding: 40px; border-radius: 12px; text-align: center; color: #666;">
        等待摄像头输入...
    </div>
    """


def create_realtime_tab() -> gr.Tab:
    """创建实时摄像头表情识别标签页"""
    with gr.Tab("📹 实时识别") as tab:
        gr.Markdown("### 开启摄像头，实时识别面部表情情感")
        gr.Markdown("*点击下方摄像头区域开启，系统将实时分析你的表情*")

        with gr.Row():
            with gr.Column(scale=1):
                webcam = gr.Image(
                    label="摄像头",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    mirror_webcam=True,
                )

            with gr.Column(scale=1):
                annotated_output = gr.Image(label="检测结果", type="numpy")
                bar_output = gr.HTML(label="实时情感分析")

        webcam.stream(
            fn=process_frame,
            inputs=[webcam],
            outputs=[annotated_output, bar_output],
        )

    return tab
