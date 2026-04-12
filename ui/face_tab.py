"""面部表情识别标签页"""

import cv2
import gradio as gr
import numpy as np

from models.face_emotion import FaceEmotionRecognizer
from ui.components import format_emotion_result
from visualization.bar_chart import create_bar_chart
from visualization.radar_chart import create_radar_chart

_face_model = None


def _get_model() -> FaceEmotionRecognizer:
    global _face_model
    if _face_model is None:
        _face_model = FaceEmotionRecognizer()
    return _face_model


def analyze_face(image_input):
    """分析面部表情"""
    if image_input is None:
        return None, None, None, ""

    model = _get_model()

    # Gradio Image 组件返回 numpy 数组 (RGB)
    if isinstance(image_input, np.ndarray):
        # Gradio 默认是 RGB，转为 BGR 给 OpenCV
        cv_image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
    else:
        cv_image = cv2.imread(image_input)

    scores, annotated, faces = model.predict_with_annotation(cv_image)

    # 转回 RGB 给 Gradio 显示
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    radar = create_radar_chart(scores, "面部表情分布")
    bar = create_bar_chart(scores, "面部表情置信度")
    result_md = format_emotion_result(scores)

    face_info = f"检测到 {len(faces)} 张人脸" if faces else "未检测到人脸，已对整张图片分析"
    result_md = f"*{face_info}*\n\n" + result_md

    return annotated_rgb, radar, bar, result_md


def create_face_tab() -> gr.Tab:
    """创建面部表情识别标签页"""
    with gr.Tab("😊 面部表情识别") as tab:
        gr.Markdown("### 上传人脸图片或拍照，识别面部表情情感")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="上传图片或拍照",
                    type="numpy",
                    sources=["upload", "webcam"],
                )
                analyze_btn = gr.Button("🔍 开始分析", variant="primary", size="lg")

            with gr.Column(scale=2):
                result_md = gr.Markdown(label="分析结果")
                annotated_image = gr.Image(label="人脸检测结果", type="numpy")
                with gr.Row():
                    radar_plot = gr.Plot(label="表情雷达图")
                    bar_plot = gr.Plot(label="表情置信度")

        analyze_btn.click(
            fn=analyze_face,
            inputs=[image_input],
            outputs=[annotated_image, radar_plot, bar_plot, result_md],
        )

    return tab
