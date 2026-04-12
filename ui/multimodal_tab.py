"""多模态融合分析标签页"""

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from config import EMOTION_EMOJIS, EMOTION_LABELS, EMOTION_LABELS_CN
from models.face_emotion import FaceEmotionRecognizer
from models.fusion import MultimodalFusion
from models.speech_emotion import SpeechEmotionRecognizer
from models.text_emotion import TextEmotionRecognizer
from processing.text_processor import is_valid_text
from utils.label_mapper import get_dominant_emotion
from visualization.bar_chart import create_comparison_bars
from visualization.fusion_viz import create_fusion_weight_pie, create_result_card, create_sankey_diagram
from visualization.radar_chart import create_multi_radar, create_radar_chart

# 全局模型实例
_models = {}
_fusion = None


def _get_models():
    global _models, _fusion
    if "text" not in _models:
        _models["text"] = TextEmotionRecognizer()
    if "speech" not in _models:
        _models["speech"] = SpeechEmotionRecognizer()
    if "face" not in _models:
        _models["face"] = FaceEmotionRecognizer()
    if _fusion is None:
        _fusion = MultimodalFusion()
    return _models, _fusion


def analyze_multimodal(text_input, audio_input, image_input):
    """多模态融合分析"""
    models, fusion = _get_models()

    text_scores = None
    speech_scores = None
    face_scores = None

    # 文本分析
    if text_input and is_valid_text(text_input):
        text_scores = models["text"].predict(text_input)

    # 语音分析
    if audio_input is not None:
        if isinstance(audio_input, tuple):
            sr, audio_data = audio_input
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if audio_data.max() > 1.0:
                    audio_data = audio_data / 32768.0
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            speech_scores = models["speech"].predict(audio_data, sr)
        else:
            speech_scores = models["speech"].predict(audio_input)

    # 面部分析
    if image_input is not None:
        if isinstance(image_input, np.ndarray):
            cv_image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            cv_image = cv2.imread(image_input)
        face_scores = models["face"].predict(cv_image)

    # 检查是否有有效输入
    if text_scores is None and speech_scores is None and face_scores is None:
        empty_msg = "请至少提供一种模态的输入（文本/语音/图片）"
        return None, None, None, None, empty_msg, None

    # 融合
    fused_scores, weights = fusion.fuse(text_scores, speech_scores, face_scores)
    dominant, confidence = get_dominant_emotion(fused_scores)

    # 可视化
    # 1. 各模态对比雷达图
    scores_list = []
    labels = []
    scores_dict = {}
    if text_scores:
        scores_list.append(text_scores)
        labels.append("文本")
        scores_dict["文本"] = text_scores
    if speech_scores:
        scores_list.append(speech_scores)
        labels.append("语音")
        scores_dict["语音"] = speech_scores
    if face_scores:
        scores_list.append(face_scores)
        labels.append("面部")
        scores_dict["面部"] = face_scores
    scores_list.append(fused_scores)
    labels.append("融合结果")
    scores_dict["融合"] = fused_scores

    multi_radar = create_multi_radar(scores_list, labels, "多模态情感对比")

    # 2. 融合结果雷达图
    fused_radar = create_radar_chart(fused_scores, "融合情感分布")

    # 3. 融合权重饼图
    weight_pie = create_fusion_weight_pie(weights) if weights else None

    # 4. 桑基图
    modality_scores_for_sankey = {}
    if text_scores:
        modality_scores_for_sankey["文本"] = text_scores
    if speech_scores:
        modality_scores_for_sankey["语音"] = speech_scores
    if face_scores:
        modality_scores_for_sankey["面部"] = face_scores
    sankey = create_sankey_diagram(modality_scores_for_sankey, fused_scores, weights)

    # 5. 结果卡片
    result_html = create_result_card(dominant, confidence, EMOTION_EMOJIS[dominant])

    # 6. 详细得分表
    table_data = {"情感": [f"{EMOTION_EMOJIS[e]} {EMOTION_LABELS_CN[e]}" for e in EMOTION_LABELS]}
    if text_scores:
        table_data["文本"] = [f"{text_scores.get(e, 0):.1%}" for e in EMOTION_LABELS]
    if speech_scores:
        table_data["语音"] = [f"{speech_scores.get(e, 0):.1%}" for e in EMOTION_LABELS]
    if face_scores:
        table_data["面部"] = [f"{face_scores.get(e, 0):.1%}" for e in EMOTION_LABELS]
    table_data["融合结果"] = [f"{fused_scores.get(e, 0):.1%}" for e in EMOTION_LABELS]
    df = pd.DataFrame(table_data)

    return multi_radar, fused_radar, weight_pie, sankey, result_html, df


def create_multimodal_tab() -> gr.Tab:
    """创建多模态融合分析标签页"""
    with gr.Tab("🔗 多模态融合") as tab:
        gr.Markdown("### 同时输入文本、语音和图片，进行多模态融合情感分析")
        gr.Markdown("*可以输入任意组合（至少一种），系统会自动调整融合权重*")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="📝 文本输入",
                    placeholder="输入中文文本...",
                    lines=3,
                )
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="🎤 语音输入",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="😊 面部图片",
                    type="numpy",
                    sources=["upload", "webcam"],
                )

        analyze_btn = gr.Button("🚀 多模态融合分析", variant="primary", size="lg")

        # 结果区域
        result_html = gr.HTML(label="融合结果")

        with gr.Row():
            multi_radar = gr.Plot(label="多模态对比雷达图")
            fused_radar = gr.Plot(label="融合情感分布")

        with gr.Row():
            weight_pie = gr.Plot(label="模态融合权重")
            sankey_plot = gr.Plot(label="融合流程桑基图")

        score_table = gr.Dataframe(label="详细得分表", interactive=False)

        analyze_btn.click(
            fn=analyze_multimodal,
            inputs=[text_input, audio_input, image_input],
            outputs=[multi_radar, fused_radar, weight_pie, sankey_plot, result_html, score_table],
        )

    return tab
