"""视频情感分析标签页"""

import cv2
import gradio as gr
import numpy as np

from config import EMOTION_EMOJIS, EMOTION_LABELS, EMOTION_LABELS_CN
from models.face_emotion import FaceEmotionRecognizer
from models.fusion import MultimodalFusion
from models.speech_emotion import SpeechEmotionRecognizer
from processing.audio_processor import load_audio
from processing.video_processor import extract_audio_from_video, extract_frames, get_video_info
from utils.label_mapper import get_dominant_emotion
from visualization.fusion_viz import create_result_card
from visualization.radar_chart import create_radar_chart
from visualization.timeline_chart import create_dual_timeline, create_timeline

_face_model = None
_speech_model = None
_fusion = None


def _get_models():
    global _face_model, _speech_model, _fusion
    if _face_model is None:
        _face_model = FaceEmotionRecognizer()
    if _speech_model is None:
        _speech_model = SpeechEmotionRecognizer()
    if _fusion is None:
        _fusion = MultimodalFusion()
    return _face_model, _speech_model, _fusion


def analyze_video(video_path, progress=gr.Progress()):
    """分析视频情感"""
    if video_path is None:
        return None, None, None, "", []

    face_model, speech_model, fusion = _get_models()

    progress(0.1, desc="读取视频信息...")
    info = get_video_info(video_path)

    # 提取帧
    progress(0.2, desc="提取视频帧...")
    frames = extract_frames(video_path)

    # 提取音频
    progress(0.3, desc="提取音频...")
    audio_path = extract_audio_from_video(video_path)

    # 面部表情分析（逐帧）
    progress(0.4, desc="分析面部表情...")
    face_segments = {}  # segment_idx -> scores list
    key_frames = []

    for frame_info in frames:
        seg_idx = frame_info["segment_idx"]
        cv_image = frame_info["frame"]
        scores = face_model.predict(cv_image)

        if seg_idx not in face_segments:
            face_segments[seg_idx] = {
                "start": frame_info["timestamp"],
                "end": frame_info["timestamp"],
                "scores_list": [],
            }
        face_segments[seg_idx]["end"] = frame_info["timestamp"]
        face_segments[seg_idx]["scores_list"].append(scores)

    # 平均每段的面部分数
    face_timeline = []
    for seg_idx in sorted(face_segments.keys()):
        seg = face_segments[seg_idx]
        avg_scores = {}
        for emotion in EMOTION_LABELS:
            avg_scores[emotion] = np.mean([s.get(emotion, 0) for s in seg["scores_list"]])
        # 归一化
        total = sum(avg_scores.values())
        if total > 0:
            avg_scores = {k: v / total for k, v in avg_scores.items()}
        face_timeline.append({
            "start": seg["start"],
            "end": seg["end"] + 0.1,
            "scores": avg_scores,
        })

    # 语音情感分析（分段）
    progress(0.6, desc="分析语音情感...")
    speech_timeline = []
    if audio_path:
        audio, sr = load_audio(audio_path)
        speech_timeline = speech_model.predict_segments(audio, sr)

    # 创建可视化
    progress(0.8, desc="生成可视化...")

    # 双轨时间线
    timeline = create_dual_timeline(face_timeline, speech_timeline)

    # 整体综合情感（所有段的平均）
    all_face = [s["scores"] for s in face_timeline] if face_timeline else []
    all_speech = [s["scores"] for s in speech_timeline] if speech_timeline else []

    overall_face = None
    if all_face:
        overall_face = {e: np.mean([s.get(e, 0) for s in all_face]) for e in EMOTION_LABELS}
        total = sum(overall_face.values())
        if total > 0:
            overall_face = {k: v / total for k, v in overall_face.items()}

    overall_speech = None
    if all_speech:
        overall_speech = {e: np.mean([s.get(e, 0) for s in all_speech]) for e in EMOTION_LABELS}
        total = sum(overall_speech.values())
        if total > 0:
            overall_speech = {k: v / total for k, v in overall_speech.items()}

    fused_scores, _ = fusion.fuse(None, overall_speech, overall_face)
    fused_radar = create_radar_chart(fused_scores, "视频综合情感分布")

    dominant, conf = get_dominant_emotion(fused_scores)
    result_html = create_result_card(dominant, conf, EMOTION_EMOJIS[dominant])

    # 关键帧画廊（每段取一帧，标注情感）
    gallery_images = []
    for frame_info in frames:
        if frame_info["timestamp"] == 0 or frame_info == frames[0]:
            continue
        # 每段取首帧
        seg_idx = frame_info["segment_idx"]
        if any(g[1] == seg_idx for g in gallery_images):
            continue
        rgb = cv2.cvtColor(frame_info["frame"], cv2.COLOR_BGR2RGB)
        if seg_idx < len(face_timeline):
            emo, _ = get_dominant_emotion(face_timeline[seg_idx]["scores"])
            caption = f"{frame_info['timestamp']:.1f}s - {EMOTION_EMOJIS[emo]} {EMOTION_LABELS_CN[emo]}"
        else:
            caption = f"{frame_info['timestamp']:.1f}s"
        gallery_images.append((rgb, seg_idx, caption))

    gallery = [(img, cap) for img, _, cap in gallery_images[:12]]

    progress(1.0, desc="完成!")
    return timeline, fused_radar, result_html, f"视频时长: {info['duration']:.1f}s | 分辨率: {info['width']}x{info['height']}", gallery


def create_video_tab() -> gr.Tab:
    """创建视频情感分析标签页"""
    with gr.Tab("🎬 视频分析") as tab:
        gr.Markdown("### 上传视频，分析面部表情和语音情感随时间的变化")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="上传视频")
                analyze_btn = gr.Button("🔍 开始分析", variant="primary", size="lg")
                video_info = gr.Markdown(label="视频信息")

            with gr.Column(scale=2):
                result_html = gr.HTML(label="综合情感结果")
                timeline_plot = gr.Plot(label="情感时间线")
                fused_radar = gr.Plot(label="综合情感分布")
                gallery = gr.Gallery(label="关键帧", columns=4, height=250)

        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input],
            outputs=[timeline_plot, fused_radar, result_html, video_info, gallery],
        )

    return tab
