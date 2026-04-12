"""全局配置：模型ID、标签映射、配色方案"""

import torch

# ==================== 设备配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    "text": {
        "model_id": "iic/nlp_structbert_emotion-classification_chinese-base",
        "source": "ModelScope (阿里达摩院)",
        "description": "StructBERT 中文情感7分类模型",
    },
    "speech": {
        "model_id": "iic/emotion2vec_plus_large",
        "source": "ModelScope / FunASR (Ma et al., 2024)",
        "description": "emotion2vec+ 语音情感表征模型 (2024 SOTA)",
    },
    "face": {
        "model_id": "trpakov/vit-face-expression",
        "source": "HuggingFace",
        "description": "ViT-base 面部表情识别模型 (FER2013)",
    },
}

# ==================== 统一情感标签 ====================
# 7类基本情感 (Ekman 6 + Neutral)
EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]

EMOTION_LABELS_CN = {
    "happy": "高兴",
    "sad": "悲伤",
    "angry": "愤怒",
    "fearful": "恐惧",
    "disgusted": "厌恶",
    "surprised": "惊讶",
    "neutral": "中性",
}

# ==================== 各模型标签到统一标签的映射 ====================
TEXT_LABEL_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "disgusted": "disgusted",
    "surprised": "surprised",
    "neutral": "neutral",
    # ModelScope StructBERT 可能的中文标签
    "高兴": "happy",
    "悲伤": "sad",
    "愤怒": "angry",
    "恐惧": "fearful",
    "厌恶": "disgusted",
    "惊讶": "surprised",
    "中性": "neutral",
    # 额外可能的标签
    "happiness": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fearful",
    "disgust": "disgusted",
    "surprise": "surprised",
    "like": "happy",
    # HuggingFace 三分类模型标签
    "positive": "happy",
    "negative": "sad",
    # StructBERT 可能的额外标签
    "喜好": "happy",
    "乐": "happy",
    "哀": "sad",
    "怒": "angry",
    "惧": "fearful",
    "恶": "disgusted",
    "惊": "surprised",
}

SPEECH_LABEL_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "disgusted": "disgusted",
    "surprised": "surprised",
    "neutral": "neutral",
    # emotion2vec 标签
    "生气/angry": "angry",
    "厌恶/disgusted": "disgusted",
    "恐惧/fearful": "fearful",
    "开心/happy": "happy",
    "中立/neutral": "neutral",
    "其他/other": "neutral",
    "难过/sad": "sad",
    "吃惊/surprised": "surprised",
    "<unk>": "neutral",
    "other": "neutral",
    "unknown": "neutral",
}

FACE_LABEL_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "disgust": "disgusted",
    "surprise": "surprised",
    "neutral": "neutral",
}

# ==================== 可视化配色 ====================
EMOTION_COLORS = {
    "happy": "#FFD700",      # 金色
    "sad": "#4169E1",        # 皇家蓝
    "angry": "#DC143C",      # 猩红
    "fearful": "#9932CC",    # 暗紫
    "disgusted": "#228B22",  # 森林绿
    "surprised": "#FF8C00",  # 暗橙
    "neutral": "#808080",    # 灰色
}

EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgusted": "🤢",
    "surprised": "😲",
    "neutral": "😐",
}

# ==================== Gradio 配置 ====================
GRADIO_CONFIG = {
    "title": "🎭 多模态情感识别系统",
    "description": "基于 StructBERT + emotion2vec + ViT 的中文多模态情感识别与可视化系统",
    "server_port": 7860,
    "share": False,
}

# ==================== 视频处理配置 ====================
VIDEO_CONFIG = {
    "segment_duration": 2.0,    # 每个分析片段的秒数
    "sample_rate": 16000,       # 音频采样率
    "max_frames_per_segment": 5,  # 每段最多分析帧数
}
