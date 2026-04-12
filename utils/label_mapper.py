"""跨模态情感标签统一映射"""

import numpy as np

from config import (
    EMOTION_LABELS,
    FACE_LABEL_MAP,
    SPEECH_LABEL_MAP,
    TEXT_LABEL_MAP,
)


def _get_label_map(modality: str) -> dict:
    maps = {
        "text": TEXT_LABEL_MAP,
        "speech": SPEECH_LABEL_MAP,
        "face": FACE_LABEL_MAP,
    }
    return maps[modality]


def normalize_scores(raw_scores: dict[str, float], modality: str) -> dict[str, float]:
    """将模型原始输出映射到统一的7类情感标签。

    Args:
        raw_scores: 模型输出的 {原始标签: 概率} 字典
        modality: "text", "speech", 或 "face"

    Returns:
        统一格式的 {情感标签: 概率} 字典，所有值之和为1.0
    """
    label_map = _get_label_map(modality)
    unified = {e: 0.0 for e in EMOTION_LABELS}

    for raw_label, score in raw_scores.items():
        # 尝试映射标签（不区分大小写）
        key = raw_label.lower().strip()
        mapped = label_map.get(key) or label_map.get(raw_label)
        if mapped and mapped in unified:
            unified[mapped] += score

    # 归一化确保概率和为1
    total = sum(unified.values())
    if total > 0:
        unified = {k: v / total for k, v in unified.items()}
    else:
        # 如果无法映射任何标签，默认均匀分布
        n = len(EMOTION_LABELS)
        unified = {e: 1.0 / n for e in EMOTION_LABELS}

    return unified


def get_dominant_emotion(scores: dict[str, float]) -> tuple[str, float]:
    """获取概率最高的情感标签及其概率"""
    dominant = max(scores, key=scores.get)
    return dominant, scores[dominant]


def scores_to_array(scores: dict[str, float]) -> np.ndarray:
    """将情感得分字典转为与 EMOTION_LABELS 对齐的 numpy 数组"""
    return np.array([scores.get(e, 0.0) for e in EMOTION_LABELS])


def array_to_scores(arr: np.ndarray) -> dict[str, float]:
    """将 numpy 数组转回情感得分字典"""
    return {e: float(arr[i]) for i, e in enumerate(EMOTION_LABELS)}
