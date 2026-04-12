"""共享 UI 组件和工具函数"""

import gradio as gr

from config import EMOTION_EMOJIS, EMOTION_LABELS, EMOTION_LABELS_CN
from utils.label_mapper import get_dominant_emotion


def format_emotion_result(scores: dict[str, float]) -> str:
    """格式化情感预测结果为 Markdown 文本"""
    dominant, confidence = get_dominant_emotion(scores)
    emoji = EMOTION_EMOJIS[dominant]
    cn_name = EMOTION_LABELS_CN[dominant]

    lines = [f"### {emoji} 主导情感: **{cn_name}** ({confidence:.1%})", "", "| 情感 | 概率 |", "|------|------|"]
    for e in sorted(EMOTION_LABELS, key=lambda x: scores.get(x, 0), reverse=True):
        lines.append(f"| {EMOTION_EMOJIS[e]} {EMOTION_LABELS_CN[e]} | {scores.get(e, 0):.1%} |")
    return "\n".join(lines)


def create_examples_text() -> list[list[str]]:
    """中文情感分析示例文本"""
    return [
        ["今天天气真好，心情特别愉快！"],
        ["听到这个消息我真的很难过，眼泪都快掉下来了。"],
        ["这也太过分了吧！凭什么这样对我！"],
        ["前面的路好黑啊，我一个人有点害怕。"],
        ["这个味道太恶心了，我真的受不了。"],
        ["天哪！你居然来了！太意外了！"],
        ["今天是周三，下午有个会议。"],
    ]
