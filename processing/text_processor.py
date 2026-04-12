"""文本预处理：清洗、分词等"""

import re


def clean_text(text: str) -> str:
    """清洗中文文本：去除多余空白、特殊字符等"""
    if not text or not text.strip():
        return ""
    # 去除 URL
    text = re.sub(r"https?://\S+", "", text)
    # 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_text(text: str) -> bool:
    """检查文本是否有效（非空、有实际内容）"""
    cleaned = clean_text(text)
    return len(cleaned) >= 2
