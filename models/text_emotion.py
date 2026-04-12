"""中文文本情感识别模型

基于阿里达摩院 StructBERT 中文情感分类模型。
模型来源: ModelScope - iic/nlp_structbert_emotion-classification_chinese-base
参考论文: StructBERT (Wang et al., 2020)

增量贡献: 统一标签映射、注意力提取用于可视化
"""

import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEVICE, EMOTION_LABELS
from models.base import BaseEmotionRecognizer
from processing.text_processor import clean_text
from utils.label_mapper import normalize_scores


class TextEmotionRecognizer(BaseEmotionRecognizer):

    def __init__(self):
        super().__init__(modality="text")
        self._tokenizer = None

    def load_model(self):
        """加载中文情感分类模型。

        按优先级尝试:
        1. 从 ModelScope 缓存直接用 transformers 加载 StructBERT
        2. 用 ModelScope pipeline 加载
        3. HuggingFace 备选模型
        """
        # 方案1: 从 ModelScope 缓存目录直接用 transformers 加载
        ms_cache = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
        local_path = os.path.join(ms_cache, "iic", "nlp_structbert_emotion-classification_chinese-base")
        # FunASR/ModelScope 有时用 models/ 子目录
        local_path_alt = os.path.join(ms_cache, "models", "iic", "nlp_structbert_emotion-classification_chinese-base")

        for path in [local_path, local_path_alt]:
            if os.path.isdir(path) and any(f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(path)):
                try:
                    print(f"[TextModel] 从本地缓存加载: {path}")
                    self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        path, trust_remote_code=True, attn_implementation="eager"
                    )
                    self._model.to(DEVICE)
                    self._model.eval()
                    self._backend = "structbert_local"
                    self._loaded = True
                    print(f"[TextModel] StructBERT 加载成功 (本地缓存)")
                    return
                except Exception as e:
                    print(f"[TextModel] 本地缓存加载失败: {e}")

        # 方案2: ModelScope pipeline
        try:
            from modelscope.pipelines import pipeline as ms_pipeline
            self._model = ms_pipeline(
                "text-classification",
                model="iic/nlp_structbert_emotion-classification_chinese-base",
            )
            self._backend = "modelscope"
            self._loaded = True
            print("[TextModel] StructBERT 加载成功 (ModelScope pipeline)")
            return
        except Exception as e:
            print(f"[TextModel] ModelScope pipeline 加载失败: {e}")

        # 方案3: HuggingFace 中文情感模型
        try:
            model_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
            print(f"[TextModel] 降级到 HuggingFace 备选: {model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_id, attn_implementation="eager"
            )
            self._model.to(DEVICE)
            self._model.eval()
            self._backend = "huggingface"
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"无法加载文本情感模型: {e}")

    def predict(self, text: str) -> dict[str, float]:
        """对中文文本进行情感预测。"""
        self.ensure_loaded()
        text = clean_text(text)
        if not text:
            return {e: 1.0 / 7 for e in EMOTION_LABELS}

        if self._backend == "modelscope":
            return self._predict_modelscope(text)
        else:
            return self._predict_transformers(text)

    def _predict_modelscope(self, text: str) -> dict[str, float]:
        result = self._model(text)
        raw_scores = {}
        if isinstance(result, dict):
            labels = result.get("labels", result.get("label", []))
            scores = result.get("scores", result.get("score", []))
            if isinstance(labels, str):
                labels, scores = [labels], [scores]
            for label, score in zip(labels, scores):
                raw_scores[label] = float(score)
        elif isinstance(result, list):
            for item in result:
                raw_scores[item["label"]] = float(item["score"])
        return normalize_scores(raw_scores, "text")

    def _predict_transformers(self, text: str) -> dict[str, float]:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        label_names = self._model.config.id2label
        raw_scores = {label_names[i]: float(probs[i]) for i in range(len(probs))}
        return normalize_scores(raw_scores, "text")

    def predict_with_attention(self, text: str) -> tuple[dict[str, float], list[tuple[str, float]]]:
        """预测情感并提取注意力权重（用于可视化）。"""
        self.ensure_loaded()
        text = clean_text(text)
        if not text:
            return {e: 1.0 / 7 for e in EMOTION_LABELS}, []

        if self._backend == "modelscope":
            scores = self.predict(text)
            char_weights = self._make_pseudo_attention(text, scores)
            return scores, char_weights
        else:
            return self._predict_with_attention_transformers(text)

    def _predict_with_attention_transformers(self, text: str) -> tuple[dict[str, float], list[tuple[str, float]]]:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)

        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        label_names = self._model.config.id2label
        raw_scores = {label_names[i]: float(probs[i]) for i in range(len(probs))}
        emotion_scores = normalize_scores(raw_scores, "text")

        # 提取注意力
        if outputs.attentions and len(outputs.attentions) > 0:
            attention = outputs.attentions[-1]  # (batch, heads, seq, seq)
            cls_attention = attention[0, :, 0, :].mean(dim=0).cpu().numpy()
            tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            char_weights = []
            for token, weight in zip(tokens, cls_attention):
                if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                    continue
                display = token.replace("##", "").replace("▁", "")
                if display:
                    char_weights.append((display, float(weight)))
            if char_weights:
                total = sum(w for _, w in char_weights)
                if total > 0:
                    char_weights = [(c, w / total) for c, w in char_weights]
        else:
            char_weights = self._make_pseudo_attention(text, emotion_scores)

        return emotion_scores, char_weights

    def _make_pseudo_attention(self, text: str, scores: dict[str, float]) -> list[tuple[str, float]]:
        """当无法提取真实注意力时，基于字符重要性生成伪注意力。

        使用逐字遮蔽法：依次遮蔽每个字符，观察预测变化。
        对于短文本有效，长文本降级为均匀分布。
        """
        chars = list(text)
        if len(chars) > 50:
            return [(c, 1.0 / len(chars)) for c in chars]

        from utils.label_mapper import get_dominant_emotion
        dominant, base_conf = get_dominant_emotion(scores)

        weights = []
        for i, char in enumerate(chars):
            masked = text[:i] + text[i+1:]
            if not masked.strip():
                weights.append(0.0)
                continue
            masked_scores = self.predict(masked)
            masked_conf = masked_scores.get(dominant, 0)
            # 遮蔽后置信度下降越多，说明这个字越重要
            importance = max(0, base_conf - masked_conf)
            weights.append(importance)

        total = sum(weights)
        if total > 0:
            return [(c, w / total) for c, w in zip(chars, weights)]
        return [(c, 1.0 / len(chars)) for c in chars]
