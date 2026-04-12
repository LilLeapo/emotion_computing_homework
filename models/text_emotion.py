"""中文文本情感识别模型

基于阿里达摩院 StructBERT 中文情感分类模型。
模型来源: ModelScope - iic/nlp_structbert_emotion-classification_chinese-base
参考论文: StructBERT (Wang et al., 2020)

增量贡献: 统一标签映射、注意力提取用于可视化
"""

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
        """加载 StructBERT 中文情感分类模型。

        优先从 ModelScope 加载，失败则使用 HuggingFace 备选方案。
        """
        try:
            # 方案1: ModelScope pipeline
            from modelscope.pipelines import pipeline as ms_pipeline
            self._model = ms_pipeline(
                "text-classification",
                model="iic/nlp_structbert_emotion-classification_chinese-base",
                device=DEVICE,
            )
            self._backend = "modelscope"
        except Exception:
            try:
                # 方案2: HuggingFace 多语言情感模型
                model_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_id, attn_implementation="eager"
                )
                self._model.to(DEVICE)
                self._model.eval()
                self._backend = "huggingface"
            except Exception as e:
                raise RuntimeError(f"无法加载文本情感模型: {e}")
        self._loaded = True

    def predict(self, text: str) -> dict[str, float]:
        """对中文文本进行情感预测。

        Args:
            text: 输入中文文本

        Returns:
            7类情感的概率分布字典
        """
        self.ensure_loaded()
        text = clean_text(text)
        if not text:
            return {e: 1.0 / 7 for e in EMOTION_LABELS}

        if self._backend == "modelscope":
            return self._predict_modelscope(text)
        else:
            return self._predict_huggingface(text)

    def _predict_modelscope(self, text: str) -> dict[str, float]:
        result = self._model(text)
        # ModelScope 返回格式: {"labels": [...], "scores": [...]} 或类似
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

    def _predict_huggingface(self, text: str) -> dict[str, float]:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        # distilbert 多语言情感模型: positive, negative, neutral
        label_names = self._model.config.id2label
        raw_scores = {label_names[i]: float(probs[i]) for i in range(len(probs))}
        return normalize_scores(raw_scores, "text")

    def predict_with_attention(self, text: str) -> tuple[dict[str, float], list[tuple[str, float]]]:
        """预测情感并提取注意力权重（用于可视化）。

        Returns:
            (情感概率字典, [(字符, 注意力权重), ...])
        """
        self.ensure_loaded()
        text = clean_text(text)
        if not text:
            return {e: 1.0 / 7 for e in EMOTION_LABELS}, []

        if self._backend == "huggingface":
            return self._predict_with_attention_hf(text)
        else:
            # ModelScope 模型不易提取注意力，使用均匀伪注意力
            scores = self.predict(text)
            chars = list(text)
            uniform_attn = [(c, 1.0 / len(chars)) for c in chars]
            return scores, uniform_attn

    def _predict_with_attention_hf(self, text: str) -> tuple[dict[str, float], list[tuple[str, float]]]:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)

        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        label_names = self._model.config.id2label
        raw_scores = {label_names[i]: float(probs[i]) for i in range(len(probs))}
        emotion_scores = normalize_scores(raw_scores, "text")

        # 提取最后一层注意力 [CLS] -> 所有 token
        attention = outputs.attentions[-1]  # (batch, heads, seq, seq)
        cls_attention = attention[0, :, 0, :].mean(dim=0).cpu().numpy()  # 多头平均

        # 将 token 注意力映射回字符
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        char_weights = []
        for token, weight in zip(tokens, cls_attention):
            if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                continue
            # 去掉 ## 前缀
            display = token.replace("##", "").replace("▁", "")
            if display:
                char_weights.append((display, float(weight)))

        # 归一化
        if char_weights:
            total = sum(w for _, w in char_weights)
            if total > 0:
                char_weights = [(c, w / total) for c, w in char_weights]

        return emotion_scores, char_weights
