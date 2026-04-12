"""语音情感识别模型

基于 emotion2vec+ (Ma et al., 2024) 语音情感表征模型。
模型来源: ModelScope / FunASR - iic/emotion2vec_plus_large
参考论文: "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation"

增量贡献: 统一标签映射、音频分段处理、与多模态融合集成
"""

import tempfile

import numpy as np
import soundfile as sf

from config import DEVICE, EMOTION_LABELS, VIDEO_CONFIG
from models.base import BaseEmotionRecognizer
from utils.label_mapper import normalize_scores


class SpeechEmotionRecognizer(BaseEmotionRecognizer):

    def __init__(self):
        super().__init__(modality="speech")
        self._label_list = None

    def load_model(self):
        """加载 emotion2vec+ 模型。

        优先使用 FunASR 加载，失败则使用备选方案。
        """
        try:
            from funasr import AutoModel
            self._model = AutoModel(
                model="iic/emotion2vec_plus_large",
                device=DEVICE,
            )
            self._backend = "funasr"
        except Exception:
            try:
                # 备选: 使用 transformers 加载 wav2vec2 情感模型
                from transformers import pipeline as hf_pipeline
                self._model = hf_pipeline(
                    "audio-classification",
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    device=0 if DEVICE == "cuda" else -1,
                )
                self._backend = "huggingface"
            except Exception as e:
                raise RuntimeError(f"无法加载语音情感模型: {e}")
        self._loaded = True

    def predict(self, audio_input, sr: int = 16000) -> dict[str, float]:
        """对音频进行情感预测。

        Args:
            audio_input: 音频文件路径(str) 或 numpy 数组
            sr: 采样率（仅当 audio_input 为数组时使用）

        Returns:
            7类情感的概率分布字典
        """
        self.ensure_loaded()

        if self._backend == "funasr":
            return self._predict_funasr(audio_input, sr)
        else:
            return self._predict_huggingface(audio_input, sr)

    def _predict_funasr(self, audio_input, sr: int) -> dict[str, float]:
        # FunASR emotion2vec 接受文件路径
        if isinstance(audio_input, np.ndarray):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio_input, sr)
            audio_path = tmp.name
        else:
            audio_path = audio_input

        result = self._model.generate(audio_path, granularity="utterance",
                                      extract_embedding=False)

        raw_scores = {}
        if result and len(result) > 0:
            rec = result[0]
            # emotion2vec 返回格式: {"labels": [...], "scores": [...]}
            labels = rec.get("labels", [])
            scores_arr = rec.get("scores", [])
            if labels and scores_arr:
                for label, score in zip(labels, scores_arr):
                    raw_scores[label] = float(score)
            elif "label" in rec:
                raw_scores[rec["label"]] = 1.0

        return normalize_scores(raw_scores, "speech")

    def _predict_huggingface(self, audio_input, sr: int) -> dict[str, float]:
        if isinstance(audio_input, str):
            import librosa
            audio_input, sr = librosa.load(audio_input, sr=16000, mono=True)

        result = self._model({"raw": audio_input, "sampling_rate": sr})
        raw_scores = {}
        for item in result:
            raw_scores[item["label"]] = float(item["score"])

        return normalize_scores(raw_scores, "speech")

    def predict_segments(self, audio: np.ndarray, sr: int,
                         segment_duration: float = None) -> list[dict]:
        """对长音频进行分段情感预测。

        Returns:
            列表，每项: {"start": float, "end": float, "scores": dict}
        """
        self.ensure_loaded()
        seg_dur = segment_duration or VIDEO_CONFIG["segment_duration"]
        seg_samples = int(seg_dur * sr)

        results = []
        for i, start in enumerate(range(0, len(audio), seg_samples)):
            segment = audio[start:start + seg_samples]
            if len(segment) < sr * 0.5:  # 太短则跳过
                continue
            scores = self.predict(segment, sr)
            results.append({
                "start": start / sr,
                "end": min((start + seg_samples) / sr, len(audio) / sr),
                "scores": scores,
            })
        return results
