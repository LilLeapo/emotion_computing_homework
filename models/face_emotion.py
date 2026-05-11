"""面部表情识别模型

基于 ViT-base 在 FER2013 上微调的表情识别模型。
模型来源: HuggingFace - trpakov/vit-face-expression
参考: Vision Transformer (Dosovitskiy et al., 2020) + FER2013 数据集

增量贡献: 集成 MediaPipe 人脸检测、标签统一映射、批量帧处理
"""

import cv2
import numpy as np
from PIL import Image

from config import DEVICE, EMOTION_LABELS
from models.base import BaseEmotionRecognizer
from processing.face_detector import face_detector
from utils.label_mapper import normalize_scores


class FaceEmotionRecognizer(BaseEmotionRecognizer):

    def __init__(self):
        super().__init__(modality="face")

    def load_model(self):
        """加载 ViT 面部表情识别模型"""
        try:
            from transformers import pipeline as hf_pipeline
            self._model = hf_pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",
                framework="pt",
                device=0 if DEVICE == "cuda" else -1,
            )
            self._backend = "huggingface"
        except Exception as e:
            raise RuntimeError(f"无法加载面部表情模型: {e}")
        self._loaded = True

    def predict(self, image_input) -> dict[str, float]:
        """对图像中的人脸进行表情识别。

        Args:
            image_input: PIL Image, numpy 数组 (BGR/RGB), 或文件路径

        Returns:
            7类情感的概率分布字典（取最大置信度的人脸）
        """
        self.ensure_loaded()
        pil_image, cv_image = self._prepare_image(image_input)

        # 检测人脸
        faces = face_detector.detect_faces(cv_image)
        if not faces:
            # 无法检测到人脸时，直接对整张图片分类
            return self._classify_image(pil_image)

        # 取最大置信度的人脸
        best_face = max(faces, key=lambda f: f["confidence"])
        face_crop = face_detector.crop_face(cv_image, best_face["bbox"])
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        return self._classify_image(face_pil)

    def predict_with_annotation(self, image_input) -> tuple[dict[str, float], np.ndarray, list[dict]]:
        """预测并返回标注后的图像。

        Returns:
            (情感概率字典, 标注图像 numpy BGR, 检测到的人脸列表)
        """
        self.ensure_loaded()
        pil_image, cv_image = self._prepare_image(image_input)
        faces = face_detector.detect_faces(cv_image)

        if not faces:
            scores = self._classify_image(pil_image)
            return scores, cv_image, []

        # 对每张人脸分类
        face_results = []
        for face in faces:
            face_crop = face_detector.crop_face(cv_image, face["bbox"])
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            scores = self._classify_image(face_pil)
            face_results.append({"face": face, "scores": scores})

        # 主结果取最大人脸
        best = max(face_results, key=lambda r: r["face"]["confidence"])
        main_scores = best["scores"]

        # 标注图像
        from config import EMOTION_LABELS_CN, EMOTION_EMOJIS
        from utils.label_mapper import get_dominant_emotion
        labels = []
        for fr in face_results:
            emo, conf = get_dominant_emotion(fr["scores"])
            labels.append(f"{EMOTION_EMOJIS[emo]} {EMOTION_LABELS_CN[emo]} {conf:.0%}")

        annotated = face_detector.draw_faces(cv_image, [fr["face"] for fr in face_results], labels)
        return main_scores, annotated, faces

    def _classify_image(self, pil_image: Image.Image) -> dict[str, float]:
        result = self._model(pil_image)
        raw_scores = {}
        for item in result:
            raw_scores[item["label"]] = float(item["score"])
        return normalize_scores(raw_scores, "face")

    def _prepare_image(self, image_input) -> tuple[Image.Image, np.ndarray]:
        """将各种输入格式转为 (PIL Image, BGR numpy) 对"""
        if isinstance(image_input, str):
            cv_img = cv2.imread(image_input)
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                cv_img = image_input
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            else:
                cv_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                pil_img = Image.fromarray(image_input)
        else:
            raise ValueError(f"不支持的图像输入类型: {type(image_input)}")
        return pil_img, cv_img
