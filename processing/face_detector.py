"""人脸检测与裁剪 (基于 MediaPipe)"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


class FaceDetector:
    """使用 MediaPipe Face Detection 进行人脸检测和裁剪"""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._detector = None

    def _ensure_loaded(self):
        if self._detector is None:
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 全距离模型
                min_detection_confidence=self.min_confidence,
            )

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """检测图像中的所有人脸。

        Args:
            image: BGR 格式的 numpy 数组 (OpenCV 格式)

        Returns:
            人脸列表，每项包含 bbox 和 confidence
        """
        self._ensure_loaded()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                # 扩展边界框 20% 以包含更多面部上下文
                pad_x = int(bw * 0.2)
                pad_y = int(bh * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": det.score[0],
                })
        return faces

    def crop_face(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        """根据边界框裁剪人脸区域"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    def draw_faces(self, image: np.ndarray, faces: list[dict],
                   labels: list[str] = None) -> np.ndarray:
        """在图像上绘制人脸框和标签。

        Args:
            image: 原始图像
            faces: detect_faces 返回的人脸列表
            labels: 每张人脸对应的标签（如情感）

        Returns:
            标注后的图像
        """
        annotated = image.copy()
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if labels and i < len(labels):
                label = labels[i]
                # 标签背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return annotated


# 全局实例
face_detector = FaceDetector()
