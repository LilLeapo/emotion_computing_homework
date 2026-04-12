"""人脸检测与裁剪 (基于 OpenCV DNN)"""

import cv2
import numpy as np


class FaceDetector:
    """使用 OpenCV Haar Cascade 进行人脸检测和裁剪"""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._detector = None

    def _ensure_loaded(self):
        if self._detector is None:
            self._detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """检测图像中的所有人脸。

        Args:
            image: BGR 格式的 numpy 数组 (OpenCV 格式)

        Returns:
            人脸列表，每项包含 bbox 和 confidence
        """
        self._ensure_loaded()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        detections = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        faces = []
        for (x, y, bw, bh) in detections:
            # 扩展边界框 20%
            pad_x = int(bw * 0.2)
            pad_y = int(bh * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": 0.99,  # Haar cascade 不提供置信度
            })
        return faces

    def crop_face(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        """根据边界框裁剪人脸区域"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    def draw_faces(self, image: np.ndarray, faces: list[dict],
                   labels: list[str] = None) -> np.ndarray:
        """在图像上绘制人脸框和标签。"""
        annotated = image.copy()
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if labels and i < len(labels):
                label = labels[i]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return annotated


# 全局实例
face_detector = FaceDetector()
