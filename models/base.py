"""情感识别器抽象基类"""

from abc import ABC, abstractmethod


class BaseEmotionRecognizer(ABC):
    """所有情感识别模型的抽象基类。

    子类需实现 predict() 方法，返回统一格式的情感概率分布。
    """

    def __init__(self, modality: str):
        self.modality = modality
        self._model = None
        self._loaded = False

    @abstractmethod
    def load_model(self):
        """加载预训练模型到内存"""

    @abstractmethod
    def predict(self, input_data) -> dict[str, float]:
        """对输入数据进行情感预测。

        Returns:
            dict: 7种情感的概率分布，如 {"happy": 0.8, "sad": 0.1, ...}
                  所有值之和应为1.0
        """

    def ensure_loaded(self):
        """确保模型已加载"""
        if not self._loaded:
            self.load_model()
            self._loaded = True
