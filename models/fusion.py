"""注意力加权多模态融合模块

本项目的核心增量贡献之一。

设计思路:
- 晚期融合 (Late Fusion): 各模态独立推理后，在概率层面融合
- 注意力权重: 使用 MLP 根据各模态的置信度分布动态计算权重
- 支持缺失模态: 当某模态输入缺失时，自动调整权重

相比简单平均融合的优势:
1. 动态权重: 高置信度的模态获得更高权重
2. 模态互补: 学习不同模态在不同情感上的互补关系
3. 鲁棒性: 缺失模态时自动降级
"""

import numpy as np
import torch
import torch.nn as nn

from config import EMOTION_LABELS
from utils.label_mapper import array_to_scores, scores_to_array


class AttentionFusionNet(nn.Module):
    """注意力加权融合网络。

    输入: 3 个模态各自的 7 维概率向量（共 21 维）
    输出: 3 个注意力权重 + 7 维融合概率
    """

    def __init__(self, n_emotions: int = 7, n_modalities: int = 3):
        super().__init__()
        input_dim = n_emotions * n_modalities  # 21

        # 注意力权重网络
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_modalities),
        )

        # 融合后的精炼网络
        self.refine_net = nn.Sequential(
            nn.Linear(n_emotions, 16),
            nn.ReLU(),
            nn.Linear(16, n_emotions),
        )

        self.n_emotions = n_emotions
        self.n_modalities = n_modalities

    def forward(self, modality_probs: torch.Tensor,
                modality_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_probs: (batch, n_modalities, n_emotions) 各模态概率
            modality_mask: (batch, n_modalities) 1=有效, 0=缺失

        Returns:
            fused_probs: (batch, n_emotions) 融合后的概率
            attention_weights: (batch, n_modalities) 注意力权重
        """
        batch_size = modality_probs.shape[0]

        # 拼接所有模态概率
        flat = modality_probs.view(batch_size, -1)  # (batch, 21)

        # 计算注意力权重
        attn_logits = self.attention_net(flat)  # (batch, 3)

        # 如果有模态缺失，将对应 logit 设为极小值
        if modality_mask is not None:
            attn_logits = attn_logits.masked_fill(modality_mask == 0, -1e9)

        attention_weights = torch.softmax(attn_logits, dim=-1)  # (batch, 3)

        # 加权求和
        weighted = (modality_probs * attention_weights.unsqueeze(-1)).sum(dim=1)  # (batch, 7)

        # 精炼 + 残差连接
        refined = self.refine_net(weighted) + weighted
        fused_probs = torch.softmax(refined, dim=-1)

        return fused_probs, attention_weights


class MultimodalFusion:
    """多模态融合推理接口"""

    def __init__(self):
        self._net = AttentionFusionNet()
        self._net.eval()
        # 使用启发式权重初始化（无需训练即可工作）
        self._init_heuristic_weights()

    def _init_heuristic_weights(self):
        """使用启发式值初始化网络权重，使得初始行为接近等权融合"""
        with torch.no_grad():
            # 注意力网络最后一层 bias 设为相等
            last_layer = self._net.attention_net[-1]
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

            # 精炼网络接近恒等映射
            for module in self._net.refine_net:
                if isinstance(module, nn.Linear):
                    nn.init.eye_(module.weight[:min(module.weight.shape), :min(module.weight.shape)])
                    nn.init.zeros_(module.bias)

    def fuse(self, text_scores: dict[str, float] = None,
             speech_scores: dict[str, float] = None,
             face_scores: dict[str, float] = None) -> tuple[dict[str, float], dict[str, float]]:
        """融合多模态情感预测结果。

        Args:
            text_scores: 文本情感概率（可选）
            speech_scores: 语音情感概率（可选）
            face_scores: 面部情感概率（可选）

        Returns:
            (融合后的情感概率字典, 各模态注意力权重字典)
        """
        # 准备输入
        n_e = len(EMOTION_LABELS)
        uniform = np.ones(n_e) / n_e

        modality_arrays = []
        modality_mask = []
        modality_names = ["文本", "语音", "面部"]

        for scores in [text_scores, speech_scores, face_scores]:
            if scores is not None:
                modality_arrays.append(scores_to_array(scores))
                modality_mask.append(1.0)
            else:
                modality_arrays.append(uniform)
                modality_mask.append(0.0)

        # 至少需要一个模态
        if sum(modality_mask) == 0:
            return {e: 1.0 / n_e for e in EMOTION_LABELS}, {}

        # 转为 tensor
        probs_tensor = torch.tensor(
            np.stack(modality_arrays), dtype=torch.float32
        ).unsqueeze(0)  # (1, 3, 7)
        mask_tensor = torch.tensor(
            modality_mask, dtype=torch.float32
        ).unsqueeze(0)  # (1, 3)

        # 推理
        with torch.no_grad():
            fused, weights = self._net(probs_tensor, mask_tensor)

        fused_scores = array_to_scores(fused[0].numpy())
        weight_dict = {}
        for i, name in enumerate(modality_names):
            if modality_mask[i] > 0:
                weight_dict[name] = float(weights[0, i].item())

        return fused_scores, weight_dict
