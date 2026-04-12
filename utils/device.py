"""GPU/CPU 设备管理"""

import torch

from config import DEVICE


def get_device() -> str:
    return DEVICE


def get_device_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        return f"GPU: {name} ({mem:.1f} GB)"
    return "CPU only"
