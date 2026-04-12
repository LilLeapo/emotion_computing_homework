"""音频预处理：加载、重采样、格式转换"""

import tempfile

import librosa
import numpy as np
import soundfile as sf

from config import VIDEO_CONFIG


def load_audio(audio_path: str, sr: int = None) -> tuple[np.ndarray, int]:
    """加载音频文件并重采样。

    Args:
        audio_path: 音频文件路径
        sr: 目标采样率，None 则使用配置默认值

    Returns:
        (音频数据 ndarray, 采样率)
    """
    target_sr = sr or VIDEO_CONFIG["sample_rate"]
    audio, orig_sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, target_sr


def save_temp_wav(audio: np.ndarray, sr: int) -> str:
    """将音频数组保存为临时 WAV 文件"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


def get_audio_duration(audio_path: str) -> float:
    """获取音频时长（秒）"""
    duration = librosa.get_duration(path=audio_path)
    return duration


def split_audio_segments(audio: np.ndarray, sr: int,
                         segment_duration: float = None) -> list[np.ndarray]:
    """将长音频切分为固定时长的片段。

    Args:
        audio: 音频数据
        sr: 采样率
        segment_duration: 每段时长（秒），默认使用配置值

    Returns:
        音频片段列表
    """
    seg_dur = segment_duration or VIDEO_CONFIG["segment_duration"]
    seg_samples = int(seg_dur * sr)
    segments = []
    for start in range(0, len(audio), seg_samples):
        seg = audio[start:start + seg_samples]
        if len(seg) >= sr * 0.5:  # 至少 0.5 秒
            segments.append(seg)
    return segments
