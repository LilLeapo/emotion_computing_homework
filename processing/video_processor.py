"""视频处理：帧提取、音频分离"""

import tempfile

import cv2
import numpy as np

from config import VIDEO_CONFIG
from processing.audio_processor import load_audio


def extract_audio_from_video(video_path: str) -> str | None:
    """从视频中提取音频并保存为临时 WAV 文件。

    使用 ffmpeg-python 进行音视频分离。
    """
    try:
        import ffmpeg
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        (
            ffmpeg
            .input(video_path)
            .output(tmp.name, acodec="pcm_s16le", ac=1, ar=VIDEO_CONFIG["sample_rate"])
            .overwrite_output()
            .run(quiet=True)
        )
        return tmp.name
    except Exception:
        return None


def extract_frames(video_path: str, segment_duration: float = None,
                   max_frames_per_segment: int = None) -> list[dict]:
    """从视频中按时间段提取关键帧。

    Returns:
        列表，每项包含:
        - "timestamp": 时间戳（秒）
        - "frame": BGR numpy 数组
        - "segment_idx": 所属时间段索引
    """
    seg_dur = segment_duration or VIDEO_CONFIG["segment_duration"]
    max_fps = max_frames_per_segment or VIDEO_CONFIG["max_frames_per_segment"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    results = []
    seg_idx = 0
    current_time = 0.0

    while current_time < duration:
        seg_end = min(current_time + seg_dur, duration)
        # 在每个段内均匀取帧
        seg_frame_count = min(max_fps, int((seg_end - current_time) * fps))
        if seg_frame_count < 1:
            seg_frame_count = 1

        for i in range(seg_frame_count):
            t = current_time + (seg_end - current_time) * i / max(seg_frame_count, 1)
            frame_no = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if ret:
                results.append({
                    "timestamp": t,
                    "frame": frame,
                    "segment_idx": seg_idx,
                })

        seg_idx += 1
        current_time = seg_end

    cap.release()
    return results


def get_video_info(video_path: str) -> dict:
    """获取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
