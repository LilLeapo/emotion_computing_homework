"""Serve the provided frontend directly and connect it to local models."""

import base64
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import GRADIO_CONFIG
from models.face_emotion import FaceEmotionRecognizer
from models.fusion import MultimodalFusion
from models.speech_emotion import SpeechEmotionRecognizer
from models.text_emotion import TextEmotionRecognizer
from processing.text_processor import is_valid_text
from utils.label_mapper import get_dominant_emotion


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / ".cache" / "modelscope"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

app = FastAPI(title="emotion.diag")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_text_model: TextEmotionRecognizer | None = None
_speech_model: SpeechEmotionRecognizer | None = None
_face_model: FaceEmotionRecognizer | None = None
_fusion: MultimodalFusion | None = None

BACKEND_TO_FRONTEND = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "neutral": "neutral",
}


def _text() -> TextEmotionRecognizer:
    global _text_model
    if _text_model is None:
        _text_model = TextEmotionRecognizer()
    return _text_model


def _speech() -> SpeechEmotionRecognizer:
    global _speech_model
    if _speech_model is None:
        _speech_model = SpeechEmotionRecognizer()
    return _speech_model


def _face() -> FaceEmotionRecognizer:
    global _face_model
    if _face_model is None:
        _face_model = FaceEmotionRecognizer()
    return _face_model


def _fusion_model() -> MultimodalFusion:
    global _fusion
    if _fusion is None:
        _fusion = MultimodalFusion()
    return _fusion


def _frontend_scores(scores: dict[str, float]) -> dict[str, float]:
    mapped = {v: 0.0 for v in BACKEND_TO_FRONTEND.values()}
    for backend_key, value in scores.items():
        mapped[BACKEND_TO_FRONTEND.get(backend_key, backend_key)] = float(value)
    total = sum(mapped.values())
    if total > 0:
        mapped = {k: round(v / total, 4) for k, v in mapped.items()}
    return mapped


def _backend_scores(scores: dict[str, float]) -> dict[str, float]:
    reverse = {v: k for k, v in BACKEND_TO_FRONTEND.items()}
    return {reverse.get(k, k): float(v) for k, v in scores.items()}


def _dominant_payload(scores: dict[str, float]) -> dict:
    backend_scores = _backend_scores(scores)
    dominant, confidence = get_dominant_emotion(backend_scores)
    return {
        "dominant": BACKEND_TO_FRONTEND.get(dominant, dominant),
        "confidence": round(float(confidence), 4),
    }


def _response(scores: dict[str, float], started_at: float, **extra) -> JSONResponse:
    payload = {
        "dist": _frontend_scores(scores),
        "latency": int((time.perf_counter() - started_at) * 1000),
    }
    payload.update(_dominant_payload(payload["dist"]))
    payload.update(extra)
    return JSONResponse(payload)


async def _save_upload(upload: UploadFile, suffix: str | None = None) -> str:
    ext = suffix or Path(upload.filename or "").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await upload.read())
        return tmp.name


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "demo.html")


@app.get("/demo")
def demo():
    return FileResponse(FRONTEND_DIR / "demo.html")


@app.get("/showcase")
def showcase():
    return FileResponse(FRONTEND_DIR / "app.html")


@app.get("/landing")
def landing():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/text")
async def analyze_text(text: str = Form(...)):
    started = time.perf_counter()
    if not text.strip():
        return JSONResponse({"error": "请输入文本"}, status_code=400)

    scores, attention = _text().predict_with_attention(text)
    attn = [{"ch": ch, "w": float(weight)} for ch, weight in attention]
    return _response(scores, started, attention=attn, modelName="StructBERT-base-zh", sourceLabel="text")


@app.post("/api/speech")
async def analyze_speech(file: UploadFile = File(...)):
    started = time.perf_counter()
    path = await _save_upload(file)
    try:
        scores = _speech().predict(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return _response(scores, started, modelName="emotion2vec_plus_large", sourceLabel="speech")


@app.post("/api/face")
async def analyze_face(file: UploadFile = File(...)):
    started = time.perf_counter()
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    cv_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if cv_image is None:
        return JSONResponse({"error": "无法读取图片"}, status_code=400)

    scores, annotated, faces = _face().predict_with_annotation(cv_image)
    ok, encoded = cv2.imencode(".png", annotated)
    image_data = None
    if ok:
        image_data = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return _response(
        scores,
        started,
        annotatedImage=image_data,
        faces=len(faces),
        modelName="vit-face-expression",
        sourceLabel="face",
    )


@app.post("/api/live-face")
async def analyze_live_face(file: UploadFile = File(...)):
    started = time.perf_counter()
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    cv_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if cv_image is None:
        return JSONResponse({"error": "无法读取摄像头帧"}, status_code=400)

    scores, _, faces = _face().predict_with_annotation(cv_image)
    return _response(
        scores,
        started,
        faces=len(faces),
        modelName="vit-face-expression",
        sourceLabel="live-camera",
    )


@app.post("/api/multimodal")
async def analyze_multimodal(
    text: str = Form(""),
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
):
    started = time.perf_counter()
    text_scores = None
    speech_scores = None
    face_scores = None

    if text and is_valid_text(text):
        text_scores = _text().predict(text)

    if audio is not None:
        audio_path = await _save_upload(audio)
        try:
            speech_scores = _speech().predict(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    if image is not None:
        raw = await image.read()
        cv_image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if cv_image is not None:
            face_scores = _face().predict(cv_image)

    if text_scores is None and speech_scores is None and face_scores is None:
        return JSONResponse({"error": "请至少提供一种模态输入"}, status_code=400)

    fused_scores, weights = _fusion_model().fuse(text_scores, speech_scores, face_scores)
    active_labels = []
    active_weights = []
    if text_scores is not None:
        active_labels.append("text")
        active_weights.append(float(weights.get("text", 0.0)))
    if speech_scores is not None:
        active_labels.append("speech")
        active_weights.append(float(weights.get("speech", 0.0)))
    if face_scores is not None:
        active_labels.append("vision")
        active_weights.append(float(weights.get("face", 0.0)))

    return _response(
        fused_scores,
        started,
        W=active_weights,
        labels=active_labels,
        modalityDists={
            "text": _frontend_scores(text_scores) if text_scores else None,
            "speech": _frontend_scores(speech_scores) if speech_scores else None,
            "vision": _frontend_scores(face_scores) if face_scores else None,
        },
        modelName="attention late fusion",
        sourceLabel="multimodal",
    )


@app.get("/{asset_path:path}")
def frontend_asset(asset_path: str):
    file_path = FRONTEND_DIR / asset_path
    if file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(FRONTEND_DIR / "demo.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=GRADIO_CONFIG["server_port"])
