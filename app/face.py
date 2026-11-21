"""Face engine: YuNet detector + SFace embedder behind a tiny interface.

To swap engines later, keep the same two methods:
    detect(frame) -> list of face rows (x, y, w, h, 10 landmark coords, score)
    embed(frame, face) -> L2-normalised float32 vector
"""
import threading
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
DET_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
REC_MODEL = MODELS / "face_recognition_sface_2021dec.onnx"


class FaceEngine:
    def __init__(self, score_threshold: float = 0.8):
        for p in (DET_MODEL, REC_MODEL):
            if not p.exists():
                raise FileNotFoundError(f"{p} missing - run: python scripts/download_models.py")
        self._det = cv2.FaceDetectorYN.create(str(DET_MODEL), "", (320, 320), score_threshold, 0.3, 5000)
        self._rec = cv2.FaceRecognizerSF.create(str(REC_MODEL), "")
        # OpenCV DNN objects are not thread-safe; the live loop and uploads share them.
        self._lock = threading.Lock()

    def detect(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        with self._lock:
            self._det.setInputSize((w, h))
            _, faces = self._det.detect(frame)
        return np.zeros((0, 15), dtype=np.float32) if faces is None else faces

    def embed(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        with self._lock:
            aligned = self._rec.alignCrop(frame, face)
            feat = self._rec.feature(aligned)
        v = feat.flatten().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)
