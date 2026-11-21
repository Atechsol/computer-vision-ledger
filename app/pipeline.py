"""Live loop: grab frame -> detect -> embed -> match -> alert rules -> notify."""
import threading
import time
from typing import Callable

import cv2
import numpy as np

from . import db
from .alerts import AlertEngine
from .camera import Camera
from .face import FaceEngine
from .matcher import Matcher

DETECT_WIDTH = 640     # detect on a downscaled frame, embed on the full-res one
TARGET_FPS = 8
MIN_FACE_PX = 50       # ignore faces smaller than this (full-res px)

RED, GREEN, GREY = (60, 60, 230), (80, 190, 80), (170, 170, 170)  # BGR


def placeholder(text: str) -> np.ndarray:
    img = np.full((480, 640, 3), 30, np.uint8)
    cv2.putText(img, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    return img


class Pipeline(threading.Thread):
    def __init__(self, camera: Camera, engine: FaceEngine, on_alert: Callable[[dict], None]):
        super().__init__(daemon=True, name="pipeline")
        self.camera, self.engine, self.on_alert = camera, engine, on_alert
        self.matcher, self.alerts = Matcher(), AlertEngine()
        self._boxes: list[tuple[tuple[int, int, int, int], str, tuple]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.fps = 0.0

    def stop(self):
        self._stop.set()

    def annotated(self) -> tuple[np.ndarray, int]:
        """Latest camera frame with the most recent face boxes drawn on it."""
        frame, seq = self.camera.latest()
        if frame is None:
            msg = "Waiting for camera..." if self.camera.ok else "Camera unavailable - check permission"
            return placeholder(msg), seq
        with self._lock:
            boxes = list(self._boxes)
        return draw(frame, boxes), seq

    def run(self):
        last_seq, last_t = -1, 0.0
        while not self._stop.is_set():
            frame, seq = self.camera.latest()
            if frame is None or seq == last_seq or time.time() - last_t < 1 / TARGET_FPS:
                time.sleep(0.01)
                continue
            last_seq, last_t = seq, time.time()
            try:
                self._process(frame)
            except Exception as e:  # keep the loop alive no matter what
                print("pipeline error:", repr(e))
            self.fps = 1 / max(time.time() - last_t, 1e-3)

    def _process(self, frame: np.ndarray):
        settings = db.get_settings()
        h, w = frame.shape[:2]
        scale = min(1.0, DETECT_WIDTH / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        faces = self.engine.detect(small)
        if scale < 1:
            faces = faces.copy()
            faces[:, :14] /= scale

        outstanding = db.outstanding_map()
        names = db.names_map()
        matched: dict[int, float] = {}
        boxes = []
        for f in faces:
            x, y, fw, fh = (int(v) for v in f[:4])
            if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
                continue
            m = self.matcher.match(self.engine.embed(frame, f), settings["threshold"])
            if m is None:
                boxes.append(((x, y, fw, fh), "unknown", GREY))
                continue
            bid, score = m
            matched[bid] = max(score, matched.get(bid, 0))
            owes = outstanding.get(bid, 0) > settings["min_due"]
            boxes.append(((x, y, fw, fh), f"{names.get(bid, '?')} {score:.2f}", RED if owes else GREEN))
        with self._lock:
            self._boxes = boxes

        for bid, score in self.alerts.update(matched, outstanding, settings):
            self._fire(frame, boxes, bid, score, outstanding[bid])

    def _fire(self, frame, boxes, bid, score, owed):
        rel = f"alerts/{bid}_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(db.PHOTOS / rel), draw(frame, boxes))
        alert_id = db.insert_alert(bid, score, owed, rel)
        self.on_alert(db.get_alert(alert_id))


def draw(frame: np.ndarray, boxes) -> np.ndarray:
    out = frame.copy()
    for (x, y, w, h), label, color in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
