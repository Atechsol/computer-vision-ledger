"""Background capture thread that always holds the latest frame."""
import os
import threading
import time

import cv2
import numpy as np


def _source():
    s = os.environ.get("CAMERA_SOURCE", "0")  # "0" = Mac webcam, or an rtsp:// URL
    return int(s) if s.isdigit() else s


class Camera:
    def __init__(self):
        self.source = _source()
        self._cap = None
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.ok = False

    def open(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap = cap if cap.isOpened() else None
        self.ok = False  # becomes True only once frames actually arrive
        return self._cap is not None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="camera").start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            if self._cap is None:
                time.sleep(2)
                self.open()
                continue
            ok, frame = self._cap.read()
            if not ok:
                self.ok = False
                self._cap.release()
                self._cap = None
                continue
            self.ok = True
            with self._lock:
                self._frame = frame
                self._seq += 1
        if self._cap is not None:
            self._cap.release()

    def latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy()), self._seq
