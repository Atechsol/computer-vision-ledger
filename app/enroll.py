"""Enrollment from uploaded photos."""
import time

import cv2
import numpy as np

from . import db
from .face import FaceEngine

MAX_DETECT_SIDE = 1600
MAX_STORED_SIDE = 800


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a readable image")
    return img


def _fit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def embed_photo(engine: FaceEngine, data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return (embedding, image) for the single face in the photo, or raise ValueError."""
    img = _fit(_decode(data), MAX_DETECT_SIDE)
    faces = engine.detect(img)
    if len(faces) == 0:
        raise ValueError("no face found")
    areas = faces[:, 2] * faces[:, 3]
    order = np.argsort(-areas)
    if len(faces) > 1 and areas[order[1]] > 0.4 * areas[order[0]]:
        raise ValueError("more than one face - use a photo of one person")
    face = faces[order[0]]
    if face[2] < 80 or face[3] < 80:
        raise ValueError("face too small - use a closer photo")
    return engine.embed(img, face), img


def enroll_photos(engine: FaceEngine, buyer_id: int, blobs: list[tuple[str, bytes]]) -> dict:
    """Embed each photo and store it. blobs = [(filename, bytes)]."""
    buyer = db.get_buyer(buyer_id)
    enrolled, errors = 0, []
    for filename, data in blobs:
        try:
            emb, img = embed_photo(engine, data)
        except ValueError as e:
            errors.append(f"{filename}: {e}")
            continue
        db.add_embedding(buyer_id, emb)
        enrolled += 1
        if not buyer["photo_path"]:
            rel = f"buyers/{buyer_id}_{int(time.time())}.jpg"
            cv2.imwrite(str(db.PHOTOS / rel), _fit(img, MAX_STORED_SIDE))
            db.set_photo(buyer_id, rel)
            buyer["photo_path"] = rel
    return {"enrolled": enrolled, "errors": errors}
