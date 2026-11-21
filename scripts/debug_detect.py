"""Debug: run YuNet on one image at several thresholds and sizes.

    python scripts/debug_detect.py photo.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from app.face import DET_MODEL  # noqa: E402

img = cv2.imread(sys.argv[1])
print("shape", img.shape)
for thr in (0.8, 0.6, 0.4):
    det = cv2.FaceDetectorYN.create(str(DET_MODEL), "", (320, 320), thr, 0.3, 5000)
    for w in (1500, 1000, 640, 480, 320):
        s = w / img.shape[1]
        im = cv2.resize(img, (w, int(img.shape[0] * s)))
        det.setInputSize((im.shape[1], im.shape[0]))
        _, f = det.detect(im)
        found = None if f is None else [(int(r[2]), int(r[3]), round(float(r[14]), 2)) for r in f]
        print(f"thr={thr} width={w}: {found}")
