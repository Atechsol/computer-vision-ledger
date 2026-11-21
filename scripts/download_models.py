"""Download the YuNet (detector) and SFace (recognizer) ONNX models from opencv_zoo."""
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
MIN_BYTES = 100_000  # guards against saving an LFS pointer / HTML error page


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > MIN_BYTES:
            print(f"ok      {name} (already present)")
            continue
        print(f"fetch   {name}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < MIN_BYTES:
            tmp.unlink()
            print(f"FAILED  {name}: download too small", file=sys.stderr)
            return 1
        tmp.rename(dest)
        print(f"saved   {dest} ({dest.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
