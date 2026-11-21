"""Seed demo data.

    python scripts/seed_demo.py [photo_of_demo_merchant.jpg ...]

Creates "Demo Merchant" (Rs 25,000 credit) and "Paid-Up Customer" (credit fully paid).
Photos, if given, are enrolled for Demo Merchant. Otherwise enroll in the dashboard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.enroll import enroll_photos  # noqa: E402
from app.face import FaceEngine  # noqa: E402


def main(photos: list[str]):
    db.init()
    merchant = db.create_buyer("Demo Merchant", "", 25000)
    paid = db.create_buyer("Paid-Up Customer", "", 5000)
    db.add_ledger(paid, "payment", 5000, "Settled")
    print(f"created Demo Merchant (#{merchant}) and Paid-Up Customer (#{paid})")
    if photos:
        blobs = [(p, Path(p).read_bytes()) for p in photos]
        print(enroll_photos(FaceEngine(), merchant, blobs))


if __name__ == "__main__":
    main(sys.argv[1:])
