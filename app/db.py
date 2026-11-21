"""SQLite schema + queries."""
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
PHOTOS = DATA / "photos"
DB_PATH = DATA / "app.db"

DEFAULT_SETTINGS = {
    "min_due": 1000.0,      # rupees; watchlist = outstanding > min_due
    "threshold": 0.363,     # cosine similarity (OpenCV SFace default)
    "cooldown": 60.0,       # seconds between alerts for the same buyer
    "confirm_n": 3,         # must match in N of the last M frames
    "confirm_m": 5,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS buyers(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  phone TEXT NOT NULL DEFAULT '',
  photo_path TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  buyer_id INTEGER NOT NULL REFERENCES buyers(id) ON DELETE CASCADE,
  vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  buyer_id INTEGER NOT NULL REFERENCES buyers(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK(kind IN ('credit','payment')),
  amount REAL NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  buyer_id INTEGER NOT NULL REFERENCES buyers(id) ON DELETE CASCADE,
  score REAL NOT NULL,
  outstanding REAL NOT NULL DEFAULT 0,
  snapshot_path TEXT,
  ts REAL NOT NULL,
  acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS settings(
  key TEXT PRIMARY KEY,
  value REAL NOT NULL
);
"""

OUTSTANDING_SQL = (
    "COALESCE((SELECT SUM(CASE kind WHEN 'credit' THEN amount ELSE -amount END) "
    "FROM ledger WHERE buyer_id = b.id), 0)"
)

# Bumped whenever embeddings change so the matcher knows to reload.
embeddings_version = 0
_settings_cache: dict | None = None
_lock = threading.Lock()


@contextmanager
def db():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        with c:
            yield c
    finally:
        c.close()


def init():
    PHOTOS.mkdir(parents=True, exist_ok=True)
    (PHOTOS / "buyers").mkdir(exist_ok=True)
    (PHOTOS / "alerts").mkdir(exist_ok=True)
    with db() as c:
        c.executescript(SCHEMA)


# ---- settings ----------------------------------------------------------
def get_settings() -> dict:
    global _settings_cache
    with _lock:
        if _settings_cache is None:
            s = dict(DEFAULT_SETTINGS)
            with db() as c:
                for r in c.execute("SELECT key, value FROM settings"):
                    if r["key"] in s:
                        s[r["key"]] = r["value"]
            _settings_cache = s
        return dict(_settings_cache)


def set_settings(values: dict) -> dict:
    global _settings_cache
    with db() as c:
        for k, v in values.items():
            if k in DEFAULT_SETTINGS:
                c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)", (k, float(v)))
    with _lock:
        _settings_cache = None
    return get_settings()


# ---- buyers ------------------------------------------------------------
def create_buyer(name: str, phone: str, opening_credit: float) -> int:
    now = time.time()
    with db() as c:
        cur = c.execute(
            "INSERT INTO buyers(name, phone, created_at) VALUES(?, ?, ?)", (name, phone, now)
        )
        bid = cur.lastrowid
        if opening_credit > 0:
            c.execute(
                "INSERT INTO ledger(buyer_id, kind, amount, note, ts) VALUES(?, 'credit', ?, 'Opening credit', ?)",
                (bid, opening_credit, now),
            )
    return bid


def _buyer_row(r) -> dict:
    d = dict(r)
    d["outstanding"] = round(d["outstanding"], 2)
    return d


def list_buyers() -> list[dict]:
    with db() as c:
        rows = c.execute(
            f"SELECT b.id, b.name, b.phone, b.photo_path, b.created_at, "
            f"{OUTSTANDING_SQL} AS outstanding, "
            f"(SELECT COUNT(*) FROM embeddings e WHERE e.buyer_id = b.id) AS n_embeddings "
            f"FROM buyers b ORDER BY b.id DESC"
        ).fetchall()
    return [_buyer_row(r) for r in rows]


def get_buyer(buyer_id: int) -> dict | None:
    with db() as c:
        r = c.execute(
            f"SELECT b.id, b.name, b.phone, b.photo_path, b.created_at, "
            f"{OUTSTANDING_SQL} AS outstanding, "
            f"(SELECT COUNT(*) FROM embeddings e WHERE e.buyer_id = b.id) AS n_embeddings "
            f"FROM buyers b WHERE b.id = ?",
            (buyer_id,),
        ).fetchone()
    return _buyer_row(r) if r else None


def delete_buyer(buyer_id: int):
    global embeddings_version
    with db() as c:
        c.execute("DELETE FROM buyers WHERE id = ?", (buyer_id,))
    embeddings_version += 1


def set_photo(buyer_id: int, path: str):
    with db() as c:
        c.execute("UPDATE buyers SET photo_path = ? WHERE id = ?", (path, buyer_id))


def add_ledger(buyer_id: int, kind: str, amount: float, note: str):
    with db() as c:
        c.execute(
            "INSERT INTO ledger(buyer_id, kind, amount, note, ts) VALUES(?, ?, ?, ?, ?)",
            (buyer_id, kind, amount, note, time.time()),
        )


def outstanding_map() -> dict[int, float]:
    with db() as c:
        rows = c.execute(f"SELECT b.id, {OUTSTANDING_SQL} AS o FROM buyers b").fetchall()
    return {r["id"]: r["o"] for r in rows}


def names_map() -> dict[int, str]:
    with db() as c:
        return {r["id"]: r["name"] for r in c.execute("SELECT id, name FROM buyers")}


# ---- embeddings --------------------------------------------------------
def add_embedding(buyer_id: int, vec: np.ndarray):
    global embeddings_version
    with db() as c:
        c.execute(
            "INSERT INTO embeddings(buyer_id, vector) VALUES(?, ?)",
            (buyer_id, vec.astype(np.float32).tobytes()),
        )
    embeddings_version += 1


def clear_embeddings(buyer_id: int):
    global embeddings_version
    with db() as c:
        c.execute("DELETE FROM embeddings WHERE buyer_id = ?", (buyer_id,))
    embeddings_version += 1


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    with db() as c:
        rows = c.execute("SELECT buyer_id, vector FROM embeddings").fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 128), dtype=np.float32)
    ids = np.array([r["buyer_id"] for r in rows], dtype=np.int64)
    mat = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    return ids, mat


# ---- alerts ------------------------------------------------------------
def insert_alert(buyer_id: int, score: float, outstanding: float, snapshot_path: str | None) -> int:
    with db() as c:
        cur = c.execute(
            "INSERT INTO alerts(buyer_id, score, outstanding, snapshot_path, ts) VALUES(?, ?, ?, ?, ?)",
            (buyer_id, score, outstanding, snapshot_path, time.time()),
        )
        return cur.lastrowid


_ALERT_SELECT = (
    "SELECT a.id, a.buyer_id, a.score, a.outstanding, a.snapshot_path, a.ts, a.acknowledged, "
    "b.name, b.phone, b.photo_path FROM alerts a JOIN buyers b ON b.id = a.buyer_id"
)


def get_alert(alert_id: int) -> dict | None:
    with db() as c:
        r = c.execute(_ALERT_SELECT + " WHERE a.id = ?", (alert_id,)).fetchone()
    return dict(r) if r else None


def list_alerts(limit: int = 50, unacked_only: bool = False) -> list[dict]:
    where = " WHERE a.acknowledged = 0" if unacked_only else ""
    with db() as c:
        rows = c.execute(_ALERT_SELECT + where + " ORDER BY a.id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def ack_alert(alert_id: int) -> bool:
    with db() as c:
        cur = c.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
        return cur.rowcount > 0
