"""Cosine match of a face embedding against all enrolled buyers."""
import numpy as np

from . import db


class Matcher:
    def __init__(self):
        self._version = -1
        self._ids = np.zeros(0, dtype=np.int64)
        self._mat = np.zeros((0, 128), dtype=np.float32)

    def _refresh(self):
        if self._version != db.embeddings_version:
            self._version = db.embeddings_version
            self._ids, self._mat = db.load_embeddings()

    def match(self, emb: np.ndarray, threshold: float) -> tuple[int, float] | None:
        """Best (buyer_id, score) at or above threshold, else None.

        Matches against everyone enrolled (not just debtors) so a paid-up
        customer is never mistaken for a look-alike debtor.
        """
        self._refresh()
        if len(self._ids) == 0:
            return None
        sims = self._mat @ emb
        i = int(np.argmax(sims))
        return (int(self._ids[i]), float(sims[i])) if sims[i] >= threshold else None
