"""Alert rules: N-of-M frame confirmation + per-buyer cooldown."""
import time
from collections import deque


class AlertEngine:
    def __init__(self):
        self._history: deque[set[int]] = deque(maxlen=5)
        self._last_alert: dict[int, float] = {}

    def update(self, matched: dict[int, float], outstanding: dict[int, float], settings: dict) -> list[tuple[int, float]]:
        """matched = {buyer_id: score} for this frame. Returns [(buyer_id, score)] to alert on."""
        m, n = int(settings["confirm_m"]), int(settings["confirm_n"])
        if self._history.maxlen != m:
            self._history = deque(self._history, maxlen=m)
        self._history.append(set(matched))

        now = time.time()
        fire = []
        for bid, score in matched.items():
            # Dues are read fresh every frame, so a payment stops alerts immediately.
            if outstanding.get(bid, 0) <= settings["min_due"]:
                continue
            if sum(bid in h for h in self._history) < n:
                continue
            if now - self._last_alert.get(bid, 0) < settings["cooldown"]:
                continue
            self._last_alert[bid] = now
            fire.append((bid, score))
        return fire
