"""In-memory request limits for the public demo.

Single-process and deliberately simple: counters reset when the server
restarts, which is acceptable for a demo. The global budget uses UTC
calendar days ("the stove relights tomorrow"); per-IP limits use rolling
windows. Rejected requests consume nothing.
"""

import time
from typing import Callable

_HOUR = 3600.0
_DAY = 86400.0


class RateLimited(Exception):
    """A request was refused by the demo limits."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RateLimiter:
    def __init__(
        self,
        *,
        per_hour: int,
        per_day: int,
        daily_budget: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._per_hour = per_hour
        self._per_day = per_day
        self._daily_budget = daily_budget
        self._clock = clock
        self._hits_by_ip: dict[str, list[float]] = {}
        self._budget_day = -1
        self._budget_used = 0

    def check(self, ip: str) -> str | None:
        """Admit and record the request, or return a refusal code.

        Codes: "budget" (global daily budget spent), "ip_day", "ip_hour".
        """
        now = self._clock()
        today = int(now // _DAY)  # epoch days are UTC days
        if today != self._budget_day:
            self._budget_day = today
            self._budget_used = 0
        if self._budget_used >= self._daily_budget:
            return "budget"

        hits = self._hits_by_ip.setdefault(ip, [])
        hits[:] = [stamp for stamp in hits if now - stamp < _DAY]
        if len(hits) >= self._per_day:
            return "ip_day"
        if sum(1 for stamp in hits if now - stamp < _HOUR) >= self._per_hour:
            return "ip_hour"

        hits.append(now)
        self._budget_used += 1
        return None
