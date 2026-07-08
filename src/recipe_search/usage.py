"""Optional usage recording for the public demo: who, what, and how many.

One SQLite row per request — a salted IP hash (never the raw address), the
query text, and how it turned out — plus the aggregate readers behind
GET /stats. Recording is additive by contract: it activates only when a
database path is configured, a recorder that cannot open its file degrades
to a no-op, and record() never raises.

No FastAPI imports; main.py owns the wiring.
"""

import asyncio
import hashlib
import logging
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    ip_hash TEXT,
    user_agent TEXT,
    referer TEXT,
    query TEXT,
    outcome TEXT,
    dish TEXT,
    source TEXT,
    duration_ms INTEGER
)
"""

_COLUMNS = (
    "ts",
    "endpoint",
    "ip_hash",
    "user_agent",
    "referer",
    "query",
    "outcome",
    "dish",
    "source",
    "duration_ms",
)


class UsageRecorder:
    """Append-only request log in SQLite; inert when the file can't be opened."""

    def __init__(self, path: str, *, salt: str | None = None) -> None:
        # Without a configured salt, a random per-process one still allows
        # unique-visitor counting within a run — never a raw-IP fallback.
        self._salt = salt or secrets.token_hex(16)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)
            conn.commit()
            self._conn = conn
        except Exception as exc:
            logger.warning("Usage recording disabled: cannot open %r (%s)", path, exc)

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def hash_ip(self, ip: str) -> str:
        """Salted, truncated hash: distinct-visitor counts without raw IPs."""
        return hashlib.sha256((self._salt + ip).encode()).hexdigest()[:16]

    async def record(
        self,
        *,
        endpoint: str,
        ip_hash: str | None = None,
        user_agent: str | None = None,
        referer: str | None = None,
        query: str | None = None,
        outcome: str | None = None,
        dish: str | None = None,
        source: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Write one usage row. Never raises: failures are logged and dropped."""
        if self._conn is None:
            return
        row = (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            endpoint,
            ip_hash,
            user_agent,
            referer,
            query,
            outcome,
            dish,
            source,
            duration_ms,
        )
        try:
            # A volume-backed fsync can take milliseconds; keep it off the loop.
            await asyncio.to_thread(self._insert, row)
        except Exception as exc:
            logger.warning("Failed to record usage event: %s", exc)

    def _insert(self, row: tuple) -> None:
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO usage_events ({', '.join(_COLUMNS)})"
                f" VALUES ({placeholders})",
                row,
            )
            self._conn.commit()

    def stats(self, days: int = 7) -> dict:
        """Aggregates over the trailing window; {} when recording is disabled.

        "Asks" are the POST endpoints (real usage); "visits" are home-page
        loads, an upper bound that includes bots and link previews.
        """
        if self._conn is None:
            return {}
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        with self._lock:
            asks_total, ask_visitors = self._conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT ip_hash) FROM usage_events"
                " WHERE ts >= ? AND endpoint != 'home'",
                (since,),
            ).fetchone()
            visits_total, visit_visitors = self._conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT ip_hash) FROM usage_events"
                " WHERE ts >= ? AND endpoint = 'home'",
                (since,),
            ).fetchone()
            by_day = [
                {"day": day, "requests": requests, "unique_visitors": visitors}
                for day, requests, visitors in self._conn.execute(
                    "SELECT substr(ts, 1, 10) AS day, COUNT(*),"
                    " COUNT(DISTINCT ip_hash) FROM usage_events"
                    " WHERE ts >= ? AND endpoint != 'home'"
                    " GROUP BY day ORDER BY day",
                    (since,),
                )
            ]
            outcomes = dict(
                self._conn.execute(
                    "SELECT outcome, COUNT(*) FROM usage_events"
                    " WHERE ts >= ? AND endpoint != 'home' GROUP BY outcome",
                    (since,),
                )
            )
            top_queries = [
                {"query": query, "count": count}
                for query, count in self._conn.execute(
                    "SELECT query, COUNT(*) AS n FROM usage_events"
                    " WHERE ts >= ? AND query IS NOT NULL"
                    " GROUP BY query ORDER BY n DESC LIMIT 10",
                    (since,),
                )
            ]
        return {
            "window_days": days,
            "asks": {
                "total": asks_total,
                "unique_visitors": ask_visitors,
                "by_day": by_day,
                "outcomes": outcomes,
                "top_queries": top_queries,
            },
            "visits": {"total": visits_total, "unique_visitors": visit_visitors},
        }

    def recent(self, limit: int = 50) -> list[dict]:
        """Newest rows first; [] when recording is disabled."""
        if self._conn is None:
            return []
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM usage_events ORDER BY id DESC LIMIT ?", (limit,)
            )
            names = [description[0] for description in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.close()
        except Exception:  # pragma: no cover — nothing useful to do at shutdown
            pass
        self._conn = None
