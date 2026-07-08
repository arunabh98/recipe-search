"""Unit tests for the optional SQLite usage recorder."""

import sqlite3

from recipe_search.usage import UsageRecorder


def read_rows(path: str, columns: str) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            f"SELECT {columns} FROM usage_events ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


async def test_record_writes_a_row(tmp_path):
    db = str(tmp_path / "usage.db")
    recorder = UsageRecorder(db, salt="test-salt")

    await recorder.record(
        endpoint="recipes/recommend",
        ip_hash=recorder.hash_ip("203.0.113.9"),
        query="kimchi and rice",
        outcome="recommended",
        dish="Kimchi Fried Rice",
        source="justonecookbook.com",
        duration_ms=41000,
    )
    recorder.close()

    rows = read_rows(db, "endpoint, query, outcome, dish, source, duration_ms")
    assert rows == [
        (
            "recipes/recommend",
            "kimchi and rice",
            "recommended",
            "Kimchi Fried Rice",
            "justonecookbook.com",
            41000,
        )
    ]


async def test_unopenable_path_degrades_to_a_noop(tmp_path):
    recorder = UsageRecorder(str(tmp_path / "no-such-dir" / "usage.db"))

    assert recorder.enabled is False
    await recorder.record(endpoint="home")  # must not raise
    assert recorder.stats() == {}
    assert recorder.recent() == []
    recorder.close()  # must not raise


def test_hash_ip_is_salted_and_opaque():
    a = UsageRecorder(":memory:", salt="salt-a")
    b = UsageRecorder(":memory:", salt="salt-b")

    assert a.hash_ip("203.0.113.9") == a.hash_ip("203.0.113.9")
    assert a.hash_ip("203.0.113.9") != b.hash_ip("203.0.113.9")
    assert "203.0.113.9" not in a.hash_ip("203.0.113.9")
    assert len(a.hash_ip("203.0.113.9")) == 16


def test_missing_salt_falls_back_to_a_random_one():
    a = UsageRecorder(":memory:")
    b = UsageRecorder(":memory:")
    # Distinct per process/instance: still never a raw-IP fallback.
    assert a.hash_ip("203.0.113.9") != b.hash_ip("203.0.113.9")


async def test_stats_and_recent(tmp_path):
    recorder = UsageRecorder(str(tmp_path / "usage.db"), salt="s")
    await recorder.record(endpoint="home", ip_hash="visitor-1")
    await recorder.record(
        endpoint="recipes/recommend",
        ip_hash="visitor-1",
        query="kimchi",
        outcome="recommended",
    )
    await recorder.record(
        endpoint="recipes/recommend",
        ip_hash="visitor-2",
        query="kimchi",
        outcome="null_recommendation",
    )

    stats = recorder.stats(days=7)
    assert stats["asks"]["total"] == 2
    assert stats["asks"]["unique_visitors"] == 2
    assert stats["asks"]["outcomes"] == {
        "recommended": 1,
        "null_recommendation": 1,
    }
    assert stats["asks"]["top_queries"] == [{"query": "kimchi", "count": 2}]
    assert stats["asks"]["by_day"][0]["requests"] == 2
    assert stats["visits"] == {"total": 1, "unique_visitors": 1}

    recent = recorder.recent(limit=2)  # newest first
    assert [row["ip_hash"] for row in recent] == ["visitor-2", "visitor-1"]
    assert recent[0]["endpoint"] == "recipes/recommend"
    recorder.close()
