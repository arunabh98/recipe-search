"""Unit tests for the demo rate limiter, driven by a fake clock."""

from recipe_search.limits import RateLimiter


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make(per_hour=2, per_day=3, daily_budget=100):
    clock = Clock()
    limiter = RateLimiter(
        per_hour=per_hour, per_day=per_day, daily_budget=daily_budget, clock=clock
    )
    return limiter, clock


def test_hourly_window():
    limiter, clock = make(per_hour=2)
    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert limiter.check("a") == "ip_hour"
    clock.advance(3601)
    assert limiter.check("a") is None


def test_daily_window():
    limiter, clock = make(per_hour=10, per_day=3)
    for _ in range(3):
        assert limiter.check("a") is None
    assert limiter.check("a") == "ip_day"
    clock.advance(86_401)
    assert limiter.check("a") is None


def test_ips_are_independent():
    limiter, _ = make(per_hour=1)
    assert limiter.check("a") is None
    assert limiter.check("a") == "ip_hour"
    assert limiter.check("b") is None


def test_global_budget_resets_next_utc_day():
    limiter, clock = make(per_hour=10, per_day=10, daily_budget=2)
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("c") == "budget"
    clock.advance(86_400)
    assert limiter.check("c") is None


def test_rejected_requests_consume_nothing():
    limiter, _ = make(per_hour=1, per_day=10, daily_budget=2)
    assert limiter.check("a") is None  # budget: 1 of 2
    assert limiter.check("a") == "ip_hour"  # rejected; budget untouched
    assert limiter.check("b") is None  # budget: 2 of 2
    assert limiter.check("c") == "budget"
