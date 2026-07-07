"""Endpoint tests for the /search API, with the Exa client faked out."""

import pytest
from fastapi.testclient import TestClient

from recipe_search.evaluation import (
    EvaluationAPIError,
    EvaluationAuthError,
    EvaluationRateLimitError,
    EvaluationTimeoutError,
    MissingItem,
    Recommendation,
    RecipeCandidate,
    SearchPlan,
    SourceLink,
)
from recipe_search.exa_search import (
    ExaAPIError,
    ExaAuthError,
    ExaRateLimitError,
    ExaTimeoutError,
    SearchResult,
)
from recipe_search.limits import RateLimiter
from recipe_search.main import app, get_evaluator, get_search_client

EXAMPLE_QUERY = "I have eggs, salsa, tortillas, and cheese. I want something quick."


class FakeExaClient:
    def __init__(self):
        self.results: list[SearchResult] = []
        self.error: Exception | None = None
        self.calls: list[dict] = []

    async def search(self, query: str, *, num_results: int = 8) -> list[SearchResult]:
        self.calls.append({"query": query, "num_results": num_results})
        if self.error is not None:
            raise self.error
        return self.results


class FakeEvaluator:
    def __init__(self):
        self.candidates: list[RecipeCandidate] = []
        self.recommendation: Recommendation | None = None
        self.plan: SearchPlan | None = None
        self.error: Exception | None = None
        self.calls: list[dict] = []
        self.plan_calls: list[dict] = []
        self.recommend_calls: list[dict] = []

    async def plan_searches(
        self, query: str, *, feedback: str | None = None
    ) -> SearchPlan:
        self.plan_calls.append({"query": query, "feedback": feedback})
        if self.plan is not None:
            return self.plan
        return SearchPlan(queries=[query])  # pass-through plan

    async def evaluate(
        self, query: str, results: list[SearchResult]
    ) -> list[RecipeCandidate]:
        self.calls.append({"query": query, "results": results})
        if self.error is not None:
            raise self.error
        return self.candidates

    async def recommend(
        self, query: str, candidates: list[RecipeCandidate]
    ) -> Recommendation:
        self.recommend_calls.append({"query": query, "candidates": candidates})
        return self.recommendation


@pytest.fixture
def fake_exa():
    return FakeExaClient()


@pytest.fixture
def fake_evaluator():
    return FakeEvaluator()


@pytest.fixture
def client(fake_exa, fake_evaluator, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    app.dependency_overrides[get_search_client] = lambda: fake_exa
    app.dependency_overrides[get_evaluator] = lambda: fake_evaluator
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_search_returns_normalized_results(client, fake_exa):
    fake_exa.results = [
        SearchResult(
            title="10-Minute Migas",
            url="https://www.seriouseats.com/migas",
            source="seriouseats.com",
            snippet="Crispy tortillas with eggs and salsa.",
            published_date="2023-05-01T00:00:00.000Z",
        )
    ]

    response = client.post("/search", json={"query": EXAMPLE_QUERY})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "title": "10-Minute Migas",
                "url": "https://www.seriouseats.com/migas",
                "source": "seriouseats.com",
                "snippet": "Crispy tortillas with eggs and salsa.",
                "published_date": "2023-05-01T00:00:00.000Z",
            }
        ]
    }
    assert fake_exa.calls == [{"query": EXAMPLE_QUERY, "num_results": 8}]


def test_search_passes_num_results(client, fake_exa):
    response = client.post("/search", json={"query": "quick dinner", "num_results": 5})
    assert response.status_code == 200
    assert fake_exa.calls[0]["num_results"] == 5


def test_search_with_no_results(client, fake_exa):
    response = client.post("/search", json={"query": "asdfghjkl zxcvbnm"})
    assert response.status_code == 200
    assert response.json() == {"results": []}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "ok", "num_results": 0},
        {"query": "ok", "num_results": 11},
        {"query": "x" * 501},
    ],
)
def test_search_rejects_invalid_requests(client, fake_exa, body):
    response = client.post("/search", json=body)
    assert response.status_code == 422
    assert fake_exa.calls == []


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ExaAuthError("bad key"), 500),
        (ExaRateLimitError("slow down"), 429),
        (ExaTimeoutError("timed out"), 504),
        (ExaAPIError("boom"), 502),
    ],
)
def test_search_maps_upstream_errors(client, fake_exa, error, expected_status):
    fake_exa.error = error
    response = client.post("/search", json={"query": "quick dinner"})
    assert response.status_code == expected_status
    assert "detail" in response.json()


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_serves_the_demo_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Simmer" in response.text
    assert "/recipes/recommend" in response.text


MIGAS_RESULT = SearchResult(
    title="10-Minute Migas",
    url="https://www.seriouseats.com/migas",
    source="seriouseats.com",
    snippet="Crispy tortillas with eggs and salsa.",
    published_date=None,
)

MIGAS_CANDIDATE = RecipeCandidate(
    title="10-Minute Migas",
    url="https://www.seriouseats.com/migas",
    source="seriouseats.com",
    dish_name="Tex-Mex migas",
    fit_score=0.92,
    why_it_matches="Uses eggs, tortillas, salsa, and cheese; quick Tex-Mex dish.",
    matched_ingredients=["eggs", "salsa", "tortillas", "cheese"],
    possibly_missing=["onion"],
    role="best_base_recipe",
)


def test_recipes_search_returns_ranked_candidates(client, fake_exa, fake_evaluator):
    fake_exa.results = [MIGAS_RESULT]
    fake_evaluator.candidates = [MIGAS_CANDIDATE]

    response = client.post("/recipes/search", json={"query": EXAMPLE_QUERY})

    assert response.status_code == 200
    assert response.json() == {
        "candidates": [
            {
                "title": "10-Minute Migas",
                "url": "https://www.seriouseats.com/migas",
                "source": "seriouseats.com",
                "dish_name": "Tex-Mex migas",
                "fit_score": 0.92,
                "why_it_matches": "Uses eggs, tortillas, salsa, and cheese; quick Tex-Mex dish.",
                "matched_ingredients": ["eggs", "salsa", "tortillas", "cheese"],
                "possibly_missing": ["onion"],
                "role": "best_base_recipe",
            }
        ]
    }
    # Exa gets the planner's query (pass-through fake); the evaluator judges
    # the user's original words.
    assert fake_evaluator.plan_calls[0] == {"query": EXAMPLE_QUERY, "feedback": None}
    assert fake_exa.calls == [{"query": EXAMPLE_QUERY, "num_results": 8}]
    assert fake_evaluator.calls == [{"query": EXAMPLE_QUERY, "results": [MIGAS_RESULT]}]


def test_recipes_search_with_no_results(client, fake_evaluator):
    response = client.post("/recipes/search", json={"query": "asdfghjkl zxcvbnm"})
    assert response.status_code == 200
    assert response.json() == {"candidates": []}


def test_recipes_search_rejects_invalid_requests(client, fake_evaluator):
    response = client.post("/recipes/search", json={"query": "   "})
    assert response.status_code == 422
    assert fake_evaluator.calls == []


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (EvaluationAuthError("bad key"), 500),
        (EvaluationRateLimitError("slow down"), 429),
        (EvaluationTimeoutError("timed out"), 504),
        (EvaluationAPIError("boom"), 502),
    ],
)
def test_recipes_search_maps_evaluation_errors(
    client, fake_exa, fake_evaluator, error, expected_status
):
    fake_exa.results = [MIGAS_RESULT]
    fake_evaluator.error = error
    response = client.post("/recipes/search", json={"query": "quick dinner"})
    assert response.status_code == expected_status
    assert "detail" in response.json()


def test_recipes_search_maps_exa_errors_too(client, fake_exa, fake_evaluator):
    fake_exa.error = ExaRateLimitError("slow down")
    response = client.post("/recipes/search", json={"query": "quick dinner"})
    assert response.status_code == 429
    assert fake_evaluator.calls == []


MIGAS_RECOMMENDATION = Recommendation(
    dish_name="Tex-Mex migas",
    headline="You're fifteen minutes from a skillet of migas.",
    why_it_fits="You have all four core ingredients and it's genuinely quick.",
    missing_items=[
        MissingItem(ingredient="onion", importance="nice_to_have", note="skip it")
    ],
    primary_sources=[
        SourceLink(
            title="10-Minute Migas",
            url="https://www.seriouseats.com/migas",
            source="seriouseats.com",
            dish_name="Tex-Mex migas",
        )
    ],
    how_to_use_sources="Follow the Serious Eats page for the whole cook.",
    alternatives=[],
)


def test_recipes_recommend_returns_recommendation(client, fake_exa, fake_evaluator):
    fake_exa.results = [MIGAS_RESULT]
    fake_evaluator.candidates = [MIGAS_CANDIDATE]
    fake_evaluator.recommendation = MIGAS_RECOMMENDATION

    response = client.post("/recipes/recommend", json={"query": EXAMPLE_QUERY})

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation"]["dish_name"] == "Tex-Mex migas"
    assert body["recommendation"]["missing_items"] == [
        {"ingredient": "onion", "importance": "nice_to_have", "note": "skip it"}
    ]
    assert body["recommendation"]["primary_sources"][0]["url"] == (
        "https://www.seriouseats.com/migas"
    )
    assert body["candidates"] == [MIGAS_CANDIDATE.model_dump()]
    assert fake_evaluator.recommend_calls == [
        {"query": EXAMPLE_QUERY, "candidates": [MIGAS_CANDIDATE]}
    ]


def test_recipes_recommend_null_when_nothing_usable(client, fake_exa, fake_evaluator):
    fake_exa.results = [MIGAS_RESULT]
    fake_evaluator.candidates = [
        MIGAS_CANDIDATE.model_copy(update={"role": "ignore"})
    ]

    response = client.post("/recipes/recommend", json={"query": EXAMPLE_QUERY})

    assert response.status_code == 200
    assert response.json()["recommendation"] is None
    assert fake_evaluator.recommend_calls == []


def test_recipes_recommend_rejects_invalid_requests(client, fake_evaluator):
    response = client.post("/recipes/recommend", json={"query": "   "})
    assert response.status_code == 422
    assert fake_evaluator.recommend_calls == []


def test_off_topic_queries_get_a_friendly_422(client, fake_exa, fake_evaluator):
    fake_evaluator.plan = SearchPlan(on_topic=False, queries=[])

    response = client.post(
        "/recipes/recommend", json={"query": "write me python code"}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "off_topic"
    assert "cooking assistant" in response.json()["detail"]
    assert fake_exa.calls == []  # no search spend on off-topic requests
    assert fake_evaluator.calls == []


@pytest.mark.parametrize("path", ["/search", "/recipes/recommend"])
def test_demo_ip_limits_apply_to_costly_endpoints(
    client, fake_exa, fake_evaluator, path
):
    app.state.limiter = RateLimiter(per_hour=1, per_day=5, daily_budget=100)
    try:
        first = client.post(path, json={"query": "eggs"})
        second = client.post(path, json={"query": "eggs"})
    finally:
        app.state.limiter = None
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["code"] == "rate_limit"


def test_demo_budget_exhaustion(client, fake_exa):
    app.state.limiter = RateLimiter(per_hour=10, per_day=10, daily_budget=1)
    try:
        first = client.post("/search", json={"query": "eggs"})
        second = client.post("/search", json={"query": "eggs"})
    finally:
        app.state.limiter = None
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["code"] == "budget"


def test_recipes_search_without_configured_evaluator(fake_exa, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    app.dependency_overrides[get_search_client] = lambda: fake_exa
    try:
        with TestClient(app) as test_client:
            app.state.evaluator = None
            response = test_client.post(
                "/recipes/search", json={"query": "quick dinner"}
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert "not configured" in response.json()["detail"]
