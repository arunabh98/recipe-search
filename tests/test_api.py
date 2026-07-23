"""Endpoint tests for the /search API, with the Exa client faked out."""

import base64

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from recipe_search.evaluation import (
    EvaluationAPIError,
    EvaluationAuthError,
    EvaluationRateLimitError,
    EvaluationTimeoutError,
    MissingItem,
    PhotoIngredients,
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
from recipe_search.main import (
    _MAX_REQUEST_BODY_BYTES,
    RequestBodyLimitMiddleware,
    app,
    get_evaluator,
    get_search_client,
)
from recipe_search.usage import UsageRecorder

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
        self.photo: PhotoIngredients = PhotoIngredients(
            food_visible=True, ingredients=["eggs", "cheddar"]
        )
        self.error: Exception | None = None
        self.calls: list[dict] = []
        self.plan_calls: list[dict] = []
        self.recommend_calls: list[dict] = []
        self.photo_calls: list[dict] = []

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

    async def identify_ingredients(
        self, images: list[tuple[str, str]]
    ) -> PhotoIngredients:
        self.photo_calls.append({"images": list(images)})
        if self.error is not None:
            raise self.error
        return self.photo


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
    assert "/ingredients/from-photo" in response.text
    assert "Add photo" in response.text
    assert 'id="photoReview"' in response.text
    # Multi-photo affordances: a multi-select picker, the thumbnail strip,
    # and the "add more" action.
    assert 'accept="image/*" multiple' in response.text
    assert 'id="photoThumbs"' in response.text
    assert "Add more photos" in response.text
    assert "Simmer doesn't store them" in response.text


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


# --- /ingredients/from-photo -----------------------------------------------


PHOTO_B64 = base64.b64encode(b"fake-jpeg-bytes").decode()
OTHER_B64 = base64.b64encode(b"second-photo-bytes").decode()


def test_photo_returns_identified_ingredients(client, fake_evaluator):
    response = client.post(
        "/ingredients/from-photo",
        json={"images": [{"image_base64": PHOTO_B64, "media_type": "image/png"}]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "food_visible": True,
        "ingredients": ["eggs", "cheddar"],
    }
    assert fake_evaluator.photo_calls == [{"images": [(PHOTO_B64, "image/png")]}]


def test_photo_accepts_multiple_images_in_one_call(client, fake_evaluator):
    fake_evaluator.photo = PhotoIngredients(
        food_visible=True, ingredients=["eggs", "spinach", "cheddar"]
    )

    response = client.post(
        "/ingredients/from-photo",
        json={
            "images": [
                {"image_base64": PHOTO_B64, "media_type": "image/png"},
                {"image_base64": OTHER_B64, "media_type": "image/jpeg"},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["ingredients"] == ["eggs", "spinach", "cheddar"]
    # Every photo reaches the evaluator, in order, in a single call.
    assert fake_evaluator.photo_calls == [
        {"images": [(PHOTO_B64, "image/png"), (OTHER_B64, "image/jpeg")]}
    ]


def test_photo_media_type_defaults_to_jpeg(client, fake_evaluator):
    response = client.post(
        "/ingredients/from-photo", json={"images": [{"image_base64": PHOTO_B64}]}
    )
    assert response.status_code == 200
    assert fake_evaluator.photo_calls[0]["images"] == [(PHOTO_B64, "image/jpeg")]


def test_photo_with_no_food(client, fake_evaluator):
    fake_evaluator.photo = PhotoIngredients(food_visible=False, ingredients=[])
    response = client.post(
        "/ingredients/from-photo", json={"images": [{"image_base64": PHOTO_B64}]}
    )
    assert response.status_code == 200
    assert response.json() == {"food_visible": False, "ingredients": []}


BIG_IMAGE_B64 = base64.b64encode(b"\0" * (5 * 1024 * 1024 + 1)).decode()


@pytest.mark.parametrize(
    "body",
    [
        {},  # images key missing entirely
        {"images": []},  # at least one photo is required
        {"images": [{}]},  # a photo with no base64
        {"images": [{"image_base64": ""}]},
        {"images": [{"image_base64": "not!!valid@@base64"}]},
        {"images": [{"image_base64": PHOTO_B64, "media_type": "image/tiff"}]},
        {"images": [{"image_base64": BIG_IMAGE_B64}]},
        {"images": [{"image_base64": "A" * 7_000_001}]},
        {"images": [{"image_base64": PHOTO_B64}] * 6},  # more than five photos
    ],
)
def test_photo_rejects_invalid_requests(client, fake_evaluator, body):
    response = client.post("/ingredients/from-photo", json=body)

    assert response.status_code == 422
    assert fake_evaluator.photo_calls == []
    assert len(response.content) < 10_000
    for error in response.json()["detail"]:
        assert set(error) == {"type", "loc", "msg"}
    # No image bytes are ever echoed back in the redacted photo 422.
    for image in body.get("images", []):
        payload = image.get("image_base64")
        if payload:
            assert payload[:80] not in response.text


def test_body_limit_rejects_declared_oversize_before_the_route(
    client, fake_evaluator
):
    response = client.post(
        "/ingredients/from-photo",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "content-length": str(_MAX_REQUEST_BODY_BYTES + 1),
        },
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body is too large."}
    assert fake_evaluator.photo_calls == []


def test_body_limit_counts_streamed_bytes_without_content_length():
    limited_app = FastAPI()
    limited_app.add_middleware(RequestBodyLimitMiddleware, max_bytes=5)

    @limited_app.post("/")
    async def consume_body(request: Request):
        return {
            "body": (await request.body()).decode(),
            "content_length": request.headers.get("content-length"),
        }

    with TestClient(limited_app) as limited_client:
        allowed = limited_client.post("/", content=iter([b"12", b"345"]))
        refused = limited_client.post("/", content=iter([b"123", b"456"]))

    assert allowed.status_code == 200
    assert allowed.json() == {"body": "12345", "content_length": None}
    assert refused.status_code == 413
    assert refused.json() == {"detail": "Request body is too large."}


def test_search_422_keeps_the_default_validation_shape(client):
    response = client.post("/search", json={"query": "   "})

    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "query"]
    # The redacting handler is scoped to the photo route; everywhere else
    # deliberately keeps FastAPI's default shape, echoed input included.
    assert "input" in error


def test_photo_maps_evaluation_errors(client, fake_evaluator):
    fake_evaluator.error = EvaluationAPIError("boom")
    response = client.post(
        "/ingredients/from-photo", json={"images": [{"image_base64": PHOTO_B64}]}
    )
    assert response.status_code == 502
    assert "detail" in response.json()


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/search", {"query": "eggs"}),
        ("/recipes/recommend", {"query": "eggs"}),
        ("/ingredients/from-photo", {"images": [{"image_base64": PHOTO_B64}]}),
    ],
)
def test_demo_ip_limits_apply_to_costly_endpoints(
    client, fake_exa, fake_evaluator, path, body
):
    app.state.limiter = RateLimiter(per_hour=1, per_day=5, daily_budget=100)
    try:
        first = client.post(path, json=body)
        second = client.post(path, json=body)
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


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/recipes/search", {"query": "quick dinner"}),
        ("/ingredients/from-photo", {"images": [{"image_base64": PHOTO_B64}]}),
    ],
)
def test_evaluation_endpoints_without_configured_evaluator(
    fake_exa, monkeypatch, path, body
):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    app.dependency_overrides[get_search_client] = lambda: fake_exa
    try:
        with TestClient(app) as test_client:
            app.state.evaluator = None
            response = test_client.post(path, json=body)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert "not configured" in response.json()["detail"]


# --- usage recording -------------------------------------------------------


@pytest.fixture
def recorder(client, tmp_path):
    """A live recorder wired onto the running app, removed afterwards."""
    rec = UsageRecorder(str(tmp_path / "usage.db"), salt="test-salt")
    app.state.usage = rec
    try:
        yield rec
    finally:
        app.state.usage = None
        rec.close()


def test_recommend_records_query_and_outcome(
    client, fake_exa, fake_evaluator, recorder
):
    fake_exa.results = [MIGAS_RESULT]
    fake_evaluator.candidates = [MIGAS_CANDIDATE]
    fake_evaluator.recommendation = MIGAS_RECOMMENDATION

    response = client.post("/recipes/recommend", json={"query": EXAMPLE_QUERY})

    assert response.status_code == 200
    (row,) = recorder.recent()
    assert row["endpoint"] == "recipes/recommend"
    assert row["query"] == EXAMPLE_QUERY
    assert row["outcome"] == "recommended"
    assert row["dish"] == "Tex-Mex migas"
    assert row["source"] == "seriouseats.com"
    assert row["duration_ms"] >= 0
    # The raw client address never lands in the database, only its hash.
    assert row["ip_hash"] == recorder.hash_ip("testclient")
    assert "testclient" not in row["ip_hash"]


def test_photo_records_outcome_but_never_the_image(client, fake_evaluator, recorder):
    response = client.post(
        "/ingredients/from-photo",
        json={"images": [{"image_base64": PHOTO_B64}, {"image_base64": OTHER_B64}]},
    )

    assert response.status_code == 200
    (row,) = recorder.recent()
    assert row["endpoint"] == "ingredients/from-photo"
    assert row["outcome"] == "ingredients:2"
    assert row["query"] is None  # the photos themselves are analyzed and discarded
    # Neither photo's bytes are recorded anywhere on the row.
    assert PHOTO_B64 not in str(row) and OTHER_B64 not in str(row)
    assert row["duration_ms"] >= 0


def test_home_visits_are_recorded_with_referer(client, recorder):
    response = client.get(
        "/", headers={"referer": "https://example.com/post", "user-agent": "UA"}
    )

    assert response.status_code == 200
    (row,) = recorder.recent()
    assert row["endpoint"] == "home"
    assert row["referer"] == "https://example.com/post"
    assert row["user_agent"] == "UA"
    assert row["query"] is None


def test_rate_limited_requests_are_recorded(client, fake_exa, recorder):
    app.state.limiter = RateLimiter(per_hour=1, per_day=5, daily_budget=100)
    try:
        first = client.post("/search", json={"query": "eggs"})
        second = client.post("/search", json={"query": "eggs"})
    finally:
        app.state.limiter = None

    assert first.status_code == 200
    assert second.status_code == 429
    outcomes = [row["outcome"] for row in recorder.recent()]
    assert outcomes == ["rate_limited:rate_limit", "results:0"]


def test_broken_recorder_never_breaks_requests(client, fake_exa, fake_evaluator):
    class ExplodingRecorder:
        def hash_ip(self, ip: str) -> str:
            raise RuntimeError("boom")

        async def record(self, **fields: object) -> None:
            raise RuntimeError("boom")

    fake_exa.results = [MIGAS_RESULT]
    fake_evaluator.candidates = [MIGAS_CANDIDATE]
    fake_evaluator.recommendation = MIGAS_RECOMMENDATION
    app.state.usage = ExplodingRecorder()
    try:
        recommend = client.post("/recipes/recommend", json={"query": EXAMPLE_QUERY})
        home = client.get("/")
    finally:
        app.state.usage = None

    assert recommend.status_code == 200
    assert home.status_code == 200


# --- /stats ----------------------------------------------------------------


def test_stats_is_hidden_when_no_token_is_configured(client, recorder):
    assert client.get("/stats").status_code == 404
    assert client.get("/stats", headers={"X-Stats-Token": "guess"}).status_code == 404


def test_stats_requires_the_exact_token(client, recorder):
    app.state.stats_token = SecretStr("owner-token")
    try:
        wrong = client.get("/stats", headers={"X-Stats-Token": "wrong"})
        right = client.get("/stats", headers={"X-Stats-Token": "owner-token"})
        via_param = client.get("/stats?token=owner-token")
    finally:
        app.state.stats_token = None

    assert wrong.status_code == 404
    assert wrong.json() == {"detail": "Not Found"}  # same body as a missing route
    assert right.status_code == 200
    body = right.json()
    assert body["recording_enabled"] is True
    assert "asks" in body["stats"] and "visits" in body["stats"]
    assert via_param.status_code == 200


def test_stats_reports_when_recording_is_off(client):
    app.state.stats_token = SecretStr("owner-token")
    try:
        response = client.get("/stats", headers={"X-Stats-Token": "owner-token"})
    finally:
        app.state.stats_token = None

    assert response.status_code == 200
    assert response.json() == {"recording_enabled": False, "stats": {}, "recent": []}
