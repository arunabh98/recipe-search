"""FastAPI app exposing the Exa-backed search endpoint."""

import asyncio
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from recipe_search.config import Settings
from recipe_search.evaluation import (
    EvaluationAuthError,
    EvaluationError,
    EvaluationRateLimitError,
    EvaluationTimeoutError,
    Recommendation,
    RecipeCandidate,
    RecipeEvaluator,
)
from recipe_search.exa_search import (
    ExaAuthError,
    ExaRateLimitError,
    ExaSearchClient,
    ExaSearchError,
    ExaTimeoutError,
    SearchResult,
)
from recipe_search.limits import RateLimited, RateLimiter
from recipe_search.pipeline import OffTopicQuery, find_recipe_candidates, recommend_recipe
from recipe_search.usage import UsageRecorder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()  # fails fast at startup if EXA_API_KEY is missing
    app.state.exa = ExaSearchClient(
        api_key=settings.exa_api_key.get_secret_value(),
        base_url=settings.exa_base_url,
        timeout_seconds=settings.exa_timeout_seconds,
    )
    # Evaluation is additive: if no Anthropic credential can be resolved,
    # /search keeps working and /recipes/search reports it's not configured.
    try:
        app.state.evaluator = RecipeEvaluator(
            api_key=(
                settings.anthropic_api_key.get_secret_value()
                if settings.anthropic_api_key
                else None
            ),
            model=settings.evaluation_model,
            effort=settings.evaluation_effort,
            timeout_seconds=settings.evaluation_timeout_seconds,
        )
    except Exception:
        logger.warning(
            "Recipe evaluation disabled: no Anthropic credentials found "
            "(set ANTHROPIC_API_KEY)"
        )
        app.state.evaluator = None
    app.state.trust_proxy_headers = settings.trust_proxy_headers
    app.state.limiter = (
        RateLimiter(
            per_hour=settings.ip_requests_per_hour,
            per_day=settings.ip_requests_per_day,
            daily_budget=settings.daily_request_budget,
        )
        if settings.demo_mode
        else None
    )
    # Usage recording is additive, like evaluation: off unless a path is
    # configured, and a recorder that can't open its file degrades to a no-op.
    app.state.usage = (
        UsageRecorder(
            settings.usage_db_path,
            salt=(
                settings.usage_salt.get_secret_value()
                if settings.usage_salt
                else None
            ),
        )
        if settings.usage_db_path
        else None
    )
    app.state.stats_token = settings.stats_token
    yield
    await app.state.exa.aclose()
    if app.state.evaluator is not None:
        await app.state.evaluator.aclose()
    if app.state.usage is not None:
        app.state.usage.close()


# Read the flag at import time so API docs can be disabled for the public
# demo; the richer Settings object still governs everything else.
load_dotenv()
_DEMO_MODE = os.environ.get("DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}

app = FastAPI(
    title="Recipe Search API",
    description="Natural-language web search backed by Exa.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _DEMO_MODE else "/docs",
    redoc_url=None,
    openapi_url=None if _DEMO_MODE else "/openapi.json",
)


def get_search_client(request: Request) -> ExaSearchClient:
    return request.app.state.exa


def get_evaluator(request: Request) -> RecipeEvaluator:
    evaluator = request.app.state.evaluator
    if evaluator is None:
        raise HTTPException(
            status_code=500,
            detail="Recipe evaluation is not configured (set ANTHROPIC_API_KEY).",
        )
    return evaluator


def _client_ip(request: Request) -> str:
    if request.app.state.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _record_usage(request: Request, **fields: object) -> None:
    """Record one usage event if recording is configured.

    Usage recording is additive: no failure here may ever fail a request,
    so the whole body is guarded and the recorder itself never raises.
    """
    try:
        recorder = getattr(request.app.state, "usage", None)
        if recorder is None:
            return
        await recorder.record(ip_hash=recorder.hash_ip(_client_ip(request)), **fields)
    except Exception:
        logger.warning("Usage recording failed", exc_info=True)


def enforce_limits(request: Request) -> None:
    """Demo-mode request limits; a no-op when demo_mode is off."""
    limiter = request.app.state.limiter
    if limiter is None:
        return
    refusal = limiter.check(_client_ip(request))
    if refusal == "budget":
        raise RateLimited(
            "budget",
            "Today's demo budget is fully used. The stove relights tomorrow.",
        )
    if refusal == "ip_day":
        raise RateLimited(
            "rate_limit",
            "Simmer is a small demo, so each visitor gets a few dishes a day. "
            "Come back tomorrow.",
        )
    if refusal == "ip_hour":
        raise RateLimited(
            "rate_limit",
            "A few dishes an hour is my pace. Give it a little while, then try again.",
        )


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="Natural-language search query",
        examples=["I have eggs, salsa, tortillas, and cheese. I want something quick."],
    )
    num_results: int = Field(default=8, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty or whitespace")
        return value


class SearchResponse(BaseModel):
    results: list[SearchResult]


class RecipeSearchResponse(BaseModel):
    candidates: list[RecipeCandidate]


class RecipeRecommendationResponse(BaseModel):
    recommendation: Recommendation | None
    candidates: list[RecipeCandidate]


# Upstream failure → HTTP response policy. Walked in order with isinstance,
# so each family's specific errors must precede its base class.
_UPSTREAM_ERRORS: list[tuple[type[Exception], int, str]] = [
    (ExaAuthError, 500, "Search service is misconfigured."),
    (ExaRateLimitError, 429, "Search provider rate limit reached. Try again shortly."),
    (ExaTimeoutError, 504, "Search timed out. Try again."),
    (ExaSearchError, 502, "Search provider error. Try again."),
    (EvaluationAuthError, 500, "Recipe evaluation is misconfigured."),
    (EvaluationRateLimitError, 429, "Evaluation rate limit reached. Try again shortly."),
    (EvaluationTimeoutError, 504, "Recipe evaluation timed out. Try again."),
    (EvaluationError, 502, "Recipe evaluation failed. Try again."),
]


@app.exception_handler(ExaSearchError)
@app.exception_handler(EvaluationError)
async def handle_upstream_error(request: Request, exc: Exception) -> JSONResponse:
    status, detail = next(
        (code, message)
        for exc_type, code, message in _UPSTREAM_ERRORS
        if isinstance(exc, exc_type)
    )
    if status in (500, 502):  # misconfiguration or unexpected upstream failure
        logger.error("%s: %s", type(exc).__name__, exc)
    return JSONResponse(status_code=status, content={"detail": detail})


@app.exception_handler(RateLimited)
async def handle_rate_limited(request: Request, exc: RateLimited) -> JSONResponse:
    # The body was never parsed (limits refuse before validation), so the
    # refusal is recorded without query text.
    await _record_usage(
        request,
        endpoint=request.url.path.lstrip("/"),
        outcome=f"rate_limited:{exc.code}",
    )
    return JSONResponse(
        status_code=429, content={"detail": exc.message, "code": exc.code}
    )


@app.exception_handler(OffTopicQuery)
async def handle_off_topic(request: Request, exc: OffTopicQuery) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": (
                "I'm a cooking assistant. Tell me what's in your kitchen, "
                "what you're craving, or the kind of meal you need, and I'll "
                "find you something great to cook."
            ),
            "code": "off_topic",
        },
    )


@app.post(
    "/search", response_model=SearchResponse, dependencies=[Depends(enforce_limits)]
)
async def search(
    body: SearchRequest,
    request: Request,
    exa: ExaSearchClient = Depends(get_search_client),
) -> SearchResponse:
    """Search the web with a natural-language query and return normalized results."""
    start = time.monotonic()
    outcome = "cancelled"
    try:
        results = await exa.search(body.query, num_results=body.num_results)
        outcome = f"results:{len(results)}"
    except Exception as exc:
        outcome = f"error:{type(exc).__name__}"
        raise
    finally:
        await _record_usage(
            request,
            endpoint="search",
            query=body.query,
            outcome=outcome,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    return SearchResponse(results=results)


@app.post(
    "/recipes/search",
    response_model=RecipeSearchResponse,
    dependencies=[Depends(enforce_limits)],
)
async def search_recipes(
    body: SearchRequest,
    request: Request,
    exa: ExaSearchClient = Depends(get_search_client),
    evaluator: RecipeEvaluator = Depends(get_evaluator),
) -> RecipeSearchResponse:
    """Plan searches, retrieve, and rank the results as cooking candidates."""
    start = time.monotonic()
    outcome = "cancelled"
    try:
        candidates = await find_recipe_candidates(
            body.query, num_results=body.num_results, exa=exa, evaluator=evaluator
        )
        outcome = f"candidates:{len(candidates)}"
    except OffTopicQuery:
        outcome = "off_topic"
        raise
    except Exception as exc:
        outcome = f"error:{type(exc).__name__}"
        raise
    finally:
        await _record_usage(
            request,
            endpoint="recipes/search",
            query=body.query,
            outcome=outcome,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    return RecipeSearchResponse(candidates=candidates)


@app.post(
    "/recipes/recommend",
    response_model=RecipeRecommendationResponse,
    dependencies=[Depends(enforce_limits)],
)
async def recommend_recipes(
    body: SearchRequest,
    request: Request,
    exa: ExaSearchClient = Depends(get_search_client),
    evaluator: RecipeEvaluator = Depends(get_evaluator),
) -> RecipeRecommendationResponse:
    """Search, rank, and answer 'what should I cook?' with source links."""
    start = time.monotonic()
    outcome = "cancelled"
    dish = source = None
    try:
        recommendation, candidates = await recommend_recipe(
            body.query, num_results=body.num_results, exa=exa, evaluator=evaluator
        )
        if recommendation is None:
            outcome = "null_recommendation"
        else:
            outcome = "recommended"
            dish = recommendation.dish_name
            if recommendation.primary_sources:
                source = recommendation.primary_sources[0].source
    except OffTopicQuery:
        outcome = "off_topic"
        raise
    except Exception as exc:
        outcome = f"error:{type(exc).__name__}"
        raise
    finally:
        await _record_usage(
            request,
            endpoint="recipes/recommend",
            query=body.query,
            outcome=outcome,
            dish=dish,
            source=source,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    return RecipeRecommendationResponse(
        recommendation=recommendation, candidates=candidates
    )


_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def home(request: Request) -> FileResponse:
    """The Simmer demo UI."""
    await _record_usage(
        request,
        endpoint="home",
        user_agent=request.headers.get("user-agent"),
        referer=request.headers.get("referer"),
    )
    return FileResponse(_INDEX_HTML, media_type="text/html")


@app.get("/stats", include_in_schema=False)
async def usage_stats(request: Request, token: str | None = None) -> dict:
    """Owner-only usage aggregates; looks like a 404 unless the token matches."""
    configured = getattr(request.app.state, "stats_token", None)
    provided = request.headers.get("x-stats-token") or token or ""
    if configured is None or not hmac.compare_digest(
        provided.encode(), configured.get_secret_value().encode()
    ):
        raise HTTPException(status_code=404, detail="Not Found")
    recorder = getattr(request.app.state, "usage", None)
    if recorder is None:
        return {"recording_enabled": False, "stats": {}, "recent": []}
    return {
        "recording_enabled": recorder.enabled,
        "stats": await asyncio.to_thread(recorder.stats),
        "recent": await asyncio.to_thread(recorder.recent),
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
