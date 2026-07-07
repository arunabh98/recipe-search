"""Isolated Exa search integration — the only module that talks to Exa
(https://exa.ai/docs/reference/search). No FastAPI imports; reusable from
CLIs, jobs, or other services.

Calls the REST API directly with httpx rather than the exa-py SDK: the SDK
(2.16.0) sends requests with no timeout (sync) or a hardcoded 600s timeout
(async), and raises bare ValueError for every HTTP failure.
"""

import logging
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Bounds accepted by Exa's API; callers may impose stricter policy.
MIN_RESULTS = 1
MAX_RESULTS = 100


class SearchResult(BaseModel):
    """A single web search result, normalized from Exa's response."""

    title: str | None
    url: str
    source: str
    snippet: str | None
    published_date: str | None


class ExaSearchError(Exception):
    """Base class for failures while searching with Exa."""


class ExaAuthError(ExaSearchError):
    """Exa rejected our credentials (401/403) — a server configuration problem."""


class ExaRateLimitError(ExaSearchError):
    """Exa rate limit exceeded (429)."""


class ExaTimeoutError(ExaSearchError):
    """The request to Exa timed out."""


class ExaAPIError(ExaSearchError):
    """Exa was unreachable, returned an unexpected status, or a malformed body."""


class ExaSearchClient:
    """Thin async client for Exa's search endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.exa.ai",
        timeout_seconds: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"x-api-key": api_key},
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def search(self, query: str, *, num_results: int = 8) -> list[SearchResult]:
        """Run a natural-language web search and return normalized results.

        Returns an empty list when Exa finds nothing. Raises an
        ``ExaSearchError`` subclass on any failure.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not MIN_RESULTS <= num_results <= MAX_RESULTS:
            raise ValueError(
                f"num_results must be between {MIN_RESULTS} and {MAX_RESULTS}"
            )

        payload = {
            "query": query,
            "type": "auto",
            "numResults": num_results,
            # Highlights are query-relevant excerpts; they back the snippet
            # field and are cheaper than fetching full page text.
            "contents": {"highlights": True},
        }

        try:
            response = await self._http.post("/search", json=payload)
        except httpx.TimeoutException as exc:
            raise ExaTimeoutError(
                f"Exa search timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ExaAPIError(f"Could not reach Exa: {type(exc).__name__}") from exc

        if response.status_code in (401, 403):
            raise ExaAuthError("Exa rejected the API key (check EXA_API_KEY)")
        if response.status_code == 429:
            raise ExaRateLimitError("Exa rate limit exceeded")
        if response.status_code >= 400:
            logger.error(
                "Exa search failed: HTTP %s: %s",
                response.status_code,
                response.text[:500],
            )
            raise ExaAPIError(f"Exa returned HTTP {response.status_code}")

        try:
            raw_results = response.json()["results"]
        except (ValueError, KeyError) as exc:
            raise ExaAPIError("Exa returned an unexpected response body") from exc
        if not isinstance(raw_results, list):
            raise ExaAPIError("Exa returned an unexpected response body")

        return [
            result
            for raw in raw_results
            if (result := _normalize_result(raw)) is not None
        ]


def _normalize_result(raw: object) -> SearchResult | None:
    """Map one raw Exa result onto SearchResult, or None if unusable."""
    if not isinstance(raw, dict):
        logger.warning("Skipping malformed Exa result: %r", raw)
        return None
    url = raw.get("url")
    if not isinstance(url, str) or not url:
        logger.warning("Skipping Exa result without a url (id=%r)", raw.get("id"))
        return None

    highlights = raw.get("highlights")
    snippet = None
    if isinstance(highlights, list):
        parts = [h.strip() for h in highlights if isinstance(h, str) and h.strip()]
        snippet = " … ".join(parts) or None

    return SearchResult(
        title=_clean_str(raw.get("title")),
        url=url,
        source=_source_from_url(url),
        snippet=snippet,
        published_date=_clean_str(raw.get("publishedDate")),
    )


def _clean_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _source_from_url(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host.removeprefix("www.")
