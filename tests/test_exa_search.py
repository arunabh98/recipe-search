"""Unit tests for the isolated Exa client, using a mock HTTP transport."""

import json

import httpx
import pytest

from recipe_search.exa_search import (
    ExaAPIError,
    ExaAuthError,
    ExaRateLimitError,
    ExaSearchClient,
    ExaTimeoutError,
)

EXA_PAYLOAD = {
    "requestId": "req-123",
    "results": [
        {
            "id": "doc-1",
            "title": "10-Minute Migas",
            "url": "https://www.seriouseats.com/migas",
            "publishedDate": "2023-05-01T00:00:00.000Z",
            "author": "A Cook",
            "highlights": [
                "Crispy tortillas with eggs and salsa.",
                "Done in 10 minutes.",
            ],
            "highlightScores": [0.92, 0.85],
        },
        {
            "id": "doc-2",
            "title": None,
            "url": "https://example.org/breakfast-tacos",
            "highlights": [],
        },
        {"id": "doc-3", "title": "No URL here"},
        "not-a-dict",
    ],
}


@pytest.fixture
async def make_client():
    clients = []

    def _make(handler) -> ExaSearchClient:
        client = ExaSearchClient(
            api_key="test-key", transport=httpx.MockTransport(handler)
        )
        clients.append(client)
        return client

    yield _make
    for client in clients:
        await client.aclose()


async def test_search_sends_documented_request_shape(make_client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["api_key"] = request.headers.get("x-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    client = make_client(handler)
    await client.search("  something quick  ", num_results=5)

    assert captured["path"] == "/search"
    assert captured["api_key"] == "test-key"
    assert captured["body"] == {
        "query": "something quick",
        "type": "auto",
        "numResults": 5,
        "contents": {"highlights": True},
    }


async def test_search_normalizes_results(make_client):
    client = make_client(lambda request: httpx.Response(200, json=EXA_PAYLOAD))
    results = await client.search("eggs salsa tortillas cheese")

    assert len(results) == 2  # doc-3 (no url) and "not-a-dict" are skipped
    assert results[0].model_dump() == {
        "title": "10-Minute Migas",
        "url": "https://www.seriouseats.com/migas",
        "source": "seriouseats.com",
        "snippet": "Crispy tortillas with eggs and salsa. … Done in 10 minutes.",
        "published_date": "2023-05-01T00:00:00.000Z",
    }
    assert results[1].model_dump() == {
        "title": None,
        "url": "https://example.org/breakfast-tacos",
        "source": "example.org",
        "snippet": None,
        "published_date": None,
    }


async def test_search_returns_empty_list_when_no_results(make_client):
    client = make_client(lambda request: httpx.Response(200, json={"results": []}))
    assert await client.search("anything") == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, ExaAPIError),
        (401, ExaAuthError),
        (403, ExaAuthError),
        (429, ExaRateLimitError),
        (500, ExaAPIError),
    ],
)
async def test_search_maps_http_errors(make_client, status, expected):
    client = make_client(
        lambda request: httpx.Response(status, json={"error": "boom"})
    )
    with pytest.raises(expected):
        await client.search("anything")


async def test_search_maps_timeouts(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ExaTimeoutError):
        await make_client(handler).search("anything")


async def test_search_maps_connection_errors(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ExaAPIError):
        await make_client(handler).search("anything")


async def test_search_rejects_unexpected_body(make_client):
    client = make_client(lambda request: httpx.Response(200, json={"nope": True}))
    with pytest.raises(ExaAPIError):
        await client.search("anything")


async def test_search_validates_arguments(make_client):
    client = make_client(lambda request: httpx.Response(200, json={"results": []}))
    with pytest.raises(ValueError):
        await client.search("   ")
    with pytest.raises(ValueError):
        await client.search("ok", num_results=0)
    with pytest.raises(ValueError):
        await client.search("ok", num_results=101)
