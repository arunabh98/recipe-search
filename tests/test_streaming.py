"""Progress is delivered before work completes; disconnects stop that work."""

import asyncio
import json

import pytest
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from recipe_search.streaming import stream_response


class Result(BaseModel):
    reply: str


async def unexpected_error(exc):
    pytest.fail(f"Unexpected error: {exc}")


async def test_progress_arrives_while_operation_is_still_waiting():
    release = asyncio.Event()

    async def operation(progress):
        progress("understanding")
        await release.wait()
        return Result(reply="Yogurt works 🍲")

    response = stream_response(operation, unexpected_error)
    iterator = response.body_iterator
    first = await asyncio.wait_for(anext(iterator), timeout=1)
    assert json.loads(first) == {"type": "progress", "stage": "understanding"}
    assert not release.is_set()
    release.set()
    final = json.loads(await anext(iterator))
    assert final == {"type": "result", "data": {"reply": "Yogurt works 🍲"}}
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


async def test_closing_stream_cancels_operation_and_runs_cleanup_once():
    outcomes = []

    async def operation(progress):
        try:
            progress("searching")
            await asyncio.Event().wait()
        finally:
            outcomes.append("cancelled")

    response = stream_response(operation, unexpected_error)
    iterator = response.body_iterator
    await anext(iterator)
    await iterator.aclose()
    assert outcomes == ["cancelled"]


async def test_error_is_terminal_even_after_progress():
    async def operation(progress):
        progress("searching")
        raise ValueError("private")

    async def error_response(exc):
        return JSONResponse(status_code=502, content={"detail": "Try again."})

    response = stream_response(operation, error_response)
    events = [json.loads(line) async for line in response.body_iterator]
    assert events == [
        {"type": "progress", "stage": "searching"},
        {"type": "error", "status": 502, "data": {"detail": "Try again."}},
    ]
