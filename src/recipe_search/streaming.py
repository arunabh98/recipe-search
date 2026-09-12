"""Optional progress transport; the operation still produces one atomic result."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress

from pydantic import BaseModel
from starlette.responses import JSONResponse, StreamingResponse

from recipe_search.pipeline import Progress


def stream_response(
    operation: Callable[[Progress], Awaitable[BaseModel]],
    error_response: Callable[[Exception], Awaitable[JSONResponse]],
) -> StreamingResponse:
    """Send actual stage transitions, then exactly one result or error event.

    Work starts with consumption and is cancelled on disconnect. Admission and
    validation happen in the route before headers are sent; later failures use
    an error event because the HTTP status can no longer change.
    """

    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        def progress(stage: str) -> None:
            queue.put_nowait({"type": "progress", "stage": stage})

        async def run() -> None:
            try:
                result = await operation(progress)
                event = {"type": "result", "data": result.model_dump(mode="json")}
            except Exception as exc:
                response = await error_response(exc)
                event = {
                    "type": "error",
                    "status": response.status_code,
                    "data": json.loads(bytes(response.body)),
                }
            queue.put_nowait(event)

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] != "progress":
                    break
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
