"""Recipe search service — Exa-backed web search API."""


def main() -> None:
    """Run the dev server (entry point for `uv run recipe-search`)."""
    import logging

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("recipe_search.main:app", host="127.0.0.1", port=8000, reload=True)
