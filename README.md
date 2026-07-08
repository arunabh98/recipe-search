# recipe-search

**Simmer** — tell it what you've got, it tells you what to cook. A FastAPI
service that takes a natural-language food query, searches the web with
[Exa](https://exa.ai), ranks every result via Claude, and answers with a
source-linked cooking recommendation.

Run the server and open **http://127.0.0.1:8000/** for the demo UI — a
single self-contained page (`src/recipe_search/static/index.html`, no build
step, no CDN) with example queries, a staged progress view while the
pipeline runs, and the full recommendation experience: the dish, why it
fits, essential vs nice-to-have missing items, cook-from source cards, and
alternatives.

## Setup

```bash
uv sync
cp .env.example .env
# then paste your keys:
#   EXA_API_KEY        from https://dashboard.exa.ai
#   ANTHROPIC_API_KEY  from https://console.anthropic.com
```

## Run

```bash
uv run recipe-search                             # dev server with reload on :8000
# or: uv run uvicorn recipe_search.main:app --port 8000
```

Interactive API docs: http://127.0.0.1:8000/docs

## API

### `POST /search`

```bash
curl -s http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "I have eggs, salsa, tortillas, and cheese. I want something quick.", "num_results": 8}'
```

Request body:

| field         | type   | notes                          |
| ------------- | ------ | ------------------------------ |
| `query`       | string | required, 1–500 chars          |
| `num_results` | int    | optional, 1–10 (default **8**) |

Response (`200`):

```json
{
  "results": [
    {
      "title": "10-Minute Migas",
      "url": "https://www.seriouseats.com/migas",
      "source": "seriouseats.com",
      "snippet": "Crispy tortillas with eggs and salsa. … Done in 10 minutes.",
      "published_date": "2023-05-01T00:00:00.000Z"
    }
  ]
}
```

`title`, `snippet`, and `published_date` are `null` when Exa doesn't have them.
`snippet` comes from Exa *highlights* — query-relevant excerpts of the page.
No matches is not an error: you get `200` with `"results": []`.

Error responses are always `{"detail": "..."}`:

| status | meaning                                          |
| ------ | ------------------------------------------------ |
| `422`  | invalid request (empty query, `num_results` out of range) |
| `429`  | Exa rate limit hit — retry shortly               |
| `500`  | server misconfigured (bad/missing Exa key)       |
| `502`  | Exa unreachable or returned an unexpected error  |
| `504`  | Exa took longer than `EXA_TIMEOUT_SECONDS` (20s default) |

### `POST /recipes/search`

Same request body as `/search`, but the results come back as **ranked cooking
candidates** via an adaptive pipeline (`src/recipe_search/pipeline.py`):

1. **Plan** — a fast Claude call turns the request (any phrasing or language)
   into 1–3 retrieval-ready Exa queries: implicit goals made concrete,
   negations kept out of retrieval (embeddings can't negate; ranking
   enforces them).
2. **Search** — query variants run concurrently; results are interleaved,
   deduped by URL, capped at 12.
3. **Evaluate** — one Claude call (default `claude-opus-4-8`) judges every
   result against the user's *original* words — usable recipe page or not,
   dish, ingredient overlap, every stated constraint, spam signals — and
   ranks them.
4. **Adapt** — if nothing usable came back, the pipeline retries once,
   feeding the evaluator's own judgments to the planner; a failed planner
   falls back to a static recipe framing, and a failed search variant is
   dropped as long as one succeeds.

```bash
curl -s http://127.0.0.1:8000/recipes/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "I have eggs, salsa, tortillas, and cheese. I want something quick."}'
```

Response (`200`):

```json
{
  "candidates": [
    {
      "title": "10-Minute Migas",
      "url": "https://www.seriouseats.com/migas",
      "source": "seriouseats.com",
      "dish_name": "Tex-Mex migas",
      "fit_score": 0.92,
      "why_it_matches": "Uses eggs, tortillas, salsa, and cheese; quick Tex-Mex dish.",
      "matched_ingredients": ["eggs", "salsa", "tortillas", "cheese"],
      "possibly_missing": ["onion", "cilantro"],
      "role": "best_base_recipe"
    }
  ]
}
```

Candidates are sorted by `role` (`best_base_recipe` → `backup` → `ignore`),
then `fit_score` descending. Exactly one candidate should be
`best_base_recipe`; spammy or unusable pages stay in the list as `ignore` so
you can see *why* they were rejected. Titles/URLs/sources always come from the
search results — the model only returns judgments keyed by result index.

Latency is dominated by the Claude calls (typically 25–60s with the default
model; up to ~2× when the retry fires). Cost is roughly $0.10–0.25 per
request on `claude-opus-4-8`; set
`EVALUATION_MODEL=claude-sonnet-5` in `.env` for a ~2× cheaper drop-in.
Evaluation errors use the same `{"detail": ...}` shape: `500` bad/missing
Anthropic key, `429` Anthropic rate limit, `504` evaluation timeout
(`EVALUATION_TIMEOUT_SECONDS`, default 120), `502` other failures. If no
Anthropic credential is configured at all, `/search` keeps working and this
endpoint returns `500` with a clear message.

### `POST /recipes/recommend`

Same request body again, but the answer is the **user-facing recommendation**:
the pipeline above runs first, then one more Claude call turns the usable
candidates into a warm, source-linked "here's what to cook" answer —
dish + headline, why it fits your ingredients and constraints, missing
items each marked `essential`/`nice_to_have` with substitution notes, 1–2
primary sources (a combination when it genuinely helps), and up to three
alternatives with when-you'd-prefer-them reasons.

Guardrails, enforced structurally: the model picks sources by candidate
index (its prompt contains no URLs), links are merged server-side, ignored
candidates are never offered as sources, and the text points at the
original pages rather than retelling their steps. If nothing usable was
found, `recommendation` is `null` and the honest candidate list is still
returned. Response shape: `{"recommendation": {...} | null, "candidates":
[...]}`. Expect ~40–60s and ~$0.15–0.30 per request on the default model.

### `GET /healthz`

Liveness check, returns `{"status": "ok"}`.

## Reusing the integrations

Each external service lives in one isolated module with no FastAPI imports:
[`exa_search.py`](src/recipe_search/exa_search.py) (web search) and
[`evaluation.py`](src/recipe_search/evaluation.py) (Claude-based candidate
ranking; typed `EvaluationError` subclasses mirror the Exa error design).

```python
from recipe_search.evaluation import RecipeEvaluator

evaluator = RecipeEvaluator(api_key=...)  # or omit to use ANTHROPIC_API_KEY
candidates = await evaluator.evaluate("something quick with eggs", results)
```

All Exa-specific code lives in [`src/recipe_search/exa_search.py`](src/recipe_search/exa_search.py)
and knows nothing about FastAPI. From anywhere else in the app:

```python
from recipe_search.exa_search import ExaSearchClient, ExaSearchError

client = ExaSearchClient(api_key=...)
results = await client.search("something quick with eggs", num_results=5)  # list[SearchResult]
```

Failures raise typed `ExaSearchError` subclasses (`ExaAuthError`,
`ExaRateLimitError`, `ExaTimeoutError`, `ExaAPIError`) so callers can decide
their own handling.

> **Why not the `exa-py` SDK?** As of 2.16.0 it sends requests with no timeout
> (sync) or a hardcoded 600s timeout (async), raises bare `ValueError` for every
> HTTP failure, and pulls in heavy transitive deps (`openai`, `requests`, `tqdm`).
> The documented [REST API](https://exa.ai/docs/reference/search) is one POST
> call, so we make it directly with `httpx` and keep timeouts and error types
> under our control.

## Tests

```bash
uv run pytest
```

## Sharing it publicly (demo mode)

Set `DEMO_MODE=true` in `.env` before exposing the demo. It turns on:

- a **global daily budget** (`DAILY_REQUEST_BUDGET`, default 120 requests/day,
  UTC reset) so your maximum daily spend is a number you chose;
- **per-IP limits** (`IP_REQUESTS_PER_HOUR` / `IP_REQUESTS_PER_DAY`,
  defaults 4/8) to stop casual scripting;
- **hidden API docs** (`/docs` and `/openapi.json` return 404).

Independent of demo mode, the planner now gates topics: requests that
clearly aren't about food (code, homework, general chat) are refused after
one cheap planning call, before any search or evaluation spend. Every
refusal renders as a warm, on-brand state in the UI (off-topic, personal
limit, daily budget), not a raw error.

Limits are in-memory (single process; they reset on restart). If you deploy
behind a proxy, set `TRUST_PROXY_HEADERS=true` so per-IP limits see real
client IPs.

Also do these two things in provider dashboards — they are the real
backstop: create a **separate Anthropic API key in its own workspace with a
monthly spend limit**, and set a **usage cap/alert in the Exa dashboard**.

## Usage stats (optional)

Off by default. Setting `USAGE_DB_PATH` records one SQLite row per request —
a salted IP hash (never the raw address), the query text, the outcome
(recommended dish / null / off-topic / rate-limited / error), and timing —
plus one row per home-page visit with its referer, so you can see which
share drove traffic.

```bash
# .env locally, or Railway variables; generate secrets with `openssl rand -hex 24`
USAGE_DB_PATH=/data/usage.db   # on Railway, mount a volume at /data first
USAGE_SALT=<secret>            # keeps visitor hashes stable across restarts
STATS_TOKEN=<secret>           # enables GET /stats
```

Read it back as aggregates (asks/visits per day, unique visitors, outcomes,
top queries) plus the most recent rows:

```bash
curl -s https://your-app/stats -H "X-Stats-Token: $STATS_TOKEN"
```

Recording is additive: a failure to open or write the database never fails
a request, and with `USAGE_DB_PATH` unset the app behaves exactly as
before. `/stats` answers `404` unless the configured token matches
(constant-time compare), so the endpoint is indistinguishable from
nonexistent to probes. Raw IPs are never stored anywhere.

## Ranking eval

```bash
uv run python scripts/eval_recipes.py        # 10 queries, ~5 min, ~$1.50
uv run python scripts/eval_recipes.py 1 2 3  # subset (1-based)
```

Runs realistic queries through the real pipeline and writes a reviewable
markdown + JSON report to `evals/`, including planned Exa queries, retry
behavior, and a mechanical no-hallucinated-URLs check.

Covers request shape, response normalization, all upstream error mappings, and
endpoint validation — no network required.
