# recipe-search

**Simmer** — tell it what you've got, it tells you what to cook. A FastAPI
service that takes a natural-language food query, searches the web with
[Exa](https://exa.ai), ranks every result via Claude, and answers with a
source-linked cooking recommendation.

Run the server and open **http://127.0.0.1:8000/** for the demo UI — a
single self-contained page (`src/recipe_search/static/index.html`, no build
step, no CDN) with example queries, a camera button beside the submit
(with drag-and-drop and paste on a desktop) that turns fridge photos into
a thumbnail-backed list of removable ingredient chips,
live progress within the input card while the pipeline runs, and the full
recommendation experience: the dish, why it fits, essential vs
nice-to-have missing items, cook-from source cards, and alternatives.
The original input becomes “Ask a question or change your request…” after
the first recommendation and stays visible as you scroll. It remembers
your ingredients and preferences, with the original request shown above it.
Cooking answers appear directly below the input; refinements such as
“Something quicker?” replace the recipe cards and bring the updated dish
into view, without a duplicate summary. Conversation context stays in
browser memory. The secondary “New search” action clears it immediately.

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

Error responses carry a `detail` field: a plain sentence for upstream
failures, FastAPI's structured list for `422` validation errors, and — in
demo mode — an extra `"code"` field on rate-limit refusals:

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
then `fit_score` descending. The prompt asks for at most one
`best_base_recipe`, though the server does not enforce that count; spammy
or unusable pages stay in the list as `ignore` so
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

### `POST /recipes/follow-up`

Continue after a recommendation with either a practical question (“Can I
substitute yogurt?”) or a refinement (“Something quicker?”). Existing
search and recommendation endpoints keep their request and response shapes.

Request body:

| field | type | notes |
| --- | --- | --- |
| `message` | string | required, 1–500 characters; whitespace-only messages rejected |
| `context.current_request` | string | accumulated ingredients and preferences, 1–4,000 characters |
| `context.recommendation_query` | string | request that produced the displayed recommendation, 1–4,000 characters |
| `context.recommendation` | object | displayed recommendation, including its sources and alternatives, in the existing `/recipes/recommend` format |
| `context.exchanges` | array | last 0–6 `{message, reply}` exchanges; messages ≤500 and replies ≤1,000 characters |
| `num_results` | int | optional, 1–10 (default **8**) |

The serialized context is limited to 32,768 UTF-8 bytes. Oversized context
is rejected rather than silently dropping ingredients or constraints.
To start a conversation, set both request fields to the original query,
copy the returned recommendation, and use an empty exchanges list. On
each successful follow-up, send the returned `context` with the next message.

Response (`200`, abbreviated context):

```json
{
  "action": "answer",
  "reply": "Plain yogurt can work; check the linked recipe for the amount and when to add it.",
  "context": {
    "current_request": "…",
    "recommendation_query": "…",
    "recommendation": {"…": "existing recommendation fields"},
    "exchanges": [{"message": "Can I substitute yogurt?", "reply": "…"}]
  },
  "result": null
}
```

One low-effort structured Claude call checks the latest message's topic,
resolves references to the current dish or an alternative, and chooses
`answer` or `search`. Direct answers make no web searches, leave the
recommendation unchanged, and offer concise general cooking advice; exact
amounts and methods remain on the linked recipe pages. Hypothetical
substitution questions do not change the ingredient inventory; explicit
facts such as “I don't have cheese” can update it even in a direct answer.

For `action: "search"`, `result` has the existing
`{"recommendation": {...} | null, "candidates": [...]}` shape. The
recommendation pipeline runs with the accumulated request and retains its
server-side source validation. Earlier constraints remain unless the user
changes them. If no new recommendation fits, context retains the revised
request alongside the earlier recommendation and the request that produced
it; the UI labels that dish as the earlier recommendation.

Only the latest six exchanges are retained as model context. The UI displays
the latest practical answer, or the updated recipe itself for a successful
refinement. It keeps the current dish visible while loading, shows errors
beside the shared input, and preserves the draft and prior context for retry.
“New search” clears the conversation, recipe, draft, and photos immediately,
and prevents any pending response from restoring them. Photos are available
when starting a new search; their ingredients are included only in that
initial request.

Statuses follow `/recipes/recommend`, including `422` for invalid context
or an off-topic latest message and `429` for demo/provider limits. Each
follow-up consumes one demo-limit slot and records one usage event, using
only the latest message as query text, without the supplied conversation.

### Live progress (optional)

Both `/recipes/recommend` and `/recipes/follow-up` accept
`Accept: application/x-ndjson`. Without that header their JSON responses
and HTTP status codes stay unchanged. The browser opts into this stream:

```text
{"type":"progress","stage":"planning"}
{"type":"progress","stage":"searching"}
{"type":"progress","stage":"evaluating"}
{"type":"progress","stage":"recommending"}
{"type":"result","data":<normal endpoint response>}
```

The final `data` contains the endpoint's normal JSON response object. Progress stages
come from actual pipeline transitions: `understanding` for the follow-up
decision, then `planning`, `searching`, `evaluating`, optional `retrying`,
and `recommending` when a usable recipe is found. A direct answer only
emits `understanding` before its result; it does not claim to search.

Validation and demo-limit failures still return ordinary JSON with their
HTTP status. Failures after streaming begins end with
`{"type":"error","status":502,"data":{"detail":"…"}}` instead of a result.
Clients must wait for a terminal result before committing conversation state;
an interrupted stream is a failed request. Disconnecting cancels pending
work. Streaming uses the same single limit admission and usage event.

The input, “New search” control, errors, and progress share one card aligned
with the recipe. The current step includes elapsed time and the latest two
completed steps; slow steps get a gentle explanatory note. No timers invent
progress or claim a percentage complete. Recommendation labels live inside
the dish card, with spacing in place of decorative divider lines.

### `POST /ingredients/from-photo`

Send one to five photos, get back the food Claude can see across them —
the mechanism behind the ask bar's camera button. The UI shows a
thumbnail strip and the detected foods as removable chips (and lets you
**Add more photos** to a running list), then combines the selected chips
with any typed cravings or constraints only when the user searches. The
pipeline itself never sees an image; photos only ever become editable,
reviewable words.

```bash
curl -s http://127.0.0.1:8000/ingredients/from-photo \
  -H 'Content-Type: application/json' \
  -d "{\"images\": [{\"image_base64\": \"$(base64 < fridge.jpg | tr -d '\n')\", \"media_type\": \"image/jpeg\"}]}"
```

Request body: `images`, a list of 1 to 5 photos. Each has `image_base64`
(bare base64, no `data:` prefix, at most 5 MB decoded) and an optional
`media_type` — `image/jpeg` (default), `image/png`, or `image/webp`. All
the photos are analyzed together in a single call, so several photos cost
one request. Every HTTP request body is capped at 8 MiB before FastAPI
buffers or validates it, which also bounds the whole batch.

Response (`200`):

```json
{
  "food_visible": true,
  "ingredients": ["eggs", "cheddar", "kale", "milk"]
}
```

One merged list across every photo: lowercase common names, most
meal-worthy first, deduplicated, at most 40. Photos with no identifiable
food are not an error: `200` with `"food_visible": false` and an empty
list. The images are analyzed in memory and discarded — never stored,
never logged. Errors follow the evaluation family (`413` request body too
large, `422` empty/oversized list or undecodable/oversized/unsupported
image, `429` rate limit, `500` missing/bad Anthropic key, `502`/`504`
upstream trouble). One low-effort vision call: a few seconds and the
cheapest request in the app.

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

Covers every endpoint's request/response behavior (including demo limits,
contextual follow-ups, off-topic refusals, photo-ingredient identification,
and `/stats` auth),
Exa request shape, normalization, and error mapping, Claude
structured-output parsing and merging, pipeline retry and fallback
behavior, and usage recording — all against in-process fakes, no network
required.

Follow-up tests cover direct answers without retrieval, replacement and
no-match searches, bounded context, retained preferences and earlier
recipes, prompt rules for references and hypothetical substitutions,
failure handling, and one usage/limit event per request.

## Sharing it publicly (demo mode)

Set `DEMO_MODE=true` in `.env` before exposing the demo. It turns on:

- a **global daily budget** (`DAILY_REQUEST_BUDGET`, default 120 requests/day,
  UTC reset) so your maximum daily spend is a number you chose;
- **per-IP limits** (`IP_REQUESTS_PER_HOUR` / `IP_REQUESTS_PER_DAY`,
  defaults 4/8) to stop casual scripting;
- **hidden API docs** (`/docs` and `/openapi.json` return 404).

Independent of demo mode, the planner gates topics: requests that
clearly aren't about food (code, homework, general chat) are refused after
one cheap planning call, before any search or evaluation spend. Every
refusal renders as a warm, on-brand state in the UI (off-topic, personal
limit, daily budget), not a raw error.
Follow-ups apply the same limits once per submission, and their lightweight
decision call checks the latest message even inside an existing cooking
conversation. Off-topic follow-ups make no web searches.

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

## Recommendation eval

```bash
uv run python scripts/eval_recipes.py        # 18 queries, ~15 min, ~$3–5
uv run python scripts/eval_recipes.py 1 2 3  # subset (1-based)
```

Runs realistic queries — including the demo UI's example chips, verbatim —
through the real `/recipes/recommend` pipeline and writes a reviewable
markdown + JSON report to `evals/`: planned Exa queries, retry behavior,
and mechanical guardrail checks (every linked URL comes from the retrieved
pool, sources drawn from usable candidates only, 1–2 primary sources, no
primary/alternative overlap, no index vocabulary leaking into prose).
