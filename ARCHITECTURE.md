# recipe-search — Architecture Report

A natural-language food query goes in — *"I have eggs, salsa, tortillas, and
cheese. I want something quick."* A web search runs. Claude reads every
result against the original request and hands back a ranked, cookable
answer, or says honestly that nothing fit. This is a line-by-line account of
how that happens: every module, every prompt, every failure path, every test.

**At a glance**

| | |
|---|---|
| Language | Python 3.12+ |
| Framework | FastAPI 0.139.0 |
| HTTP endpoints | 3 |
| External services | 2 (Exa, Anthropic Claude) |
| Source modules | 6 |
| Tests | 67 / 67 passing (verified live for this report) |
| Database | none — fully stateless |
| Git history | none yet — working tree is entirely untracked |

## Contents

1. [Overview](#1-overview)
2. [Project map](#2-project-map)
3. [Configuration](#3-configuration)
4. [Application lifecycle & dependency injection](#4-application-lifecycle--dependency-injection)
5. [The Exa integration](#5-the-exa-integration)
6. [The Claude evaluation engine](#6-the-claude-evaluation-engine)
7. [The adaptive pipeline](#7-the-adaptive-pipeline)
8. [HTTP API layer](#8-http-api-layer)
9. [Full API reference](#9-full-api-reference)
10. [Test suite](#10-test-suite)
11. [Dependencies & tooling](#11-dependencies--tooling)
12. [Notable engineering decisions](#12-notable-engineering-decisions)

---

## 1. Overview

`recipe-search` is a single-process, fully async FastAPI service with three
routes and no database. Every request is independent — there is no session
state, no cache, no persistence layer anywhere in the codebase. What makes it
more than a search proxy is `POST /recipes/search`, which wraps a raw web
search in an adaptive, Claude-driven pipeline that plans queries, retrieves,
judges, and — if the first attempt comes back empty-handed — tries again
with feedback from its own failure.

| Route | What it does |
|---|---|
| `GET /healthz` | Liveness only. Always `200` while the process is up. |
| `POST /search` | A thin, typed wrapper around one Exa web search call. No model involved — whatever Exa finds, normalized, is what comes back. |
| `POST /recipes/search` | The product: plan → search → evaluate → adapt. Same request shape, but the response is a ranked list of judged cooking candidates. |

**Two request flows, at a glance**

`/search`:
1. Validate request body
2. `ExaSearchClient.search()`
3. Normalize results
4. Return

`/recipes/search`:
1. Validate request body
2. Resolve evaluator (`500` immediately if unconfigured)
3. `find_recipe_candidates()` — plan, fan out searches, merge/dedupe, evaluate, retry once if nothing usable
4. Return ranked candidates

The codebase mirrors this split structurally: `exa_search.py` and
`evaluation.py` are isolated integrations that know nothing about FastAPI or
each other. `pipeline.py` is the only module that imports both. `main.py` is
the only module that imports FastAPI at all. That layering is why the
README can show `ExaSearchClient` and `RecipeEvaluator` being used directly
from a plain Python script — the web framework is an edge, not a foundation.

**Running it locally** (from README.md):

```bash
uv sync
cp .env.example .env        # then paste EXA_API_KEY and ANTHROPIC_API_KEY

uv run recipe-search        # dev server, reload on, http://127.0.0.1:8000
# docs: http://127.0.0.1:8000/docs
```

---

## 2. Project map

Six source files, four test files, and a strict rule about which one is
allowed to know about HTTP.

```
recipe-search/
├── .env.example                # template for secrets — see §3
├── .gitignore                  # ignores __pycache__, .venv, .env, .pytest_cache
├── .python-version             # "3.12"
├── pyproject.toml              # deps, build backend, pytest config
├── uv.lock                     # resolved dependency graph
├── README.md
├── src/recipe_search/
│   ├── __init__.py             # console-script entry point
│   ├── config.py                # Settings — env / .env
│   ├── exa_search.py            # Exa REST client (no FastAPI import)
│   ├── evaluation.py            # Claude planning + judging (no FastAPI import)
│   ├── pipeline.py              # orchestrates the two above (no FastAPI import)
│   └── main.py                  # FastAPI app — the only file that imports it
└── tests/
    ├── test_api.py              # 23 cases
    ├── test_exa_search.py       # 12 cases
    ├── test_evaluation.py       # 23 cases
    └── test_pipeline.py         # 9 cases
```

| Module | Responsibility | Imports FastAPI |
|---|---|---|
| `__init__.py` | Defines `main()`, the target of the `recipe-search` console script. | no |
| `config.py` | One `Settings` class, typed and loaded from env vars / `.env`. | no |
| `exa_search.py` | Everything that talks to Exa: request shape, response normalization, typed errors. | no |
| `evaluation.py` | Everything that talks to Claude: query planning, candidate judging, typed errors. | no |
| `pipeline.py` | The plan → search → evaluate → adapt algorithm. Imports both integrations above. | no |
| `main.py` | Routes, request/response models, dependency injection, upstream-error → HTTP mapping. | **yes** — the only one |

One consequence worth noting for anyone inspecting this checkout directly:
the working tree currently has **no git commits** — every file above is
untracked. This report describes the code exactly as it sits on disk right
now, not a historical snapshot.

---

## 3. Configuration

`src/recipe_search/config.py` — one pydantic-settings class, seven fields,
two required behaviors.

`Settings` extends pydantic-settings' `BaseSettings`, reading from process
environment variables first and falling back to a `.env` file (UTF-8,
unknown keys ignored rather than rejected). Every field name maps to an
environment variable of the same name, upper-cased — no manual aliasing —
which is exactly what `.env.example` assumes.

| Field | Env var | Default | Required |
|---|---|---|---|
| `exa_api_key` | `EXA_API_KEY` | — | **yes** — app refuses to start |
| `exa_base_url` | `EXA_BASE_URL` | `https://api.exa.ai` | no |
| `exa_timeout_seconds` | `EXA_TIMEOUT_SECONDS` | `20.0` | no |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `null` | no — SDK auto-resolves |
| `evaluation_model` | `EVALUATION_MODEL` | `claude-opus-4-8` | no |
| `evaluation_effort` | `EVALUATION_EFFORT` | `medium` | no |
| `evaluation_timeout_seconds` | `EVALUATION_TIMEOUT_SECONDS` | `120.0` | no |

The key/secret fields are typed `SecretStr`, not `str` — Pydantic masks
their value in reprs and logs, so an accidental `print(settings)` or an
uncaught exception traceback can't leak a raw API key. Reaching the real
value requires an explicit `.get_secret_value()` call, which both call
sites in `main.py` make deliberately.

### Two different failure postures

`Settings()` is constructed inside the FastAPI `lifespan` handler (see §4),
which makes `EXA_API_KEY` a **hard requirement**: a missing key raises a
pydantic `ValidationError` at process startup, before uvicorn ever binds
the port. You get a loud, immediate crash rather than a service that
appears healthy until the first search.

`ANTHROPIC_API_KEY` is the opposite by design: it's optional at the
`Settings` level, and constructing the evaluator is wrapped in its own
try/except in `lifespan`. If no Anthropic credential can be resolved at
all, the app logs a warning, sets `app.state.evaluator = None`, and keeps
running — `/search` and `/healthz` stay fully functional, and only
`/recipes/search` starts returning a clear `500` per request. The code
comment in `main.py` calls this out directly: *"Evaluation is additive."*

> **Why `anthropic_api_key` can be null and still work**
>
> When `RecipeEvaluator` isn't given an explicit `api_key`, it constructs
> `anthropic.AsyncAnthropic(api_key=None, ...)` — and the Anthropic SDK
> itself then falls back to the `ANTHROPIC_API_KEY` environment variable,
> or a locally logged-in `ant auth login` profile. `Settings.anthropic_api_key`
> is really just an optional override on top of that SDK-level resolution,
> not the only path to a key.

`evaluation_effort` is passed straight through to the Anthropic API's
`output_config.effort` field, whose valid values are `low`, `medium`,
`high`, `xhigh`, and `max` (confirmed against the installed SDK's own type
definitions). Nothing in this codebase validates the env var against that
list before sending it — an invalid value would only surface as an API
error on the first evaluation call, not at startup.

---

## 4. Application lifecycle & dependency injection

`src/recipe_search/main.py` — how the two clients are built once, shared
across every request, and torn down cleanly.

FastAPI's `lifespan` context manager is the only place `Settings()` is
instantiated. It runs once at process startup, builds the two long-lived
clients, stores them on `app.state`, yields control to the running
application, and closes both clients again on shutdown:

```python
# src/recipe_search/main.py — lifespan()
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()  # fails fast at startup if EXA_API_KEY is missing
    app.state.exa = ExaSearchClient(
        api_key=settings.exa_api_key.get_secret_value(),
        base_url=settings.exa_base_url,
        timeout_seconds=settings.exa_timeout_seconds,
    )
    try:
        app.state.evaluator = RecipeEvaluator(
            api_key=(settings.anthropic_api_key.get_secret_value()
                     if settings.anthropic_api_key else None),
            model=settings.evaluation_model,
            effort=settings.evaluation_effort,
            timeout_seconds=settings.evaluation_timeout_seconds,
        )
    except Exception:
        logger.warning("Recipe evaluation disabled: no Anthropic credentials found "
                        "(set ANTHROPIC_API_KEY)")
        app.state.evaluator = None
    yield
    await app.state.exa.aclose()
    if app.state.evaluator is not None:
        await app.state.evaluator.aclose()
```

Routes never construct their own clients. Two small dependency functions
pull the shared instances back out of `app.state`:

```python
# src/recipe_search/main.py — dependency getters
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
```

This is the exact mechanism behind the README's claim that a misconfigured
evaluator fails cleanly: `get_evaluator` raises `HTTPException(500)` as a
FastAPI dependency, which runs *before* the route body executes. A request
to `/recipes/search` with no Anthropic credentials never reaches
`find_recipe_candidates()` — Exa is never called, no budget is spent, and
the failure is immediate and specific.

Storing clients on `app.state` rather than as module-level globals is also
what makes the test suite possible without touching real network calls:
`tests/test_api.py` swaps in fakes via `app.dependency_overrides[get_search_client]`
and `app.dependency_overrides[get_evaluator]`, so the same route code runs
in tests as in production, against objects that satisfy the same interface
but never leave the process.

---

## 5. The Exa integration

`src/recipe_search/exa_search.py` — the only module that talks to Exa. No
FastAPI import, reusable from a CLI or a job.

> **Why raw httpx instead of the `exa-py` SDK**
>
> The module docstring is explicit about this: as of version 2.16.0, the
> official SDK sends synchronous requests with **no timeout** at all, and
> async requests with a **hardcoded 600-second timeout**. It raises a bare
> `ValueError` for every kind of HTTP failure — auth, rate limit, and
> server error are indistinguishable to a caller — and it pulls in
> `openai`, `requests`, and `tqdm` as transitive dependencies for what is,
> per Exa's own docs, one documented POST endpoint. This file makes that
> one call directly with `httpx` instead, and keeps timeouts and error
> types under the app's own control.

### The normalized result shape

Every raw Exa result is mapped onto one Pydantic model, regardless of which
endpoint eventually consumes it:

| Field | Type | Notes |
|---|---|---|
| `title` | `str \| null` | null if Exa's title is missing or blank |
| `url` | `str` | the only field that can never be null — results without one are dropped |
| `source` | `str` | hostname derived from the URL, `www.` stripped |
| `snippet` | `str \| null` | Exa's *highlights* — query-relevant excerpts — joined with `" … "` |
| `published_date` | `str \| null` | passed through as Exa's raw ISO string |

### Error hierarchy

Four concrete errors, one shared base, one `isinstance`-friendly design:

| Exception | Raised when |
|---|---|
| `ExaSearchError` | base class — never raised directly, used for handler registration |
| `ExaAuthError` | Exa responds `401` or `403` — bad or missing API key |
| `ExaRateLimitError` | Exa responds `429` |
| `ExaTimeoutError` | `httpx.TimeoutException` — no response inside `exa_timeout_seconds` (20s default) |
| `ExaAPIError` | unreachable, any other status ≥ 400, or a response body that isn't the expected shape |

### Request construction and the `search()` walkthrough

`ExaSearchClient` owns one long-lived `httpx.AsyncClient`, built once with
the Exa base URL, an `x-api-key` header, and a single uniform
`httpx.Timeout` covering connect/read/write/pool. A `transport` parameter
can be injected — that seam exists purely so `tests/test_exa_search.py` can
hand it an `httpx.MockTransport` and never touch the network; nothing in
production code passes it.

`search(query, *, num_results=8)` runs, in order:

1. Strips `query`; a blank query raises `ValueError` (a caller-contract bug, not an upstream failure — distinct from the `ExaSearchError` family on purpose).
2. Validates `1 <= num_results <= 100` (Exa's own accepted range — the FastAPI layer imposes a stricter 1–10 policy on top of this; see §8).
3. Builds the request body: `{"query", "type": "auto", "numResults", "contents": {"highlights": true}}`. `type: "auto"` lets Exa itself choose neural vs. keyword retrieval per query; requesting *highlights* rather than full page text is what backs `snippet` and is cheaper than fetching whole pages.
4. POSTs to `/search`. An `httpx.TimeoutException` becomes `ExaTimeoutError`; any other `httpx.HTTPError` (DNS failure, connection refused, …) becomes `ExaAPIError`.
5. Maps the status code: `401`/`403` → auth, `429` → rate limit, anything else `>= 400` logs the first 500 characters of the response body at error level and raises a generic `ExaAPIError` carrying the status code.
6. Parses the JSON body and requires a top-level `results` list; a missing key, unparsable body, or wrong type all become `ExaAPIError`.
7. Maps every raw entry through `_normalize_result`, silently dropping any that come back `None` — one malformed entry never fails the whole request.

```python
# src/recipe_search/exa_search.py — _normalize_result()
def _normalize_result(raw: object) -> SearchResult | None:
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
```

`_clean_str` is what turns a blank or whitespace-only upstream string into
`null` rather than `""` — it's applied to both `title` and
`published_date`, which is exactly why the README can promise those fields
are null, never empty strings, when Exa doesn't have them.
`_source_from_url` pulls the hostname with `urlsplit` and strips a leading
`www.` — `https://www.seriouseats.com/migas` becomes `seriouseats.com`.

One normalization detail worth flagging: Exa's own request contract is
camelCase (`numResults`, `publishedDate`), but every field this service
exposes — in both `SearchResult` and `RecipeCandidate` — is snake_case. The
camelCase boundary stops at this one file.

---

## 6. The Claude evaluation engine

`src/recipe_search/evaluation.py` — the only module that talks to Claude.
Two capabilities, one client, one shared error-mapping chokepoint.

`RecipeEvaluator` does two distinct jobs through the same Anthropic client:
`plan_searches` turns a user's raw request into Exa-ready queries (cheap,
fast), and `evaluate` judges every search result comparatively against
that request (expensive, thorough). Both go through Claude's native
structured-output mechanism — `messages.parse(..., output_format=SomePydanticModel)`
— so the model's response is never hand-parsed JSON; it's a
schema-validated instance of a real Pydantic class by the time this code
sees it.

> **Verified against the installed SDK**
>
> Checked directly against `anthropic==0.116.0`'s source in this project's
> virtualenv: `messages.parse()` takes the `output_format` class, converts
> it to a JSON Schema via `pydantic.TypeAdapter(...).json_schema()`, merges
> that schema into `output_config["format"]` alongside whatever `effort`
> was requested, and sends it as native `output_config` on
> `POST /v1/messages`. A `post_parser` hook then rebuilds
> `response.parsed_output` as a genuine instance of that Pydantic class.
> Nothing in this codebase manually parses model JSON.

### The two system prompts, verbatim

A meaningful share of this system's actual behavior lives here — as
literal English instructions to Claude, not as Python control flow.

**Planner system prompt — `plan_searches()`:**

```text
You write web-search queries for a recipe app backed by Exa, a neural
search engine that matches queries to pages by meaning — it behaves like
text that would naturally precede a shared link. Given a user's food
request, produce search queries that will retrieve cookable recipe pages.

Guidelines, not rules — adapt to the request:
- Phrase each query as a statement that would precede a recipe link, e.g.
  "Here is a great quick weeknight recipe using eggs and tortillas:".
- Make implicit goals concrete in recipe terms (e.g. "high protein and
  fast" → a quick high-protein dinner recipe).
- Neural search cannot handle negation: never mention ingredients,
  equipment, or qualities the user wants to AVOID — the ranking step
  enforces those.
- Keep the user's language; if it isn't English, add one English variant.
- If the request contains several distinct intents, split them across
  queries.
- If feedback from a failed attempt is provided, diagnose why retrieval
  missed and take a genuinely different angle.

Return one to three queries, most promising first.
```

**Evaluation system prompt — `evaluate()`:**

```text
You evaluate web search results as cooking candidates for a recipe app.

You get a user's food request (ingredients on hand, cravings, constraints
like "quick", "vegetarian", or a cuisine like "Indian-ish") and a numbered
list of search results with content excerpts. Judge every result as a recipe
the user could actually cook tonight.

For each result, decide:
- usable_recipe_page: does the page contain an actual cookable recipe
  (ingredients and steps)? Category/listicle pages without a concrete
  recipe, video-only pages, forum threads, and pure blogspam are not usable.
- dish_name: the specific dish the page teaches (e.g. "Tex-Mex migas");
  null if unclear.
- fit_score: 0.0-1.0 — how well this candidate serves the request. Weigh
  ingredient overlap with what the user has, every constraint the user
  states (time, equipment, diet, cuisine, servings, occasion, things to
  avoid), how few important extra ingredients it needs, and source
  quality. Penalize spam signals: keyword stuffing, machine-generated
  filler, invisible/zero-width characters in the text, incoherent or
  repetitive prose.
- matched_ingredients: ingredients the user mentioned that the recipe uses,
  in the user's own words, lowercase. Empty if the user named no
  ingredients — goals or constraints like "high protein" are not
  ingredients.
- possibly_missing: important ingredients the recipe needs that the user
  did not mention. Skip pantry staples (salt, pepper, water, common oil).
- why_it_matches: one concrete sentence a cook would find useful.
- role: "best_base_recipe" for the single strongest candidate (at most one
  result, and only if genuinely usable), "backup" for solid alternatives,
  "ignore" for results that are unusable, off-request, or untrustworthy.

Keep fit_score consistent with role: the best_base_recipe has the highest
score, backups below it, ignores lowest. Evaluate every result index exactly
once. Interpret vague requests generously — surface the most promising
cookable matches rather than rejecting everything.
```

### The anti-hallucination boundary: two schemas, not one

The single most consequential design choice in this file is what the model
is — and isn't — allowed to return. Claude's structured output is
constrained to `_CandidateEvaluation`, an internal, index-keyed schema:

```python
# src/recipe_search/evaluation.py — the model-facing schema
class _CandidateEvaluation(BaseModel):
    """Claude's judgment of one search result, keyed by its index."""
    index: int
    usable_recipe_page: bool
    dish_name: str | None
    fit_score: float
    why_it_matches: str
    matched_ingredients: list[str]
    possibly_missing: list[str]
    role: Role
```

Notice what's *not* there: no `title`, no `url`, no `source`. The model can
only ever refer to a search result by its numeric index. The public-facing
`RecipeCandidate` that the API actually returns is assembled server-side in
`_merge_and_rank`, which pulls `title`/`url`/`source` from the original,
trusted `SearchResult` — never from anything Claude said. This isn't just
tidy plumbing: it means adversarial content on a page (prompt injection, a
fake "click this link instead" instruction hidden in the excerpt) has no
schema field through which it could redirect a user to a different URL. The
model can mis-judge a page; it cannot make up a link.

`_EvaluationOutput` is a one-field wrapper — `{evaluations: list[_CandidateEvaluation]}`
— that exists because the Anthropic structured-output contract needs a
single JSON object at the schema root, not a bare array.

### `plan_searches()` and `evaluate()` side by side

Both methods fall through the same private helper, `_parse_structured`, but
are tuned very differently for the very different jobs they do:

| | `plan_searches()` | `evaluate()` |
|---|---|---|
| `max_tokens` | 1,000 | 16,000 |
| `thinking` | omitted entirely | `{"type": "adaptive"}` |
| `output_config.effort` | `"low"` (hardcoded) | settings-configurable, default `"medium"` |
| `output_format` | `SearchPlan` | `_EvaluationOutput` |
| Skips the model call when… | never | results list is empty |

The planner runs with no extended-thinking budget at all — confirmed
directly by a test asserting `"thinking" not in call` — because rewriting a
query into 1–3 retrieval strings doesn't benefit from deliberation the way
judging a dozen full-page excerpts against every stated constraint does.
The evaluator's `thinking: adaptive` lets Claude itself decide how much
internal reasoning the judgment deserves, rather than a fixed token budget
for every call regardless of how ambiguous the request is.

`plan_searches` also cleans up after the model: it strips every returned
query, drops blanks and exact duplicates while preserving order, raises
`EvaluationAPIError` if nothing survives, and caps the result to the first
three queries.

`evaluate` guards a real cost lever, too: `_MAX_SNIPPET_CHARS = 10_000`
truncates any one excerpt before it reaches the prompt, appending
`" …[truncated]"`. The comment in the source is explicit that this is a
guardrail against pathological pages, not a token budget — real Exa
highlights run roughly 2,500–3,000 characters, so ordinary results never
come close to it (a test sends a realistic ~2,700-character snippet
through untouched, and a separate test confirms the truncation boundary
lands at exactly the 10,000th character).

### The shared error-mapping chokepoint

Every Anthropic SDK exception `_parse_structured` can see is mapped to one
of this module's own typed errors — the same hierarchy shape as
`exa_search.py`'s:

| Anthropic SDK exception | Mapped to |
|---|---|
| `AuthenticationError` | `EvaluationAuthError` |
| `PermissionDeniedError` | `EvaluationAuthError` (key valid, lacks model access) |
| `RateLimitError` | `EvaluationRateLimitError` |
| `APITimeoutError` | `EvaluationTimeoutError` |
| `APIConnectionError` | `EvaluationAPIError` |
| `APIStatusError` (any other non-2xx) | `EvaluationAPIError` (logged with status + message) |
| `pydantic.ValidationError` | `EvaluationAPIError` (output didn't match the schema) |
| `stop_reason == "max_tokens"` | `EvaluationAPIError` (truncated mid-generation) |
| `parsed_output is None` | `EvaluationAPIError` (e.g. `stop_reason="refusal"`) |

The real `anthropic.AsyncAnthropic` client (built only when no client is
injected) is constructed with `max_retries=1` — deliberately small. The
pipeline layer above already owns a retry/adapt strategy of its own (§7);
letting the SDK also retry aggressively underneath it would let two retry
policies compound and quietly blow through the 120-second evaluation
timeout budget.

### Merging the model's judgment back onto trusted data

```python
# src/recipe_search/evaluation.py — _merge_and_rank()
def _merge_and_rank(results, evaluations):
    candidates: dict[int, RecipeCandidate] = {}
    for evaluation in evaluations:
        index = evaluation.index
        if not 0 <= index < len(results):
            logger.warning("Dropping evaluation for unknown result index %s", index)
            continue
        if index in candidates:
            logger.warning("Duplicate evaluation for result index %s; keeping the first", index)
            continue
        result = results[index]
        role = evaluation.role
        if not evaluation.usable_recipe_page and role != "ignore":
            role = "ignore"
        candidates[index] = RecipeCandidate(
            title=result.title, url=result.url, source=result.source,
            dish_name=evaluation.dish_name,
            fit_score=min(max(evaluation.fit_score, 0.0), 1.0),
            why_it_matches=evaluation.why_it_matches,
            matched_ingredients=evaluation.matched_ingredients,
            possibly_missing=evaluation.possibly_missing,
            role=role,
        )
    ...
    return sorted(candidates.values(),
                  key=lambda c: (_ROLE_RANK[c.role], -c.fit_score))
```

Four defensive behaviors happen here, all covered directly by tests:

- **Out-of-range indexes are dropped**, not errored on — a model hallucinating index 5 against a 1-result list just loses that one judgment.
- **Duplicate indexes keep only the first** occurrence; later ones are dropped with a warning.
- **`usable_recipe_page: false` always forces `role = "ignore"`**, even if the model separately said `"backup"` — a code-level consistency check that doesn't fully trust the model's own internal consistency.
- **`fit_score` is clamped to `[0.0, 1.0]`** in code, even though nothing in the schema itself enforces that bound.

If the model evaluates fewer indexes than there were results, the gap is
logged as a warning and the response simply contains fewer candidates —
there's no placeholder synthesis and no hard failure for a partially-lazy
model response. The final sort key is `(role_rank, -fit_score)`: role first
(`best_base_recipe` → `backup` → `ignore`), fit_score descending within
each tier.

---

## 7. The adaptive pipeline

`src/recipe_search/pipeline.py` — the orchestration layer that turns two
isolated integrations into one adaptive product.

The module docstring states the design philosophy plainly: robustness
comes from *judgment* at each choke point rather than hardcoded rules —
Claude plans the Exa queries regardless of phrasing, language, or
vagueness; the retrieval pool is fanned out across query variants and
deduplicated; and when evaluation judges the whole pool unusable, the
pipeline retries exactly once, feeding the evaluator's own reasoning back
into the next planning call. Failures degrade instead of cascading: a
planner failure falls back to a static query template, and one failed
search variant is tolerated as long as another succeeds.

**Plan → Search → Evaluate → Adapt**

| Stage | What happens | Notes |
|---|---|---|
| 1. Plan | Claude turns the user's raw request into 1–3 retrieval-ready Exa queries. | 1 call · effort: low · no thinking budget · fallback: static template |
| 2. Search | Every planned query runs against Exa concurrently; pools interleave, dedupe by URL, cap at 12. | N calls, parallel · 1 failed variant tolerated · 0 successes → raises |
| 3. Evaluate | One Claude call judges the merged pool against the user's *original* words. | 1 call · effort: configurable · thinking: adaptive · empty pool → skipped |
| 4. Adapt | Nothing usable? Retry once — same three stages, seeded with the evaluator's own reasons. | 0 or 1 retry, never more · excludes URLs already seen |

The retry (Adapt → Plan) happens **at most once** per request. If the retry
*also* comes back with nothing usable, the pipeline does not keep trying —
it returns the **first** attempt's honest ranking, on the reasoning that a
second unusable ranking is no more trustworthy than the first, and an
unbounded retry loop has no defined stopping point.

### `find_recipe_candidates()` — the public entry point

```python
# src/recipe_search/pipeline.py
async def find_recipe_candidates(query, *, num_results, exa, evaluator):
    first, seen_urls = await _attempt(query, num_results=num_results, exa=exa, evaluator=evaluator)
    if _has_usable(first):
        return first

    feedback = _failure_feedback(first)
    logger.info("No usable candidates; retrying. Feedback: %s", feedback)
    retry, _ = await _attempt(query, num_results=num_results, exa=exa, evaluator=evaluator,
                               feedback=feedback, exclude_urls=seen_urls)
    return retry if _has_usable(retry) else first
```

This is the only function `main.py` calls. It's also, notably, the only
place in the entire codebase where a retry decision is made — neither
`exa_search.py` nor `evaluation.py` retries anything on its own (the
Anthropic client's own `max_retries=1` covers transient transport hiccups,
not semantic failure).

### Inside one attempt

`_attempt()` is the same four-line sequence whether it's the first try or
the retry: plan queries, run searches, merge the pool, evaluate. The one
detail worth pulling out explicitly — because it's easy to miss and it's
load-bearing — is **what gets evaluated**:

> `evaluator.evaluate(query, pool)` is always called with the user's
> original, untouched request text — never with the planner's rewritten
> Exa queries. Retrieval and ranking are deliberately decoupled: the
> planner is free to phrase "Here is a great quick weeknight recipe using
> eggs and tortillas:" to retrieve well, but the evaluator judges every
> candidate against what the user actually typed, constraints and all.

Two smaller functions handle the two ways retrieval can partially fail:

- **`_plan_queries`** catches any `EvaluationError` from the planner and falls back to a single query: `"Here is a great home-cooked recipe: {original query}"` — a total planner outage still produces one Exa-shaped query, so the pipeline degrades rather than failing outright.
- **`_run_searches`** runs every planned query concurrently via `asyncio.gather(..., return_exceptions=True)`. If every variant fails, it re-raises the *first* failure (a real `ExaSearchError` subclass propagates all the way up to the FastAPI error handler and gets the correct status code). If only some fail, the failures are logged as warnings and the pipeline proceeds on whatever succeeded.

### Merging pools: round-robin, not concatenation

```python
# src/recipe_search/pipeline.py — _merge_pools()
def _merge_pools(pools, *, exclude_urls):
    merged: list[SearchResult] = []
    seen = set(exclude_urls)
    for tier in itertools.zip_longest(*pools):
        for result in tier:
            if result is None or result.url in seen:
                continue
            seen.add(result.url)
            merged.append(result)
            if len(merged) == _MAX_POOL_SIZE:
                return merged
    return merged
```

`itertools.zip_longest(*pools)` walks the pools in lockstep — first result
of query 1, first of query 2, second of query 1, second of query 2, and so
on — rather than exhausting one query's results before moving to the next.
A strong hit from the second planned query is never buried behind eleven
mediocre results from the first. The same pass drops anything already seen
(cross-pool duplicates, since one page can legitimately match two
different query variants) or already excluded (URLs a prior attempt
already showed the evaluator), and stops the moment the pool hits
`_MAX_POOL_SIZE = 12` — capping both the evaluation call's input size and
its cost.

### What triggers a retry

```python
# src/recipe_search/pipeline.py
def _has_usable(candidates: list[RecipeCandidate]) -> bool:
    return any(candidate.role != "ignore" for candidate in candidates)

def _failure_feedback(candidates: list[RecipeCandidate]) -> str:
    if not candidates:
        return "The search returned no results at all."
    reasons = "; ".join(c.why_it_matches for c in candidates[:3])
    return f"Every result was judged unusable. Sample judgments: {reasons}"
```

An empty candidate list correctly counts as "not usable" — `any([])` is
`False` — so a zero-result first pool triggers a retry exactly like an
all-`ignore` pool does, just with a different feedback string. In the
all-`ignore` case, the feedback is literally the evaluator's own
`why_it_matches` text from up to three candidates, fed verbatim into the
next `plan_searches` call — this is the concrete mechanism behind the
planner prompt's instruction to "diagnose why retrieval missed and take a
genuinely different angle."

Worst case per request: 2 planning calls, up to two rounds of up to 3
concurrent Exa searches each, and 2 evaluation calls — matching the
README's note that a retry costs "up to ~2×" the base latency and, on the
default `claude-opus-4-8` model, roughly $0.10–0.25 per request.

---

## 8. HTTP API layer

`src/recipe_search/main.py` — the only file in the project that knows what
an HTTP status code is.

### Request validation

Both endpoints share one request model. Its field constraints are stricter
than what the underlying clients themselves would accept — a deliberate
split between *API policy* (enforced here) and *client capability*
(enforced in `exa_search.py`):

```python
# src/recipe_search/main.py — SearchRequest
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, ...)
    num_results: int = Field(default=8, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty or whitespace")
        return value
```

`num_results` is capped at 10 here even though `ExaSearchClient.search()`
itself accepts up to 100 — the API's public contract is intentionally
narrower than what the client underneath it can technically do. The custom
validator is also why a whitespace-only query (`"   "`) is rejected with
`422` rather than silently passed through as a "valid" one-character-after-strip
string.

### One handler, eight exception types

Every upstream failure — from either Exa or Claude — is mapped to an HTTP
response through a single ordered table and one registered handler:

```python
# src/recipe_search/main.py — _UPSTREAM_ERRORS
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
async def handle_upstream_error(request, exc):
    status, detail = next((code, message) for exc_type, code, message in _UPSTREAM_ERRORS
                           if isinstance(exc, exc_type))
    if status in (500, 502):
        logger.error("%s: %s", type(exc).__name__, exc)
    return JSONResponse(status_code=status, content={"detail": detail})
```

The ordering is what makes this correct, not just tidy. `ExaAuthError`,
`ExaRateLimitError`, and `ExaTimeoutError` are all direct subclasses of
`ExaSearchError` — so is the catch-all row itself. Because `next(...)`
returns the *first* matching row and `isinstance` against the base class
would match all four, the three specific rows have to come before the
generic `ExaSearchError` row, or every Exa failure would silently collapse
to `502`. The same constraint applies to the four `Evaluation*` rows
beneath them. One handler function, registered against just the two base
exception types, correctly covers all eight concrete error classes because
FastAPI dispatches by `isinstance`, not exact type.

Only the `500`/`502` tier gets logged at error level — `429` and `504` are
treated as expected, operational conditions rather than bugs worth an
error-level log line.

### The three routes

| Route | Dependencies | Body |
|---|---|---|
| `POST /search` | `get_search_client` | `exa.search(query, num_results)` → `SearchResponse` |
| `POST /recipes/search` | `get_search_client`, `get_evaluator` | `find_recipe_candidates(...)` → `RecipeSearchResponse` |
| `GET /healthz` | none | `{"status": "ok"}` |

Because `get_evaluator` raises inside dependency resolution, an
unconfigured-evaluator request to `/recipes/search` never executes the
route body at all — `find_recipe_candidates` is never called, and neither
Exa nor Claude is ever contacted for that request.

---

## 9. Full API reference

Every field, every status code, worked examples pulled straight from the
project's own README and test fixtures.

### `GET /healthz`

No request body. Always `200` `{"status": "ok"}` while the process is
alive — no dependency on Exa or Anthropic.

### `POST /search`

| Field | Type | Constraints |
|---|---|---|
| `query` | string | required, 1–500 characters after trimming |
| `num_results` | integer | optional, 1–10, default 8 |

```bash
curl -s http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "I have eggs, salsa, tortillas, and cheese. I want something quick.", "num_results": 8}'
```

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

No matches is not an error — a query that finds nothing returns `200` with
`{"results": []}`.

| Status | Meaning |
|---|---|
| `200` | results returned (possibly empty) |
| `422` | invalid request — empty/whitespace query, query over 500 chars, or num_results outside 1–10 |
| `429` | Exa rate limit — retry shortly |
| `500` | server misconfigured — bad or missing EXA_API_KEY |
| `502` | Exa unreachable or returned an unexpected error |
| `504` | Exa took longer than EXA_TIMEOUT_SECONDS (20s default) |

### `POST /recipes/search`

Same request body as `/search`. The response is the output of the full
adaptive pipeline from §7.

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

| Field | Meaning |
|---|---|
| `dish_name` | the specific dish the page teaches, or null if unclear |
| `fit_score` | 0.0–1.0, clamped server-side regardless of what the model returned |
| `why_it_matches` | one concrete sentence explaining the judgment |
| `matched_ingredients` | the user's own words, lowercase; empty if none were named |
| `possibly_missing` | important ingredients the recipe needs that weren't mentioned; pantry staples excluded |
| `role` | `best_base_recipe` (at most one) · `backup` · `ignore` |

Candidates are sorted by role first, then `fit_score` descending. Spammy or
unusable pages stay in the response as `ignore` rather than being removed —
the point is to show *why* a result was rejected, not to hide it.

| Status | Meaning |
|---|---|
| `200` | candidates returned (possibly empty, possibly all "ignore") |
| `422` | same request validation as /search |
| `429` | Exa *or* Anthropic rate limit |
| `500` | bad/missing Exa key, bad/missing Anthropic key, **or** no Anthropic credential configured at all |
| `502` | Exa or Anthropic upstream failure |
| `504` | Exa timeout (20s default) *or* evaluation timeout (EVALUATION_TIMEOUT_SECONDS, 120s default) |

> **Latency and cost.** Dominated entirely by the Claude calls: typically
> 25–60 seconds on the default model, up to roughly 2× when the retry
> fires. Cost runs about $0.10–0.25 per request on `claude-opus-4-8`;
> setting `EVALUATION_MODEL=claude-sonnet-5` is a documented drop-in that's
> roughly half the cost.

---

## 10. Test suite

67 cases across 4 files, 0 network calls, verified live for this report.

Running `uv run pytest -q` against this checkout right now produces
**67 passed** in about 2.4 seconds, plus one pre-existing
`StarletteDeprecationWarning` about `TestClient`'s use of `httpx`
(Starlette suggests installing `httpx2` instead) — harmless, and unrelated
to this project's own code.

Every external boundary is faked, never mocked-at-a-distance:
`test_exa_search.py` injects an `httpx.MockTransport` through the client's
own `transport=` constructor parameter; `test_evaluation.py` injects a
hand-written fake Anthropic client through `RecipeEvaluator(client=...)`;
`test_api.py` swaps whole fake clients in via FastAPI's
`app.dependency_overrides`; and `test_pipeline.py` drives
`find_recipe_candidates` against hand-written stubs that let a single test
script an exact sequence of responses across a first attempt and a retry.

### `tests/test_api.py` — 23 cases

| Test | What it proves |
|---|---|
| `test_search_returns_normalized_results` | full round trip, exact JSON shape, exact call args recorded on the fake |
| `test_search_passes_num_results` | num_results is threaded through to the client unchanged |
| `test_search_with_no_results` | empty list is 200, not an error |
| `test_search_rejects_invalid_requests` ×6 | missing/empty/whitespace query, num_results 0 or 11, 501-char query — all 422, client never called |
| `test_search_maps_upstream_errors` ×4 | Auth→500, RateLimit→429, Timeout→504, generic→502 |
| `test_healthz` | 200, `{"status": "ok"}` |
| `test_recipes_search_returns_ranked_candidates` | full round trip incl. planner call, Exa call, evaluator call, and exact response shape |
| `test_recipes_search_with_no_results` | 200, `{"candidates": []}` |
| `test_recipes_search_rejects_invalid_requests` | 422 before the evaluator is ever touched |
| `test_recipes_search_maps_evaluation_errors` ×4 | same 4-way status mapping, for the Evaluation* error family |
| `test_recipes_search_maps_exa_errors_too` | the same route also correctly maps Exa-layer failures; evaluator never called |
| `test_recipes_search_without_configured_evaluator` | `app.state.evaluator = None` → 500, "not configured" in detail |

### `tests/test_exa_search.py` — 12 cases

| Test | What it proves |
|---|---|
| `test_search_sends_documented_request_shape` | exact POST path, x-api-key header, exact JSON body against Exa's documented contract |
| `test_search_normalizes_results` | malformed entries (no url, non-dict) dropped; highlight-join and www.-stripping verified |
| `test_search_returns_empty_list_when_no_results` | — |
| `test_search_maps_http_errors` ×5 | 400→API, 401→Auth, 403→Auth, 429→RateLimit, 500→API |
| `test_search_maps_timeouts` | `httpx.ReadTimeout` → `ExaTimeoutError` |
| `test_search_maps_connection_errors` | `httpx.ConnectError` → `ExaAPIError` |
| `test_search_rejects_unexpected_body` | missing "results" key → `ExaAPIError` |
| `test_search_validates_arguments` | blank query, num_results 0, num_results 101 → `ValueError`, no HTTP call made |

### `tests/test_evaluation.py` — 23 cases

| Test | What it proves |
|---|---|
| `test_evaluate_merges_and_ranks` | mixed roles sort correctly; title/url/source proven to come from the result, not the model |
| `test_evaluate_sends_query_and_indexed_results` | exact call kwargs: model, thinking, output_config, output_format, rendered prompt content |
| `test_evaluate_ranks_backups_by_score` | pure fit_score ordering within one role |
| `test_evaluate_clamps_scores` | 1.7 → 1.0, −0.2 → 0.0 |
| `test_evaluate_drops_unknown_and_duplicate_indexes` | duplicate keeps first; out-of-range and negative indexes dropped |
| `test_unusable_page_is_forced_to_ignore` | usable_recipe_page=false overrides a model-claimed "backup" role |
| `test_oversized_snippets_are_truncated_with_marker` | precise 10,000-char boundary + truncation marker |
| `test_normal_snippets_are_sent_in_full` | a realistic ~2.7k-char snippet is sent untouched |
| `test_empty_results_skip_the_model_call` | zero Anthropic calls made for an empty result list |
| `test_blank_query_raises` | — |
| `test_plan_searches_cleans_and_caps_queries` | 6 messy inputs → 3 clean, unique, ordered queries; confirms "thinking" is entirely absent from the call |
| `test_plan_searches_includes_feedback` | — |
| `test_plan_searches_rejects_empty_plan` | all-blank queries → `EvaluationAPIError` |
| `test_plan_searches_maps_errors_via_shared_path` | proves plan_searches and evaluate share `_parse_structured` |
| `test_plan_searches_blank_query_raises` | — |
| `test_evaluate_maps_anthropic_errors` ×6 | 401/403→Auth, 429→RateLimit, 500→API, APITimeoutError→Timeout, APIConnectionError→API |
| `test_truncated_output_raises` | `stop_reason="max_tokens"` |
| `test_missing_parsed_output_raises` | `stop_reason="refusal"`, `parsed_output=None` |

### `tests/test_pipeline.py` — 9 cases

| Test | What it proves |
|---|---|
| `test_fans_out_interleaves_and_dedupes` | 2 queries → round-robin interleaved, cross-pool duplicate removed |
| `test_pool_is_capped` | 15 results in, evaluator sees exactly 12 |
| `test_planner_failure_falls_back_to_template` | planner error → Exa called with the literal fallback template string |
| `test_one_failed_search_variant_is_tolerated` | 1 of 2 query variants fails; pipeline proceeds on the survivor |
| `test_all_failed_searches_raise` | both variants fail → pipeline raises; evaluator never called |
| `test_unusable_pool_retries_with_feedback_and_exclusions` | retry feedback contains the real judgment text; re-surfaced URL excluded from the retry pool |
| `test_retry_still_unusable_returns_first_attempt` | both attempts unusable → returns the FIRST attempt specifically |
| `test_empty_pool_retries_with_no_results_feedback` | zero-result pool triggers retry with the exact "no results at all" string |
| `test_no_retry_when_first_attempt_is_usable` | fast path: exactly 1 plan call + 1 evaluate call, no wasted round trip |

---

## 11. Dependencies & tooling

Managed end to end by `uv`; versions below are what's actually resolved and
installed in this checkout's virtualenv, not just the floor pinned in
`pyproject.toml`.

**Runtime**

| Package | Installed | Role |
|---|---|---|
| `anthropic` | 0.116.0 | Claude client — structured outputs, adaptive thinking, typed errors |
| `fastapi` | 0.139.0 | HTTP layer, routing, dependency injection, request validation |
| `httpx` | 0.28.1 | async HTTP client — hand-rolled Exa integration |
| `pydantic-settings` | 2.14.2 | typed env/.env configuration (config.py) |
| `uvicorn[standard]` | 0.49.0 | ASGI server; "standard" pulls in uvloop, httptools, watchfiles, websockets |

**Dev-only** (declared under `[dependency-groups]`)

| Package | Installed | Role |
|---|---|---|
| `pytest` | 9.1.1 | test runner |
| `pytest-asyncio` | 1.4.0 | lets every `async def test_*` run as a coroutine test automatically |

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

`asyncio_mode = "auto"` means no test needs a `@pytest.mark.asyncio`
decorator — every `async def test_...` function across all four files is
picked up and awaited automatically. `asyncio_default_fixture_loop_scope = "function"`
gives each test function its own fresh event loop, so async fixtures in
one test can't leak state into another.

The package builds with `uv_build` (declared under `[build-system]`),
targets `requires-python >= 3.12`, and `.python-version` pins local tooling
to exactly `3.12`. The console script `recipe-search = "recipe_search:main"`
is what `uv run recipe-search` resolves to — and that entry point (§2)
binds to `127.0.0.1` with `reload=True`, which marks it clearly as a
**development** launcher. A production deployment would run
`uvicorn recipe_search.main:app` directly, without `reload`, under a
process manager.

---

## 12. Notable engineering decisions

The choices that show up as behavior, not just as code — gathered in one
place.

1. **Raw httpx over the exa-py SDK.** The official SDK has no timeout on sync calls, a hardcoded 600s on async calls, raises bare `ValueError` for every failure mode, and pulls in openai/requests/tqdm for one documented POST endpoint.
2. **The model can't return a URL.** Claude's structured-output schema for evaluation is index-only — no title/url/source fields exist for it to fill in. Those are always merged back from the trusted search result server-side.
3. **Planning skips "thinking"; evaluating doesn't.** Query planning omits the `thinking` parameter entirely for latency. Evaluation uses `thinking: adaptive`, letting Claude size its own reasoning budget to how ambiguous the request is.
4. **`max_retries=1` on the Anthropic client.** The pipeline already owns a retry/adapt strategy. A larger SDK-level retry budget underneath it could compound and quietly exceed the evaluation timeout.
5. **Evaluation is additive, not required.** No Anthropic credential anywhere in the environment disables only `/recipes/search`, discovered per-request via a dependency. `/search` and `/healthz` are unaffected.
6. **Round-robin pool merging.** Multi-query results interleave instead of concatenating, so a strong second-query hit is never buried behind a first query's twelfth-best result.
7. **Retrieval and ranking are decoupled.** Exa sees the planner's rewritten queries; the evaluator always judges the user's original words. Different jobs, different text, same underlying request.
8. **Negation lives in ranking, not retrieval.** The planner prompt is explicitly told never to phrase exclusions into a query, because neural search embeddings can't represent "not X" — only the evaluation step enforces what to avoid.
9. **Exactly one retry, never a loop.** `find_recipe_candidates` tries at most twice and always has a defined answer — the honest first ranking — rather than retrying indefinitely with no stopping condition.
10. **Injection seams exist only for tests.** `ExaSearchClient`'s `transport=` and `RecipeEvaluator`'s `client=` parameters are never used by production code — `main.py` never passes them. They exist so the test suite can run with zero network calls.

---

*Compiled by reading every source and test file in this checkout,
cross-checking the Anthropic SDK's own installed source for
`messages.parse` and `output_config`, and running `uv run pytest -q` live
against the working tree (67 passed, 1 pre-existing warning). No git
history exists yet for this project — every file is currently untracked.*
