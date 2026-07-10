"""Internal eval for the /recipes/recommend pipeline.

Runs realistic cooking queries through the real pipeline (in-process — the
same `recommend_recipe` the route calls) and writes a reviewable markdown
report plus raw JSON to evals/. Covers both layers: the ranked candidates
and the user-facing recommendation, with mechanical guardrail checks
(sources drawn from the usable pool only, no index vocabulary leaking into
prose, primary/alternative hygiene).

Usage:
    uv run python scripts/eval_recipes.py           # all queries (~15 min)
    uv run python scripts/eval_recipes.py 1 2 3     # a subset, 1-based

Each query costs roughly $0.15-0.30 in Claude + Exa usage.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from recipe_search.config import Settings
from recipe_search.evaluation import (
    Recommendation,
    RecipeCandidate,
    RecipeEvaluator,
    SearchPlan,
)
from recipe_search.exa_search import ExaSearchClient, SearchResult
from recipe_search.pipeline import OffTopicQuery, recommend_recipe

ROOT = Path(__file__).resolve().parents[1]

QUERIES = [
    "I have eggs, salsa, tortillas, and cheese. I want something quick.",
    "Something with chicken and eggs? Indian-ish.",
    "I have tofu, spinach, rice, and soy sauce. No oven.",
    "Use up yogurt and chickpeas. Vegetarian dinner.",
    "Something high protein with canned tuna and eggs.",
    "I have pasta, cream cheese, frozen peas, and no meat.",
    "I want something Korean-ish with ground beef and rice.",
    "I have mushrooms, eggs, and leftover rice. Fast lunch.",
    "I want dessert but only have bananas, oats, and peanut butter.",
    "I have chicken thighs but no onions or garlic.",
    "Vegan and gluten free dinner with lentils.",
    "I'm allergic to nuts. Quick Thai-style noodles?",
    "Need an appetizer and a main for a date night dinner.",
    "asdfghjkl qwerty",
    # The demo UI's example chips, verbatim (static/index.html) — keep in sync
    # so every front-door example stays covered by this eval.
    "Kimchi, leftover rice, and eggs. Something fast",
    "Chicken thighs and a jar of peanut butter?",
    "Two brown bananas and a bag of oats. Something sweet?",
    "Broccoli, some cashews, and a bunch of mint. Some Indian dish?",
]

REVIEW_CHECKLIST = """\
Manual review questions (per query):
- Is the recommended dish actually a good call, and is the tone a helpful
  cooking assistant (not a search page)?
- Are essential vs nice_to_have judgments culinarily sane? Substitution
  notes helpful?
- Do primary sources / combinations make sense? Alternatives have real
  when-you'd-prefer-them reasons?
- Are constraints (negations, allergies, diet, equipment, occasion,
  language) respected end to end?
- Is the system honest when the request is vague or gibberish?
"""


class RecordingEvaluator:
    """Pass-through wrapper that records plans and evaluation pools."""

    def __init__(self, inner: RecipeEvaluator):
        self._inner = inner
        self.attempts: list[dict] = []
        self.pools: list[list[SearchResult]] = []

    async def plan_searches(self, query: str, *, feedback: str | None = None) -> SearchPlan:
        plan = await self._inner.plan_searches(query, feedback=feedback)
        self.attempts.append({"feedback": feedback, "queries": plan.queries})
        return plan

    async def evaluate(self, query: str, results: list[SearchResult]):
        self.pools.append(list(results))
        return await self._inner.evaluate(query, results)

    async def recommend(self, query: str, candidates: list[RecipeCandidate]):
        return await self._inner.recommend(query, candidates)


def _check_recommendation(
    rec: Recommendation, candidates: list[RecipeCandidate]
) -> dict[str, bool]:
    usable_urls = {c.url for c in candidates if c.role != "ignore"}
    primary_urls = {s.url for s in rec.primary_sources}
    alternative_urls = {a.recipe.url for a in rec.alternatives}
    prose = " ".join(
        [rec.headline, rec.why_it_fits, rec.how_to_use_sources]
        + [a.reason for a in rec.alternatives]
        + [m.note or "" for m in rec.missing_items]
    ).lower()
    return {
        "sources_from_usable_pool": primary_urls <= usable_urls
        and alternative_urls <= usable_urls,
        "no_index_leak_in_prose": "candidate [" not in prose,
        "primary_count_ok": 1 <= len(rec.primary_sources) <= 2,
        "no_primary_alt_overlap": not (primary_urls & alternative_urls),
    }


async def run_query(
    number: int, query: str, exa: ExaSearchClient, evaluator: RecipeEvaluator
) -> dict:
    recorder = RecordingEvaluator(evaluator)
    start = time.time()
    off_topic = False
    try:
        recommendation, candidates = await recommend_recipe(
            query, num_results=8, exa=exa, evaluator=recorder
        )
        error = None
    except OffTopicQuery:  # the expected outcome for non-food queries
        recommendation, candidates, error, off_topic = None, [], None, True
    except Exception as exc:  # keep the eval going; report the failure
        recommendation, candidates, error = None, [], f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - start

    pool_urls = {r.url for pool in recorder.pools for r in pool}
    return {
        "number": number,
        "query": query,
        # Fewer attempts than pools => the retry planner failed and the
        # pipeline fell back to the static template.
        "attempts": recorder.attempts,
        "retried": len(recorder.pools) > 1,
        "off_topic": off_topic,
        "pool_sizes": [len(pool) for pool in recorder.pools],
        "url_integrity_ok": all(c.url in pool_urls for c in candidates),
        "rec_checks": (
            _check_recommendation(recommendation, candidates)
            if recommendation
            else None
        ),
        "elapsed_s": round(elapsed, 1),
        "error": error,
        "recommendation": recommendation.model_dump() if recommendation else None,
        "candidates": [c.model_dump() for c in candidates],
    }


def _checks_ok(entry: dict) -> str:
    if entry["error"]:
        return "ERROR"
    if entry["off_topic"]:
        return "n/a (off topic)"
    if not entry["url_integrity_ok"]:
        return "NO"
    if entry["rec_checks"] is None:
        return "n/a (null rec)"
    return "yes" if all(entry["rec_checks"].values()) else "NO"


def render_markdown(entries: list[dict]) -> str:
    lines = [
        f"# /recipes/recommend eval — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        REVIEW_CHECKLIST,
        "## Summary",
        "",
        "| # | query | recommended dish | top fit | retry | checks | time |",
        "|---|-------|------------------|---------|-------|--------|------|",
    ]
    for e in entries:
        rec = e["recommendation"]
        top = e["candidates"][0] if e["candidates"] else None
        null_dish = "— (off topic)" if e["off_topic"] else "— (null)"
        lines.append(
            f"| {e['number']} | {e['query'][:44]} | "
            f"{(rec or {}).get('dish_name') or null_dish} | "
            f"{f'{top['fit_score']:.2f}' if top else '—'} | "
            f"{'yes' if e['retried'] else 'no'} | {_checks_ok(e)} | {e['elapsed_s']}s |"
        )
    lines.append("")

    for e in entries:
        lines += [f"## {e['number']}. {e['query']}", ""]
        if e["error"]:
            lines += [f"**FAILED**: {e['error']}", ""]
            continue
        if e["off_topic"]:
            lines += [
                "**Off-topic refusal**: the planner judged this not a food "
                "request; no search or evaluation spend.",
                "",
            ]
            continue
        for i, attempt in enumerate(e["attempts"], 1):
            lines.append(f"- planned queries (attempt {i}): {attempt['queries']}")
            if attempt["feedback"]:
                lines.append(f"  - retry feedback: {attempt['feedback']}")
        if len(e["pool_sizes"]) > len(e["attempts"]):
            lines.append("- retry planner failed; fell back to the static template")
        lines += [
            f"- retry: {'yes' if e['retried'] else 'no'} | "
            f"pool sizes: {e['pool_sizes']} | checks: {e['rec_checks']} | "
            f"url integrity: {'ok' if e['url_integrity_ok'] else 'FAILED'} | "
            f"{e['elapsed_s']}s",
            "",
        ]

        rec = e["recommendation"]
        if rec is None:
            lines += ["**Recommendation: null (nothing usable)**", ""]
        else:
            lines += [
                f"### Recommendation: {rec['dish_name']}",
                "",
                f"> {rec['headline']}",
                "",
                rec["why_it_fits"],
                "",
                "Missing:",
            ]
            for item in rec["missing_items"]:
                note = f" — {item['note']}" if item["note"] else ""
                lines.append(f"- [{item['importance']}] {item['ingredient']}{note}")
            lines += ["", "Primary sources:"]
            for s in rec["primary_sources"]:
                lines.append(f"- {s['dish_name']} — {s['source']} — {s['url']}")
            lines += ["", f"How to use: {rec['how_to_use_sources']}", ""]
            if rec["alternatives"]:
                lines.append("Alternatives:")
                for alt in rec["alternatives"]:
                    lines.append(
                        f"- {alt['recipe']['dish_name']} ({alt['recipe']['source']}) — "
                        f"{alt['reason']} — {alt['recipe']['url']}"
                    )
                lines.append("")

        lines.append("### Candidates")
        lines.append("")
        for rank, c in enumerate(e["candidates"], 1):
            lines += [
                f"**{rank}. [{c['role']}] {c['fit_score']:.2f} — "
                f"{c['dish_name'] or '(no dish name)'}** ({c['source']})",
                f"- title: {c['title']}",
                f"- url: {c['url']}",
                f"- why: {c['why_it_matches']}",
                f"- matched: {c['matched_ingredients']} | missing: {c['possibly_missing']}",
                "",
            ]
    return "\n".join(lines)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    selected = [int(arg) for arg in sys.argv[1:]] or range(1, len(QUERIES) + 1)

    settings = Settings(_env_file=ROOT / ".env")
    exa = ExaSearchClient(
        api_key=settings.exa_api_key.get_secret_value(),
        base_url=settings.exa_base_url,
        timeout_seconds=settings.exa_timeout_seconds,
    )
    evaluator = RecipeEvaluator(
        api_key=(
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None
        ),
        model=settings.evaluation_model,
        effort=settings.evaluation_effort,
        timeout_seconds=settings.evaluation_timeout_seconds,
    )

    entries = []
    try:
        for number in selected:
            query = QUERIES[number - 1]
            print(f"[{number}/{len(QUERIES)}] {query}", flush=True)
            entry = await run_query(number, query, exa, evaluator)
            rec = entry["recommendation"]
            status = entry["error"] or (
                f"rec={'null' if rec is None else rec['dish_name']!r}, "
                f"checks={_checks_ok(entry)}, retry={'yes' if entry['retried'] else 'no'}, "
                f"{entry['elapsed_s']}s"
            )
            print(f"    -> {status}", flush=True)
            entries.append(entry)
    finally:
        await exa.aclose()
        await evaluator.aclose()

    out_dir = ROOT / "evals"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"eval-{stamp}.md"
    md_path.write_text(render_markdown(entries))
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(entries, indent=2))
    print(f"\nreport: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
