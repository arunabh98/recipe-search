"""Recipe-candidate pipeline: plan searches, retrieve, evaluate, adapt.

Robustness comes from judgment instead of rules at each choke point: Claude
plans the Exa queries (any phrasing, language, or vagueness), the pool is
fanned out across query variants and deduped, and when evaluation judges
the whole pool unusable the pipeline retries once, feeding the evaluator's
own reasons back to the planner. Failures degrade instead of cascading: a
failed retry planner falls back to a static recipe framing (the query
already passed the on-topic gate on the first attempt), and a failed
search variant is dropped as long as any variant succeeds. A first-attempt
planner failure propagates instead: falling back there would search
unvetted input, bypassing the on-topic gate.

No FastAPI imports; reusable from CLIs, jobs, or other services.
"""

import asyncio
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from recipe_search.evaluation import (
    EvaluationAPIError,
    EvaluationError,
    FollowUpContext,
    FollowUpExchange,
    Recommendation,
    RecipeCandidate,
    RecipeEvaluator,
)
from recipe_search.exa_search import ExaSearchClient, SearchResult

logger = logging.getLogger(__name__)

_FALLBACK_QUERY_TEMPLATE = "Here is a great home-cooked recipe: {}"
_MAX_POOL_SIZE = 12


Progress = Callable[[str], None]


def ignore_progress(stage: str) -> None:
    """Default for callers that only need the final result."""


class OffTopicQuery(Exception):
    """The planner judged the request to be not about food or cooking."""


@dataclass
class FollowUpResult:
    """One committed conversation turn, with a recipe payload only for searches."""

    action: Literal["answer", "search"]
    reply: str
    context: FollowUpContext
    recommendation: Recommendation | None = None
    candidates: list[RecipeCandidate] | None = None


async def follow_up_recipe(
    message: str,
    context: FollowUpContext,
    *,
    num_results: int,
    exa: ExaSearchClient,
    evaluator: RecipeEvaluator,
    progress: Progress = ignore_progress,
) -> FollowUpResult:
    """Answer briefly or search again without changing the caller's prior state."""
    progress("understanding")
    decision = await evaluator.follow_up(message, context)
    if not decision.on_topic:
        raise OffTopicQuery(message)

    recommendation = None
    candidates = None
    visible_recommendation = context.recommendation
    recommendation_query = context.recommendation_query
    reply = decision.reply
    if decision.action == "search":
        recommendation, candidates = await recommend_recipe(
            decision.current_request,
            num_results=num_results,
            exa=exa,
            evaluator=evaluator,
            progress=progress,
        )
        if recommendation is not None:
            visible_recommendation = recommendation
            recommendation_query = decision.current_request
            # The router has not seen the search results. Only announce a new
            # dish after retrieval succeeds, using the grounded recommendation.
            reply = recommendation.headline[:1000]
        else:
            reply = (
                "I couldn't find a recipe that fits that change. "
                "Your earlier recommendation is still here. "
                "Try another adjustment."
            )

    try:
        updated_context = FollowUpContext(
            current_request=decision.current_request,
            recommendation_query=recommendation_query,
            recommendation=visible_recommendation,
            exchanges=[
                *context.exchanges[-5:],
                FollowUpExchange(message=message, reply=reply),
            ],
        )
    except ValidationError as exc:
        # Do not return an unusable context or silently clip dietary constraints.
        raise EvaluationAPIError("Follow-up context exceeds the supported limits") from exc
    return FollowUpResult(
        action=decision.action,
        reply=reply,
        context=updated_context,
        recommendation=recommendation,
        candidates=candidates,
    )


async def find_recipe_candidates(
    query: str,
    *,
    num_results: int,
    exa: ExaSearchClient,
    evaluator: RecipeEvaluator,
    progress: Progress = ignore_progress,
) -> list[RecipeCandidate]:
    """Search the web and rank cooking candidates for a natural-language query.

    Retries once with planner feedback when the first pool yields nothing
    usable; if the retry doesn't improve things, the first (honest) ranking
    is returned.
    """
    first, seen_urls = await _attempt(
        query, num_results=num_results, exa=exa, evaluator=evaluator, progress=progress
    )
    if _has_usable(first):
        return first

    feedback = _failure_feedback(first)
    logger.info("No usable candidates; retrying. Feedback: %s", feedback)
    retry, _ = await _attempt(
        query,
        num_results=num_results,
        exa=exa,
        evaluator=evaluator,
        feedback=feedback,
        progress=progress,
        exclude_urls=seen_urls,
    )
    return retry if _has_usable(retry) else first


async def recommend_recipe(
    query: str,
    *,
    num_results: int,
    exa: ExaSearchClient,
    evaluator: RecipeEvaluator,
    progress: Progress = ignore_progress,
) -> tuple[Recommendation | None, list[RecipeCandidate]]:
    """Run the candidate pipeline, then write the user-facing recommendation.

    Only usable candidates (role != ignore) are offered as sources; when
    nothing usable came back, the recommendation is None and the (honest)
    candidate list is still returned.
    """
    candidates = await find_recipe_candidates(
        query, num_results=num_results, exa=exa, evaluator=evaluator, progress=progress
    )
    usable = [c for c in candidates if c.role != "ignore"]
    if not usable:
        logger.info("No usable candidates; skipping recommendation")
        return None, candidates
    progress("recommending")
    recommendation = await evaluator.recommend(query, usable)
    return recommendation, candidates


async def _attempt(
    query: str,
    *,
    num_results: int,
    exa: ExaSearchClient,
    evaluator: RecipeEvaluator,
    progress: Progress = ignore_progress,
    feedback: str | None = None,
    exclude_urls: frozenset[str] = frozenset(),
) -> tuple[list[RecipeCandidate], frozenset[str]]:
    progress("retrying" if feedback is not None else "planning")
    queries = await _plan_queries(evaluator, query, feedback)
    logger.info("Planned search queries: %s", queries)
    progress("searching")
    pools = await _run_searches(exa, queries, num_results)
    pool = _merge_pools(pools, exclude_urls=exclude_urls)
    progress("evaluating")
    candidates = await evaluator.evaluate(query, pool)
    return candidates, exclude_urls | {result.url for result in pool}


async def _plan_queries(
    evaluator: RecipeEvaluator, query: str, feedback: str | None
) -> list[str]:
    try:
        plan = await evaluator.plan_searches(query, feedback=feedback)
    except EvaluationError as exc:
        if feedback is None:
            # First attempt: the query hasn't passed the on-topic gate yet,
            # so searching a fallback framing of it would bypass the gate.
            raise
        logger.warning("Retry planning failed (%s); using fallback query", exc)
        return [_FALLBACK_QUERY_TEMPLATE.format(query)]
    if not plan.on_topic:
        # Stop before any search or evaluation spend.
        raise OffTopicQuery(query)
    return plan.queries


async def _run_searches(
    exa: ExaSearchClient, queries: list[str], num_results: int
) -> list[list[SearchResult]]:
    outcomes = await asyncio.gather(
        *(exa.search(q, num_results=num_results) for q in queries),
        return_exceptions=True,
    )
    pools = [outcome for outcome in outcomes if isinstance(outcome, list)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if not pools:
        raise failures[0]
    for failure in failures:
        logger.warning("Search variant failed: %s", failure)
    return pools


def _merge_pools(
    pools: list[list[SearchResult]], *, exclude_urls: frozenset[str]
) -> list[SearchResult]:
    """Interleave pools round-robin, dedupe by URL, cap the merged size."""
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


def _has_usable(candidates: list[RecipeCandidate]) -> bool:
    return any(candidate.role != "ignore" for candidate in candidates)


def _failure_feedback(candidates: list[RecipeCandidate]) -> str:
    if not candidates:
        return "The search returned no results at all."
    reasons = "; ".join(c.why_it_matches for c in candidates[:3])
    return f"Every result was judged unusable. Sample judgments: {reasons}"
