"""Unit tests for the plan → search → evaluate → retry pipeline, with stubs."""

import pytest

from recipe_search.evaluation import (
    EvaluationAPIError,
    Recommendation,
    RecipeCandidate,
    SearchPlan,
    SourceLink,
)
from recipe_search.exa_search import ExaRateLimitError, SearchResult
from recipe_search.pipeline import (
    OffTopicQuery,
    find_recipe_candidates,
    recommend_recipe,
)

RECOMMENDATION = Recommendation(
    dish_name="Migas",
    headline="Make migas.",
    why_it_fits="Fits well.",
    missing_items=[],
    primary_sources=[
        SourceLink(title="t", url="https://a", source="a.com", dish_name="Migas")
    ],
    how_to_use_sources="Follow the page.",
    alternatives=[],
)

QUERY = "something quick with eggs"


def result(url: str) -> SearchResult:
    return SearchResult(
        title=url, url=url, source="example.com", snippet="eggs", published_date=None
    )


def candidate(role: str, url: str = "https://a") -> RecipeCandidate:
    return RecipeCandidate(
        title=url,
        url=url,
        source="example.com",
        dish_name="dish",
        fit_score=0.8 if role != "ignore" else 0.1,
        why_it_matches="reason text",
        matched_ingredients=[],
        possibly_missing=[],
        role=role,
    )


class StubExa:
    def __init__(self):
        self.pools: dict[str, list[SearchResult] | Exception] = {}
        self.default: list[SearchResult] | Exception = []
        self.calls: list[dict] = []

    async def search(self, query: str, *, num_results: int = 8):
        self.calls.append({"query": query, "num_results": num_results})
        outcome = self.pools.get(query, self.default)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StubEvaluator:
    def __init__(self):
        self.plans: list[SearchPlan | Exception] = []
        self.evaluations: list[list[RecipeCandidate] | Exception] = []
        self.recommendation: Recommendation = RECOMMENDATION
        self.plan_calls: list[dict] = []
        self.evaluate_calls: list[dict] = []
        self.recommend_calls: list[dict] = []

    async def recommend(self, query: str, candidates: list[RecipeCandidate]):
        self.recommend_calls.append({"query": query, "candidates": list(candidates)})
        return self.recommendation

    async def plan_searches(self, query: str, *, feedback: str | None = None):
        self.plan_calls.append({"query": query, "feedback": feedback})
        outcome = self.plans.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def evaluate(self, query: str, results: list[SearchResult]):
        self.evaluate_calls.append({"query": query, "results": list(results)})
        outcome = self.evaluations.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def exa():
    return StubExa()


@pytest.fixture
def evaluator():
    return StubEvaluator()


async def run(exa, evaluator, num_results: int = 8):
    return await find_recipe_candidates(
        QUERY, num_results=num_results, exa=exa, evaluator=evaluator
    )


async def test_fans_out_interleaves_and_dedupes(exa, evaluator):
    a, b, c = result("https://a"), result("https://b"), result("https://c")
    evaluator.plans = [SearchPlan(queries=["q1", "q2"])]
    exa.pools = {"q1": [a, b], "q2": [b, c]}
    good = [candidate("best_base_recipe")]
    evaluator.evaluations = [good]

    candidates = await run(exa, evaluator, num_results=5)

    assert candidates == good
    assert [call["query"] for call in exa.calls] == ["q1", "q2"]
    assert all(call["num_results"] == 5 for call in exa.calls)
    assert evaluator.evaluate_calls == [{"query": QUERY, "results": [a, b, c]}]
    assert evaluator.plan_calls == [{"query": QUERY, "feedback": None}]


async def test_pool_is_capped(exa, evaluator):
    evaluator.plans = [SearchPlan(queries=["q1"])]
    exa.pools = {"q1": [result(f"https://r{i}") for i in range(15)]}
    evaluator.evaluations = [[candidate("best_base_recipe")]]

    await run(exa, evaluator)

    assert len(evaluator.evaluate_calls[0]["results"]) == 12


async def test_planner_failure_falls_back_to_template(exa, evaluator):
    evaluator.plans = [EvaluationAPIError("planner down")]
    exa.default = [result("https://a")]
    evaluator.evaluations = [[candidate("best_base_recipe")]]

    candidates = await run(exa, evaluator)

    assert len(candidates) == 1
    assert exa.calls[0]["query"] == f"Here is a great home-cooked recipe: {QUERY}"


async def test_one_failed_search_variant_is_tolerated(exa, evaluator):
    a = result("https://a")
    evaluator.plans = [SearchPlan(queries=["good", "bad"])]
    exa.pools = {"good": [a], "bad": ExaRateLimitError("slow down")}
    evaluator.evaluations = [[candidate("best_base_recipe")]]

    candidates = await run(exa, evaluator)

    assert len(candidates) == 1
    assert evaluator.evaluate_calls[0]["results"] == [a]


async def test_all_failed_searches_raise(exa, evaluator):
    evaluator.plans = [SearchPlan(queries=["bad1", "bad2"])]
    exa.pools = {
        "bad1": ExaRateLimitError("slow down"),
        "bad2": ExaRateLimitError("slow down"),
    }

    with pytest.raises(ExaRateLimitError):
        await run(exa, evaluator)
    assert evaluator.evaluate_calls == []


async def test_unusable_pool_retries_with_feedback_and_exclusions(exa, evaluator):
    a, b = result("https://a"), result("https://b")
    evaluator.plans = [SearchPlan(queries=["q1"]), SearchPlan(queries=["q2"])]
    exa.pools = {"q1": [a], "q2": [a, b]}  # retry re-surfaces a; must be excluded
    good = [candidate("best_base_recipe", url="https://b")]
    evaluator.evaluations = [[candidate("ignore", url="https://a")], good]

    candidates = await run(exa, evaluator)

    assert candidates == good
    feedback = evaluator.plan_calls[1]["feedback"]
    assert "unusable" in feedback
    assert "reason text" in feedback
    assert evaluator.evaluate_calls[1]["results"] == [b]


async def test_retry_still_unusable_returns_first_attempt(exa, evaluator):
    first = [candidate("ignore", url="https://a")]
    evaluator.plans = [SearchPlan(queries=["q1"]), SearchPlan(queries=["q2"])]
    exa.pools = {"q1": [result("https://a")], "q2": [result("https://b")]}
    evaluator.evaluations = [first, [candidate("ignore", url="https://b")]]

    assert await run(exa, evaluator) == first


async def test_empty_pool_retries_with_no_results_feedback(exa, evaluator):
    evaluator.plans = [SearchPlan(queries=["q1"]), SearchPlan(queries=["q2"])]
    exa.pools = {"q1": [], "q2": [result("https://b")]}
    good = [candidate("backup", url="https://b")]
    evaluator.evaluations = [[], good]

    candidates = await run(exa, evaluator)

    assert candidates == good
    assert (
        evaluator.plan_calls[1]["feedback"]
        == "The search returned no results at all."
    )


async def test_off_topic_plan_stops_before_any_search(exa, evaluator):
    evaluator.plans = [SearchPlan(on_topic=False, queries=[])]

    with pytest.raises(OffTopicQuery):
        await find_recipe_candidates(QUERY, num_results=8, exa=exa, evaluator=evaluator)

    assert exa.calls == []
    assert evaluator.evaluate_calls == []


async def test_recommend_recipe_offers_usable_candidates_only(exa, evaluator):
    best = candidate("best_base_recipe", url="https://a")
    ignored = candidate("ignore", url="https://b")
    evaluator.plans = [SearchPlan(queries=["q1"])]
    exa.pools = {"q1": [result("https://a"), result("https://b")]}
    evaluator.evaluations = [[best, ignored]]

    recommendation, candidates = await recommend_recipe(
        QUERY, num_results=8, exa=exa, evaluator=evaluator
    )

    assert recommendation == RECOMMENDATION
    assert candidates == [best, ignored]  # full transparency in the response
    assert evaluator.recommend_calls == [{"query": QUERY, "candidates": [best]}]


async def test_recommend_recipe_returns_none_when_nothing_usable(exa, evaluator):
    evaluator.plans = [SearchPlan(queries=["q1"]), SearchPlan(queries=["q2"])]
    exa.pools = {"q1": [result("https://a")], "q2": [result("https://b")]}
    first = [candidate("ignore", url="https://a")]
    evaluator.evaluations = [first, [candidate("ignore", url="https://b")]]

    recommendation, candidates = await recommend_recipe(
        QUERY, num_results=8, exa=exa, evaluator=evaluator
    )

    assert recommendation is None
    assert candidates == first
    assert evaluator.recommend_calls == []


async def test_no_retry_when_first_attempt_is_usable(exa, evaluator):
    evaluator.plans = [SearchPlan(queries=["q1"])]
    exa.pools = {"q1": [result("https://a")]}
    evaluator.evaluations = [[candidate("backup")]]

    await run(exa, evaluator)

    assert len(evaluator.plan_calls) == 1
    assert len(evaluator.evaluate_calls) == 1
