"""Unit tests for the recipe-candidate evaluator, with the Anthropic client faked."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from recipe_search.evaluation import (
    _PHOTO_PROMPT,
    _PLANNER_PROMPT,
    _RECOMMENDER_PROMPT,
    EvaluationAPIError,
    EvaluationAuthError,
    EvaluationRateLimitError,
    EvaluationTimeoutError,
    MissingItem,
    PhotoIngredients,
    RecipeCandidate,
    RecipeEvaluator,
    SearchPlan,
    _AlternativeRef,
    _CandidateEvaluation,
    _EvaluationOutput,
    _RecommendationOutput,
)
from recipe_search.exa_search import SearchResult


def make_result(index: int) -> SearchResult:
    return SearchResult(
        title=f"Recipe {index}",
        url=f"https://example{index}.com/recipe",
        source=f"example{index}.com",
        snippet=f"Ingredients for recipe {index}: eggs, cheese.",
        published_date=None,
    )


def make_evaluation(index: int, **overrides) -> _CandidateEvaluation:
    fields = {
        "index": index,
        "usable_recipe_page": True,
        "dish_name": f"Dish {index}",
        "fit_score": 0.5,
        "why_it_matches": "Uses the ingredients on hand.",
        "matched_ingredients": ["eggs", "cheese"],
        "possibly_missing": ["onion"],
        "role": "backup",
    }
    fields.update(overrides)
    return _CandidateEvaluation(**fields)


class FakeMessages:
    def __init__(self, owner):
        self._owner = owner

    async def parse(self, **kwargs):
        self._owner.calls.append(kwargs)
        if self._owner.error is not None:
            raise self._owner.error
        return self._owner.response


class FakeAnthropicClient:
    def __init__(self):
        self.calls = []
        self.error = None
        self.response = SimpleNamespace(parsed_output=None, stop_reason="end_turn")
        self.messages = FakeMessages(self)

    def respond_with(self, evaluations):
        self.response = SimpleNamespace(
            parsed_output=_EvaluationOutput(evaluations=evaluations),
            stop_reason="end_turn",
        )

    async def close(self):
        pass


@pytest.fixture
def fake_anthropic():
    return FakeAnthropicClient()


@pytest.fixture
def evaluator(fake_anthropic):
    return RecipeEvaluator(client=fake_anthropic)


async def test_evaluate_merges_and_ranks(evaluator, fake_anthropic):
    results = [make_result(0), make_result(1), make_result(2)]
    fake_anthropic.respond_with(
        [
            make_evaluation(0, role="ignore", fit_score=0.1, usable_recipe_page=False),
            make_evaluation(1, role="backup", fit_score=0.7),
            make_evaluation(2, role="best_base_recipe", fit_score=0.95),
        ]
    )

    candidates = await evaluator.evaluate("eggs and cheese, quick", results)

    assert [c.role for c in candidates] == ["best_base_recipe", "backup", "ignore"]
    best = candidates[0]
    # title/url/source come from the search result, never from the model
    assert best.title == "Recipe 2"
    assert best.url == "https://example2.com/recipe"
    assert best.source == "example2.com"
    assert best.dish_name == "Dish 2"
    assert best.fit_score == 0.95
    assert best.matched_ingredients == ["eggs", "cheese"]
    assert best.possibly_missing == ["onion"]


async def test_evaluate_sends_query_and_indexed_results(evaluator, fake_anthropic):
    results = [make_result(0), make_result(1)]
    fake_anthropic.respond_with([make_evaluation(0), make_evaluation(1)])

    await evaluator.evaluate("  something quick  ", results)

    call = fake_anthropic.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}
    assert call["output_format"] is _EvaluationOutput
    prompt = call["messages"][0]["content"]
    assert 'User request: "something quick"' in prompt
    assert "[0] Title: Recipe 0" in prompt
    assert "[1] Title: Recipe 1" in prompt
    assert "https://example1.com/recipe" in prompt


async def test_evaluate_ranks_backups_by_score(evaluator, fake_anthropic):
    results = [make_result(0), make_result(1), make_result(2)]
    fake_anthropic.respond_with(
        [
            make_evaluation(0, fit_score=0.4),
            make_evaluation(1, fit_score=0.9),
            make_evaluation(2, fit_score=0.6),
        ]
    )

    candidates = await evaluator.evaluate("eggs", results)

    assert [c.fit_score for c in candidates] == [0.9, 0.6, 0.4]


async def test_evaluate_clamps_scores(evaluator, fake_anthropic):
    results = [make_result(0), make_result(1)]
    fake_anthropic.respond_with(
        [
            make_evaluation(0, fit_score=1.7),
            make_evaluation(1, fit_score=-0.2),
        ]
    )

    candidates = await evaluator.evaluate("eggs", results)

    assert [c.fit_score for c in candidates] == [1.0, 0.0]


async def test_evaluate_drops_unknown_and_duplicate_indexes(evaluator, fake_anthropic):
    results = [make_result(0)]
    fake_anthropic.respond_with(
        [
            make_evaluation(0, dish_name="first"),
            make_evaluation(0, dish_name="duplicate"),
            make_evaluation(5),
            make_evaluation(-1),
        ]
    )

    candidates = await evaluator.evaluate("eggs", results)

    assert len(candidates) == 1
    assert candidates[0].dish_name == "first"


async def test_unusable_page_is_forced_to_ignore(evaluator, fake_anthropic):
    results = [make_result(0)]
    fake_anthropic.respond_with(
        [make_evaluation(0, usable_recipe_page=False, role="backup")]
    )

    candidates = await evaluator.evaluate("eggs", results)

    assert candidates[0].role == "ignore"


async def test_oversized_snippets_are_truncated_with_marker(evaluator, fake_anthropic):
    result = make_result(0).model_copy(update={"snippet": "x" * 12_000})
    fake_anthropic.respond_with([make_evaluation(0)])

    await evaluator.evaluate("eggs", [result])

    prompt = fake_anthropic.calls[0]["messages"][0]["content"]
    assert "x" * 10_000 + " …[truncated]" in prompt
    assert "x" * 10_001 not in prompt


async def test_normal_snippets_are_sent_in_full(evaluator, fake_anthropic):
    snippet = "Ingredients: eggs, cheese. " * 100  # ~2.7k chars, like real results
    result = make_result(0).model_copy(update={"snippet": snippet})
    fake_anthropic.respond_with([make_evaluation(0)])

    await evaluator.evaluate("eggs", [result])

    prompt = fake_anthropic.calls[0]["messages"][0]["content"]
    assert snippet.strip() in prompt
    assert "…[truncated]" not in prompt


async def test_empty_results_skip_the_model_call(evaluator, fake_anthropic):
    assert await evaluator.evaluate("eggs", []) == []
    assert fake_anthropic.calls == []


async def test_blank_query_raises(evaluator):
    with pytest.raises(ValueError):
        await evaluator.evaluate("   ", [make_result(0)])


async def test_plan_searches_cleans_and_caps_queries(evaluator, fake_anthropic):
    fake_anthropic.response = SimpleNamespace(
        parsed_output=SearchPlan(
            queries=[" q one ", "q one", "", "q two", "q three", "q four"]
        ),
        stop_reason="end_turn",
    )

    plan = await evaluator.plan_searches("  find dinner  ")

    assert plan.queries == ["q one", "q two", "q three"]
    call = fake_anthropic.calls[0]
    assert call["output_format"] is SearchPlan
    assert call["system"] is _PLANNER_PROMPT
    assert call["output_config"] == {"effort": "low"}
    assert "thinking" not in call  # planner runs without thinking for speed
    assert 'User request: "find dinner"' in call["messages"][0]["content"]
    assert "Feedback" not in call["messages"][0]["content"]


async def test_plan_searches_includes_feedback(evaluator, fake_anthropic):
    fake_anthropic.response = SimpleNamespace(
        parsed_output=SearchPlan(queries=["q"]), stop_reason="end_turn"
    )

    await evaluator.plan_searches("dinner", feedback="only product pages came back")

    content = fake_anthropic.calls[0]["messages"][0]["content"]
    assert "Feedback from a failed attempt: only product pages came back" in content


async def test_plan_searches_passes_off_topic_through(evaluator, fake_anthropic):
    fake_anthropic.response = SimpleNamespace(
        parsed_output=SearchPlan(on_topic=False, queries=["", "junk"]),
        stop_reason="end_turn",
    )

    plan = await evaluator.plan_searches("write me python code")

    assert plan.on_topic is False
    assert plan.queries == []


async def test_plan_searches_rejects_empty_plan(evaluator, fake_anthropic):
    fake_anthropic.response = SimpleNamespace(
        parsed_output=SearchPlan(queries=["", "   "]), stop_reason="end_turn"
    )
    with pytest.raises(EvaluationAPIError):
        await evaluator.plan_searches("dinner")


async def test_plan_searches_maps_errors_via_shared_path(evaluator, fake_anthropic):
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.anthropic.com")
    )
    fake_anthropic.error = anthropic.RateLimitError("slow", response=response, body=None)
    with pytest.raises(EvaluationRateLimitError):
        await evaluator.plan_searches("dinner")


async def test_plan_searches_blank_query_raises(evaluator):
    with pytest.raises(ValueError):
        await evaluator.plan_searches("   ")


def make_candidate(index: int, role: str = "backup") -> RecipeCandidate:
    return RecipeCandidate(
        title=f"Recipe {index}",
        url=f"https://example{index}.com/recipe",
        source=f"example{index}.com",
        dish_name=f"Dish {index}",
        fit_score=0.8,
        why_it_matches="Uses your ingredients.",
        matched_ingredients=["eggs"],
        possibly_missing=["onion"],
        role=role,
    )


def make_recommendation_output(**overrides) -> _RecommendationOutput:
    fields = {
        "dish_name": "Migas",
        "headline": "Make migas tonight.",
        "why_it_fits": "You have everything for it.",
        "missing_items": [
            MissingItem(ingredient="onion", importance="nice_to_have", note="skip it")
        ],
        "primary_indexes": [0],
        "how_to_use_sources": "Follow the first page.",
        "alternatives": [],
    }
    fields.update(overrides)
    return _RecommendationOutput(**fields)


async def test_recommend_prompts_with_indexes_not_urls(evaluator, fake_anthropic):
    candidates = [make_candidate(0, role="best_base_recipe"), make_candidate(1)]
    fake_anthropic.response = SimpleNamespace(
        parsed_output=make_recommendation_output(), stop_reason="end_turn"
    )

    await evaluator.recommend("  eggs and tortillas  ", candidates)

    call = fake_anthropic.calls[0]
    assert call["output_format"] is _RecommendationOutput
    assert call["system"] is _RECOMMENDER_PROMPT
    prompt = call["messages"][0]["content"]
    assert 'User request: "eggs and tortillas"' in prompt
    assert "[0] Dish 0" in prompt
    assert "[1] Dish 1" in prompt
    assert "https://" not in prompt  # links merge server-side, never via model


async def test_recommend_merges_sources_server_side(evaluator, fake_anthropic):
    candidates = [make_candidate(0), make_candidate(1), make_candidate(2)]
    fake_anthropic.response = SimpleNamespace(
        parsed_output=make_recommendation_output(
            primary_indexes=[1],
            alternatives=[_AlternativeRef(index=2, reason="slower but richer")],
        ),
        stop_reason="end_turn",
    )

    recommendation = await evaluator.recommend("eggs", candidates)

    assert [s.url for s in recommendation.primary_sources] == [
        "https://example1.com/recipe"
    ]
    assert recommendation.alternatives[0].recipe.url == "https://example2.com/recipe"
    assert recommendation.alternatives[0].reason == "slower but richer"
    assert recommendation.missing_items[0].importance == "nice_to_have"


async def test_recommend_invalid_indexes_fall_back_to_top(evaluator, fake_anthropic):
    candidates = [make_candidate(0), make_candidate(1)]
    fake_anthropic.response = SimpleNamespace(
        parsed_output=make_recommendation_output(
            primary_indexes=[9],
            alternatives=[_AlternativeRef(index=7, reason="nope")],
        ),
        stop_reason="end_turn",
    )

    recommendation = await evaluator.recommend("eggs", candidates)

    assert recommendation.primary_sources[0].url == "https://example0.com/recipe"
    assert recommendation.alternatives == []


async def test_recommend_dedupes_primary_and_alternatives(evaluator, fake_anthropic):
    candidates = [make_candidate(0), make_candidate(1), make_candidate(2)]
    fake_anthropic.response = SimpleNamespace(
        parsed_output=make_recommendation_output(
            primary_indexes=[0, 0, 1],
            alternatives=[
                _AlternativeRef(index=0, reason="already primary"),
                _AlternativeRef(index=2, reason="different mood"),
            ],
        ),
        stop_reason="end_turn",
    )

    recommendation = await evaluator.recommend("eggs", candidates)

    assert [s.url for s in recommendation.primary_sources] == [
        "https://example0.com/recipe",
        "https://example1.com/recipe",
    ]
    assert [a.recipe.url for a in recommendation.alternatives] == [
        "https://example2.com/recipe"
    ]


async def test_recommend_validates_inputs(evaluator):
    with pytest.raises(ValueError):
        await evaluator.recommend("   ", [make_candidate(0)])
    with pytest.raises(ValueError):
        await evaluator.recommend("eggs", [])


PHOTO_B64 = "ZmFrZS1qcGVnLWJ5dGVz"  # the fake client never decodes it


def photo_response(**overrides):
    fields = {"food_visible": True, "ingredients": ["eggs", "cheddar"]}
    fields.update(overrides)
    return SimpleNamespace(
        parsed_output=PhotoIngredients(**fields), stop_reason="end_turn"
    )


async def test_identify_ingredients_sends_the_image_block(evaluator, fake_anthropic):
    fake_anthropic.response = photo_response()

    found = await evaluator.identify_ingredients([(PHOTO_B64, "image/png")])

    assert found == PhotoIngredients(food_visible=True, ingredients=["eggs", "cheddar"])
    call = fake_anthropic.calls[0]
    assert call["system"] is _PHOTO_PROMPT
    assert call["output_format"] is PhotoIngredients
    assert call["output_config"] == {"effort": "low"}
    assert "thinking" not in call  # like the planner: a fast, cheap call
    image_block, text_block = call["messages"][0]["content"]
    assert image_block == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": PHOTO_B64,
        },
    }
    assert text_block["type"] == "text"


async def test_identify_ingredients_sends_every_image_in_one_call(
    evaluator, fake_anthropic
):
    fake_anthropic.response = photo_response()

    await evaluator.identify_ingredients(
        [(PHOTO_B64, "image/png"), ("c2Vjb25k", "image/jpeg")]
    )

    # One call, both photos in it, followed by a single text nudge.
    assert len(fake_anthropic.calls) == 1
    content = fake_anthropic.calls[0]["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "image", "text"]
    assert content[0]["source"]["data"] == PHOTO_B64
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"] == "c2Vjb25k"
    assert content[1]["source"]["media_type"] == "image/jpeg"


async def test_identify_ingredients_cleans_and_caps_the_list(
    evaluator, fake_anthropic
):
    noisy = [" Eggs ", "eggs", "", "  "] + [f"item {i}" for i in range(45)]
    fake_anthropic.response = photo_response(ingredients=noisy)

    found = await evaluator.identify_ingredients([(PHOTO_B64, "image/jpeg")])

    assert found.food_visible is True
    assert len(found.ingredients) == 40
    assert found.ingredients[:2] == ["eggs", "item 0"]  # deduped, lowercased


@pytest.mark.parametrize(
    ("food_visible", "ingredients"),
    [(False, ["junk"]), (True, ["", "   "])],
)
async def test_identify_ingredients_returns_no_food_for_unusable_results(
    evaluator, fake_anthropic, food_visible, ingredients
):
    fake_anthropic.response = photo_response(
        food_visible=food_visible, ingredients=ingredients
    )

    found = await evaluator.identify_ingredients([(PHOTO_B64, "image/jpeg")])

    assert found == PhotoIngredients(food_visible=False, ingredients=[])


async def test_identify_ingredients_empty_list_raises(evaluator):
    with pytest.raises(ValueError):
        await evaluator.identify_ingredients([])


async def test_identify_ingredients_empty_image_raises(evaluator):
    with pytest.raises(ValueError):
        await evaluator.identify_ingredients([("", "image/jpeg")])


def _status_error(cls, status: int):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.anthropic.com")
    )
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_status_error(anthropic.AuthenticationError, 401), EvaluationAuthError),
        (_status_error(anthropic.PermissionDeniedError, 403), EvaluationAuthError),
        (_status_error(anthropic.RateLimitError, 429), EvaluationRateLimitError),
        (_status_error(anthropic.InternalServerError, 500), EvaluationAPIError),
        (
            anthropic.APITimeoutError(
                request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            EvaluationTimeoutError,
        ),
        (
            anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            EvaluationAPIError,
        ),
    ],
)
async def test_evaluate_maps_anthropic_errors(
    evaluator, fake_anthropic, error, expected
):
    fake_anthropic.error = error
    with pytest.raises(expected):
        await evaluator.evaluate("eggs", [make_result(0)])


async def test_truncated_output_raises(evaluator, fake_anthropic):
    fake_anthropic.response = SimpleNamespace(
        parsed_output=None, stop_reason="max_tokens"
    )
    with pytest.raises(EvaluationAPIError):
        await evaluator.evaluate("eggs", [make_result(0)])


async def test_missing_parsed_output_raises(evaluator, fake_anthropic):
    fake_anthropic.response = SimpleNamespace(
        parsed_output=None, stop_reason="refusal"
    )
    with pytest.raises(EvaluationAPIError):
        await evaluator.evaluate("eggs", [make_result(0)])
