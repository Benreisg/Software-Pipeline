"""The Google provider — Gemini's usage accounting, cost, and the guards.

Stubs `client.models.generate_content` and feeds the provider the SDK's own
`GenerateContentResponseUsageMetadata` type, so what is exercised is the code a
live run runs. The numbers in `_usage()` follow Gemini's actual contract, which
differs from the other two vendors in three ways worth stating once:

    total_token_count = prompt + candidates + thoughts
    thoughts are OUTSIDE candidates_token_count but billed at the output rate
    cached_content_token_count is the cached share OF prompt_token_count
    tool_use_prompt_token_count is a tool-use prompt breakdown, not another term

The provider also accepts the Python SDK contract that adds tool-use prompt
tokens to total_token_count, using the returned total to distinguish the shapes.
"""
from __future__ import annotations

import types

import pytest

import config as cfg
from google.genai.types import (
    GenerateContentResponseUsageMetadata, MediaModality, ModalityTokenCount,
)
from providers import (
    GoogleProvider, _GOOGLE_ACCOUNTED, _google_modality_warning,
    _unaccounted_usage_fields,
)

FLASH = cfg.model_by_key("google_standard")     # gemini-3.6-flash      $0.75/$3.75 intro
PRO31 = cfg.model_by_key("google_advanced")     # gemini-3.1-pro-preview $2.00/$12.00
LITE = cfg.model_by_key("google_efficient")     # gemini-3.5-flash-lite $0.30/$2.50


def _usage(prompt=10_000, candidates=2_000, thoughts=1_200, cached=0, tool_prompt=0,
           modality=MediaModality.TEXT, total=None):
    if total is None:
        total = prompt + candidates + thoughts
    return GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        cached_content_token_count=cached or None,
        tool_use_prompt_token_count=tool_prompt or None,
        total_token_count=total,
        prompt_tokens_details=[ModalityTokenCount(modality=modality,
                                                  token_count=prompt)],
    )


def _stub(usage, text="digraph g { a -> b }", finish="STOP", capture=None):
    def generate_content(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return types.SimpleNamespace(
            text=text, usage_metadata=usage, response_id="resp-123",
            candidates=[types.SimpleNamespace(finish_reason=finish)])
    return types.SimpleNamespace(
        models=types.SimpleNamespace(generate_content=generate_content))


def _provider(usage=None, **stub_kwargs):
    provider = GoogleProvider.__new__(GoogleProvider)   # no client construction
    provider._client = _stub(usage if usage is not None else _usage(), **stub_kwargs)
    provider._max_retries = 5
    return provider


# ── catalogue ─────────────────────────────────────────────────────────────────
def test_google_is_a_live_vendor_with_verified_prices():
    assert "google" in cfg.LIVE_VENDORS
    for model in (FLASH, PRO31, LITE):
        assert model.price_verified, model.key
        assert model.price_cached_in_per_mtok is not None, model.key
        assert model.model_id.startswith("gemini-3."), model.key


def test_the_retired_25_family_is_gone():
    """Gemini 2.5 answers 404 for new API keys — a run must not be configured
    to call it."""
    for model in cfg.models_by_vendor("google"):
        assert "2.5" not in model.model_id


def test_the_advanced_tier_is_the_only_pro_class_text_model(settings):
    """Google sells one Pro-class text model on the Gemini API and only as a
    preview id. Asserting the id keeps the preview status visible: if Google
    withdraws or renames it, this fails before a run spends anything."""
    assert PRO31.model_id == "gemini-3.1-pro-preview"
    assert PRO31.thinking
    # The ≤200k-prompt tier, which is the only one PMo prompts can reach.
    assert (PRO31.price_in_per_mtok, PRO31.price_out_per_mtok) == (2.00, 12.00)


def test_a_traffic_type_that_is_not_on_demand_is_flagged(settings):
    """Provisioned throughput is not billed per token at all, so a per-token
    figure for such a call would be fiction."""
    from google.genai.types import TrafficType

    usage = _usage()
    usage.traffic_type = TrafficType.PROVISIONED_THROUGHPUT
    result = _provider(usage).generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert "not the list-rate tier" in result.untracked_usage


def test_current_service_tier_takes_precedence_over_legacy_traffic_type(settings):
    """New responses expose service_tier; older SDKs expose traffic_type."""
    from google.genai.types import TrafficType

    usage = _usage()
    usage.traffic_type = TrafficType.ON_DEMAND
    usage.__dict__["service_tier"] = "flex"
    result = _provider(usage).generate(
        FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.service_tier == "flex"
    assert "not the list-rate tier" in result.untracked_usage


def test_the_pricing_sheet_is_dated():
    sheet = cfg.pricing_sheet("google")
    assert sheet["as_of"] == "2026-08-30"
    assert "ai.google.dev" in sheet["source"]


# ── request shape ─────────────────────────────────────────────────────────────
def test_the_advanced_tier_requests_thought_summaries(settings):
    sent = {}
    _provider(capture=sent).generate(PRO31, None,
                                     [{"role": "user", "content": "hi"}], settings)
    thinking = sent["config"].thinking_config
    assert thinking is not None and thinking.include_thoughts is True
    # effort="medium" since 2026-08-31; before that this tier named no level and
    # ran at Gemini's own default while the other Advanced tiers stated a depth.
    assert str(thinking.thinking_level).endswith("MEDIUM")
    # No provider writes a system prompt into its request any more
    # (removed 2026-08-31), so this asserts the slot stays empty.
    assert sent["config"].system_instruction is None
    # No service_tier on any vendor: omitting it means `auto`, and the tier
    # that served the call is read back from the response instead — see the
    # service-tier note in config.py.
    assert sent["config"].service_tier is None


def test_an_effort_becomes_a_thinking_level(settings):
    import dataclasses
    sent = {}
    model = dataclasses.replace(PRO31, effort="high")
    _provider(capture=sent).generate(model, None, [{"role": "user", "content": "hi"}], settings)
    assert str(sent["config"].thinking_config.thinking_level).endswith("HIGH")


def test_a_non_thinking_tier_sends_no_thinking_config(settings):
    sent = {}
    _provider(capture=sent).generate(LITE, None, [{"role": "user", "content": "hi"}], settings)
    assert sent["config"].thinking_config is None


# ── the token split ───────────────────────────────────────────────────────────
def test_thinking_tokens_are_added_to_output_not_left_out(settings):
    """Gemini reports thoughts outside candidates_token_count, but bills them at
    the output rate — leaving them out would understate every reasoning call."""
    result = _provider(_usage(prompt=10_000, candidates=2_000, thoughts=1_200)) \
        .generate(PRO31, None, [{"role": "user", "content": "hi"}], settings)
    assert result.output_tokens == 3_200
    assert result.thinking_tokens == 1_200
    assert result.input_tokens == 10_000
    assert result.total_tokens == 13_200


def test_cached_tokens_are_split_out_of_the_prompt(settings):
    """cached_content_token_count is part of prompt_token_count, so the uncached
    remainder is the difference — counting both would bill the prompt twice."""
    result = _provider(_usage(prompt=10_000, cached=6_000)) \
        .generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.input_tokens == 4_000
    assert result.cached_tokens == 6_000
    assert result.billable_input_tokens == 10_000


def test_tool_use_prompt_tokens_are_not_double_counted(settings):
    """The REST total formula does not add this prompt breakdown separately."""
    plain = _provider(_usage(tool_prompt=0)) \
        .generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    withtool = _provider(_usage(tool_prompt=500)) \
        .generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert withtool.input_tokens == plain.input_tokens
    assert withtool.total_tokens == plain.total_tokens
    assert withtool.cost_usd == plain.cost_usd


def test_the_reported_prompt_count_survives_the_normalisation(settings):
    """The API's inclusive prompt and total counts survive normalisation."""
    result = _provider(_usage(prompt=10_000, cached=6_000, tool_prompt=500)) \
        .generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.reported_input_tokens == 10_000   # exactly prompt_token_count
    assert result.reported_total_tokens == 13_200   # exactly total_token_count
    assert result.input_tokens == 4_000             # 10_000 - 6_000 cached
    assert result.tool_use_prompt_tokens == 500     # its own breakdown


def test_geminis_own_output_count_is_kept_apart_from_the_billable_one(settings):
    """`candidates_token_count` is the answer alone. output_tokens adds the
    thoughts Gemini bills at the output rate, so on this vendor — and only this
    one — the reported figure and the billable one are different numbers."""
    result = _provider(_usage(candidates=2_000, thoughts=1_200)) \
        .generate(PRO31, None, [{"role": "user", "content": "hi"}], settings)
    assert result.reported_output_tokens == 2_000   # exactly candidates_token_count
    assert result.output_tokens == 3_200            # + thoughts, which are billed
    assert result.output_tokens - result.thinking_tokens == result.reported_output_tokens


def test_tool_definition_tokens_are_stated_on_their_own(settings):
    """The breakdown is retained without being added to the inclusive prompt."""
    result = _provider(_usage(prompt=10_000, tool_prompt=500)) \
        .generate(PRO31, None, [{"role": "user", "content": "hi"}], settings)
    assert result.tool_use_prompt_tokens == 500
    assert result.input_tokens == 10_000           # already inside prompt count
    plain = _provider(_usage(prompt=10_000, tool_prompt=0)) \
        .generate(PRO31, None, [{"role": "user", "content": "hi"}], settings)
    assert plain.tool_use_prompt_tokens == 0


def test_sdk_tool_prompt_total_shape_is_supported_without_losing_tokens(settings):
    """Some official SDK schemas add tool-use prompts as a fourth total term."""
    usage = _usage(prompt=10_000, candidates=2_000, thoughts=1_200,
                   tool_prompt=500, total=13_700)
    result = _provider(usage).generate(
        PRO31, None, [{"role": "user", "content": "hi"}], settings)
    assert result.input_tokens == 10_500
    assert result.total_tokens == result.reported_total_tokens == 13_700
    assert result.untracked_usage == ""


def test_the_whole_usage_object_is_archived_as_json(settings):
    """The modality breakdowns carry nothing this text-only run can act on, but
    they are the part of Gemini's accounting no column holds — so the raw object
    is archived verbatim instead of being summarised away."""
    import json

    result = _provider(_usage(prompt=10_000, tool_prompt=500)) \
        .generate(PRO31, None, [{"role": "user", "content": "hi"}], settings)
    archived = json.loads(result.raw_usage)
    assert archived["prompt_token_count"] == 10_000
    assert archived["candidates_token_count"] == 2_000
    assert archived["tool_use_prompt_token_count"] == 500
    # The modality breakdown survives, enums resolved to plain JSON.
    assert archived["prompt_tokens_details"][0]["token_count"] == 10_000
    assert "TEXT" in str(archived["prompt_tokens_details"][0]["modality"])


# ── cost ──────────────────────────────────────────────────────────────────────
def test_cost_uses_the_intro_rate_inside_the_window(settings):
    """gemini-3.6-flash bills $0.75/$3.75 through 2026-12-31, not the list rate."""
    usage = cfg.TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    intro, basis = cfg.cost_usd(FLASH, usage, on_date="2026-12-31")
    listed, listed_basis = cfg.cost_usd(FLASH, usage, on_date="2027-01-01")
    assert (intro, basis) == (pytest.approx(0.75 + 3.75), "intro")
    assert (listed, listed_basis) == (pytest.approx(1.50 + 7.50), "list")


def test_the_intro_window_also_discounts_cache_reads():
    """Google discounts input, output and caching together; pricing a cache read
    at the list rate inside the window would overstate the cost."""
    usage = cfg.TokenUsage(cache_read_tokens=1_000_000)
    inside, _ = cfg.cost_usd(FLASH, usage, on_date="2026-12-31")
    outside, _ = cfg.cost_usd(FLASH, usage, on_date="2027-01-01")
    assert inside == pytest.approx(0.075)
    assert outside == pytest.approx(0.15)


def test_every_token_class_moves_the_cost(settings):
    """No billable quantity may be silently free."""
    base = _provider(_usage(prompt=1_000, candidates=100, thoughts=0)) \
        .generate(PRO31, None, [{"role": "user", "content": "hi"}], settings).cost_usd
    for bumped in (_usage(prompt=2_000, candidates=100, thoughts=0),
                   _usage(prompt=1_000, candidates=200, thoughts=0),
                   _usage(prompt=1_000, candidates=100, thoughts=100)):
        cost = _provider(bumped).generate(
            PRO31, None, [{"role": "user", "content": "hi"}], settings).cost_usd
        assert cost > base


def test_google_search_grounding_is_priced_at_googles_own_rate():
    """$14 per 1,000 grounded prompts, not Anthropic's $10 per 1,000 searches."""
    usage = cfg.TokenUsage(web_search_requests=1_000)
    cost, _ = cfg.cost_usd(FLASH, usage)
    assert cost == pytest.approx(14.00)
    assert cfg.web_search_price("google") == 14.00
    assert cfg.web_search_price("anthropic") == 10.00


# ── the guards ────────────────────────────────────────────────────────────────
def test_a_complete_usage_object_reports_nothing_untracked(settings):
    result = _provider().generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.untracked_usage == ""


def test_a_usage_field_the_cost_model_does_not_price_is_flagged():
    usage = _usage()
    leaked = usage.model_copy(update={"tool_use_prompt_token_count": 700})
    assert "tool_use_prompt_token_count" in _GOOGLE_ACCOUNTED
    # ...and one that is not accounted for must be reported.
    assert _unaccounted_usage_fields(leaked, _GOOGLE_ACCOUNTED - {"tool_use_prompt_token_count"})


def test_audio_input_is_flagged_because_it_costs_more_than_text():
    """Gemini charges a higher input rate for audio; the cost model prices text."""
    audio = _usage(modality=MediaModality.AUDIO)
    assert "AUDIO" in _google_modality_warning(audio)
    assert _google_modality_warning(_usage()) == ""


def test_a_total_that_does_not_reconcile_is_reported(settings):
    """Gemini states its own total. If the split cannot reproduce it, a token
    class exists on this response that the cost model does not know about."""
    odd = _usage(prompt=1_000, candidates=100, thoughts=0, total=9_999)
    result = _provider(odd).generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert "does not reconcile" in result.untracked_usage
    assert "9999" in result.untracked_usage.replace(",", "")


# ── retries ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("exc, retry", [
    (Exception("503 UNAVAILABLE. This model is currently experiencing high demand"), True),
    (Exception("429 RESOURCE_EXHAUSTED"), True),
    (Exception("500 INTERNAL"), True),
    (Exception("404 NOT_FOUND. no longer available to new users"), False),
    (Exception("400 INVALID_ARGUMENT"), False),
    (Exception("403 PERMISSION_DENIED"), False),
])
def test_only_transient_failures_are_retried(exc, retry):
    assert GoogleProvider._is_retryable(exc) is retry


def test_the_response_id_is_captured(settings):
    """Gemini sets no request-id header — response_id is the only handle."""
    result = _provider().generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.response_id == "resp-123"
