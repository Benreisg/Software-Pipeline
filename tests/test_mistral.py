"""The Mistral provider — usage accounting, the chunked reasoning reply, cost.

Stubs `client.chat.complete` and feeds the provider the SDK's own `UsageInfo`
type, so what is exercised is the code a live run runs.

Mistral's usage object reports prompt, completion, total, a cached share of the
prompt under three documented names, and a service tier. Two consequences drive
most of what is tested here:

    * no reasoning-token count exists at all, so a reasoning call's thinking is
      inside completion_tokens with no way to separate it;
    * with `reasoning_effort` set, `message.content` is a list of chunks rather
      than a string, and the ThinkChunk must not reach the generated DOT.
"""
from __future__ import annotations

import json
import types

import pytest

import config as cfg
from mistralai.client.models import UsageInfo
from providers import (
    MistralProvider, _MISTRAL_ACCOUNTED, _mistral_cache_counts, _mistral_text,
    _unaccounted_usage_fields,
)

LARGE = cfg.model_by_key("mistral_standard")    # mistral-large-latest   $0.50/$1.50
MEDIUM = cfg.model_by_key("mistral_advanced")   # mistral-medium-latest  $1.50/$7.50
SMALL = cfg.model_by_key("mistral_efficient")   # mistral-small-latest   $0.15/$0.60


def _usage(prompt=10_000, completion=2_000, cached=0, tier="standard", total=None,
           singular_cached=None, num_cached=None, top_level_cached=None,
           reasoning=None, audio_seconds=None, audio_tokens=None,
           request_count=None, messages=None):
    """A Chat usage object. Everything past `service_tier` is a field the
    schema documents but no observed response has carried — `UsageInfo`
    declares five fields and accepts the rest as pydantic extras, which is
    exactly how a live response would deliver them."""
    kwargs = dict(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
        service_tier=tier,
    )
    details = {}
    if cached is not None:
        details["cached_tokens"] = cached
    if audio_tokens is not None:
        details["audio_tokens"] = audio_tokens
    if messages is not None:
        details["messages"] = messages
    if details:
        kwargs["prompt_tokens_details"] = details
    if singular_cached is not None:
        kwargs["prompt_token_details"] = {"cached_tokens": singular_cached}
    if num_cached is not None:
        kwargs["num_cached_tokens"] = num_cached
    if top_level_cached is not None:
        kwargs["cached_tokens"] = top_level_cached
    if reasoning is not None:
        kwargs["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    if audio_seconds is not None:
        kwargs["prompt_audio_seconds"] = audio_seconds
    if request_count is not None:
        kwargs["request_count"] = request_count
    return UsageInfo(**kwargs)


class _Think:
    """Stand-in for the SDK's ThinkChunk."""
    type = "thinking"

    def __init__(self, text):
        self.thinking = [types.SimpleNamespace(text=text)]


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


def _stub(usage, content="digraph g { a -> b }", finish="stop", capture=None):
    def complete(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(
            id="cmpl-123", usage=usage,
            choices=[types.SimpleNamespace(message=message, finish_reason=finish)])
    return types.SimpleNamespace(chat=types.SimpleNamespace(complete=complete))


def _provider(usage=None, **stub_kwargs):
    provider = MistralProvider.__new__(MistralProvider)   # no client construction
    provider._client = _stub(usage if usage is not None else _usage(), **stub_kwargs)
    provider._max_retries = 5
    return provider


# ── catalogue ─────────────────────────────────────────────────────────────────
def test_mistral_is_a_live_vendor_with_verified_prices():
    assert "mistral" in cfg.LIVE_VENDORS
    for model in (LARGE, MEDIUM, SMALL):
        assert model.price_verified, model.key
        assert model.price_cached_in_per_mtok is not None, model.key


def test_the_retired_magistral_medium_is_gone():
    """`magistral-medium-latest` no longer exists in the API."""
    for model in cfg.models_by_vendor("mistral"):
        assert "magistral" not in model.model_id


def test_the_tiers_follow_price_not_the_name():
    """Mistral's naming is no longer its price ladder — Large 3 is the cheap
    fast model and Medium 3.5 the premium one."""
    assert LARGE.price_in_per_mtok < MEDIUM.price_in_per_mtok
    assert SMALL.price_in_per_mtok < LARGE.price_in_per_mtok
    assert "large" in LARGE.model_id and "medium" in MEDIUM.model_id


def test_the_pricing_sheet_is_dated():
    sheet = cfg.pricing_sheet("mistral")
    assert sheet["as_of"] == "2026-08-30"
    assert "mistral.ai" in sheet["source"]


# ── the chunked reasoning reply ───────────────────────────────────────────────
def test_thinking_chunks_never_reach_the_generated_text():
    """With reasoning on, content is a chunk list. Passing it through would put
    the model's monologue into the .gv file."""
    content = [_Think("Okay, the user wants a digraph..."), _Text("digraph { a -> b }")]
    text, thinking_chars = _mistral_text(content)
    assert text == "digraph { a -> b }"
    assert "Okay" not in text
    assert thinking_chars == len("Okay, the user wants a digraph...")


def test_a_plain_string_reply_is_passed_through():
    assert _mistral_text("digraph { a }") == ("digraph { a }", 0)
    assert _mistral_text(None) == ("", 0)


def test_the_advanced_tier_returns_only_the_answer(settings):
    content = [_Think("thinking hard"), _Text("digraph g { a -> b }")]
    result = _provider(content=content).generate(
        MEDIUM, None, [{"role": "user", "content": "hi"}], settings)
    assert result.text == "digraph g { a -> b }"
    assert "thinking hard" not in result.text


# ── request shape ─────────────────────────────────────────────────────────────
def test_the_advanced_tier_uses_high_reasoning_effort(settings):
    """Medium 3.5 explicitly receives the requested high reasoning effort."""
    sent = {}
    _provider(capture=sent).generate(MEDIUM, None,
                                     [{"role": "user", "content": "hi"}], settings)
    assert sent["reasoning_effort"] == "high"
    assert MEDIUM.thinking and not MEDIUM.reasoning_off
    # No provider writes a system prompt into its request any more
    # (removed 2026-08-31), so this asserts the slot stays empty.
    assert all(m["role"] != "system" for m in sent["messages"])
    # No service_tier on any vendor: omitting it means `auto`, and the tier
    # that served the call is read back from the response instead — see the
    # service-tier note in config.py.
    assert "service_tier" not in sent
    # prompt_mode is the other reasoning switch and Medium 3.5 rejects it with
    # a 400 — it must not be sent.
    assert "prompt_mode" not in sent


def test_the_efficient_tier_asks_for_no_thinking_explicitly(settings):
    """Small 4 is reasoning-capable, and Mistral documents `none` but not what
    an omitted parameter does. Sending nothing would leave the tier's reasoning
    state undocumented — measured, but not known."""
    sent = {}
    _provider(capture=sent).generate(SMALL, None, [{"role": "user", "content": "hi"}], settings)
    assert sent["reasoning_effort"] == "none"


def test_a_model_without_a_reasoning_switch_is_sent_none_of_it(settings):
    """Large 3 is not documented as taking `reasoning_effort`, so the parameter
    stays off the request rather than being sent on the chance it is ignored."""
    sent = {}
    _provider(capture=sent).generate(LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert "reasoning_effort" not in sent
    assert not LARGE.reasoning_off and not LARGE.thinking


def test_temperature_still_reaches_a_tier_with_reasoning_switched_off(settings):
    """Switching thinking off must not cost the tier its sampling setting."""
    sent = {}
    _provider(capture=sent).generate(
        SMALL, None, [{"role": "user", "content": "hi"}],
        cfg.RunSettings(temperature=0.3, max_output_tokens=4096, n_few_shot=2))
    assert sent["temperature"] == 0.3
    assert sent["reasoning_effort"] == "none"


def test_a_service_tier_the_cost_model_cannot_price_is_flagged(settings):
    """Priority costs 1.75x standard on Mistral. This project requests no tier,
    so one showing up means an assumption broke — and cost_usd is then not the
    amount charged."""
    result = _provider(_usage(tier="priority")).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.service_tier == "priority"
    assert "not the list-rate tier" in result.untracked_usage

    standard = _provider(_usage(tier="standard")).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert standard.untracked_usage == ""


# ── the token split ───────────────────────────────────────────────────────────
def test_cached_tokens_are_split_out_of_the_prompt(settings):
    result = _provider(_usage(prompt=10_000, cached=6_000)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.input_tokens == 4_000
    assert result.cached_tokens == 6_000
    assert result.billable_input_tokens == 10_000
    assert result.reported_input_tokens == 10_000   # prompt_tokens, verbatim
    assert result.reported_total_tokens == 12_000   # total_tokens, verbatim


@pytest.mark.parametrize("usage", [
    _usage(cached=6_000),
    _usage(cached=None, singular_cached=6_000),
    _usage(cached=None, num_cached=6_000),
    _usage(cached=6_000, singular_cached=6_000, num_cached=6_000),
])
def test_every_documented_cache_alias_is_collected_once(settings, usage):
    """The Chat schema currently names the same cache share three ways. A
    response may use any one of them, or repeat the value under all of them."""
    result = _provider(usage).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 6_000
    assert result.input_tokens == 4_000
    assert result.billable_input_tokens == 10_000
    assert result.untracked_usage == ""


def test_every_cache_alias_survives_in_the_raw_usage_for_the_results_report(settings):
    usage = _usage(cached=6_000, singular_cached=6_000, num_cached=6_000)
    result = _provider(usage).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    archived = json.loads(result.raw_usage)
    assert archived["prompt_tokens_details"]["cached_tokens"] == 6_000
    assert archived["prompt_token_details"]["cached_tokens"] == 6_000
    assert archived["num_cached_tokens"] == 6_000


def test_disagreeing_cache_aliases_are_not_guessed_at(settings):
    result = _provider(_usage(cached=6_000, singular_cached=5_000)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 0
    assert result.input_tokens == 10_000
    assert "cached-token aliases do not reconcile" in result.untracked_usage
    assert "prompt_tokens_details.cached_tokens=6000" in result.untracked_usage
    assert "prompt_token_details.cached_tokens=5000" in result.untracked_usage


def test_a_cache_count_larger_than_the_prompt_is_not_priced_as_real(settings):
    result = _provider(_usage(prompt=1_000, cached=None, num_cached=1_001)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 0
    assert result.input_tokens == 1_000
    assert "does not fit inside prompt_tokens=1000" in result.untracked_usage


def test_no_reasoning_token_count_is_invented(settings):
    """No observed response carries one; reporting a number here is a guess."""
    result = _provider().generate(
        MEDIUM, None, [{"role": "user", "content": "hi"}], settings)
    assert result.thinking_tokens is None
    assert "carried none" in result.reasoning_note
    assert "completion_tokens_details.reasoning_tokens" in result.reasoning_note


# ── cost ──────────────────────────────────────────────────────────────────────
def test_cost_matches_the_published_rates():
    usage = cfg.TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cfg.cost_usd(LARGE, usage)[0] == pytest.approx(0.50 + 1.50)
    assert cfg.cost_usd(MEDIUM, usage)[0] == pytest.approx(1.50 + 7.50)
    assert cfg.cost_usd(SMALL, usage)[0] == pytest.approx(0.15 + 0.60)


def test_a_cache_read_costs_a_tenth_of_the_input_rate():
    """Mistral advertises -90% on cached input; the per-model rates encode it."""
    for model in (LARGE, MEDIUM, SMALL):
        assert model.price_cached_in_per_mtok == pytest.approx(
            model.price_in_per_mtok * 0.10)


def test_every_token_class_moves_the_cost(settings):
    base = _provider(_usage(prompt=1_000, completion=100)).generate(
        MEDIUM, None, [{"role": "user", "content": "hi"}], settings).cost_usd
    for bumped in (_usage(prompt=2_000, completion=100),
                   _usage(prompt=1_000, completion=200),
                   _usage(prompt=2_000, completion=100, cached=500)):
        cost = _provider(bumped).generate(
            MEDIUM, None, [{"role": "user", "content": "hi"}], settings).cost_usd
        assert cost > base


def test_the_batch_tier_is_half_price(settings):
    """Mistral sells batch at half price; a batch row must not be priced as a
    standard one."""
    standard = _provider(_usage(tier="standard")).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    batch = _provider(_usage(tier="batch")).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert batch.cost_usd == pytest.approx(standard.cost_usd / 2)
    assert "batch" in batch.cost_basis


# ── the guards ────────────────────────────────────────────────────────────────
def test_a_complete_usage_object_reports_nothing_untracked(settings):
    result = _provider().generate(LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.untracked_usage == ""


def test_a_usage_field_the_cost_model_does_not_price_is_flagged():
    usage = _usage(cached=500)
    documented_cache_paths = {
        "prompt_tokens_details.cached_tokens",
        "prompt_token_details.cached_tokens",
        "num_cached_tokens",
        "cached_tokens",
    }
    assert documented_cache_paths <= _MISTRAL_ACCOUNTED
    assert _mistral_cache_counts(usage) == {
        "prompt_tokens_details.cached_tokens": 500,
    }
    assert _unaccounted_usage_fields(
        usage, _MISTRAL_ACCOUNTED - {"prompt_tokens_details.cached_tokens"})


def test_a_total_that_does_not_reconcile_is_reported(settings):
    odd = _usage(prompt=1_000, completion=100, total=9_999)
    result = _provider(odd).generate(LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert "does not reconcile" in result.untracked_usage


# ── retries ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("exc, retry", [
    (Exception("API error occurred: Status 429"), True),
    (Exception("API error occurred: Status 503"), True),
    (Exception("API error occurred: Status 400. Reasoning prompt mode is not enabled"), False),
    (Exception("API error occurred: Status 401"), False),
])
def test_only_transient_failures_are_retried(exc, retry):
    assert MistralProvider._is_retryable(exc) is retry


def test_the_response_id_is_captured(settings):
    result = _provider().generate(LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.response_id == "cmpl-123"


# ── the rest of the documented usage schema ───────────────────────────────────
# None of these fields has been seen on a chat completion: checked against the
# live API on 2026-08-29 (plain, reasoning on, reasoning off, and a 15k-token
# prompt sent twice to provoke a cache hit), on the wire as well as through the
# SDK, every response carried prompt/completion/total, one cached alias and the
# service tier. They are read anyway — the day Mistral starts sending one, it
# has to arrive as a measurement rather than as a surprise on the invoice.
def test_the_fourth_cached_alias_is_read_and_priced_once(settings):
    """The schema also spells the cached share at the top level."""
    usage = _usage(prompt=1_000, cached=None, top_level_cached=400)
    assert _mistral_cache_counts(usage) == {"cached_tokens": 400}
    result = _provider(usage).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 400
    assert result.input_tokens == 600
    assert result.untracked_usage == ""


def test_all_four_cached_aliases_are_reconciled_not_summed(settings):
    result = _provider(_usage(prompt=1_000, cached=400, singular_cached=400,
                              num_cached=400, top_level_cached=400)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 400
    assert result.input_tokens == 600
    assert result.untracked_usage == ""


def test_a_fourth_alias_that_disagrees_is_not_priced(settings):
    result = _provider(_usage(prompt=1_000, cached=400, top_level_cached=900)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 0
    assert "do not reconcile" in result.untracked_usage


def test_a_reported_reasoning_count_is_carried(settings):
    """`completion_tokens_details.reasoning_tokens` is a breakdown of the
    completion count: informational, already billed at the output rate."""
    plain = _provider(_usage(prompt=1_000, completion=500)).generate(
        MEDIUM, None, [{"role": "user", "content": "hi"}], settings)
    withreasoning = _provider(_usage(prompt=1_000, completion=500, reasoning=300)).generate(
        MEDIUM, None, [{"role": "user", "content": "hi"}], settings)
    assert withreasoning.thinking_tokens == 300
    assert withreasoning.output_tokens == 500       # inside, not beside
    assert withreasoning.cost_usd == plain.cost_usd  # so it must not add cost
    assert withreasoning.untracked_usage == ""
    assert "reasoning_tokens=300" in withreasoning.reasoning_note


def test_audio_seconds_are_measured_and_flagged(settings):
    """The one quantity billed in a unit this project has no rate for."""
    result = _provider(_usage(audio_seconds=12)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.audio_input_seconds == 12
    assert "prompt_audio_seconds=12" in result.untracked_usage
    assert "floor" in result.untracked_usage


def test_a_text_call_reports_no_audio(settings):
    result = _provider().generate(LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.audio_input_seconds == 0.0


def test_audio_prompt_tokens_are_flagged_as_priced_at_the_text_rate(settings):
    """They are inside prompt_tokens and therefore already charged — but at the
    text rate, which is not what Mistral bills audio at."""
    result = _provider(_usage(prompt=1_000, audio_tokens=250)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert "audio_tokens=250" in result.untracked_usage
    assert "understated" in result.untracked_usage


def test_a_usage_object_covering_several_requests_is_flagged(settings):
    result = _provider(_usage(request_count=3)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert "request_count=3" in result.untracked_usage
    result = _provider(_usage(request_count=1)).generate(
        LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.untracked_usage == ""


def test_the_per_message_prompt_breakdown_is_not_read_as_an_extra_charge(settings):
    """`prompt_tokens_details.messages[]` breaks prompt_tokens down per message.
    The guard walks lists now, so it would double-report these without the
    accounted paths that name them."""
    result = _provider(_usage(prompt=1_000, messages=[
        {"total_tokens": 600, "usage_count": 1, "truncated": False},
        {"total_tokens": 400, "usage_count": 1, "truncated": False},
    ])).generate(LARGE, None, [{"role": "user", "content": "hi"}], settings)
    assert result.untracked_usage == ""
    assert result.input_tokens == 1_000


def test_a_billable_unit_that_is_not_tokens_cannot_pass_unnoticed():
    """The guard's reason for existing: a unit nobody has seen yet. It matches
    on the unit in the name, so seconds, characters and requests are caught the
    same way tokens are."""
    for field, value in (("prompt_video_seconds", 30), ("output_characters", 900),
                         ("image_count", 4), ("connector_calls", 2)):
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                          **{field: value})
        assert f"{field}={value}" in _unaccounted_usage_fields(
            usage, _MISTRAL_ACCOUNTED)
