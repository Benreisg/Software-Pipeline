"""The OpenAI provider — Responses API request shape, usage split, cost.

Stubs `client.responses.create` and feeds the provider the SDK's own
`ResponseUsage` type, so what is exercised is the code a live run runs.
"""
from __future__ import annotations

import types

import pytest

import config as cfg
import prompts
from openai.types.responses.response_usage import (
    InputTokensDetails, OutputTokensDetails, ResponseUsage,
)
from providers import OpenAIProvider, _OPENAI_ACCOUNTED, _unaccounted_usage_fields, \
    _is_permanent_quota_error, is_fatal_account_error

TERRA = cfg.model_by_key("openai_standard")    # gpt-5.6-terra  $2/$12,   cached $0.20
SOL = cfg.model_by_key("openai_advanced")      # gpt-5.6-sol    $5/$30,   cached $0.50
                                               # (promo $4/$20, cached $0.40, to 2026-11-21)
LUNA = cfg.model_by_key("openai_efficient")    # gpt-5.6-luna   $0.20/$1.20, cached $0.02


def _usage(input_tokens=10_000, cached=6_000, write=1_000, output=2_000, reasoning=1_200,
           total=None):
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached,
                                                cache_write_tokens=write),
        output_tokens=output,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning),
        total_tokens=input_tokens + output if total is None else total,
    )


def _stub(usage, status="completed", incomplete=None, capture=None,
          service_tier="default"):
    def create(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return types.SimpleNamespace(output_text="digraph g { a -> b }", usage=usage,
                                     status=status, incomplete_details=incomplete,
                                     service_tier=service_tier, model="stub")
    return types.SimpleNamespace(responses=types.SimpleNamespace(create=create))


def _provider(usage=None, **stub_kwargs):
    provider = OpenAIProvider(api_key="sk-test-not-used")
    provider._client = _stub(usage or _usage(), **stub_kwargs)
    return provider


# ── catalogue ─────────────────────────────────────────────────────────────────
def test_openai_is_a_live_vendor_with_verified_prices():
    assert "openai" in cfg.LIVE_VENDORS
    for model in (TERRA, SOL, LUNA):
        assert model.price_verified
        assert model.price_cached_in_per_mtok is not None
        assert not model.supports_temperature   # unconfirmed on gpt-5.6 → never sent


# ── request shape ─────────────────────────────────────────────────────────────
def test_the_advanced_tier_sends_effort_and_the_system_prompt_as_instructions(settings):
    sent = {}
    _provider(capture=sent).generate(SOL, "SYSTEM TEXT",
                                     [{"role": "user", "content": "hi"}], settings)
    assert sent["instructions"] == "SYSTEM TEXT"
    assert sent["reasoning"] == {"effort": "high"}
    assert sent["max_output_tokens"] == settings.max_output_tokens
    assert "temperature" not in sent


def test_a_lower_tier_sends_no_reasoning_parameter(settings):
    sent = {}
    _provider(capture=sent).generate(LUNA, None,
                                     [{"role": "user", "content": "hi"}], settings)
    assert "reasoning" not in sent
    assert "instructions" not in sent


def test_few_shot_transcript_survives_as_input_items(settings):
    """The Responses API takes assistant turns in `input` — few-shot depends on it."""
    exemplars = prompts.load_exemplars(
        [cfg.PMO_DATASET_DIR / "graphviz" / f"{i}.dot" for i in ("01", "02")])
    system, messages = prompts.build_messages("few_shot", "DESCRIPTION", exemplars=exemplars)
    sent = {}
    _provider(capture=sent).generate(TERRA, system, messages, settings)
    assert [m["role"] for m in sent["input"]] == \
        ["user", "assistant", "user", "assistant", "user"]


# ── usage split and cost ──────────────────────────────────────────────────────
def test_inclusive_input_tokens_are_split_into_uncached_cached_and_written(settings):
    """OpenAI's input_tokens covers the whole prompt; Anthropic's is the uncached
    remainder. The provider normalises so the column means one thing."""
    gen = _provider().generate(SOL, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.input_tokens == 3_000          # 10_000 - 6_000 cached - 1_000 written
    assert gen.cached_tokens == 6_000
    assert gen.cache_write_tokens == 1_000
    assert gen.billable_input_tokens == 10_000
    assert gen.total_tokens == 12_000
    assert gen.untracked_usage == ""

    # Rates come from the basis the row itself recorded rather than being frozen
    # here: sol is inside a promotional window until 2026-11-21, and a hard-coded
    # rate would turn the day it closes into a test failure about arithmetic that
    # never changed. What is asserted is the split above times the published
    # rates for that basis — the same check a reader of the CSV would make.
    rates = cfg.rates_for(SOL, gen.cost_basis)
    expected = (3_000 * rates["input_tokens"] + 6_000 * rates["cached_tokens"]
                + 1_000 * rates["cache_write_tokens"]
                + 2_000 * rates["output_tokens"]) / 1e6
    assert gen.cost_usd == pytest.approx(expected, abs=1e-12)


def test_the_vendors_own_input_count_survives_the_normalisation(settings):
    """`input_tokens` is normalised down to the uncached remainder, so OpenAI's
    own figure would otherwise be unrecoverable from the row."""
    gen = _provider().generate(SOL, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.reported_input_tokens == 10_000     # exactly usage.input_tokens
    assert gen.input_tokens == 3_000               # the normalised column
    assert (gen.reported_input_tokens
            == gen.input_tokens + gen.cached_tokens + gen.cache_write_tokens)


def test_the_serving_tier_is_recorded_and_checked(settings):
    """The Responses API states the tier on the response, not in usage. `flex`
    is cheaper and `priority` dearer than the rates this cost model holds, so
    either one makes cost_usd the wrong number — it has to be said."""
    default = _provider().generate(SOL, None, [{"role": "user", "content": "hi"}],
                                   settings)
    assert default.service_tier == "default"
    assert default.untracked_usage == ""

    provider = _provider()
    provider._client = _stub(_usage(), service_tier="priority")
    priority = provider.generate(SOL, None, [{"role": "user", "content": "hi"}],
                                 settings)
    assert priority.service_tier == "priority"
    assert "not the list-rate tier" in priority.untracked_usage


def test_the_vendors_own_total_survives_and_is_checked(settings):
    """OpenAI states a grand total, so it is kept verbatim next to this
    project's own sum — and the two are compared rather than assumed equal."""
    gen = _provider().generate(SOL, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.reported_total_tokens == 12_000     # exactly usage.total_tokens
    assert gen.reported_output_tokens == gen.output_tokens == 2_000   # no split to make
    assert '"total_tokens": 12000' in gen.raw_usage                   # archived verbatim
    assert gen.total_tokens == 12_000              # the priced sum agrees
    assert gen.untracked_usage == ""


def test_a_total_that_input_and_output_do_not_account_for_is_flagged(settings):
    """A total the split cannot explain means a token class this provider does
    not know about — the guard Mistral, DeepSeek and Gemini already had."""
    gen = _provider(_usage(total=13_500)).generate(
        SOL, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.reported_total_tokens == 13_500
    assert "token split does not reconcile" in gen.untracked_usage
    assert "total_tokens=13500" in gen.untracked_usage


def test_one_anomaly_is_not_reported_twice(settings):
    """A cache split that does not fit already says the input is unreliable;
    the total check would only restate it."""
    gen = _provider(_usage(input_tokens=100, cached=90, write=90)).generate(
        SOL, None, [{"role": "user", "content": "hi"}], settings)
    assert "exceed" in gen.untracked_usage
    assert "token split does not reconcile" not in gen.untracked_usage


def test_a_split_that_does_not_reconcile_is_flagged(settings):
    """Rather than silently pick a reading, price the safe one and say so."""
    gen = _provider(_usage(input_tokens=100, cached=90, write=90)).generate(
        TERRA, None, [{"role": "user", "content": "hi"}], settings)
    assert "exceed" in gen.untracked_usage


def test_reasoning_tokens_are_reported_and_sit_inside_output(settings):
    gen = _provider().generate(SOL, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.thinking_tokens == 1_200 < gen.output_tokens


# ── completion state ──────────────────────────────────────────────────────────
def test_completed_maps_to_end_turn(settings):
    gen = _provider().generate(TERRA, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.stop_reason == "end_turn"


def test_truncation_maps_onto_the_same_column_as_anthropic(settings):
    gen = _provider(status="incomplete",
                    incomplete=types.SimpleNamespace(reason="max_output_tokens")).generate(
        TERRA, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.stop_reason == "max_output_tokens"


# ── guards ────────────────────────────────────────────────────────────────────
def test_guard_covers_the_openai_usage_object():
    usage = types.SimpleNamespace(input_tokens=10, output_tokens=20, audio_input_tokens=42)
    assert "audio_input_tokens=42" in _unaccounted_usage_fields(usage, _OPENAI_ACCOUNTED)


def test_exhausted_credit_is_not_retried_but_a_rate_limit_is():
    """A 429 for an empty balance never succeeds; retrying it burns the run."""
    empty_balance = types.SimpleNamespace(
        body={"error": {"code": "credit_balance_exhausted",
                        "message": "You have no credits remaining."}})
    assert _is_permanent_quota_error(empty_balance)
    assert not _is_permanent_quota_error(Exception("429 rate_limit_exceeded, slow down"))


def test_account_errors_are_fatal_for_the_whole_run():
    assert is_fatal_account_error("Error code: 429 ... insufficient_quota")
    assert is_fatal_account_error("authentication failed — check the API key")
    assert not is_fatal_account_error("overloaded_error, please retry")
