"""Cost accounting — the formula, the rate basis, and the Anthropic live path.

The rule these tests defend: **no cost of an API call may go untracked.** Every
billable token class the vendor reports has to reach `cost_usd`, and anything
the cost model does not price has to surface in `untracked_usage` rather than
being silently worth zero.
"""
from __future__ import annotations

import types

import pytest

import config as cfg
from providers import AnthropicProvider, _ANTHROPIC_ACCOUNTED, _unaccounted_usage_fields

SONNET = cfg.model_by_key("anthropic_standard")   # $2/$10 list, promo retired 2026-08-30
OPUS = cfg.model_by_key("anthropic_advanced")     # $5/$25, no promo
# The promotional window moved vendors on 2026-08-30: Sonnet 5's became the list
# price, and gpt-5.6-sol opened one. These tests follow whichever model actually
# has a window, because an intro window that no model exercises is untested code.
SOL = cfg.model_by_key("openai_advanced")         # $5/$30 list, $4/$20 promo to 2026-11-21


# ── the formula ───────────────────────────────────────────────────────────────
def test_list_rate_applies_after_the_promotional_window():
    usage = cfg.TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cfg.cost_usd(SOL, usage, on_date="2026-11-22") == (35.0, "list")


def test_promotional_rate_applies_inside_the_window_and_on_its_last_day():
    usage = cfg.TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cfg.cost_usd(SOL, usage, on_date="2026-08-30") == (24.0, "intro")
    assert cfg.cost_usd(SOL, usage, on_date=SOL.intro_until)[1] == "intro"


def test_sonnet_5_no_longer_has_a_promotional_window():
    """Regression guard on the 2026-08-30 sheet check. Sonnet 5's $2/$10 launch
    rate BECAME the list price; the increase to $3/$15 announced for 2026-09-01
    was cancelled. While this file still encoded that increase, every Sonnet 5
    call from 2026-09-01 on would have been priced 50 % above what it cost.
    """
    usage = cfg.TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert SONNET.intro_until is None
    assert cfg.cost_usd(SONNET, usage, on_date="2026-09-01") == (12.0, "list")


def test_every_billable_token_class_is_priced():
    usage = cfg.TokenUsage(
        input_tokens=100_000,          # 100k * $5      = 0.5000
        cache_read_tokens=200_000,     # 200k * $5*0.10 = 0.1000
        cache_write_tokens=50_000,  #  50k * $5*1.25 = 0.3125
        cache_write_1h_tokens=10_000,  #  10k * $5*2.00 = 0.1000
        output_tokens=20_000,          #  20k * $25     = 0.5000
    )
    cost, _ = cfg.cost_usd(OPUS, usage)
    assert cost == pytest.approx(1.5125, abs=1e-12)
    # The old input+output-only formula missed a third of this bill.
    naive = (usage.input_tokens * 5.0 + usage.output_tokens * 25.0) / 1e6
    assert cost > naive
    assert usage.billable_input_tokens == 360_000
    assert usage.total_tokens == 380_000


def test_billable_input_is_the_whole_prompt_not_the_uncached_remainder():
    usage = cfg.TokenUsage(input_tokens=10, cache_read_tokens=20, cache_write_tokens=30)
    assert usage.billable_input_tokens == 60


def test_batch_tier_halves_the_bill():
    usage = cfg.TokenUsage(input_tokens=1_000_000, batch=True)
    assert cfg.cost_usd(OPUS, usage) == (2.5, "list+batch")


def test_unverified_prices_give_none_not_zero():
    """0.0 would read as 'this call was free'. It wasn't — we just can't price it."""
    # Every model in the catalogue is priced now, so the rule is exercised on a
    # spec built for the purpose — it is the mechanism that must hold, not the
    # accident of some vendor still being unconfirmed.
    import dataclasses
    unpriced = dataclasses.replace(cfg.model_by_key("anthropic_standard"),
                                   price_in_per_mtok=None, price_out_per_mtok=None,
                                   intro_price_in_per_mtok=None, price_verified=False)
    cost, basis = cfg.cost_usd(unpriced,
                               cfg.TokenUsage(input_tokens=1_000, output_tokens=1_000))
    assert cost is None and basis == "unverified"


def test_every_live_vendor_can_actually_be_priced():
    """The other side of the same rule: a vendor that performs real API calls
    must have a rate for every one of its models, or a paid run produces rows
    with no cost at all."""
    for model in cfg.ALL_MODELS:
        if model.vendor in cfg.LIVE_VENDORS:
            assert model.price_verified, model.key
            cost, basis = cfg.cost_usd(model, cfg.TokenUsage(input_tokens=1_000))
            assert cost is not None and basis != "unverified", model.key


def test_server_tools_are_priced_per_request():
    cost, _ = cfg.cost_usd(OPUS, cfg.TokenUsage(web_search_requests=250))
    assert cost == pytest.approx(2.50, abs=1e-12)


def test_per_model_cache_rate_beats_the_flat_multiplier():
    """OpenAI publishes a cached rate per model; Anthropic uses a flat 0.10x."""
    luna = cfg.model_by_key("openai_efficient")   # $0.20 in, cached $0.02
    usage = cfg.TokenUsage(input_tokens=1_000_000, cache_read_tokens=1_000_000)
    cost, _ = cfg.cost_usd(luna, usage)
    assert cost == pytest.approx(0.22, abs=1e-12)


# ── the untracked-usage guard ─────────────────────────────────────────────────
def test_guard_flags_a_usage_field_the_cost_model_does_not_price():
    usage = types.SimpleNamespace(input_tokens=10, output_tokens=20,
                                  some_future_tier_input_tokens=999)
    assert "some_future_tier_input_tokens=999" in _unaccounted_usage_fields(
        usage, _ANTHROPIC_ACCOUNTED)


def test_guard_stays_silent_when_the_accounting_is_complete():
    usage = types.SimpleNamespace(input_tokens=10, output_tokens=20,
                                  cache_read_input_tokens=0)
    assert _unaccounted_usage_fields(usage, _ANTHROPIC_ACCOUNTED) == ""


# ── the live Anthropic path, transport stubbed ────────────────────────────────
def _anthropic_stub(usage):
    return types.SimpleNamespace(messages=types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="digraph g { a -> b }")],
            usage=usage)))


def test_anthropic_prices_the_full_usage_object(settings):
    from anthropic.types import Usage
    from anthropic.types.cache_creation import CacheCreation
    from anthropic.types.output_tokens_details import OutputTokensDetails

    provider = AnthropicProvider(api_key="sk-test-not-used")
    provider._client = _anthropic_stub(Usage(
        input_tokens=1_000, output_tokens=2_000,
        cache_read_input_tokens=8_000,
        cache_creation_input_tokens=5_000,
        cache_creation=CacheCreation(ephemeral_5m_input_tokens=4_000,
                                     ephemeral_1h_input_tokens=1_000),
        output_tokens_details=OutputTokensDetails(thinking_tokens=1_500),
        service_tier="standard",
    ))
    gen = provider.generate(OPUS, None, [{"role": "user", "content": "hi"}], settings)

    expected = (1_000 * 5 + 8_000 * 5 * 0.10 + 4_000 * 5 * 1.25
                + 1_000 * 5 * 2.0 + 2_000 * 25) / 1e6
    assert gen.cost_usd == pytest.approx(expected, abs=1e-12)
    # Anthropic already reports the uncached remainder, so here the raw figure
    # and the normalised one coincide — they do on no other vendor.
    assert gen.reported_input_tokens == gen.input_tokens == 1_000
    # And its usage object states no grand total at all, so there is none to
    # keep: 0 here means "not reported", not "reported as zero".
    assert gen.reported_total_tokens == 0
    # The flat writes total is kept beside its 5m/1h split, and the two agree.
    assert gen.reported_cache_creation_tokens == 5_000
    # Anthropic states no separate answer count, so reported == billable here.
    assert gen.reported_output_tokens == gen.output_tokens == 2_000
    assert gen.tool_use_prompt_tokens == 0      # no such class on this vendor
    assert gen.cache_write_tokens + gen.cache_write_1h_tokens == 5_000
    assert gen.billable_input_tokens == 14_000
    assert gen.total_tokens == 16_000
    assert gen.thinking_tokens == 1_500
    assert gen.untracked_usage == ""


def test_a_priority_tier_call_is_flagged_rather_than_priced_as_standard(settings):
    """Anthropic serves standard, priority or batch. Batch is priced (0.5x);
    priority bills above the list rate this model holds, so a priority call
    would be understated — it is reported instead."""
    from anthropic.types import Usage

    provider = AnthropicProvider(api_key="sk-test-not-used")
    provider._client = _anthropic_stub(Usage(input_tokens=100, output_tokens=100,
                                             service_tier="priority"))
    gen = provider.generate(OPUS, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.service_tier == "priority"
    assert "not the list-rate tier" in gen.untracked_usage


def test_cache_write_without_a_ttl_split_is_charged_and_flagged(settings):
    """Charging the cheaper 5m rate is a choice, so it has to be visible."""
    from anthropic.types import Usage

    provider = AnthropicProvider(api_key="sk-test-not-used")
    provider._client = _anthropic_stub(Usage(input_tokens=100, output_tokens=100,
                                             cache_creation_input_tokens=1_000))
    gen = provider.generate(OPUS, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.cache_write_tokens == 1_000
    assert gen.reported_cache_creation_tokens == 1_000
    assert "5m rate" in gen.untracked_usage
    # The unsplit case is reported once, by the branch above — the arithmetic
    # guard must not restate it.
    assert "cache write split does not reconcile" not in gen.untracked_usage


def test_a_write_split_that_does_not_add_up_to_the_total_is_flagged(settings):
    """A total larger than 5m + 1h means a cache tier beyond the two TTLs — the
    way the 1-hour tier itself once arrived. Its tokens would go unpriced."""
    from anthropic.types import Usage
    from anthropic.types.cache_creation import CacheCreation

    provider = AnthropicProvider(api_key="sk-test-not-used")
    provider._client = _anthropic_stub(Usage(
        input_tokens=100, output_tokens=100,
        cache_creation_input_tokens=5_000,
        cache_creation=CacheCreation(ephemeral_5m_input_tokens=4_000,
                                     ephemeral_1h_input_tokens=500),
    ))
    gen = provider.generate(OPUS, None, [{"role": "user", "content": "hi"}], settings)
    assert gen.reported_cache_creation_tokens == 5_000
    assert "cache write split does not reconcile" in gen.untracked_usage
    assert "ephemeral_5m=4000 + ephemeral_1h=500 = 4500" in gen.untracked_usage


# ── output budget ─────────────────────────────────────────────────────────────
def test_the_default_output_budget_fits_a_reasoning_call():
    """Reasoning is billed inside the output allowance. At 4096 the Mistral and
    DeepSeek Advanced tiers spent the whole budget thinking and returned
    nothing — billed in full, 0/4 valid. Measured needs were 6k-11k tokens."""
    assert cfg.DEFAULTS.max_output_tokens >= 16384


def test_a_truncated_empty_call_is_reported(capsys):
    """A paid call that returned nothing must never pass silently: it reads as a
    quality failure, but no prompt change fixes it."""
    import pipeline
    rows = [{"stop_reason": "length", "generated_gv": "",
             "model_key": "mistral_advanced", "cost_usd": 0.0307}] * 4
    rows.append({"stop_reason": "stop", "generated_gv": "a.gv",
                 "model_key": "mistral_standard", "cost_usd": 0.001})
    pipeline._warn_about_truncated_calls(rows, cfg.RunSettings(max_output_tokens=4096))
    out = capsys.readouterr().out
    assert "4 call(s) hit the output cap" in out
    assert "mistral_advanced (4)" in out
    assert "0.1228" in out              # the money spent on nothing
    assert "--max-output-tokens" in out


def test_nothing_is_printed_when_no_call_was_truncated(capsys):
    import pipeline
    pipeline._warn_about_truncated_calls(
        [{"stop_reason": "stop", "generated_gv": "a.gv", "cost_usd": 0.1}],
        cfg.RunSettings())
    assert capsys.readouterr().out == ""


def test_a_large_budget_is_streamed_not_sent_as_one_request(settings):
    """The Anthropic SDK refuses a non-streaming request whose result could take
    over ten minutes — it rejects anything above 128_000 * 10/60 output tokens
    outright. Raising the default budget for the reasoning tiers put every
    Anthropic call over that line, so a large request is streamed instead."""
    import dataclasses

    import providers
    assert providers.ANTHROPIC_NONSTREAMING_MAX_TOKENS == 21_333
    assert cfg.DEFAULTS.max_output_tokens > providers.ANTHROPIC_NONSTREAMING_MAX_TOKENS,         "the default budget must exercise the streaming path"

    from anthropic.types import Usage

    class _Stream:
        def __init__(self, message): self._message = message
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def get_final_message(self): return self._message

    message = types.SimpleNamespace(
        stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text="digraph g { a -> b }")],
        usage=Usage(input_tokens=10, output_tokens=20), id="msg_streamed")

    used = []
    provider = AnthropicProvider(api_key="sk-test-not-used")
    provider._client = types.SimpleNamespace(messages=types.SimpleNamespace(
        create=lambda **kw: used.append("create") or message,
        stream=lambda **kw: used.append("stream") or _Stream(message)))

    provider.generate(OPUS, None, [{"role": "user", "content": "hi"}], cfg.RunSettings())
    assert used == ["stream"]

    used.clear()
    small = dataclasses.replace(cfg.RunSettings(), max_output_tokens=4096)
    provider.generate(OPUS, None, [{"role": "user", "content": "hi"}], small)
    assert used == ["create"]
