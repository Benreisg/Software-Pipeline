"""The DeepSeek provider — cache hit/miss accounting, the reasoning switch,
and time-of-day pricing.

DeepSeek speaks OpenAI's Chat Completions dialect through the `openai` SDK with
a different base URL, but three things about it are its own and each one costs
money if read wrong:

    * `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` are DeepSeek's own
      split of the prompt and are authoritative;
    * a cache hit costs about a thirtieth of a miss, not the usual tenth;
    * both models reason by default, so a non-reasoning tier must switch it off
      or it pays for thinking it never asked for;
    * rates DOUBLE during peak hours (01:00-04:00 and 06:00-10:00 UTC).
"""
from __future__ import annotations

import types
from datetime import datetime, timezone

import pytest

import config as cfg
from providers import (
    DeepSeekProvider, _DEEPSEEK_ACCOUNTED, _unaccounted_usage_fields,
)

FLASH = cfg.model_by_key("deepseek_efficient")  # deepseek-v4-flash, reasoning OFF
PRO = cfg.model_by_key("deepseek_advanced")      # deepseek-v4-pro, reasoning ON

# Since 2026-08-31 the Efficient tier takes the reasoning-off branch again, so
# the explicit switch is covered by a live tier and needs no probe double.
REASONING_OFF = FLASH

PEAK = datetime(2026, 8, 18, 2, 0, tzinfo=timezone.utc)      # 02:00 UTC — peak
OFFPEAK = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)  # 12:00 UTC — off-peak


class _Usage(types.SimpleNamespace):
    """A usage double that also answers `model_dump()`.

    The untracked-usage guard walks `model_dump()` when the object offers one,
    which every vendor SDK's pydantic usage object does — a plain namespace
    would send the guard down its fallback path and quietly never see the
    nested fields, so the test would not exercise what production runs.
    """

    def model_dump(self):
        out = {}
        for key, value in vars(self).items():
            out[key] = vars(value).copy() if isinstance(value, types.SimpleNamespace) else value
        return out


def _usage(prompt=10_000, completion=2_000, hit=0, miss=None, reasoning=0,
           total=None):
    """DeepSeek's usage object as its API reference documents it: the hit/miss
    pair. A live response also carries OpenAI's `prompt_tokens_details` block
    beside it (the same cached count under OpenAI's name); the tests that care
    about it add it explicitly, because it is the pair that is authoritative."""
    if miss is None:
        miss = prompt - hit
    return _Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
        prompt_cache_hit_tokens=hit,
        prompt_cache_miss_tokens=miss,
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning),
    )


def _stub(usage, content="digraph g { a -> b }", finish="stop", capture=None):
    def create(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        message = types.SimpleNamespace(content=content, reasoning_content=None)
        return types.SimpleNamespace(
            id="chatcmpl-ds-1", usage=usage,
            choices=[types.SimpleNamespace(message=message, finish_reason=finish)])
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def _provider(usage=None, **stub_kwargs):
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider._client = _stub(usage if usage is not None else _usage(), **stub_kwargs)
    provider._max_retries = 5
    return provider


# ── catalogue ─────────────────────────────────────────────────────────────────
def test_deepseek_is_a_live_vendor_with_verified_prices():
    assert "deepseek" in cfg.LIVE_VENDORS
    for model in (FLASH, PRO):
        assert model.price_verified, model.key
        assert model.enabled, model.key
        assert model.model_id.startswith("deepseek-v4-"), model.key


def test_the_retired_chat_and_reasoner_ids_are_gone():
    """`deepseek-chat` and `deepseek-reasoner` no longer exist in the API."""
    ids = {m.model_id for m in cfg.models_by_vendor("deepseek")}
    assert "deepseek-chat" not in ids and "deepseek-reasoner" not in ids


def test_both_deepseek_models_are_enabled():
    """The default run and interactive selector must offer both current ids."""
    live = [m for m in cfg.models_by_vendor("deepseek") if m.enabled]
    assert [(m.key, m.model_id, m.tier) for m in live] == [
        ("deepseek_efficient", "deepseek-v4-flash", "efficient"),
        ("deepseek_advanced", "deepseek-v4-pro", "advanced"),
    ]
    assert PRO in cfg.enabled_models()

    # Reasoning was switched on and back off here on 2026-08-31 (see config.py).
    # The two tiers now differ by reasoning state as well as by model class,
    # which is the arrangement every other vendor's Efficient tier has.
    assert FLASH.reasoning_off and not FLASH.thinking
    assert FLASH.effort is None      # nothing to state when nothing reasons
    assert PRO.thinking and not PRO.reasoning_off
    assert PRO.effort == "high"      # DeepSeek has no "medium" to select


def test_a_cache_hit_is_far_cheaper_than_the_usual_tenth():
    """Deriving DeepSeek's cached rate from CACHE_READ_MULTIPLIER would
    overcharge a cache hit roughly threefold."""
    for model in (FLASH, PRO):
        ratio = model.price_cached_in_per_mtok / model.price_in_per_mtok
        assert ratio == pytest.approx(1 / 30, rel=0.05), model.key
        assert ratio < cfg.CACHE_READ_MULTIPLIER


def test_the_pricing_sheet_is_dated():
    sheet = cfg.pricing_sheet("deepseek")
    assert sheet["as_of"] == "2026-08-30"
    assert "deepseek.com" in sheet["source"]


# ── time-of-day pricing ───────────────────────────────────────────────────────
@pytest.mark.parametrize("hour, peak", [
    (0, False), (1, True), (3, True), (4, False), (5, False),
    (6, True), (9, True), (10, False), (23, False),
])
def test_the_peak_windows_are_the_published_ones(hour, peak):
    at = datetime(2026, 8, 18, hour, 30, tzinfo=timezone.utc)   # a Tuesday
    assert cfg.is_peak_hour("deepseek", at) is peak


@pytest.mark.parametrize("day, weekday, peak", [
    (17, "Monday", True), (21, "Friday", True),
    (22, "Saturday", False), (23, "Sunday", False),
])
def test_the_peak_windows_do_not_apply_at_the_weekend(day, weekday, peak):
    """DeepSeek's published window is 01:00-04:00 and 06:00-10:00 UTC *Monday
    through Friday*. Reading the hours without the weekday — which this project
    did until 2026-08-30 — priced every Saturday and Sunday call at the peak
    rate, twice what DeepSeek charges, and a long run is most likely to be left
    going over a weekend.
    """
    at = datetime(2026, 8, day, 2, 30, tzinfo=timezone.utc)
    assert at.strftime("%A") == weekday          # the fixture dates say what they mean
    assert cfg.is_peak_hour("deepseek", at) is peak


def test_every_other_vendor_is_always_at_full_rate():
    for vendor in ("anthropic", "openai", "google", "mistral"):
        assert cfg.is_peak_hour(vendor, OFFPEAK) is True


def test_the_same_call_costs_half_as_much_off_peak():
    usage = cfg.TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    peak, peak_basis = cfg.cost_usd(FLASH, usage, at_utc=PEAK)
    off, off_basis = cfg.cost_usd(FLASH, usage, at_utc=OFFPEAK)
    assert peak == pytest.approx(0.44 + 1.32)
    assert off == pytest.approx(peak / 2)
    assert peak_basis.endswith("+peak") and off_basis.endswith("+offpeak")


def test_the_off_peak_discount_reaches_cache_reads_too():
    usage = cfg.TokenUsage(cache_read_tokens=1_000_000)
    peak, _ = cfg.cost_usd(FLASH, usage, at_utc=PEAK)
    off, _ = cfg.cost_usd(FLASH, usage, at_utc=OFFPEAK)
    assert peak == pytest.approx(0.014)
    assert off == pytest.approx(0.007)


def test_a_vendor_without_time_pricing_records_no_peak_basis():
    """Only DeepSeek carries the peak/off-peak suffix — it would be noise on a
    vendor that charges one rate all day."""
    usage = cfg.TokenUsage(input_tokens=1_000)
    _, basis = cfg.cost_usd(cfg.model_by_key("openai_standard"), usage, at_utc=PEAK)
    assert "peak" not in basis


# ── the cache split ───────────────────────────────────────────────────────────
def test_deepseeks_own_hit_miss_split_is_authoritative(settings):
    result = _provider(_usage(prompt=10_000, hit=6_000)).generate(
        FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 6_000
    assert result.input_tokens == 4_000
    assert result.billable_input_tokens == 10_000
    assert result.reported_input_tokens == 10_000   # prompt_tokens, verbatim
    assert result.reported_total_tokens == 12_000   # total_tokens, verbatim


def test_a_response_without_the_split_is_reported_not_reconstructed(settings):
    """The hit/miss pair is the prompt split DeepSeek documents as its own. A
    response without it used to be rebuilt from the OpenAI-shaped
    `prompt_tokens_details` beside it — guessing at the cache state of a prompt
    whose hit rate decides a 30x price difference, from a block that is absent
    exactly when the pair is. It is now said out loud instead."""
    usage = _usage(prompt=10_000)
    usage.prompt_cache_hit_tokens = 0
    usage.prompt_cache_miss_tokens = 0
    result = _provider(usage).generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    # Priced as entirely uncached: a miss costs ~30x a hit, so this is the
    # direction that overstates the cost. Pricing it as 0 would have made the
    # prompt free, which is the one reading that hides the error.
    assert result.cached_tokens == 0
    assert result.input_tokens == 10_000
    assert result.cost_usd > 0
    assert "no cache split reported" in result.untracked_usage
    assert "may overstate the cost" in result.untracked_usage


def test_an_openai_shaped_cache_field_is_flagged_rather_than_priced(settings):
    """DeepSeek charges for hits and misses only — it has no cache-write class.
    A `cache_write_tokens` arriving anyway must not be priced at OpenAI's 1.25x,
    which would invent a charge the vendor does not make; the guard reports it
    so the discrepancy is investigated rather than billed."""
    usage = _usage(prompt=10_000, hit=6_000)
    usage.prompt_tokens_details = types.SimpleNamespace(cache_write_tokens=500)
    result = _provider(usage).generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cache_write_tokens == 0
    assert "prompt_tokens_details.cache_write_tokens=500" in result.untracked_usage


def test_the_openai_dialect_cache_alias_is_not_reported_as_untracked(settings):
    """A live DeepSeek response carries both names for the cached prompt:
    `prompt_cache_hit_tokens` and OpenAI's `prompt_tokens_details.cached_tokens`
    with the same value. They are one quantity, priced once as a cache read —
    reporting the alias as unpriced would say the cost is a lower bound when it
    is exact."""
    usage = _usage(prompt=139, hit=128, miss=11, completion=842)
    usage.prompt_tokens_details = types.SimpleNamespace(
        cached_tokens=128, cache_write_tokens=None, audio_tokens=None,
        image_tokens=None, text_tokens=None)
    result = _provider(usage).generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 128
    assert result.input_tokens == 11
    assert "not priced" not in result.untracked_usage
    assert "cached_tokens" not in result.untracked_usage


def test_the_two_cached_token_names_disagreeing_is_a_finding(settings):
    """Equal is the normal case and says nothing. If the alias ever states a
    different number, one of the two names no longer describes what is priced —
    and the hit/miss pair, which is DeepSeek's own, stays the one that is."""
    usage = _usage(prompt=139, hit=128, miss=11, completion=842)
    usage.prompt_tokens_details = types.SimpleNamespace(cached_tokens=64)
    result = _provider(usage).generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.cached_tokens == 128
    assert "cached-token names disagree" in result.untracked_usage
    assert "prompt_tokens_details.cached_tokens=64" in result.untracked_usage


def test_a_cache_split_that_does_not_add_up_is_reported(settings):
    odd = _usage(prompt=10_000, hit=6_000, miss=1_000)
    result = _provider(odd).generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert "cache split does not reconcile" in result.untracked_usage


# ── the reasoning switch ──────────────────────────────────────────────────────
def test_the_off_switch_is_the_documented_one(settings):
    """DeepSeek's off switch is the `thinking` object — `reasoning_effort`
    accepts low|high|max only. The unrecognised value this sent until
    2026-08-20 was silently ignored, so the tier kept reasoning and kept
    billing it while the row said otherwise. No enabled tier uses this branch
    now, so a probe keeps the switch covered."""
    sent = {}
    _provider(capture=sent).generate(
        REASONING_OFF, None, [{"role": "user", "content": "hi"}], settings)
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in sent      # never at the top level, never "none"


def test_the_thinking_branch_asks_for_thinking_at_a_stated_effort(settings):
    """Sent explicitly rather than left to DeepSeek's default, so the run
    records the effort it actually asked for — even where the value happens to
    match the default, as "high" does. Only Pro reasons since 2026-08-31."""
    model = PRO
    sent = {}
    _provider(capture=sent).generate(model, None, [{"role": "user", "content": "hi"}], settings)
    # `thinking` can only travel in extra_body; `reasoning_effort` is part of
    # OpenAI's own schema and stays top-level, as DeepSeek's own example shows.
    assert sent["extra_body"] == {"thinking": {"type": "enabled"}}
    assert sent["reasoning_effort"] == "high"
    # No service_tier here either — no vendor is sent one.
    assert "service_tier" not in sent
    assert model.effort in ("low", "high", "max")   # DeepSeek's scale, not OpenAI's


def test_a_reported_zero_is_not_the_same_as_no_report(settings):
    """0 means DeepSeek measured no reasoning on this call; None means it stated
    no such field. Collapsing them would make a working off-switch look exactly
    like an API that said nothing — which is how the 2026-08-20 toggle bug went
    unnoticed in the first place."""
    zero = _provider(_usage(reasoning=0)).generate(
        REASONING_OFF, None, [{"role": "user", "content": "hi"}], settings)
    assert zero.thinking_tokens == 0
    assert "reasoning_tokens=0 confirms it took effect" in zero.reasoning_note

    usage = _usage()
    del usage.completion_tokens_details
    silent = _provider(usage).generate(
        REASONING_OFF, None, [{"role": "user", "content": "hi"}], settings)
    assert silent.thinking_tokens is None
    assert "no reasoning-token count" in silent.reasoning_note


def test_temperature_never_travels_with_thinking_on(settings):
    """DeepSeek ignores temperature in thinking mode, so sending it would record
    a sampling setting the run did not actually have. Flash is covered by the
    opposite test below, which asserts temperature DOES reach a non-thinking
    tier."""
    sent = {}
    _provider(capture=sent).generate(
        PRO, None, [{"role": "user", "content": "hi"}],
        cfg.RunSettings(temperature=0.7, max_output_tokens=4096, n_few_shot=2))
    assert "temperature" not in sent
    assert sent["extra_body"] == {"thinking": {"type": "enabled"}}


def test_temperature_reaches_a_model_with_thinking_switched_off():
    sent = {}
    _provider(capture=sent).generate(
        REASONING_OFF, None, [{"role": "user", "content": "hi"}],
        cfg.RunSettings(temperature=0.7, max_output_tokens=4096, n_few_shot=2))
    assert sent["temperature"] == 0.7
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}


def test_reasoning_tokens_are_recorded(settings):
    result = _provider(_usage(reasoning=1_500)).generate(
        FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.thinking_tokens == 1_500
    assert "reasoning_tokens=1500" in result.reasoning_note


def test_residual_reasoning_on_a_switched_off_tier_is_flagged(settings):
    """If reasoning tokens turn up despite effort=none, they were still billed
    and the note must say so rather than imply the tier was free of them."""
    result = _provider(_usage(reasoning=40)).generate(
        REASONING_OFF, None, [{"role": "user", "content": "hi"}], settings)
    assert "residual reasoning_tokens=40" in result.reasoning_note


# ── cost and the guards ───────────────────────────────────────────────────────
def test_every_token_class_moves_the_cost(settings):
    base = _provider(_usage(prompt=1_000, completion=100)).generate(
        FLASH, None, [{"role": "user", "content": "hi"}], settings).cost_usd
    for bumped in (_usage(prompt=2_000, completion=100),
                   _usage(prompt=1_000, completion=200),
                   # the same 1_000 uncached tokens plus 400 served from cache
                   _usage(prompt=1_400, completion=100, hit=400)):
        cost = _provider(bumped).generate(
            FLASH, None, [{"role": "user", "content": "hi"}], settings).cost_usd
        assert cost > base


def test_a_complete_usage_object_reports_nothing_untracked(settings):
    result = _provider().generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.untracked_usage == ""


def test_a_usage_field_the_cost_model_does_not_price_is_flagged():
    usage = _usage(reasoning=900)
    assert "completion_tokens_details.reasoning_tokens" in _DEEPSEEK_ACCOUNTED
    assert _unaccounted_usage_fields(
        usage, _DEEPSEEK_ACCOUNTED - {"completion_tokens_details.reasoning_tokens"})


def test_the_response_id_is_captured(settings):
    result = _provider().generate(FLASH, None, [{"role": "user", "content": "hi"}], settings)
    assert result.response_id == "chatcmpl-ds-1"
