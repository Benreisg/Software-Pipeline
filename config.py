"""
config.py — single source of truth (multi-vendor branch)
============================================================
Models, prices, strategies, run settings, and paths. Everything else in the
pipeline reads from here (DC1 Reproducibility).

Five vendors per the thesis decision log (Efficient / Standard / Advanced
models where the vendor offers them): Anthropic, OpenAI, Mistral, Google,
DeepSeek.

⚠ LIVE vs MOCK — only vendors listed in LIVE_VENDORS below perform real API
  calls. All others are selectable in the run but are served by the mock
  provider (instant, free, `is_mock=True` in results.csv). Today all five
  configured vendors are live.

⚠ VERIFICATION STATUS — read before any paid run:
  All five vendors' rates were re-read from their live pricing pages on
  2026-08-30 (the URLs are in PRICING_SHEETS below). Three things had moved
  since the previous sheets, and each is commented at the model it affects:
    - Sonnet 5's $2/$10 introductory rate BECAME the list price. The increase
      to $3/$15 announced for 2026-09-01 was cancelled — this file had the
      increase encoded and would have started overpricing Sonnet 5 by 50 %
      on 2026-09-01, one day after this check.
    - gpt-5.6-sol dropped to a promotional $4/$20 (from $5/$30) through at
      least 2026-11-21; it is now an intro window, not a new list price.
    - DeepSeek's peak window is Monday-Friday only. `is_peak_hour` ignored
      the weekday and priced Saturday and Sunday calls at double.
  Google and Mistral re-verified unchanged.

  The rule this file has always followed still stands: a rate that cannot be
  read off a vendor's current page stays `None`, `cost_usd` comes out `None`,
  and the row is flagged `price_verified=False`. NEVER fabricate a dollar
  figure — an invented price is worse than a missing one, because it looks
  like a measurement. Confirm model IDs the same way (a typo'd model ID fails
  loudly with a 404 on first real call, so test each vendor with --limit 1
  before a full run).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "runs"

# ── dataset ──────────────────────────────────────────────────────────────────
# PMo dataset: 55 items, each a pair of files sharing one number —
#   <PMO_DATASET_DIR>/descriptions/<id>.txt  → the LLM input
#   <PMO_DATASET_DIR>/graphviz/<id>.dot      → the ground-truth model
# The dataset lives outside the project (it is data, not code), so this default
# is machine-specific. Override it per machine with the PMO_DATASET_DIR
# environment variable, or per run with `--dataset-dir`.
PMO_DATASET_DIR = Path(
    os.environ.get("PMO_DATASET_DIR")
    # Renamed from "Data Sets" to "Data_Sets" after the runs of 2026-08-21; the
    # old spelling made the interactive setup exit before its first question.
    or r"C:\Users\breis\Desktop\Data_Sets\pmo-dataset"
)

# Items shown as in-context examples by the few-shot strategies. They are NOT
# held out of the evaluated set: a run covers all 55 items, these two included.
# "Never generate for an item the model was just shown" is enforced per item
# instead of per run — when the item being generated for IS one of the
# exemplars, that exemplar is swapped for a stand-in from FEW_SHOT_FALLBACK_IDS
# (prompts.exemplars_for_item), so the answer is never in its own prompt.
FEW_SHOT_IDS: List[str] = ["01", "02"]

# Stand-ins used, in order, when an evaluated item collides with one of
# FEW_SHOT_IDS. An item can collide with at most one exemplar, so a single
# reserve covers the default n_few_shot=2; the second leaves room for wider
# exemplar sets and for a reserve that is itself the item being generated for.
# Must not overlap FEW_SHOT_IDS.
FEW_SHOT_FALLBACK_IDS: List[str] = ["03", "04"]

# Items that get a *complete* alternate exemplar set instead of the per-slot
# swap above. `01` and `02` are the exemplars themselves, so their own model
# would otherwise sit in their own prompt; `03` does not leak (it is not an
# exemplar) but it is the stand-in `01` and `02` are shown, so covering it too
# means all three run on the *same* exemplar pair rather than on three
# differently-swapped ones. FEW_SHOT_ALT_IDS must therefore be disjoint from
# both FEW_SHOT_IDS and FEW_SHOT_FALLBACK_IDS.
#
# Emptying either list switches the behaviour off; the per-slot swap then
# covers the collisions on its own, as before.
FEW_SHOT_ALT_FOR_IDS: List[str] = ["01", "02", "03"]
FEW_SHOT_ALT_IDS: List[str] = ["40", "41"]

# ── vendors ──────────────────────────────────────────────────────────────────
VENDORS: List[str] = ["anthropic", "openai", "mistral", "google", "deepseek"]

# Vendors whose provider integration is actually wired up and may perform real
# API calls. Every other vendor in VENDORS is selectable, but is served by the
# MockProvider: it returns instantly, spends nothing, and its rows are flagged
# `is_mock=True` in results.csv so a placeholder can never be mistaken for a
# measurement. Move a vendor in here only once its provider in providers.py is
# implemented AND its model IDs/prices in ALL_MODELS are verified.
LIVE_VENDORS: List[str] = ["anthropic", "openai", "google", "mistral", "deepseek"]

# Env var each vendor's SDK conventionally reads, offered as the default when
# the interactive key prompt runs (see run.py `_prompt_for_keys`).
VENDOR_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def vendor_is_live(vendor: str) -> bool:
    """True if this vendor performs real API calls; False if it is mocked."""
    return vendor in LIVE_VENDORS


# ── service tier ─────────────────────────────────────────────────────────────
# NO tier is requested, on any vendor — there is deliberately no parameter and
# no mapping here. Requested 2026-08-31, removed the same day once the vendors'
# own documentation was read; the findings are kept because they are the reason:
#
#   * Omitting the parameter does NOT mean "standard". Every vendor defaults to
#     `auto`, i.e. "let the account decide" — which is only equivalent to
#     standard while the organisation holds no purchased capacity.
#   * Anthropic: Priority Tier "commitments are no longer available for
#     purchase", and Priority is unsupported on Claude Opus 5 and Claude Sonnet 5
#     regardless. Only Haiku 4.5 could route there, and only under a legacy
#     contract (platform.claude.com/docs/en/api/service-tiers, read 2026-08-31).
#   * OpenAI: `scale` is NOT a per-request class. Scale Tier is prepaid capacity
#     — token units per minute bought upfront for one model snapshot, 30-day
#     minimum — and it carries no per-token price at all, so a call served that
#     way could not be priced by `cost_usd` even in principle. Worse, with Scale
#     Tier enabled on the project, OMITTING the parameter consumes it; only
#     `default` forces shared PAYG. This is the one vendor where a dashboard
#     toggle can reroute a running experiment's billing without a code change.
#   * Mistral takes Anthropic's shape (`auto | standard_only`), Google's is
#     `unspecified | flex | standard | priority`, DeepSeek documents none.
#
# What guards the cost model is therefore not the request but the RESPONSE:
# every provider reads back the tier that actually served the call and
# `providers._service_tier_note` flags anything that is not a list-rate tier.
# Across every row in `runs/` that has ever been recorded, that check has never
# fired — Anthropic and Mistral reported "standard", OpenAI "default", Google
# and DeepSeek none.


# ── API keys ─────────────────────────────────────────────────────────────────
# Keys live in api_keys.py (plaintext, git-ignored, never printed). They are
# applied automatically — nothing is ever asked for interactively. A vendor left
# as None there falls back to its conventional environment variable, so a
# machine that prefers env vars keeps working without editing any file.
try:
    from api_keys import API_KEYS as _CONFIGURED_KEYS
except ImportError:      # api_keys.py absent → env vars only
    _CONFIGURED_KEYS = {}


def api_key_for(vendor: str) -> Optional[str]:
    """The API key to use for `vendor`, or None if none is configured."""
    key = (_CONFIGURED_KEYS.get(vendor) or "").strip()
    if key:
        return key
    return (os.environ.get(VENDOR_ENV_VAR.get(vendor, ""), "") or "").strip() or None


def api_key_source(vendor: str) -> str:
    """Where this vendor's key comes from — for status output. Never returns
    the key itself."""
    if (_CONFIGURED_KEYS.get(vendor) or "").strip():
        return "api_keys.py"
    if (os.environ.get(VENDOR_ENV_VAR.get(vendor, ""), "") or "").strip():
        return f"${VENDOR_ENV_VAR[vendor]}"
    return "not configured"


# ── models ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    key: str                            # unique CLI / output-table key, "{vendor}_{tier}"
    vendor: str                         # one of VENDORS
    model_id: str                       # exact vendor API model id
    tier: str                           # "efficient" | "standard" | "advanced"
                                        # (retired specs keep the tier they ran under)
    price_in_per_mtok: Optional[float]  # USD per 1,000,000 input tokens; None = unverified
    price_out_per_mtok: Optional[float] # USD per 1,000,000 output tokens; None = unverified
    price_verified: bool = False
    enabled: bool = True
    thinking: bool = False              # use extended/adaptive reasoning if the vendor supports it
    effort: Optional[str] = None        # reasoning depth hint (vendor-specific interpretation)
    supports_temperature: bool = False
    # Time-limited introductory rate, when the vendor is running one. Billing
    # applies it automatically, so a run inside the window that priced at the
    # list rate would overstate its own cost — `prices_for()` resolves which of
    # the two applies and every row records the basis it used.
    intro_price_in_per_mtok: Optional[float] = None
    intro_price_out_per_mtok: Optional[float] = None
    intro_until: Optional[str] = None   # ISO date, inclusive — last day of the intro rate
    # Explicit cache-read rate, when the vendor publishes one per model rather
    # than as a fixed fraction of the input rate. Anthropic's is a flat 0.10x
    # across the line; OpenAI's is 0.10x for the gpt-5.6 family but 0.5x on o1
    # and 0.25x on o3 — so the rate is stated per model and only falls back to
    # CACHE_READ_MULTIPLIER when the vendor really does apply a uniform factor.
    price_cached_in_per_mtok: Optional[float] = None
    # The cache-read rate that goes with the introductory input/output rates.
    # Google discounts all three together during a launch window, so pricing a
    # cache read at the list rate inside that window would overstate the cost.
    intro_price_cached_in_per_mtok: Optional[float] = None
    # Time-of-day pricing. DeepSeek bills its full rate during peak hours and a
    # fraction of it the rest of the day, so the same call costs a different
    # amount depending on when it ran. The prices above are the peak/full rates;
    # this multiplier is applied off-peak and the basis is recorded per row.
    offpeak_multiplier: Optional[float] = None
    # Reasoning is on by default on this model and must be switched off for a
    # non-reasoning tier — the inverse of `thinking`, which asks for it.
    reasoning_off: bool = False


ALL_MODELS: List[ModelSpec] = [
    # Three tiers per vendor, in ascending order: Efficient -> Standard ->
    # Advanced. The tier is this project's own grouping, not the vendors' — it
    # is what makes "the cheap model" comparable across five price sheets. The
    # ADVANCED tier is the reasoning-capable one wherever the vendor sells such
    # a model; it was called "reasoning" until 2026-08-20 and was renamed
    # because on two vendors the slot is filled by a model class rather than by
    # a reasoning switch (see the retired entries at the end of this list).
    #
    # ⚠ Not every vendor fills every tier — DeepSeek sells two models, so it has
    # no Standard tier. Nothing in the pipeline assumes three per vendor.

    # ── Anthropic — rates re-verified 2026-08-30, platform.claude.com ──────────────────
    ModelSpec(
        key="anthropic_efficient", vendor="anthropic", model_id="claude-haiku-4-5",
        tier="efficient", price_in_per_mtok=1.00, price_out_per_mtok=5.00, price_verified=True,
        supports_temperature=True,
    ),
    ModelSpec(
        key="anthropic_standard", vendor="anthropic", model_id="claude-sonnet-5",
        tier="standard", price_in_per_mtok=2.00, price_out_per_mtok=10.00, price_verified=True,
        # ⚠ Changed 2026-08-30: $2/$10 is now the LIST price, not an intro rate.
        # Sonnet 5 launched at $2/$10 "through 2026-08-31", with $3/$15 to follow
        # on 2026-09-01 — which is what this spec encoded. Anthropic cancelled
        # that increase and made $2/$10 standard (pricing page, note
        # `claude-sonnet-5-introductory-pricing`, read 2026-08-30). The intro
        # fields are therefore gone: had they stayed, every Sonnet 5 call from
        # 2026-09-01 on would have been priced 50 % above what it cost.
        # Rows from runs inside the old window recorded cost_basis="intro" and
        # are unaffected — `rates_for` falls back to the list rate for a basis
        # whose intro fields no longer exist, and that rate is the same $2/$10
        # they were billed at.
    ),
    ModelSpec(
        key="anthropic_advanced", vendor="anthropic", model_id="claude-opus-5",
        tier="advanced", price_in_per_mtok=5.00, price_out_per_mtok=25.00, price_verified=True,
        # Opus 5 replaced Opus 4.8 in this slot on 2026-08-18, at the identical
        # $5/$25 — the newer generation for the same money. The call shape is
        # unchanged: both take adaptive thinking and reject `budget_tokens`.
        # effort lowered high -> medium on 2026-08-31 at the author's request,
        # together with the other three reasoning tiers, so the Advanced tiers
        # are compared at one stated depth instead of four vendor-specific ones.
        thinking=True, effort="medium",
    ),

    # ── OpenAI — rates re-verified 2026-08-30, developers.openai.com ────────
    # The gpt-5.6 family is one architecture in three sizes, so the tiers map by
    # size: luna ≈ Haiku's slot, terra ≈ Sonnet's, sol ≈ Opus's. All three are
    # reasoning-capable and default to effort "medium"; only the Advanced tier
    # raises it, mirroring `thinking=True` on the Anthropic side.
    # ⚠ `supports_temperature` is False for all three: temperature support on
    # gpt-5.6 could not be confirmed from the docs, and the pipeline sends no
    # sampling parameter it has not verified. RunSettings.temperature is None by
    # default, so nothing is lost until someone sets it deliberately.
    ModelSpec(
        key="openai_efficient", vendor="openai", model_id="gpt-5.6-luna",
        tier="efficient", price_in_per_mtok=0.20, price_out_per_mtok=1.20,
        price_cached_in_per_mtok=0.02, price_verified=True,
    ),
    ModelSpec(
        key="openai_standard", vendor="openai", model_id="gpt-5.6-terra",
        tier="standard", price_in_per_mtok=2.00, price_out_per_mtok=12.00,
        price_cached_in_per_mtok=0.20, price_verified=True,
    ),
    ModelSpec(
        key="openai_advanced", vendor="openai", model_id="gpt-5.6-sol",
        tier="advanced", price_in_per_mtok=5.00, price_out_per_mtok=30.00,
        price_cached_in_per_mtok=0.50, price_verified=True,
        # Promotional rate since this sheet was last read: $4/$20 with cache
        # reads at $0.40, announced as a 20 % input / 33 % output reduction —
        # which is what confirms $5/$30 above as the rate it comes off, and so
        # as the rate that returns when the promotion ends. OpenAI states it
        # runs "at least through November 21, 2026" and names no successor
        # rate; `intro_until` therefore takes the only date on the page. If the
        # promotion is extended, this window closes too early and calls are
        # priced ABOVE what they cost — the safe direction, and visible in
        # `cost_basis` flipping from "intro" to "list" overnight.
        intro_price_in_per_mtok=4.00, intro_price_out_per_mtok=20.00,
        intro_price_cached_in_per_mtok=0.40, intro_until="2026-11-21",
        # ⚠ medium (set 2026-08-31) is also gpt-5.6's own default, which luna and
        # terra inherit by sending no `reasoning` at all. The three OpenAI tiers
        # now differ by model size only, not by reasoning depth — the parameter
        # is still sent explicitly so the manifest records the depth rather than
        # leaving it to an undocumented default.
        thinking=True, effort="medium",
    ),

    # ── Mistral — rates re-verified 2026-08-30, docs.mistral.ai (unchanged) ─
    # Mistral's naming does not track its price ladder: Large 3 is the cheap
    # fast model at $0.50/$1.50 and Medium 3.5 is the premium one at $1.50/$7.50.
    # The tiers therefore follow price and capability, not the words in the name
    # — Medium 3.5 is the Advanced tier and Large 3 the Standard one, which
    # reads backwards and is worth a sentence wherever the tiers are compared.
    #
    # There is no reasoning-specialised model to fill the Advanced slot either:
    # `magistral-medium-latest`, which this project had configured, no longer
    # exists, and Mistral Small 4 absorbed Magistral's reasoning into the general
    # line. `magistral-small-latest` still resolves but carries no published
    # price, and an unpriced model cannot be used where cost is a result. The
    # Advanced tier is therefore Medium 3.5 driven by `reasoning_effort="high"`,
    # which is a genuine reasoning switch — see MistralProvider.
    ModelSpec(
        key="mistral_efficient", vendor="mistral", model_id="mistral-small-latest",
        tier="efficient", price_in_per_mtok=0.15, price_out_per_mtok=0.60,
        price_cached_in_per_mtok=0.015, price_verified=True,
        # Small 4 absorbed Magistral's reasoning, so it is reasoning-capable and
        # accepts `reasoning_effort`. Mistral documents "none" but NOT what an
        # omitted parameter does — so this tier switches thinking off explicitly
        # instead of measuring an undocumented default (added 2026-08-20; before
        # that the tier sent nothing and its reasoning state was unknown).
        supports_temperature=True, reasoning_off=True,
    ),
    ModelSpec(
        key="mistral_standard", vendor="mistral", model_id="mistral-large-latest",
        tier="standard", price_in_per_mtok=0.50, price_out_per_mtok=1.50,
        price_cached_in_per_mtok=0.05, price_verified=True,
        supports_temperature=True,
    ),
    ModelSpec(
        key="mistral_advanced", vendor="mistral", model_id="mistral-medium-latest",
        tier="advanced", price_in_per_mtok=1.50, price_out_per_mtok=7.50,
        price_cached_in_per_mtok=0.15, price_verified=True,
        # Reasoning explicitly uses high effort as requested on 2026-08-31.
        thinking=True, effort="high",
    ),

    # ── Google — rates re-verified 2026-08-30, ai.google.dev (unchanged) ────
    # The Gemini 2.5 family this project started with is closed to new API keys
    # (404 "no longer available to new users"), so all three tiers come from the
    # 3.x line.
    #
    # ⚠ The Advanced tier runs on a PREVIEW model. `gemini-3.1-pro-preview` is
    # the only Pro-class *text* model the Gemini API offers — `gemini-3-pro-image`
    # generates images — and Google publishes it under that id alone, with no
    # stable alias (checked 2026-08-20). Preview models can change or be
    # withdrawn without notice, which is a reproducibility risk this project has
    # to name rather than hide. A withdrawal fails loudly in `check_connection`
    # rather than quietly measuring something else.
    #
    # ⚠ 3.1 Pro is the only model here with prompt-size-tiered pricing: $2/$12
    # up to a 200k-token prompt, $4/$18 above it (cache read $0.20 / $0.40).
    # The rates below are the ≤200k tier. PMo prompts are a few hundred tokens
    # and the whole dataset is far under the threshold, so the upper tier cannot
    # apply here — but `cost_usd` has no notion of prompt-size tiers, so a
    # long-context run on this model would be priced too low.
    #
    # Every Gemini 3.x model reasons by default and bills those tokens at the
    # output rate, so `thinking` here only decides whether thought summaries are
    # requested; the token count is captured on all three (see GoogleProvider).
    ModelSpec(
        key="google_efficient", vendor="google", model_id="gemini-3.5-flash-lite",
        tier="efficient", price_in_per_mtok=0.30, price_out_per_mtok=2.50,
        price_cached_in_per_mtok=0.03, price_verified=True,
        supports_temperature=True,
    ),
    # Swapped 2026-08-30 at the author's request: `gemini-3.7-flash` →
    # `gemini-3.6-flash`, Google's previous-generation Flash (released
    # 2026-07-21, status stable). The RATES ARE IDENTICAL — Google runs the same
    # promotional window on both models, verified on ai.google.dev/gemini-api/
    # docs/pricing 2026-08-30 — so nothing below changed except the model id,
    # and rows from runs on 3.7 keep pricing correctly under this spec.
    # ⚠ They are still a different model: a quality comparison must not pool
    # 3.6 and 3.7 rows just because their cost columns line up. `model_id` is
    # recorded per row, which is what tells them apart.
    ModelSpec(
        key="google_standard", vendor="google", model_id="gemini-3.6-flash",
        tier="standard", price_in_per_mtok=1.50, price_out_per_mtok=7.50,
        price_cached_in_per_mtok=0.15, price_verified=True,
        # Launch rate, still running: $0.75/$3.75 through 2026-12-31, and the
        # cache read is discounted with them ($0.075 against $0.15).
        intro_price_in_per_mtok=0.75, intro_price_out_per_mtok=3.75,
        intro_until="2026-12-31", intro_price_cached_in_per_mtok=0.075,
        supports_temperature=True,
    ),
    ModelSpec(
        key="google_advanced", vendor="google", model_id="gemini-3.1-pro-preview",
        tier="advanced", price_in_per_mtok=2.00, price_out_per_mtok=12.00,
        price_cached_in_per_mtok=0.20, price_verified=True,
        # effort added 2026-08-31: until then this tier sent `include_thoughts`
        # with no `thinking_level`, so it ran at Gemini's own default while the
        # other Advanced tiers named a depth — the one tier whose reasoning
        # depth was unstated. "medium" maps to thinking_level MEDIUM
        # (google-genai 2.18.1: MINIMAL|LOW|MEDIUM|HIGH, case-insensitive).
        thinking=True, effort="medium",
    ),

    # ── DeepSeek — rates re-verified 2026-08-30, api-docs.deepseek.com ──────
    # `deepseek-chat` and `deepseek-reasoner`, which this project had
    # configured, no longer exist: the API serves exactly two models,
    # `deepseek-v4-pro` and `deepseek-v4-flash`.
    #
    # ⚠ TWO DeepSeek models are configured again since 2026-08-31: Flash as the
    # Efficient tier and Pro as the Advanced tier, both reasoning at effort
    # "high".
    # Pro had been removed on 2026-08-25 over its runtime — a median 269.6 s per
    # call at effort "high" (n=3, range 137-402 s) against 43.4 s for the
    # next-slowest Advanced tier, with ~10.2k output tokens of which ~93 % were
    # thinking. The reinstated spec now uses that same reasoning setting, so the
    # runtime warning applies: time one call before committing to a sweep.
    # DeepSeek has no Standard tier, so it contributes TWO rows where every
    # other vendor contributes
    # three.
    #
    # Until 2026-08-20 the two models were stretched over three tiers by running
    # one of them twice, with and without a reasoning switch that did not in
    # fact switch anything (see DeepSeekProvider). The retired specs at the end
    # of this list keep the rates those runs were billed at.
    #
    # Two pricing features of DeepSeek's that no other vendor here has:
    #   * a cache hit costs ~1/30 of a miss, not the usual 1/10 — $0.014 against
    #     $0.44 per Mtok on Flash, $0.044 against $1.32 on Pro (peak; re-checked
    #     against the published sheet 2026-08-20). The cached rate is therefore
    #     stated per model and never derived from a multiplier. DeepSeek charges
    #     nothing at all for cache WRITES — it has no such class;
    #   * rates DOUBLE during peak hours (01:00-04:00 and 06:00-10:00 UTC).
    #     The prices below are the peak/full rates; `cost_usd` halves them
    #     off-peak and records which applied in `cost_basis`.
    ModelSpec(
        key="deepseek_efficient", vendor="deepseek", model_id="deepseek-v4-flash",
        tier="efficient", price_in_per_mtok=0.44, price_out_per_mtok=1.32,
        price_cached_in_per_mtok=0.014, price_verified=True,
        offpeak_multiplier=0.50,
        # ⚠ Reasoning switched OFF again on 2026-08-31 at the author's request,
        # hours after it was switched on the same day. `reasoning_off=True` sends
        # `thinking={"type": "disabled"}` explicitly — the shape this tier had
        # until that morning — rather than omitting the object and measuring an
        # undocumented default.
        #
        # What the short reasoning window measured, and why it was reverted: the
        # 62 reasoning calls in `runs/` sit at a median 81.0 s and 10082 output
        # tokens (9505 of them reasoning, 94 %) at $0.00673, against 5.6 s / 640
        # tokens / $0.00047 with it off — 14x the latency and 14x the cost. The
        # per-token throughput was never the problem (~108 tok/s, mid-field and
        # ahead of OpenAI); the token count was. Two comparability faults came
        # with it: DeepSeek's reasoning_effort accepts low|high|max only, so this
        # tier ran "high" while the other four vendors were set to "medium" on the
        # same day, and every other vendor's efficient tier does not reason at
        # all. With reasoning off, `tier="efficient"` describes the setting again.
        #
        # Rows recorded inside that window are therefore NOT comparable with rows
        # on either side of it. They stay identifiable per row: `thinking_tokens`
        # > 0 and `reasoning_note` naming `thinking={"type": "enabled"}`, plus
        # `params` in each `prompts/<tag>.json`.
        #
        # `supports_temperature=True` is unchanged, but it now has an effect
        # again: the provider withholds temperature only while thinking is on.
        supports_temperature=True, reasoning_off=True,
    ),
    # ⚠ REINSTATED 2026-08-31 at the author's request. Removed on 2026-08-25 over
    # its runtime — median 269.6 s per call at effort "high" against 43.4 s for
    # the next-slowest tier, which over a 477-call sweep is ~35 h for this model
    # alone. This spec now uses the same reasoning setting (`thinking=True`,
    # `effort="high"`), so that runtime estimate is directly relevant. Time one
    # call before a sweep.
    #
    # ⚠ THE RATES BELOW ARE ARCHIVE RATES. They are the ones this spec ran under
    # in August and were frozen when it was retired; the 2026-08-30 price check
    # covered the models that were live then, which did not include Pro. Re-read
    # api-docs.deepseek.com before a paid run — `price_verified=True` here refers
    # to the day it ran, not to today.
    #
    # Since 2026-08-31 this is the ONLY DeepSeek tier that reasons — the Flash
    # tier went back to `thinking={"type": "disabled"}` the same day. The two
    # tiers therefore differ by reasoning state as well as by model class and
    # price, which is the arrangement every other vendor's Efficient tier has.
    ModelSpec(
        key="deepseek_advanced", vendor="deepseek", model_id="deepseek-v4-pro",
        tier="advanced", price_in_per_mtok=1.32, price_out_per_mtok=3.96,
        price_cached_in_per_mtok=0.044, price_verified=True,
        enabled=True, offpeak_multiplier=0.50, supports_temperature=True,
        thinking=True, effort="high",
    ),
    # ── Retired ────────────────────────────────────────────────────────────
    # Not selectable (`enabled=False`) and never run. They exist so that rows in
    # `runs/` recorded before 2026-08-20 can still be priced and split by token
    # class: `model_by_key` finds them, `enabled_models()` does not. Three of
    # them carry live measurements, which is why each keeps the rates it ran
    # under instead of being mapped onto a renamed tier.
    ModelSpec(
        key="anthropic_reasoning", vendor="anthropic", model_id="claude-opus-5",
        tier="reasoning", price_in_per_mtok=5.00, price_out_per_mtok=25.00,
        price_verified=True, thinking=True, effort="high", enabled=False,
    ),
    ModelSpec(
        key="openai_reasoning", vendor="openai", model_id="gpt-5.6-sol",
        tier="reasoning", price_in_per_mtok=5.00, price_out_per_mtok=30.00,
        price_cached_in_per_mtok=0.50, price_verified=True,
        thinking=True, effort="high", enabled=False,
    ),
    ModelSpec(
        key="mistral_reasoning", vendor="mistral", model_id="mistral-medium-latest",
        tier="reasoning", price_in_per_mtok=1.50, price_out_per_mtok=7.50,
        price_cached_in_per_mtok=0.15, price_verified=True,
        thinking=True, effort="high", enabled=False,
    ),
    # Google's old Advanced slot: gemini-3.5-flash, configured while the Gemini
    # API was believed to sell no Pro-class text model. Different model and
    # different rates, so it cannot be folded into google_advanced.
    ModelSpec(
        key="google_reasoning", vendor="google", model_id="gemini-3.5-flash",
        tier="reasoning", price_in_per_mtok=1.50, price_out_per_mtok=9.00,
        price_cached_in_per_mtok=0.15, price_verified=True,
        thinking=True, enabled=False,
    ),
    # DeepSeek's old three-tiers-from-two-models arrangement kept `deepseek_pro`
    # here as well; that spec is active again since 2026-08-31 and now lives with
    # the other selectable models above. Flash with reasoning on as Reasoning:
    ModelSpec(
        key="deepseek_reasoning", vendor="deepseek", model_id="deepseek-v4-flash",
        tier="reasoning", price_in_per_mtok=0.44, price_out_per_mtok=1.32,
        price_cached_in_per_mtok=0.014, price_verified=True,
        offpeak_multiplier=0.50, thinking=True, enabled=False,
    ),
]


def enabled_models() -> List[ModelSpec]:
    return [m for m in ALL_MODELS if m.enabled]


def model_by_key(key: str) -> ModelSpec:
    for m in ALL_MODELS:
        if m.key == key:
            return m
    raise KeyError(f"Unknown model key {key!r}. Known: {[m.key for m in ALL_MODELS]}")


def models_by_vendor(vendor: str) -> List[ModelSpec]:
    return [m for m in ALL_MODELS if m.vendor == vendor]


def vendors_used_by(models: List[ModelSpec]) -> List[str]:
    seen = []
    for m in models:
        if m.vendor not in seen:
            seen.append(m.vendor)
    return seen


# ── pricing ───────────────────────────────────────────────────────────────────
# Date of the price sheet the per-model rates above were taken from. Recorded
# in the manifest: a cost figure without the date of the price list behind it
# is not reproducible.
PRICING_SHEETS: Dict[str, Dict[str, str]] = {
    "anthropic": {"as_of": "2026-08-30",
                  "source": "platform.claude.com/docs/en/about-claude/pricing"},
    "openai": {"as_of": "2026-08-30",
               "source": "developers.openai.com/api/docs/pricing"},
    "google": {"as_of": "2026-08-30",
               "source": "ai.google.dev/gemini-api/docs/pricing"},
    "mistral": {"as_of": "2026-08-30",
                "source": "docs.mistral.ai/inference/pricing"},
    "deepseek": {"as_of": "2026-08-30",
                 "source": "api-docs.deepseek.com/quick_start/pricing"},
}


def pricing_sheet(vendor: str) -> Dict[str, str]:
    return PRICING_SHEETS.get(vendor, {"as_of": "unknown", "source": "not recorded"})


def pricing_summary(vendors: List[str]) -> str:
    """'anthropic 2026-08-30, openai 2026-08-30' — for the manifest, so a cost
    figure is never separated from the sheet behind it."""
    return ", ".join(f"{v} {pricing_sheet(v)['as_of']}" for v in sorted(set(vendors)))


# Kept as the project-wide fallback for anything not vendor-scoped.
PRICING_AS_OF = PRICING_SHEETS["anthropic"]["as_of"]
PRICING_SOURCE = PRICING_SHEETS["anthropic"]["source"]

# An API call is billed for more than input+output. These are the multipliers on
# a model's *input* rate for the other billable token classes; they are vendor
# policy, not per-model, and identical across the Claude line.
#
#   cache write  1.25x base input at the standard rate (OpenAI's single write
#                tier, Anthropic's default 5-minute TTL), 2x for Anthropic's
#                1-hour TTL
#   cache read   0.10x base input
#   batch        0.5x everything (Message Batches API; this pipeline runs
#                synchronous requests, so it never applies — kept so a batch
#                run cannot be priced as a standard one by omission)
#
# ⚠ `usage.input_tokens` is the UNCACHED remainder only: the full prompt is
# input + cache_read + cache_write. Pricing input_tokens alone therefore both
# undercounts tokens and undercounts cost the moment caching is switched on.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.00
CACHE_READ_MULTIPLIER = 0.10
BATCH_MULTIPLIER = 0.50

# Server-side tools bill per request, not per token, and the rate differs by
# vendor: Anthropic charges $10 per 1,000 web searches, Google $14 per 1,000
# grounded prompts on the Gemini 3.x line (after a monthly free allowance this
# project never touches). None of the prompting strategies here use a tool — the
# entry exists so that a call that somehow made one is priced rather than
# silently free.
WEB_SEARCH_USD_PER_1K_REQUESTS = 10.00          # default / Anthropic
WEB_SEARCH_USD_PER_1K_BY_VENDOR: Dict[str, float] = {
    "anthropic": 10.00,
    "google": 14.00,
}


# Hours (UTC) during which a vendor charges its full rate. Outside them the
# model's `offpeak_multiplier` applies. Each entry is a [start, end) hour range.
# DeepSeek: "Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through
# Friday (all other hours are off-peak)" — api-docs.deepseek.com/quick_start/
# pricing, re-read 2026-08-30.
PEAK_HOURS_UTC: Dict[str, List[Tuple[int, int]]] = {
    "deepseek": [(1, 4), (6, 10)],
}

# ⚠ Added 2026-08-30. The weekday half of the same sentence was being dropped:
# the hour windows alone made every Saturday and Sunday call peak-priced, at
# exactly double what DeepSeek charges — and weekends are when a long run is
# most likely to be left going. Weekdays are Python's Mon=0 … Sun=6. A vendor
# absent here has no weekday restriction on the windows above.
PEAK_WEEKDAYS_UTC: Dict[str, Tuple[int, ...]] = {
    "deepseek": (0, 1, 2, 3, 4),
}


def is_peak_hour(vendor: str, at_utc: Optional[datetime] = None) -> bool:
    """Whether `vendor` is charging its full rate at `at_utc` (default: now).

    A vendor with no peak windows is always at full rate, which is what every
    vendor but DeepSeek does. A vendor with peak windows is at full rate only
    inside one of them AND on a day the windows apply to.
    """
    windows = PEAK_HOURS_UTC.get(vendor)
    if not windows:
        return True
    moment = at_utc or datetime.now(timezone.utc)
    weekdays = PEAK_WEEKDAYS_UTC.get(vendor)
    if weekdays is not None and moment.weekday() not in weekdays:
        return False
    return any(start <= moment.hour < end for start, end in windows)


def web_search_price(vendor: str) -> float:
    """USD per 1,000 server-side search requests for `vendor`.

    Falls back to the Anthropic rate for a vendor whose tool pricing was never
    recorded — an overestimate rather than a free call, so an unpriced tool use
    shows up as cost instead of disappearing.
    """
    return WEB_SEARCH_USD_PER_1K_BY_VENDOR.get(vendor, WEB_SEARCH_USD_PER_1K_REQUESTS)


def prices_for(model: ModelSpec, on_date: Optional[str] = None) -> Tuple[Optional[float], Optional[float], str]:
    """The input/output rates actually billed for `model` on `on_date`
    (ISO date, default today), plus which basis they came from.

    Returns `(price_in, price_out, basis)` where basis is "intro", "list" or
    "unverified". Rates are None for a model whose prices were never confirmed —
    the caller must then report no cost rather than a guess.
    """
    if model.price_in_per_mtok is None or model.price_out_per_mtok is None:
        return None, None, "unverified"
    if model.intro_until and model.intro_price_in_per_mtok is not None:
        day = on_date or date.today().isoformat()
        if day <= model.intro_until:
            return model.intro_price_in_per_mtok, model.intro_price_out_per_mtok, "intro"
    return model.price_in_per_mtok, model.price_out_per_mtok, "list"


def rates_for(model: ModelSpec, basis: str) -> Optional[Dict[str, float]]:
    """The per-Mtok rates behind a recorded `cost_basis`, one per token class.

    `cost_usd` returns a single number; a reader who wants to see *where* a call's
    cost came from needs the rates that produced it. Recomputing them from the
    model alone would be wrong for a row priced in the past — an intro window
    that has since closed, or an off-peak hour — so the basis string the row
    recorded is what selects them. Returns None when the model has no rates.

    Keep in step with `cost_usd`: same inputs, same multipliers, same order.
    """
    if model.price_in_per_mtok is None or model.price_out_per_mtok is None:
        return None
    basis = basis or ""

    if "intro" in basis and model.intro_price_in_per_mtok is not None:
        price_in = model.intro_price_in_per_mtok
        price_out = model.intro_price_out_per_mtok
        price_cached = model.intro_price_cached_in_per_mtok
    else:
        price_in = model.price_in_per_mtok
        price_out = model.price_out_per_mtok
        price_cached = model.price_cached_in_per_mtok
    if price_cached is None:
        price_cached = price_in * CACHE_READ_MULTIPLIER

    if "offpeak" in basis and model.offpeak_multiplier is not None:
        f = model.offpeak_multiplier
        price_in, price_out, price_cached = price_in * f, price_out * f, price_cached * f

    rates = {
        "input_tokens": price_in,
        "cached_tokens": price_cached,
        "cache_write_tokens": price_in * CACHE_WRITE_MULTIPLIER,
        "cache_write_1h_tokens": price_in * CACHE_WRITE_1H_MULTIPLIER,
        "output_tokens": price_out,
    }
    if "batch" in basis:
        rates = {k: v * BATCH_MULTIPLIER for k, v in rates.items()}
    return rates


@dataclass(frozen=True)
class TokenUsage:
    """Every billable quantity one API call can produce.

    Kept separate from the vendor SDKs' own usage objects so that the cost
    formula has exactly one input shape, and so a field the pipeline does not
    know about cannot silently reach the cost function as a zero.
    """
    input_tokens: int = 0            # uncached prompt tokens, at 1x
    output_tokens: int = 0           # completion incl. thinking/reasoning, at 1x output rate
    cache_read_tokens: int = 0       # prompt served from cache, at 0.10x input
    cache_write_tokens: int = 0      # prompt written to cache, at 1.25x input
    cache_write_1h_tokens: int = 0   # written to Anthropic's 1-hour cache, at 2x input
    web_search_requests: int = 0     # server-side tool calls, priced per request
    batch: bool = False              # served by the Batches API → half price

    @property
    def billable_input_tokens(self) -> int:
        """The whole prompt as billed — the sum the vendor's own invoice shows,
        not `input_tokens`, which is only its uncached part."""
        return (self.input_tokens + self.cache_read_tokens
                + self.cache_write_tokens + self.cache_write_1h_tokens)

    @property
    def total_tokens(self) -> int:
        return self.billable_input_tokens + self.output_tokens


def cost_usd(model: ModelSpec, usage: TokenUsage, on_date: Optional[str] = None,
             at_utc: Optional[datetime] = None) -> Tuple[Optional[float], str]:
    """USD for one API call, over every billable token class.

    Returns `(cost, basis)`; cost is None when the model's prices are
    unverified — never 0.0, which would read as "this call was free".

    `at_utc` is the moment the call was billed, and matters only for a vendor
    with time-of-day pricing: DeepSeek charges its full rate during peak hours
    and half of it the rest of the day, so the same call costs twice as much at
    02:00 UTC as at 12:00. Defaults to now, which is when a live call is priced.
    """
    price_in, price_out, basis = prices_for(model, on_date)
    if price_in is None or price_out is None:
        return None, basis

    # Cache reads: the model's own published rate when the vendor sets one per
    # model, otherwise the vendor-wide multiplier. Cache writes are 1.25x the
    # uncached input rate on both vendors (Anthropic 5-minute TTL; OpenAI's
    # single tier) — verified against each vendor's price sheet. Inside an
    # introductory window the discounted cache rate applies with the rest.
    price_cached = model.price_cached_in_per_mtok
    if basis == "intro" and model.intro_price_cached_in_per_mtok is not None:
        price_cached = model.intro_price_cached_in_per_mtok
    if price_cached is None:
        price_cached = price_in * CACHE_READ_MULTIPLIER

    # Time-of-day pricing, applied to every token class alike.
    if model.offpeak_multiplier is not None and not is_peak_hour(model.vendor, at_utc):
        factor = model.offpeak_multiplier
        price_in, price_out, price_cached = (price_in * factor, price_out * factor,
                                             price_cached * factor)
        basis = f"{basis}+offpeak"
    elif model.offpeak_multiplier is not None:
        basis = f"{basis}+peak"
    tokens_usd = (
        usage.input_tokens * price_in
        + usage.cache_read_tokens * price_cached
        + usage.cache_write_tokens * price_in * CACHE_WRITE_MULTIPLIER
        + usage.cache_write_1h_tokens * price_in * CACHE_WRITE_1H_MULTIPLIER
        + usage.output_tokens * price_out
    ) / 1_000_000
    tools_usd = (usage.web_search_requests
                 * web_search_price(model.vendor) / 1_000)

    total = tokens_usd + tools_usd
    if usage.batch:
        total *= BATCH_MULTIPLIER
        basis = f"{basis}+batch"
    return total, basis


# ── prompting strategies (Li et al., 2025, plus user-supplied strategies) ────
STRATEGIES: List[str] = ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "tree_of_thought", "role_prompt_1", "role_prompt_2", "zero_shot_graph_type_tn_rules", "zero_shot__short_bpmn_description"]

# Everything a run may select. `zero_shot_system` — Li et al.'s "Fine-Tuned
# LLMs" system-role variant — was **deactivated on 2026-08-17** at the author's
# instruction: it is no longer selectable in the wizard and `--strategies
# zero_shot_system` is rejected as unknown. Its template and provenance stay in
# prompts.py, so re-enabling it is adding the name back here and in
# prompts.ALL_STRATEGIES (both lists, they are not derived from one another).
# It is the only Li-et-al. strategy not in the run set — the thesis has to say
# so; see README "Open decisions" #1.
ALL_STRATEGIES: List[str] = list(STRATEGIES)


# ── repetitions ───────────────────────────────────────────────────────────────
# How often one (item × model × strategy) combination is generated. LLM output
# is non-deterministic, so a single generation is one draw, not the setting:
# repeating it is what turns token count, cost, latency and quality into values
# with a spread instead of single points. Asked for by the interactive wizard
# and settable with --repetitions. The ceiling is a cost guard — N repetitions
# cost N times as much and take N times as long.
MIN_REPETITIONS = 1
MAX_REPETITIONS = 20
DEFAULT_REPETITIONS = 1


# ── run settings ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RunSettings:
    temperature: Optional[float] = None  # only applied to models with supports_temperature=True
    # 4096 was too small once reasoning models entered the run: their thinking
    # is billed inside the output budget, so a cap that fits the answer does not
    # fit the thinking that precedes it. Measured 2026-08-18 on the PMo prompts:
    # mistral-medium-latest spent all 4096 tokens reasoning and returned nothing
    # on 4 calls out of 4 — billed in full, stop_reason "length", 0/4 valid —
    # and reasoning-on DeepSeek runs need 6k-11k. A cap only costs money when it
    # is consumed, so a generous one is free for the models that finish early
    # and is the difference between a result and a paid blank for the ones that
    # do not.
    max_output_tokens: int = 32768
    n_few_shot: int = 2
    # Quality-scoring settings (GED timeout, BERT toggle) were removed together
    # with the metric layer — add the new ones here so they land in the manifest.


DEFAULTS = RunSettings()
