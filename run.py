#!/usr/bin/env python3
"""
run.py — CLI for the PMo / Graphviz-DOT cost & token experiment (multi-vendor)
=================================================================================

Input is the PMo dataset: `descriptions/<id>.txt` (the LLM input) paired with
`graphviz/<id>.dot` (the ground truth) by file number. Its location defaults to
cfg.PMO_DATASET_DIR and can be overridden with --dataset-dir.

Examples
--------
No arguments at all — the run is configured interactively (providers, models,
item count, strategies, repetitions), then confirmed before anything is spent.
The wizard always runs on the PMo dataset; another location or a CSV is a
scripted run:
    python run.py

Providers without a wired-up integration (everything except Anthropic today,
see cfg.LIVE_VENDORS) can already be selected: they are served by the mock
provider, return instantly, cost nothing, and their result rows carry
is_mock=True so placeholders are never read as measurements.

Offline smoke test (no API key, no network), 1 held-out PMo item across all
vendors' models:
    python run.py --demo

Real run against the PMo dataset, every vendor, all enabled models, the nine
default strategies, first 50 items, BERT off. The API key is read from
api_keys.py automatically — nothing is asked for:
    python run.py --limit 50

Run against a single vendor, and a PMo copy in a different location:
    python run.py --vendors anthropic --dataset-dir /path/to/pmo-dataset --limit 50

Generate every (item × model × strategy) combination five times over, to get a
spread for tokens, cost, latency and quality instead of a single draw:
    python run.py --limit 10 --repetitions 5

Run on a CSV of process descriptions (one row per item). The cross product
Prompt × Model × rows is executed; tokens/cost are recorded for every row, and
quality is scored too for any row that supplies a ground-truth .gv path:
    python run.py --vendors anthropic,openai --csv my_descriptions.csv

Re-score an existing run and rebuild its page (free, no API calls):
    python score_run.py runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import config as cfg
import csv_export
import dataset
import prompts
import quality
import runcontrol
import results_report
from dataset import load_dataset, load_dataset_from_csv
from pipeline import run_experiment
from providers import get_provider

# Some Windows consoles default stdout/stderr to a legacy code page (e.g.
# cp1252) that can't encode the ✓/✗/⚠ symbols used in status output below —
# printing one would crash the run before it starts. Force UTF-8 with a safe
# fallback so status output never takes the process down.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["live", "mock"], default=None,
                   help="'live' calls real vendor APIs (prompts for keys); 'mock' is offline. "
                        "Default: mock for --demo, else live.")
    p.add_argument("--vendors", default=None,
                   help=f"Comma list of vendors to run (default: all). Known: {', '.join(cfg.VENDORS)}")

    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="PMo dataset location — the folder containing "
                        "descriptions/ and graphviz/ (either subfolder is also "
                        f"accepted). Default: {cfg.PMO_DATASET_DIR}")
    p.add_argument("--csv", type=Path, default=None,
                   help="CSV of process descriptions (one row per item). "
                        "Alternative to --dataset-dir. Columns are auto-detected: "
                        "a description column is required; optional id and "
                        "ground-truth-path columns are used if present.")
    p.add_argument("--csv-description-col", default=None,
                   help="Name of the description column (override auto-detection)")
    p.add_argument("--csv-id-col", default=None,
                   help="Name of the id column (override auto-detection)")
    p.add_argument("--csv-ground-truth-col", default=None,
                   help="Name of the ground-truth .gv path column (override auto-detection)")
    p.add_argument("--csv-delimiter", default=None,
                   help="Force the CSV delimiter (default: auto-sniff; handles ';')")
    p.add_argument("--demo", action="store_true",
                   help="Offline demo on one PMo item (implies --provider mock)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of evaluated items (cost control)")
    p.add_argument("--repetitions", type=int, default=cfg.DEFAULT_REPETITIONS, metavar="N",
                   help=f"How often each (item × model × strategy) combination is "
                        f"generated, {cfg.MIN_REPETITIONS}-{cfg.MAX_REPETITIONS} "
                        f"(default {cfg.DEFAULT_REPETITIONS}). Repetitions differ only "
                        f"by the model's own non-determinism and are numbered in the "
                        f"`repetition` column. N repetitions cost N times as much.")

    p.add_argument("--models", default=None,
                   help="Comma list of model keys (default: all enabled models for --vendors). "
                        f"Known: {[m.key for m in cfg.ALL_MODELS]}")
    p.add_argument("--strategies", default=",".join(cfg.STRATEGIES),
                   help=f"Comma list. Known: {cfg.ALL_STRATEGIES}")
    p.add_argument("--few-shot-ids", default=",".join(cfg.FEW_SHOT_IDS),
                   help="Comma list of PMo item ids to show as few-shot exemplars. "
                        "They are still evaluated themselves — when the item being "
                        "generated for is one of them, that exemplar is swapped for "
                        "a --few-shot-fallback-id. "
                        f"Default: {','.join(cfg.FEW_SHOT_IDS)}")
    p.add_argument("--few-shot-fallback-ids", default=",".join(cfg.FEW_SHOT_FALLBACK_IDS),
                   help="Comma list of PMo item ids used as stand-in exemplars when an "
                        "evaluated item is itself an exemplar, so no item is shown its "
                        f"own answer. Default: {','.join(cfg.FEW_SHOT_FALLBACK_IDS)}")
    p.add_argument("--few-shot-alt-ids", default=",".join(cfg.FEW_SHOT_ALT_IDS),
                   help="Comma list of PMo item ids used as the COMPLETE exemplar set "
                        "for the items in --few-shot-alt-for-ids, instead of swapping "
                        "one slot out of --few-shot-ids. Pass an empty string to switch "
                        f"this off. Default: {','.join(cfg.FEW_SHOT_ALT_IDS)}")
    p.add_argument("--few-shot-alt-for-ids", default=",".join(cfg.FEW_SHOT_ALT_FOR_IDS),
                   help="Comma list of PMo item ids that run on --few-shot-alt-ids "
                        "instead of on a per-slot swap, so they all see the same "
                        "exemplars and none of them sees its own answer. "
                        f"Default: {','.join(cfg.FEW_SHOT_ALT_FOR_IDS)}")
    p.add_argument("--hold-out-few-shot-items", action="store_true",
                   help="Old behaviour: drop the --few-shot-ids items from the "
                        "evaluated set (55 → 53) instead of evaluating them with a "
                        "swapped exemplar.")
    p.add_argument("--few-shot-files", default=None,
                   help="Comma list of ground-truth .dot/.gv paths to use as few-shot "
                        "exemplars. Overrides --few-shot-ids; use it for exemplars "
                        "from outside the dataset directory.")

    # settings overrides
    p.add_argument("--temperature", type=float, default=cfg.DEFAULTS.temperature)
    p.add_argument("--max-output-tokens", type=int, default=cfg.DEFAULTS.max_output_tokens)
    p.add_argument("--n-few-shot", type=int, default=cfg.DEFAULTS.n_few_shot)
    p.add_argument("--parallel-models", action="store_true",
                   help="Run one request per model at a time, all models at "
                        "once, instead of one call at a time. A model never "
                        "competes with itself, so the latency columns stay "
                        "comparable BETWEEN models - but they are measured "
                        "beside the other models, not alone, and every row "
                        "records how many were in flight (concurrent_models). "
                        "Report latency from a sequential run; use this for "
                        "token, cost and quality sweeps.")
    p.add_argument("--tag", default=None, help="Label appended to the run id")
    p.add_argument("--resume", type=Path, default=None, metavar="RUN_DIR",
                   help="Continue an interrupted run instead of starting a new "
                        "one. The plan is read back out of that run's "
                        "manifest.json and every generation already in its "
                        "results.jsonl is skipped, so only the missing calls "
                        "are paid for. The flags that define the plan "
                        "(--models, --strategies, --repetitions, the few-shot "
                        "options, --temperature, ...) come from the manifest "
                        "and cannot be given alongside it; --limit and "
                        "--hold-out-few-shot-items can, because a run does not "
                        "record them.")
    p.add_argument("--resume-rewind", type=int, default=0, metavar="N",
                   help="With --resume: run the last N generations of every "
                        "model again instead of continuing straight on. The "
                        "replies that arrived while a run was going down are "
                        "the ones worth distrusting. The rows are moved to "
                        "rewound.jsonl in the run directory, not deleted, and "
                        "cost N x (number of models) extra calls.")
    p.add_argument("--no-score", action="store_true",
                   help="Skip the quality metrics (score later with score_run.py)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip building results.html (build it later with score_run.py)")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip the csv/ export (write it later with csv_export.py)")
    return p.parse_args(argv)


def _resolve_vendors(arg: str) -> List[str]:
    if not arg:
        return list(cfg.VENDORS)
    vendors = [v.strip() for v in arg.split(",") if v.strip()]
    bad = [v for v in vendors if v not in cfg.VENDORS]
    if bad:
        sys.exit(f"Unknown vendors: {bad}. Known: {cfg.VENDORS}")
    return vendors


def _resolve_models(arg: str, vendors: List[str]) -> List[cfg.ModelSpec]:
    if not arg:
        models = [m for m in cfg.enabled_models() if m.vendor in vendors]
        if not models:
            sys.exit(f"No enabled models for vendors {vendors}.")
        return models
    keys = [k.strip() for k in arg.split(",") if k.strip()]
    return [cfg.model_by_key(k) for k in keys]


def _resolve_strategies(arg: str) -> List[str]:
    strats = [s.strip() for s in arg.split(",") if s.strip()]
    bad = [s for s in strats if not prompts.strategy_is_known(s)]
    if bad:
        sys.exit(f"Unknown strategies: {bad}. Known: {cfg.ALL_STRATEGIES}")
    return strats


def _resolve_keys(vendors: List[str]) -> Dict[str, str]:
    """API keys for the live vendors of this run, taken from api_keys.py (or
    the vendor's env var as a fallback). Nothing is asked interactively; a
    missing key aborts before any call with a message saying where to put it.
    Keys are held in memory only and never printed."""
    keys: Dict[str, str] = {}
    missing: List[str] = []
    for v in vendors:
        key = cfg.api_key_for(v)
        if key:
            keys[v] = key
        else:
            missing.append(v)

    if missing:
        sys.exit(
            "\n  No API key configured for: " + ", ".join(missing) +
            "\n  Add it to api_keys.py (API_KEYS[\"" + missing[0] + "\"] = \"sk-...\"), "
            "or set $" + cfg.VENDOR_ENV_VAR[missing[0]] + " in the environment."
            "\n  Aborting before any call — nothing was spent."
        )

    for v in vendors:
        print(f"  Using the {v} API key from {cfg.api_key_source(v)}.")
    return keys


# ══════════════════════════════════════════════════════════════════════════
# Interactive setup — what plain `python run.py` (no arguments) runs.
# Every answer maps onto the exact same flags the CLI already accepts, so the
# scripted and interactive paths configure the run through one code path.
# ══════════════════════════════════════════════════════════════════════════

def _ask(question: str, default: str = "") -> str:
    """One prompt. Enter accepts `default`. A non-interactive stdin (piped or
    redirected) also falls back to the default, so the wizard can never hang
    an unattended run waiting for a keystroke that will not come."""
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{question}{suffix}: ").strip()
    except EOFError:
        print(f"    (stdin is not interactive — using default: {default!r})")
        return default
    return raw or default


# Returned by a step the user backed out of. A sentinel rather than None, so it
# can never be confused with "the user chose nothing".
BACK = object()
_BACK_WORDS = ("b", "back")


def _is_back(raw: str) -> bool:
    return raw.strip().lower() in _BACK_WORDS


def _select_vendors():
    print("\n  ── 1/6  Providers ──────────────────────────────────────────────")
    print("  Which providers do you want to investigate?\n")
    for i, v in enumerate(cfg.VENDORS, 1):
        n = len([m for m in cfg.models_by_vendor(v) if m.enabled])
        # No "live" tag: every vendor is wired up now, so it sat on every line
        # and distinguished nothing. The mock tag stays — a mocked vendor is the
        # exception worth flagging. The cost warning is not lost either; the
        # confirmation step still names every provider the run will pay for.
        tag = "" if cfg.vendor_is_live(v) else "   (mock — not wired up yet, free and instant)"
        print(f"    {i}. {v:<10} {n} model(s){tag}")

    while True:
        raw = _ask("\n  Numbers or names, comma-separated ('all' for every provider)", "1")
        if _is_back(raw):
            return BACK
        picked = _parse_vendor_selection(raw)
        if picked:
            return picked
        print("  ✗ Not a valid selection. Try e.g. '1', '1,5', 'anthropic,deepseek' or 'all'.")


def _parse_vendor_selection(raw: str) -> List[str]:
    """Accepts numbers, vendor names, or 'all'. Returns [] on any bad token so
    the caller re-asks rather than silently running a different set."""
    if raw.strip().lower() in ("all", "*"):
        return list(cfg.VENDORS)
    picked: List[str] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if not 1 <= idx <= len(cfg.VENDORS):
                return []
            vendor = cfg.VENDORS[idx - 1]
        elif token in cfg.VENDORS:
            vendor = token
        else:
            return []
        if vendor not in picked:
            picked.append(vendor)
    return picked


def _model_label(model: cfg.ModelSpec) -> str:
    """`claude-opus-5 (Advanced)` — the model that will actually be called,
    with the tier it stands for.

    The id leads because that is what the API is asked for and what the results
    are attributed to; the tier follows because it is this project's own
    grouping, not the vendor's. A vendor need not offer all three: DeepSeek
    lists two models and no Standard tier, and the gap is worth seeing.
    """
    return f"{model.model_id} ({model.tier.capitalize()})"


def _select_models_for_vendor(vendor: str, position: int, total: int):
    """The tier question, asked once for one provider.

    Returns the chosen ModelSpecs, BACK to re-open the previous provider (or the
    provider list, at the first one), or [] when the provider is skipped.
    """
    available = [m for m in cfg.models_by_vendor(vendor) if m.enabled]
    if not available:
        print(f"\n  ⓘ  {vendor}: no enabled model — skipped.")
        return []

    print(f"\n  {vendor}  ({position} of {total})")
    for i, model in enumerate(available, 1):
        print(f"    {i}. {_model_label(model)}")

    while True:
        raw = _ask("\n  Numbers, comma-separated "
                   "('all', 's' to skip this provider, 'b' to go back)", "all")
        if _is_back(raw):
            return BACK
        if raw.strip().lower() in ("s", "skip"):
            return []
        picked = _parse_model_selection(raw, available)
        if picked:
            return picked
        print(f"  ✗ Not a valid selection. Enter numbers 1-{len(available)} "
              "(e.g. '1', '1,3'), 'all', or 's' to skip.")


def _parse_model_selection(raw: str, available: List[cfg.ModelSpec]) -> List[cfg.ModelSpec]:
    """Accepts numbers, tier names, or 'all'. Returns [] on any bad token so the
    caller re-asks rather than silently running a different set."""
    if raw.strip().lower() in ("all", "*"):
        return list(available)
    picked: List[cfg.ModelSpec] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if not 1 <= idx <= len(available):
                return []
            model = available[idx - 1]
        else:
            matches = [m for m in available if m.tier == token]
            if not matches:
                return []
            model = matches[0]
        if model not in picked:
            picked.append(model)
    return picked


def _select_models(vendors: List[str]):
    """Step 2, walked provider by provider.

    Backing out of the first provider returns to the provider list; backing out
    of a later one re-opens the provider before it, so a mistyped answer costs
    one question rather than the whole step.
    """
    print("\n  ── 2/6  Models ─────────────────────────────────────────────────")
    print("  Asked once per provider. Pick the models to run.")

    chosen: Dict[str, List[cfg.ModelSpec]] = {}
    i = 0
    while i < len(vendors):
        picked = _select_models_for_vendor(vendors[i], i + 1, len(vendors))
        if picked is BACK:
            if i == 0:
                return BACK
            i -= 1
            chosen.pop(vendors[i], None)
            continue
        chosen[vendors[i]] = picked
        i += 1

    return [m for v in vendors for m in chosen.get(v, [])]


def _resolve_wizard_input() -> int:
    """The wizard always runs on the PMo dataset — there is nothing to choose,
    so it is validated up front instead of asked for. Returns the item count.

    Checked before any question so a missing dataset fails immediately, rather
    than after the run has been configured and the spend confirmed. Another
    location, or a CSV, is a scripted run: `--dataset-dir` / `--csv`.
    """
    try:
        _, gv_dir = dataset.resolve_dirs(cfg.PMO_DATASET_DIR)
    except ValueError as exc:
        sys.exit(f"\n  {exc}\n  The interactive setup runs on the PMo dataset. "
                 f"Set $PMO_DATASET_DIR, or pass --dataset-dir <path> "
                 f"(which skips this wizard).")
    return len([p for p in gv_dir.iterdir() if p.suffix in (".dot", ".gv")])


def _select_limit(n_items: int):
    """Returns the item cap (None = all), or BACK."""
    print("\n  ── 3/6  How many process generations? ──────────────────────────")
    while True:
        raw = _ask(f"  Number of processes, up to {n_items} of the PMo dataset "
                   "('all' for every one, 'b' to go back)", "all")
        if _is_back(raw):
            return BACK
        if raw.strip().lower() in ("all", "*"):
            return None
        if raw.strip().isdigit() and int(raw) > 0:
            n = int(raw)
            if n > n_items:
                print(f"  ⓘ  PMo holds only {n_items} — running all of them.")
                return None
            return n
        print("  ✗ Enter a positive number, or 'all'.")


def _select_repetitions():
    """Returns how often each (item × model × strategy) cell is generated, or
    BACK. Bounded by cfg.MIN/MAX_REPETITIONS — the ceiling is a cost guard."""
    print("\n  ── 5/6  Repetitions ────────────────────────────────────────────")
    print("  How often should each combination of process description, model and")
    print("  prompting strategy be generated? The model is non-deterministic, so")
    print("  repeating a setting is what gives tokens, cost, latency and quality a")
    print("  spread instead of a single value — at N times the price.\n")

    while True:
        raw = _ask(f"  Repetitions per combination "
                   f"({cfg.MIN_REPETITIONS}-{cfg.MAX_REPETITIONS}, 'b' to go back)",
                   str(cfg.DEFAULT_REPETITIONS))
        if _is_back(raw):
            return BACK
        raw = raw.strip()
        if raw.isdigit() and cfg.MIN_REPETITIONS <= int(raw) <= cfg.MAX_REPETITIONS:
            return int(raw)
        print(f"  ✗ Enter a whole number between {cfg.MIN_REPETITIONS} and "
              f"{cfg.MAX_REPETITIONS}.")


def _select_strategies():
    """Returns the chosen strategy list, or BACK."""
    print("\n  ── 4/6  Prompting strategies ───────────────────────────────────")
    print("  The nine thesis defaults.\n")
    for i, name in enumerate(cfg.ALL_STRATEGIES, 1):
        note = "" if name in cfg.STRATEGIES else "   (optional, not in 'all')"
        print(f"    {i:2}. {name}{note}")

    while True:
        raw = _ask("\n  Numbers or names, comma-separated ('all' for the nine "
                   "defaults, 'b' to go back)", "all")
        if _is_back(raw):
            return BACK
        picked = _parse_strategy_selection(raw)
        if picked:
            return picked
        print(f"  ✗ Not a valid selection. Enter numbers 1-{len(cfg.ALL_STRATEGIES)} "
              "(e.g. '1', '1,3'), strategy names, or 'all'.")


def _parse_strategy_selection(raw: str) -> List[str]:
    """Accepts numbers, strategy names, or 'all'. Returns [] on any bad token so
    the caller re-asks rather than silently running a different set.

    'all' is the nine-strategy thesis default (cfg.STRATEGIES) — the same set
    `--strategies` defaults to, so the wizard and the scripted path stay in
    step. Since `zero_shot_system` was deactivated it is also the whole of
    cfg.ALL_STRATEGIES, so 'all' and the numbered list now cover the same nine.
    """
    if raw.strip().lower() in ("all", "*"):
        return list(cfg.STRATEGIES)
    picked: List[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if not 1 <= idx <= len(cfg.ALL_STRATEGIES):
                return []
            name = cfg.ALL_STRATEGIES[idx - 1]
        elif prompts.strategy_is_known(token):
            name = token
        else:
            return []
        if name not in picked:
            picked.append(name)
    return picked


def _select_execution(models):
    """Returns True to run the models in parallel, False for one call at a
    time, or BACK.

    Only asked when there is more than one model: with a single model the two
    regimes are the same run, and offering the choice would suggest otherwise.
    """
    if len(models) < 2:
        return False

    print("\n  ── 6/6  Execution ──────────────────────────────────────────────")
    print("  Parallel gives every model one worker, so all models run at once")
    print("  but no model ever has two calls in flight. It is far faster and")
    print("  leaves tokens, cost and quality untouched — those are counted per")
    print("  request. What it does change is latency: each call is measured")
    print(f"  beside the other {len(models) - 1}, not alone. Every row records")
    print("  `concurrent_models`, so the two regimes can never be pooled by")
    print("  accident.\n")
    print("  Run sequentially if the latency numbers are going into the thesis.\n")

    while True:
        raw = _ask(f"  Run the {len(models)} models in parallel? "
                   f"(y/n, 'b' to go back)", "y")
        if raw is BACK:
            return BACK
        answer = raw.strip().lower()
        if answer in ("y", "yes", "j", "ja", ""):
            return True
        if answer in ("n", "no", "nein"):
            return False
        print("  ✗ Enter 'y' or 'n'.")


def _confirm_plan(models, used_vendors, strategies, limit, repetitions, n_items=None,
                  parallel_models=False):
    """Show the run plan. Returns True to start, False to abort, or BACK."""
    live = [v for v in used_vendors if cfg.vendor_is_live(v)]
    mocked = [v for v in used_vendors if not cfg.vendor_is_live(v)]
    live_models = [m for m in models if m.vendor in live]
    per_item = len(models) * len(strategies) * repetitions
    live_per_item = len(live_models) * len(strategies) * repetitions

    print("\n  ── Run plan ────────────────────────────────────────────────────")
    for v in used_vendors:
        n = len([m for m in models if m.vendor == v])
        print(f"    {v:<10} {n} model(s)   {'LIVE (paid)' if v in live else 'mock (free)'}")
    print(f"    strategies : {len(strategies)}")
    # Spelled out rather than left at "all": every item of the dataset is
    # generated for, few-shot exemplars included, and the count is the thing a
    # reader checks the run against afterwards.
    print(f"    items      : {limit if limit else ('all' if n_items is None else f'all {n_items}')}")
    print(f"    repetitions: {repetitions}"
          + ("" if repetitions == 1 else f"   ({repetitions}× the calls and {repetitions}× the cost)"))
    # Named in the plan because it decides how the latency columns may be read
    # afterwards, and because a reader has to be able to see it before paying.
    print(f"    execution  : " + (
        f"parallel — {len(models)} models at once, one request per model"
        if parallel_models else "sequential — one call at a time"))
    print(f"    → {per_item} generation(s) per item, "
          f"{live_per_item} of them real API calls")
    if mocked:
        print(f"\n    Mocked providers ({', '.join(mocked)}) return a stub graph instantly. "
              "\n    Their rows land in results.csv flagged is_mock=True — placeholders, "
              "\n    not measurements. Exclude them before reporting anything.")

    if live_per_item:
        print(f"\n  ⚠  This will spend money on: {', '.join(live)}")
        answer = _ask("  Type 'yes' to start, 'b' to change the configuration, "
                      "anything else to abort", "no")
        if _is_back(answer):
            return BACK
        return answer.strip().lower() in ("yes", "y")

    answer = _ask("  Press Enter to start ('b' to change the configuration)", "")
    return BACK if _is_back(answer) else True


def _interactive_setup(args) -> None:
    print("\n  ════════════════════════════════════════════════════════════════")
    print("   PMo / DOT experiment — interactive setup")
    print("   No arguments given, so this run is configured here.")
    print("   Press Enter at any step to accept the [default], 'b' to go back.")
    print("  ════════════════════════════════════════════════════════════════")

    n_items = _resolve_wizard_input()
    print(f"\n  Input: PMo dataset, {n_items} items — {cfg.PMO_DATASET_DIR}")

    # Walked as a step machine rather than straight-line calls, so any step can
    # hand control back to the previous one. Answers already given are kept:
    # going back re-asks that one step, it does not reset the whole wizard.
    vendors: List[str] = []
    models: List[cfg.ModelSpec] = []
    used_vendors: List[str] = []
    strategies: List[str] = []
    limit = None
    repetitions = cfg.DEFAULT_REPETITIONS
    parallel_models = False

    step = 0
    while True:
        if step == 0:
            picked = _select_vendors()
            if picked is BACK:
                print("  ⓘ  Already at the first step.")
                continue
            vendors = picked
            step = 1

        elif step == 1:
            picked = _select_models(vendors)
            if picked is BACK:
                step = 0
                continue
            models = picked
            if not models:
                # Recoverable now that steps can be revisited: re-ask instead of
                # exiting and making the user restart the whole wizard.
                print("\n  ✗ No model selected for any provider. Pick at least "
                      "one, or 'b' for the providers.")
                continue

            # A provider drops out when it was skipped, or has no enabled model.
            # Carry only the ones that actually contribute, so the plan doesn't
            # advertise a provider that will never be called.
            used_vendors = [v for v in vendors if any(m.vendor == v for m in models)]
            dropped = [v for v in vendors if v not in used_vendors]
            if dropped:
                print(f"\n  ⓘ  No model selected for: {', '.join(dropped)} "
                      "— dropped from this run.")
            step = 2

        elif step == 2:
            picked = _select_limit(n_items)
            if picked is BACK:
                step = 1
                continue
            limit = picked
            step = 3

        elif step == 3:
            picked = _select_strategies()
            if picked is BACK:
                step = 2
                continue
            strategies = picked
            step = 4

        elif step == 4:
            picked = _select_repetitions()
            if picked is BACK:
                step = 3
                continue
            repetitions = picked
            step = 5

        elif step == 5:
            picked = _select_execution(models)
            if picked is BACK:
                step = 4
                continue
            parallel_models = picked
            step = 6

        else:
            decision = _confirm_plan(models, used_vendors, strategies, limit,
                                     repetitions, n_items, parallel_models)
            if decision is BACK:
                step = 5
                continue
            if not decision:
                sys.exit("  Aborted — nothing was called, nothing was spent.")
            break

    args.vendors = ",".join(used_vendors)
    args.models = ",".join(m.key for m in models)
    args.limit = limit
    args.strategies = ",".join(strategies)
    args.repetitions = repetitions
    args.parallel_models = parallel_models


# Flags whose value the manifest already holds. Given alongside --resume they
# would silently describe a different experiment than the one being continued,
# so they are refused rather than merged.
_RESUME_OWNED_FLAGS = (
    "--models", "--vendors", "--strategies", "--repetitions", "--dataset-dir",
    "--csv", "--few-shot-ids", "--few-shot-fallback-ids", "--few-shot-alt-ids",
    "--few-shot-alt-for-ids", "--few-shot-files", "--n-few-shot",
    "--temperature", "--max-output-tokens", "--parallel-models", "--demo",
    "--provider", "--no-score",
)


def _apply_resume(args, raw_argv: List[str]) -> None:
    """Fill `args` from the manifest of the run being resumed.

    Everything downstream then builds the plan exactly as it did the first
    time - same models, strategies, exemplars, settings - because the plan is a
    pure cross product with no random element in it. `pipeline.run_experiment`
    drops the cells already in `results.jsonl` and calls for the rest.

    Two things a run does not record are left to the caller: `--limit` and
    `--hold-out-few-shot-items` shape the item list but never reach the
    manifest, so they stay available and are checked against the recorded rows
    in `_resume_conflicts` instead of guessed at here.
    """
    run_dir = Path(args.resume)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"--resume: no manifest.json in {run_dir} - that is not a run directory.")
    if not (run_dir / "results.jsonl").exists():
        sys.exit(f"--resume: no results.jsonl in {run_dir} - nothing was generated, "
                 f"so there is nothing to continue. Start a new run instead.")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))

    given = [f for f in _RESUME_OWNED_FLAGS if f in raw_argv]
    if given:
        sys.exit(f"--resume takes the plan from {manifest_path}; drop "
                 f"{', '.join(given)}. A resumed run has to be the same "
                 f"experiment, or its rows cannot be pooled.")

    args.models = ",".join(x["key"] for x in m["models"])
    args.vendors = ",".join(sorted({x["vendor"] for x in m["models"]}))
    args.strategies = ",".join(m["strategies"])
    args.repetitions = int(m["repetitions"])
    # The execution width is part of what the latency columns mean, so it is
    # carried over rather than re-chosen: rows from the two sittings are only
    # comparable if they were measured under the same regime.
    args.parallel_models = bool(m.get("parallel_models"))
    settings = m.get("settings", {})
    args.temperature = settings.get("temperature")
    args.max_output_tokens = settings.get("max_output_tokens", args.max_output_tokens)
    args.n_few_shot = settings.get("n_few_shot", args.n_few_shot)
    if m.get("dataset_dir"):
        args.dataset_dir = Path(m["dataset_dir"])
    else:
        # A CSV run: the path is in the manifest, the column mapping is not, so
        # the --csv-*-col flags stay the caller's to repeat.
        args.csv = Path(m["input_source"])
    # Exemplars are recorded as paths; the ids are what the resolution takes.
    args.few_shot_ids = ",".join(Path(f).stem for f in m.get("exemplar_files", []))
    args.few_shot_fallback_ids = ",".join(
        Path(f).stem for f in m.get("fallback_exemplar_files", []))
    args.few_shot_alt_ids = ",".join(
        Path(f).stem for f in m.get("alt_exemplar_files", []))
    args.few_shot_alt_for_ids = ",".join(m.get("few_shot_alt_for_ids", []))
    # The item cap is the one plan axis a run records only as a count. Taking it
    # from `n_items` reproduces a `--limit` run (the cap takes the first N items,
    # in dataset order, the same way it did the first time) and is a no-op for a
    # run over the whole set. An explicit --limit still wins: it is how a run
    # that used --hold-out-few-shot-items, whose item *set* differs from the
    # first N, is resumed. `_resume_conflicts` checks the result either way.
    if "--limit" not in raw_argv:
        args.limit = m.get("n_items") or args.limit
    # Scoring has to match: one results.jsonl cannot hold rows that carry the
    # metric columns beside rows that do not - `quality.already_scored` reads
    # the frame, not the row, and would take the whole file for scored.
    args.no_score = not m.get("scored_inline", True)
    if m.get("mode") == "mock":
        args.provider = "mock"

    print(f"  Resuming {run_dir} - plan read from its manifest: "
          f"{len(m['models'])} model(s), {len(m['strategies'])} strategies, "
          f"{args.repetitions} repetition(s)"
          + (", scoring off" if args.no_score else ""))


def _resume_conflicts(run_dir: Path, items, models, strategies,
                      repetitions: int) -> List[str]:
    """Where the plan just built does not contain what the run already holds.

    `pipeline.run_experiment` refuses such a resume outright - it would double
    some cells and never call others. This says which axis is at fault while
    the message can still name it.
    """
    rows = [json.loads(line) for line
            in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    unknown = {
        "item(s)": {str(r.get("item_id")) for r in rows} - {i.item_id for i in items},
        "model(s)": {r.get("model_key") for r in rows} - {m.key for m in models},
        "strategy/strategies": {r.get("strategy") for r in rows} - set(strategies),
    }
    out = [f"{label} in the run but not in this plan: {sorted(v)}"
           for label, v in unknown.items() if v]
    # A plan that merely *contains* the recorded rows is not the same plan: a
    # wider item set would call for cells the original run never planned, at
    # full price. The count is what the manifest records, so it is what is
    # compared.
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    n_items = int(manifest.get("n_items", 0))
    if n_items and n_items != len(items):
        out.append(f"the run ran on {n_items} item(s), this plan has {len(items)}")
    top = max((int(r.get("repetition", 1)) for r in rows), default=1)
    if top > repetitions:
        out.append(f"the run holds repetition r{top:02d}, this plan has "
                   f"{repetitions}")
    return out


def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = _parse_args(argv)

    # Plain `python run.py` → configure interactively instead of erroring out
    # on a missing input source. Any flag at all keeps the scripted behaviour.
    if not raw_argv:
        try:
            _interactive_setup(args)
        except KeyboardInterrupt:
            sys.exit("\n  Aborted.")

    if args.resume is not None:
        _apply_resume(args, raw_argv)
    elif args.resume_rewind:
        sys.exit("--resume-rewind only means something with --resume.")

    # Checked here, before any dataset work or API key is touched, so a typo in
    # --repetitions costs nothing. The wizard validates its own input already.
    if not cfg.MIN_REPETITIONS <= args.repetitions <= cfg.MAX_REPETITIONS:
        sys.exit(f"--repetitions must be between {cfg.MIN_REPETITIONS} and "
                 f"{cfg.MAX_REPETITIONS} (got {args.repetitions}). "
                 f"The upper bound is a cost guard: N repetitions cost N times as much.")

    mode = "mock" if (args.demo or args.provider == "mock") else "live"

    vendors = _resolve_vendors(args.vendors)
    models = _resolve_models(args.models, vendors)
    strategies = _resolve_strategies(args.strategies)
    needs_exemplars = any("few_shot" in s for s in strategies)

    if args.csv and args.dataset_dir and not args.demo:
        sys.exit("Use either --csv or --dataset-dir, not both.")

    # ── dataset location ──
    # A CSV replaces the dataset as the *input*, but few-shot exemplars still
    # come from the PMo dataset, so its location is resolved either way.
    use_csv = bool(args.csv) and not args.demo
    dataset_dir = None if use_csv else Path(args.dataset_dir or cfg.PMO_DATASET_DIR)
    exemplar_source = dataset_dir or cfg.PMO_DATASET_DIR
    if dataset_dir is not None:
        try:
            dataset.resolve_dirs(dataset_dir)
        except ValueError as exc:
            sys.exit(f"\n  {exc}\n  Pass --dataset-dir <pmo-dataset>, or set "
                     f"$PMO_DATASET_DIR / cfg.PMO_DATASET_DIR to your copy.")

    # ── few-shot exemplars (shown as in-context examples) ──
    # These items are evaluated like every other one. What keeps a model from
    # being shown the answer to the item it is generating for is the per-item
    # swap in prompts.exemplars_for_item, which pulls a stand-in from
    # `fallback_files` for that one prompt.
    few_shot_ids = [s.strip() for s in args.few_shot_ids.split(",") if s.strip()]
    fallback_ids = [s.strip() for s in args.few_shot_fallback_ids.split(",") if s.strip()]
    fallback_ids = [i for i in fallback_ids if i not in few_shot_ids]
    fallback_files: List[Path] = []
    # The complete alternate set (see cfg.FEW_SHOT_ALT_*). An id that is itself
    # one of the items receiving the set would put that item's own answer back
    # into its prompt, so it is dropped here rather than silently shown.
    alt_for_ids = [s.strip() for s in args.few_shot_alt_for_ids.split(",") if s.strip()]
    alt_ids = [s.strip() for s in args.few_shot_alt_ids.split(",") if s.strip()]
    alt_ids = [i for i in alt_ids if i not in alt_for_ids]
    alt_files: List[Path] = []
    if args.few_shot_files:
        exemplar_files = [Path(p.strip()) for p in args.few_shot_files.split(",") if p.strip()]
        missing = [str(p) for p in exemplar_files if not p.exists()]
        if missing:
            sys.exit(f"--few-shot-files: file(s) not found: {missing}")
        # Explicit files may come from outside the dataset, so there is no id
        # list to draw a stand-in from: whatever is given beyond --n-few-shot
        # is the reserve.
        fallback_files = exemplar_files[args.n_few_shot:]
        exemplar_files = exemplar_files[: args.n_few_shot]
    elif dataset_dir is not None or needs_exemplars:
        try:
            exemplar_files = dataset.exemplar_paths(exemplar_source, few_shot_ids)
        except (ValueError, FileNotFoundError) as exc:
            sys.exit(f"\n  Could not resolve the few-shot exemplars: {exc}")
        try:
            fallback_files = dataset.exemplar_paths(exemplar_source, fallback_ids)
        except (ValueError, FileNotFoundError) as exc:
            # Not fatal: without a stand-in the colliding exemplar is dropped
            # for those items rather than shown, and every item still runs.
            print(f"  ⓘ  Few-shot stand-ins unavailable ({exc}) — an item that is "
                  f"itself an exemplar runs with one exemplar fewer.")
            fallback_files = []
    else:
        exemplar_files = []

    if exemplar_files and alt_ids and alt_for_ids:
        try:
            alt_files = dataset.exemplar_paths(exemplar_source, alt_ids)
        except (ValueError, FileNotFoundError) as exc:
            # Not fatal, same as the stand-ins: without the alternate set those
            # items fall back to the per-slot swap, which still keeps every
            # item out of its own prompt.
            print(f"  ⓘ  Alternate few-shot exemplars unavailable ({exc}) — items "
                  f"{','.join(alt_for_ids)} fall back to the per-slot swap.")
            alt_files = []

    # ── input items ──
    # Nothing is held out: "all" means all 55 PMo items, exemplars included.
    # The per-strategy comparison stays sound because every strategy sees the
    # same item set — the few-shot strategies just swap their own exemplar out
    # for the items where it would otherwise be the answer (see above).
    # --hold-out-few-shot-items restores the old 53-item behaviour.
    exclude = {p.stem for p in exemplar_files} if args.hold_out_few_shot_items else set()
    if args.demo:
        source_label = f"demo — PMo, mock provider ({dataset_dir})"
        items = load_dataset(dataset_dir, limit=args.limit or 1, exclude_ids=exclude)
    elif use_csv:
        source_label = str(args.csv)
        items = load_dataset_from_csv(
            args.csv, limit=args.limit, exclude_ids=exclude,
            description_col=args.csv_description_col,
            id_col=args.csv_id_col,
            ground_truth_col=args.csv_ground_truth_col,
            delimiter=args.csv_delimiter,
        )
    else:
        source_label = str(dataset_dir)
        items = load_dataset(dataset_dir, limit=args.limit, exclude_ids=exclude)

    if not items:
        sys.exit(f"No usable items found in {source_label}.")

    # Before any key is read or any call is made: does the plan just built
    # actually contain the run being resumed? --limit and
    # --hold-out-few-shot-items are the two things the manifest cannot hand
    # back, so they are the usual answer when this fires.
    if args.resume is not None:
        conflicts = _resume_conflicts(Path(args.resume), items, models,
                                      strategies, args.repetitions)
        if conflicts:
            sys.exit("\n  --resume: the plan does not match "
                     f"{args.resume}:\n    - " + "\n    - ".join(conflicts) +
                     "\n\n  Nothing was called. If the original run used --limit "
                     "or --hold-out-few-shot-items, pass the same flag again - a "
                     "run does not record either.")
    if needs_exemplars and len(exemplar_files) < args.n_few_shot:
        sys.exit(f"A few-shot strategy is selected but only {len(exemplar_files)} "
                 f"exemplar file(s) are available; need {args.n_few_shot}. "
                 f"Pass --few-shot-files, or run zero-shot strategies only.")

    # ── settings ──
    settings = cfg.RunSettings(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        n_few_shot=args.n_few_shot,
    )

    # ── providers ──
    # A run is not uniformly live or mocked. Vendors whose provider is wired up
    # (cfg.LIVE_VENDORS) call their real API; the rest are served by the mock
    # provider so they can already be selected, compared and reported on before
    # their integration exists. --demo / --provider mock forces everything off.
    needed_vendors = sorted({m.vendor for m in models})
    if mode == "mock":
        live_vendors, mocked_vendors = [], needed_vendors
    else:
        live_vendors = [v for v in needed_vendors if cfg.vendor_is_live(v)]
        mocked_vendors = [v for v in needed_vendors if not cfg.vendor_is_live(v)]

    mock_provider = get_provider("mock")
    active_providers = {"mock": mock_provider}
    for v in mocked_vendors:
        active_providers[v] = mock_provider

    if mocked_vendors:
        print(f"\n  ⓘ  Mocked this run (no live API): {', '.join(mocked_vendors)}. "
              "Their rows are flagged is_mock=True in results.csv — tokens, cost and "
              "latency for them are stub values, not measurements.")

    if live_vendors:
        keys = _resolve_keys(live_vendors)
        for v in live_vendors:
            active_providers[v] = get_provider(v, api_key=keys[v])

        live_models = [m for m in models if m.vendor in live_vendors]

        unverified = sorted({m.vendor for m in live_models if not m.price_verified})
        if unverified:
            print(f"\n  ⚠  Reminder: model IDs and prices for {unverified} are UNVERIFIED "
                  "placeholders in config.py — confirm against each vendor's current docs "
                  "before trusting cost_usd, and before any paid run.")

        # ── connectivity self-check ──
        # One free, read-only call per live model, before the real run: a
        # broken network/proxy/auth for any vendor fails fast with one clear
        # message instead of silently burning 5 retries x every row for that
        # vendor's models. Mocked vendors are skipped — nothing to check.
        print("\n  Checking connectivity for each live vendor before the real run...")
        failures = []
        for model in live_models:
            provider = active_providers[model.vendor]
            ok, msg = provider.check_connection(model.model_id)
            status = "✓" if ok else "✗"
            print(f"    {status} [{model.vendor}] {model.model_id}: {msg}")
            if not ok:
                failures.append(f"{model.vendor}/{model.model_id}: {msg}")
        if failures:
            sys.exit(
                "\n  Connectivity/auth check failed for:\n    " + "\n    ".join(failures) +
                "\n\n  Aborting before spending any generation calls. Common causes: no "
                "network access from this shell/proxy blocking python's outbound requests, "
                "an invalid/expired API key, or a stale model id in config.py."
            )

    # Recorded in the manifest: "mixed" whenever live and mocked vendors ran
    # side by side, so a later reader is never told a run was fully "live".
    if not live_vendors:
        mode = "mock"
    elif mocked_vendors:
        mode = "mixed"
    else:
        mode = "live"

    # ── run ──
    # Keyboard pause/stop, and Ctrl-C folded into the same stop. On for every
    # run, not just the paid ones: a mock run is where a reader tries the keys
    # out, and controls that exist only sometimes are controls nobody trusts.
    # It still turns itself off when stdin is not a terminal.
    control = runcontrol.RunControl()
    control.start()
    try:
        run_dir = run_experiment(
            providers=active_providers,
            mode=mode,
            models=models,
            strategies=strategies,
            items=items,
            exemplar_files=exemplar_files,
            fallback_exemplar_files=fallback_files,
            alt_exemplar_files=alt_files,
            alt_for_ids=alt_for_ids,
            settings=settings,
            dataset_dir=dataset_dir,
            tag=args.tag,
            input_source=source_label,
            repetitions=args.repetitions,
            parallel_models=args.parallel_models,
            score_inline=not args.no_score,
            resume_dir=args.resume,
            resume_rewind=args.resume_rewind,
            control=control,
        )
    finally:
        control.close()

    # Quality scoring is a local pass over what was just written — no API
    # calls, nothing spent, and re-runnable later via score_run.py.
    # Scoring is a local pass over what was just written and hands its result
    # straight to the report — the run stores no table in between, so nothing
    # on disk can disagree with the metric definitions.
    scored = None
    if not args.no_score:
        # The run scored each reply as it arrived, so this normally just reads
        # the table back. It still falls through to a full scoring pass for a
        # run that did not (an older run directory, or --no-score followed by
        # score_run.py). Changing the metrics and re-applying them is
        # score_run.py's job, and that one always recomputes.
        scored = quality.scored_frame(run_dir)

    if not args.no_report:
        out = results_report.build(run_dir, scored=scored)
        print(f"  [report] wrote {out}")

    # The same scored frame as one CSV per table, for anything that reads
    # tables rather than a page. Written from the scoring pass, never read back
    # (csv_export.py), so it cannot fall out of step with the metrics.
    if not args.no_csv:
        out = csv_export.build(run_dir, scored=scored)
        print(f"  [csv] wrote {out}")

    print(f"\n  Done. Run directory: {run_dir}\n")


if __name__ == "__main__":
    main()
