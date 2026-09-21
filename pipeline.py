"""
pipeline.py — the model x strategy x item loop and all persistence (multi-vendor)
=====================================================================================
For every (model, strategy, item) triple — `repetitions` times over, numbered
in the `repetition` column, so a setting can be measured as a distribution
instead of a single draw:
  1. Build the prompt for that strategy.
  2. Call that model's vendor provider, capturing tokens/cost and the round
     trip of the request (`api_latency_s`, plus the wider `latency_s` and
     `call_wall_s` spans — see providers.CallTiming).
  3. Extract the DOT from the reply and validate it parses.
  4. Persist the raw reply and the extracted .gv, then append a flat row to
     results.csv / results.jsonl.

Each model carries its own vendor (config.ModelSpec.vendor); the caller
supplies one Provider instance per vendor actually in use (`providers` dict),
so a single run can mix Anthropic, OpenAI, Mistral, and Google models freely.

NOTE — quality scoring was removed and is being restructured. This loop
measures cost/tokens/latency and records whether the output is extractable and
syntactically parseable DOT; it makes no judgement about *model quality*. Every
row carries `generated_gv` and `ground_truth_path`, so a scorer can run over a
finished run directory afterwards and join its results back onto results.csv.
"""
from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config as cfg
import postprocess
import prompts
from dataset import ProcessItem, description_path_for
import providers as providers_mod
import runcontrol
from quality import score as quality_score
from providers import Provider

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(s: str) -> str:
    return _SAFE.sub("_", s)


def _sha256(text: str) -> str:
    """Content hash of a prompt or an input, so a row names the exact text
    behind it and not just the file that happened to hold it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _new_run_dir(tag: Optional[str]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{_safe(tag)}" if tag else ts
    run_dir = cfg.RUNS_DIR / name
    # prompts/ holds one record per generation: the prompt that went out, the
    # input it was built from and the parameters and rates it ran under (O6).
    for sub in ("raw", "generated", "prompts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_provider(providers: Dict[str, Provider], vendor: str) -> Provider:
    """The provider serving `vendor` in this run.

    A run is not uniformly live or mocked: the caller passes one entry per
    vendor it wired up, and anything missing falls back to the mock provider
    (offline demo runs pass only a "mock" entry, so everything falls back)."""
    provider = providers.get(vendor) or providers.get("mock")
    if provider is None:
        raise KeyError(
            f"No provider supplied for vendor {vendor!r} and no 'mock' fallback "
            f"in the providers dict (got: {sorted(providers)})."
        )
    return provider


ABANDONED_ERROR = ("abandoned: stop requested while this call was in flight — the "
                   "vendor may still have produced and billed it, and the reply was "
                   "not waited for")

# How often the waiting loop below looks up from the call to see whether a stop
# has been asked for. Short enough that `end` feels immediate, long enough that
# a run of hundreds of calls does not spend its time polling.
_ABANDON_POLL_SECONDS = 0.1


def _generate(provider: Provider, model, system, messages, settings,
              control=None):
    """One generation, on a thread the loop can stop waiting for.

    Returns `(result, error, abandoned)`. A stop asked for while the request is
    on the wire **drops it**: the loop gets control back at once instead of
    sitting out a call that can take minutes on a reasoning model, and the row
    records what happened rather than a reply that never arrived.

    What that costs, stated plainly because it is a real cost: the request is
    already sent, so the vendor may well complete and bill it. The run keeps no
    reply, no tokens and no cost for it — the row carries `ABANDONED_ERROR`, and
    the run's totals are therefore a lower bound on what the account was charged.
    The worker thread is a daemon: it is left to finish or to be killed with the
    process, and nothing it returns afterwards is read.

    Without a `control` there is nobody to ask for a stop, so the call is made
    on this thread exactly as before.
    """
    if control is None:
        try:
            return provider.generate(model, system, messages, settings), "", False
        except Exception as exc:                       # noqa: BLE001 — recorded, not raised
            return None, str(exc), False

    done: Dict[str, object] = {}

    def work() -> None:
        try:
            done["gen"] = provider.generate(model, system, messages, settings)
        except Exception as exc:                       # noqa: BLE001 — recorded, not raised
            done["error"] = str(exc)

    worker = threading.Thread(target=work, name="generate", daemon=True)
    worker.start()
    while True:
        worker.join(timeout=_ABANDON_POLL_SECONDS)
        if not worker.is_alive():
            return done.get("gen"), str(done.get("error", "")), False
        if control.stopped:
            return None, ABANDONED_ERROR, True


@dataclass
class _CallResult:
    """One finished generation and everything the writer needs to record it.

    Deliberately inert. A worker builds one of these and touches no shared
    state, so the only thread that ever appends to `rows`, writes the JSONL or
    updates `substitutions` is the one draining the results.
    """
    row: Dict[str, Any]
    tag: str
    item_id: str
    swap: Optional[Tuple[str, str]] = None   # (exemplar dropped, stand-in used)
    abandoned: bool = False
    fatal_error: str = ""      # non-empty: nothing further can succeed
    untracked: str = ""        # non-empty: cost_usd for this row is a floor


# Put on the results queue by a worker that has no calls left. Counting these
# is how the consumer knows every model is finished without joining threads it
# may still need to abandon.
_WORKER_DONE = object()


class _WorkerCrashed(Exception):
    """A worker hit something `run_one` does not handle.

    Provider failures are not this: those are caught per call and recorded in
    the row. Reaching here means a defect - and the reason it gets its own
    exception rather than a log line is that the failure mode it replaces was
    silent. A worker whose thread died still ran its `finally`, posted
    _WORKER_DONE, and left the run looking like that model had simply finished:
    a partial run reporting `stopped_early: false`, with 3 rows of a planned 39
    and nothing anywhere saying why.
    """


def _iter_sequential(plan, run_one, control, stop_flag):
    """One call at a time, in plan order.

    The regime every run used before parallel execution existed, and the only
    one whose `api_latency_s` carries no load this run created itself. Latency
    reported in the thesis should come from here.
    """
    for entry in plan:
        if stop_flag.is_set():
            return
        # Checked before the call as well as during it: this is where a pause
        # takes effect, and where a stop that arrived between two calls ends
        # the run without sending another one.
        if control is not None and not control.wait_if_paused():
            stop_flag.set()
            return
        yield run_one(entry)


def _iter_per_model(sub_plans, run_one, control, stop_flag,
                    heartbeat=None, heartbeat_every=10.0):
    """One worker per model, each stepping through that model's own calls in
    order — so exactly one request per model is ever in flight.

    That restriction is the whole point. A model is never measured against
    copies of itself, and every model sees the same set of neighbours, so the
    load falls on all of them alike and `api_latency_s` stays comparable
    *between* models. What it is not is an unloaded measurement: the models do
    run beside each other, which is why every row records how many were in
    flight (`concurrent_models`) and why a latency figure from a parallel run
    must never be pooled with one from a sequential run.

    Results arrive in COMPLETION order, not plan order. They are yielded to the
    caller, which performs every write, so the run directory and the JSONL are
    still written by exactly one thread.

    While the calls are on the wire nothing completes, and on the slow models
    that silence can last minutes. `heartbeat(n_in_flight)` is therefore called
    every `heartbeat_every` seconds that pass without a result, so the reader
    keeps seeing how many calls are finished and how many are still out. It runs
    on this thread, in the gap where the consumer would otherwise block, which is
    why it needs no timer thread of its own and cannot interleave with a write.

    A stop reaches the workers through `stop_flag` and takes effect before each
    worker's next call; a request already on the wire is dropped by `_generate`
    itself. With one worker per model that means at most one abandoned request
    per model rather than one per queued call — the reason this arrangement is
    cheaper to interrupt than a shared pool of the same width.
    """
    out: "queue.Queue" = queue.Queue()
    # How many requests are on the wire right now. Only ever touched under the
    # lock, and only so the heartbeat can report it.
    in_flight = [0]
    flight_lock = threading.Lock()

    def work(sub_plan):
        try:
            for entry in sub_plan:
                if stop_flag.is_set():
                    break
                if control is not None and not control.wait_if_paused():
                    stop_flag.set()
                    break
                with flight_lock:
                    in_flight[0] += 1
                try:
                    result = run_one(entry)
                finally:
                    with flight_lock:
                        in_flight[0] -= 1
                out.put(result)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the consumer
            # Stop the other workers before surfacing it: whatever broke here
            # will almost certainly break them too, one paid call at a time.
            stop_flag.set()
            out.put(_WorkerCrashed(exc))
        finally:
            out.put(_WORKER_DONE)

    workers = [
        threading.Thread(target=work, args=(sub_plan,), daemon=True,
                         name=f"gen-{key}")
        for key, sub_plan in sub_plans.items()
    ]
    for w in workers:
        w.start()

    remaining = len(workers)
    while remaining:
        try:
            result = out.get(timeout=heartbeat_every)
        except queue.Empty:
            if heartbeat is not None:
                with flight_lock:
                    heartbeat(in_flight[0])
            continue
        if result is _WORKER_DONE:
            remaining -= 1
            continue
        if isinstance(result, _WorkerCrashed):
            raise result.args[0]
        yield result


def run_experiment(
    *,
    providers: Dict[str, Provider],
    mode: str,  # "mock" | "live" | "mixed" — see also manifest["vendor_modes"]
    models: List[cfg.ModelSpec],
    strategies: List[str],
    items: List[ProcessItem],
    exemplar_files: List[Path],
    settings: cfg.RunSettings,
    fallback_exemplar_files: Optional[List[Path]] = None,
    # A complete alternate exemplar set, and the items that get it instead of
    # the per-slot swap (cfg.FEW_SHOT_ALT_IDS / cfg.FEW_SHOT_ALT_FOR_IDS).
    alt_exemplar_files: Optional[List[Path]] = None,
    alt_for_ids: Optional[List[str]] = None,
    dataset_dir,
    tag: Optional[str],
    input_source: str,
    repetitions: int = cfg.DEFAULT_REPETITIONS,
    # One worker per model instead of one call at a time. Never more than one
    # request per model, so no model competes with itself; see _iter_per_model
    # for what that does and does not buy. Off by default: a sequential run is
    # the only one whose latency needs no caveat.
    parallel_models: bool = False,
    # Score each reply on the worker that fetched it, instead of in a pass over
    # the finished run. The metrics are local and cost nothing; computing them
    # while the other models are still waiting on HTTP uses time the run was
    # spending idle anyway, and every row is complete the moment it is written.
    # `score_run.py` still re-scores a finished run when the metrics change.
    score_inline: bool = True,
    control=None,   # runcontrol.RunControl, or None to run uninterruptibly
) -> Path:
    if not cfg.MIN_REPETITIONS <= repetitions <= cfg.MAX_REPETITIONS:
        raise ValueError(
            f"repetitions must be between {cfg.MIN_REPETITIONS} and "
            f"{cfg.MAX_REPETITIONS} (got {repetitions})."
        )

    run_dir = _new_run_dir(tag)

    # How many models have a request in flight at the same time. One slot per
    # model and never more than one, so no model is ever measured against copies
    # of itself and every model sees the same set of neighbours — that symmetry
    # is what keeps api_latency_s comparable between models. A sequential run
    # records 1.
    #
    # It goes onto every row on purpose. A latency read back months later cannot
    # otherwise tell which regime produced it, and pooling the two would compare
    # a lone call against one that ran beside a dozen others.
    concurrent_models = len(models) if parallel_models and len(models) > 1 else 1

    needs_exemplars = any(prompts.needs_exemplars(s) for s in strategies)
    exemplars = prompts.load_exemplars(exemplar_files) if needs_exemplars else []
    exemplars = exemplars[: settings.n_few_shot]
    # Stand-ins for the case below, where the item being generated for is one
    # of the exemplars. Loaded once here rather than per item; unused (and so
    # unread) when no few-shot strategy runs.
    reserve = (prompts.load_exemplars(fallback_exemplar_files or [])
               if needs_exemplars else [])
    # The complete alternate set for `alt_ids`. Takes precedence over the
    # per-slot swap, so those items all run on the same exemplars instead of on
    # a differently-swapped default set each. Empty either half and the swap
    # alone applies, exactly as before.
    alt_exemplars = (prompts.load_exemplars(alt_exemplar_files or [])[: settings.n_few_shot]
                     if needs_exemplars else [])
    alt_ids = set(alt_for_ids or []) if alt_exemplars else set()

    manifest = {
        "created_at": datetime.now().isoformat(),
        "mode": mode,
        "vendors_used": sorted({m.vendor for m in models}),
        # Per-vendor live/mock resolution. `mode` alone is not enough: a single
        # run can mix a live vendor with mocked ones (vendors without a wired-up
        # provider yet), and the reader has to be able to tell which numbers in
        # results.csv are measurements and which are placeholders.
        "vendor_modes": {
            v: ("mock" if _resolve_provider(providers, v).name == "mock" else "live")
            for v in sorted({m.vendor for m in models})
        },
        "input_source": input_source,
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "n_items": len(items),
        # What the cost column was computed with. A price without the date of
        # the price sheet behind it cannot be reproduced — and the rate actually
        # billed can differ from the list rate while a vendor runs a promotion,
        # so the resolved rate and its basis are recorded per model, not just
        # the catalogue entry.
        "pricing": {
            # One price sheet per vendor — they are published on different days
            # and move independently, so a single date would be a fiction.
            "as_of": cfg.pricing_summary([m.vendor for m in models]),
            "sheets": {v: cfg.pricing_sheet(v) for v in sorted({m.vendor for m in models})},
            "source": cfg.PRICING_SOURCE,
            "priced_on": datetime.now().date().isoformat(),
            "multipliers_on_input_rate": {
                "cache_write": cfg.CACHE_WRITE_MULTIPLIER,
                "cache_write_1h": cfg.CACHE_WRITE_1H_MULTIPLIER,
                "cache_read": cfg.CACHE_READ_MULTIPLIER,
                "batch": cfg.BATCH_MULTIPLIER,
            },
            "web_search_usd_per_1k_requests": cfg.WEB_SEARCH_USD_PER_1K_REQUESTS,
            "rates_applied": {
                m.key: dict(zip(("input_per_mtok", "output_per_mtok", "basis"),
                                cfg.prices_for(m)))
                for m in models
            },
        },
        # How often every (item × model × strategy) cell was generated. Each
        # repetition is an independent draw from the same setting, numbered in
        # the `repetition` column of results.csv.
        "repetitions": repetitions,
        # The execution regime, because it decides how the latency columns may
        # be read. concurrent_models == 1 is the unloaded case.
        "parallel_models": concurrent_models > 1,
        "concurrent_models": concurrent_models,
        # Whether results.jsonl already carries the quality columns. False means
        # the metrics still have to be derived from the .gv files.
        "scored_inline": bool(score_inline),
        "models": [asdict(m) for m in models],
        "strategies": strategies,
        "exemplar_files": [str(p) for p in exemplar_files],
        # Every item is evaluated, exemplars included. When the item being
        # generated for is itself an exemplar, that one slot is filled from
        # this reserve instead, so no item is ever shown its own answer — the
        # substitutions that actually happened are listed per item in
        # `few_shot_substitutions` (written after the run).
        "fallback_exemplar_files": [str(p) for p in (fallback_exemplar_files or [])],
        # The items that bypass that swap and run on a complete alternate
        # exemplar set instead, and the files it comes from. Empty list =
        # every collision was handled by the per-slot swap above.
        "alt_exemplar_files": [str(p) for p in (alt_exemplar_files or [])],
        "few_shot_alt_for_ids": sorted(alt_ids),
        "settings": asdict(settings),
        # Per-strategy provenance, so a run's own metadata says where each
        # prompt's wording came from (see the strategy table in README.md).
        "prompt_source": {
            "zero_shot": "Li et al. (2025), Sec. 4 — verbatim",
            "zero_shot_cot": "Li et al. (2025), Sec. 4 — verbatim",
            "few_shot": "Li et al. (2025), Sec. 4 — verbatim",
            "few_shot_cot": "Li et al. (2025), Sec. 4 + Appendix A.1 — verbatim",
            "zero_shot_system": "Li et al. (2025), Sec. 4 ('Fine-Tuned LLMs' box) — verbatim",
            "zero_shot_graph_type_tn_rules": "Klievtsova et al., 'Conversational Process Modeling' — verbatim apart from one formatting change (2026-08-27): description and briefing are separated by a line break instead of '. '",
            "zero_shot__short_bpmn_description": "Klievtsova et al., 'Conversational Process Modeling' — verbatim apart from one formatting change (2026-08-27): description and briefing are separated by a line break instead of '. '",
            "tree_of_thought": "author-supplied template — no published source on file",
            "role_prompt_1": "author-supplied template — no published source on file",
            "role_prompt_2": "author-supplied template — no published source on file",
            "_papers": [
                "Li et al. (2025), 'LLM-based Business Process Models Generation "
                "from Textual Descriptions' (IJCNLP-AACL, 523-533)",
                "Klievtsova et al., 'Conversational Process Modeling: Can Generative "
                "AI Empower Domain Experts in Creating and Redesigning Process Models?'",
            ],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows = []
    # item_id -> the exemplar id shown in its place ("" = none was available and
    # the slot was dropped). Reported in the manifest so a reader can see which
    # items ran on a different exemplar set than the rest.
    substitutions: Dict[str, str] = {}
    jsonl_path = run_dir / "results.jsonl"

    # Execution order: the repetition is the OUTERMOST loop. One full sweep of
    # (model × strategy × item) finishes before the next repetition starts,
    # instead of repeating each cell back to back — consecutive calls for one
    # cell would share the same network and API conditions and understate the
    # very spread the repetitions exist to measure. It also means a run stopped
    # halfway holds repetition 1 for everything, rather than some cells
    # repeated in full and others never attempted.
    plan = [
        (rep, model, strategy, item)
        for rep in range(1, repetitions + 1)
        for model in models
        for strategy in strategies
        for item in items
    ]
    total = len(plan)
    done = 0
    stopped_early = False

    # Artefact names carry the repetition only when there is more than one, so
    # single-repetition runs keep exactly the file names every earlier run used.
    def _rep_suffix(rep: int) -> str:
        return f"__r{rep:02d}" if repetitions > 1 else ""

    def _write_provenance(tag_str, rep, model, strategy, item, provider,
                          system, messages, exemplars_shown, swap,
                          gen, gen_error) -> Dict[str, Any]:
        """Write this generation's conditions beside its reply, and hand back
        the columns that point at them (O6, traceable experimentation).

        `results.jsonl` carries the measurements and `manifest.json` the
        run-wide configuration. What neither of them held is the prompt that
        actually went out and the input it was built from — so a row could be
        re-run but not re-derived, and a template edited between two runs left
        no trace in either of them. One JSON file per generation closes that:
        input, prompt (with its template version), model, parameters,
        timestamps, the rates it was priced at and the artefacts it produced.

        A call that never returned gets a record too — the conditions are what
        they were, and the failure is part of what they produced.
        """
        name = _safe(tag_str)
        prompt_blob = json.dumps({"system": system, "messages": messages},
                                 ensure_ascii=False, sort_keys=True)
        prompt_chars = len(system or "") + sum(len(m.get("content") or "") for m in messages)

        # The description file it came from, when the input is a dataset pair.
        # CSV input has no such file, and then the input_source is the honest
        # answer — the text itself is in the record either way.
        input_path = ""
        if item.ground_truth_path:
            try:
                input_path = str(description_path_for(item.ground_truth_path))
            except Exception:
                input_path = ""

        # The rates this call was *actually* priced at, selected by the basis
        # the cost carries rather than by today's catalogue: an intro window
        # that closes, or an off-peak hour, prices two identical calls
        # differently and the record has to say which one applied.
        basis = (gen.cost_basis if gen is not None else "") or ""
        rates = cfg.rates_for(model, basis)
        if rates is None:
            price_in = price_out = None
        else:
            price_in, price_out = rates["input_tokens"], rates["output_tokens"]
        sheet = cfg.pricing_sheet(model.vendor) or {}

        record = {
            "run_id": run_dir.name,
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "call": {
                "item_id": item.item_id,
                "vendor": model.vendor,
                "model_key": model.key,
                "model_id": model.model_id,
                "model_tier": model.tier,
                "strategy": strategy,
                "repetition": rep,
                "provider": provider.name,
                "is_mock": provider.name == "mock",
                "request_sent_at": gen.request_sent_at if gen is not None else "",
                "response_received_at": gen.response_received_at if gen is not None else "",
                "request_id": gen.request_id if gen is not None else None,
                "response_id": gen.response_id if gen is not None else None,
                "generation_error": gen_error or "",
            },
            "input": {
                "item_id": item.item_id,
                "source": input_path or input_source,
                "is_file": bool(input_path),
                "sha256": _sha256(item.description),
                "chars": len(item.description),
                "text": item.description,
            },
            "prompt": {
                "strategy": strategy,
                "source": manifest["prompt_source"].get(strategy, ""),
                "template_version": prompts.template_version(strategy),
                "sha256": _sha256(prompt_blob),
                "chars": prompt_chars,
                "n_turns": len(messages),
                "system": system,
                "messages": messages,
                "exemplar_item_ids": [ex.item_id for ex in (exemplars_shown or [])],
                # (dropped_id, replacement_id) when this item's own model sat in
                # its own prompt and was swapped out; None when nothing collided.
                "exemplar_substitution": list(swap) if swap else None,
            },
            "parameters": {
                "temperature": settings.temperature if model.supports_temperature else None,
                "temperature_supported": model.supports_temperature,
                "max_output_tokens": settings.max_output_tokens,
                "thinking": model.thinking,
                "effort": model.effort,
                "reasoning_off": model.reasoning_off,
                "n_few_shot": settings.n_few_shot,
                "concurrent_models": concurrent_models,
            },
            "pricing": {
                "input_per_mtok": price_in,
                "output_per_mtok": price_out,
                "basis": basis,
                "price_verified": model.price_verified,
                "sheet_as_of": sheet.get("as_of"),
                "sheet_source": sheet.get("source"),
                "priced_on": manifest["pricing"]["priced_on"],
                "rates_per_mtok": rates,
            },
            "artefacts": {
                "raw_output": f"raw/{name}.txt",
                "vendor_usage": f"raw/{name}.usage.json",
                "generated_gv": f"generated/{name}.gv",
            },
        }
        (run_dir / "prompts" / f"{name}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "prompt_file": f"prompts/{name}.json",
            "prompt_sha256": record["prompt"]["sha256"],
            "prompt_template_version": record["prompt"]["template_version"],
            "prompt_source": record["prompt"]["source"],
            "input_file": record["input"]["source"],
            "input_sha256": record["input"]["sha256"],
            "raw_output_file": f"raw/{name}.txt",
            "price_in_per_mtok": price_in,
            "price_out_per_mtok": price_out,
            "pricing_as_of": sheet.get("as_of", ""),
        }

    def run_one(entry, announce=None) -> _CallResult:
        """One generation, start to finish, touching nothing shared.

        Everything it needs is either read-only (the exemplars, the settings) or
        unique to this call (its own two artefact paths), which is what makes it
        safe on a worker thread. What it must NOT do is append to `rows`, write
        the JSONL or update `substitutions` - those belong to the single thread
        draining the results, so the run directory keeps exactly one writer
        whether the run is sequential or parallel.
        """
        rep, model, strategy, item = entry
        provider = _resolve_provider(providers, model.vendor)
        ex_for_strategy = None
        swap_seen = None
        if prompts.needs_exemplars(strategy):
            ex_for_strategy, swap_seen = prompts.exemplars_for_item(
                exemplars, item.item_id, reserve,
                alternates=alt_exemplars if item.item_id in alt_ids else None)

        tag_str = f"{item.item_id}__{model.key}__{strategy}{_rep_suffix(rep)}"
        if announce is not None:
            announce(tag_str)

        system, messages = prompts.build_messages(
            strategy, item.description, exemplars=ex_for_strategy
        )

        t0 = time.monotonic()
        gen, gen_error, abandoned = _generate(provider, model, system, messages,
                                              settings, control)
        wall_s = time.monotonic() - t0

        raw_path = run_dir / "raw" / f"{_safe(tag_str)}.txt"
        raw_path.write_text(gen.text if gen else f"[ERROR] {gen_error}", encoding="utf-8")
        # The vendor's usage object verbatim, beside the reply it belongs
        # to. The results table carries the classes this project prices and
        # compares; this carries everything the API said, modality
        # breakdowns included, for an audit that outlives the schema — and
        # is what the report's per-model token cards are built from.
        if gen is not None and gen.raw_usage:
            (run_dir / "raw" / f"{_safe(tag_str)}.usage.json").write_text(
                gen.raw_usage, encoding="utf-8")

        # The conditions this reply was produced under, on disk beside it.
        prov = _write_provenance(tag_str, rep, model, strategy, item, provider,
                                 system, messages, ex_for_strategy, swap_seen,
                                 gen, gen_error)

        generated_path = None
        extract = postprocess.ExtractResult(dot_text=None, parse_ok=False, error=gen_error or "generation failed")
        if gen is not None:
            extract = postprocess.extract_and_validate(gen.text)
            if extract.dot_text:
                generated_path = run_dir / "generated" / f"{_safe(tag_str)}.gv"
                generated_path.write_text(extract.dot_text, encoding="utf-8")

        row = {
            "item_id": item.item_id,
            "vendor": model.vendor,
            "model_key": model.key,
            "model_id": model.model_id,
            "model_tier": model.tier,
            "strategy": strategy,
            # Which of the run's `repetitions` draws of this setting this
            # row is (1-based). Same prompt, same model, same parameters —
            # only the model's own non-determinism differs. Part of the key
            # that identifies a row; see quality/score.py KEY_COLS.
            "repetition": rep,
            # How many models had a request in flight while this row was
            # produced. 1 = the call ran alone. Anything higher means the
            # latency columns carry load from this run itself and must not be
            # pooled with rows recorded at a different width.
            "concurrent_models": concurrent_models,
            # Which provider actually produced this row. is_mock=True
            # means tokens/cost/latency/quality are placeholders from
            # the offline stub, NOT measurements — filter these out
            # before reporting any result.
            "provider": provider.name,
            "is_mock": provider.name == "mock",
            # ── billable quantities (see config.TokenUsage) ──
            # input_tokens is the UNCACHED prompt only — the vendors report
            # it that way. billable_input_tokens is the whole prompt the
            # invoice charges for: input + cache read + cache writes, each
            # at its own multiplier. total_tokens is that plus output.
            # The vendor's own prompt count, verbatim — provenance for the
            # normalised columns below, never priced. Its meaning differs by
            # vendor (see providers.GenerationResult), so do not compare it.
            "reported_input_tokens": gen.reported_input_tokens if gen else None,
            "reported_total_tokens": gen.reported_total_tokens if gen else None,
            "reported_cache_creation_tokens": (
                gen.reported_cache_creation_tokens if gen else None),
            "reported_output_tokens": gen.reported_output_tokens if gen else None,
            # Gemini only: the vendor's tool-use prompt count. It is a
            # prompt breakdown in the REST contract; the provider also
            # handles SDK responses that put it outside prompt_token_count.
            # Never price this informational column directly.
            "tool_use_prompt_tokens": gen.tool_use_prompt_tokens if gen else None,
            "input_tokens": gen.input_tokens if gen else None,
            "output_tokens": gen.output_tokens if gen else None,
            "cached_tokens": gen.cached_tokens if gen else None,
            "cache_write_tokens": gen.cache_write_tokens if gen else None,
            "cache_write_1h_tokens": gen.cache_write_1h_tokens if gen else None,
            "billable_input_tokens": gen.billable_input_tokens if gen else None,
            "thinking_tokens": gen.thinking_tokens if gen else None,
            "web_search_requests": gen.web_search_requests if gen else None,
            # Mistral only: audio handed to the model, billed by duration
            # rather than by token. 0 on every text call; non-zero always comes
            # with an `untracked_usage` note, because no audio rate is priced
            # here (see providers.MistralProvider).
            "audio_input_seconds": gen.audio_input_seconds if gen else None,
            "service_tier": gen.service_tier if gen else "",
            "total_tokens": gen.total_tokens if gen else None,
            "cost_usd": gen.cost_usd if gen else None,
            "cost_basis": gen.cost_basis if gen else "",
            # Non-empty = the vendor billed a quantity this cost model does
            # not price, so cost_usd is a floor, not the price. Never ignore.
            "untracked_usage": gen.untracked_usage if gen else "",
            "price_verified": model.price_verified,
            # ── timing, three nested spans (see providers.CallTiming) ──
            # api_latency_s answers "how long did this prompt take":
            # request handed to the vendor SDK → complete reply back,
            # for the attempt that actually answered. latency_s is the
            # same span plus failed attempts and their backoff (equal
            # unless api_attempts > 1); call_wall_s adds this loop's
            # own provider-wrapper overhead. A generation that raised
            # never received a reply, so its api_* fields stay empty
            # and only the wall clock until the failure is kept.
            "latency_s": gen.latency_s if gen else wall_s,
            "api_latency_s": gen.api_latency_s if gen else None,
            "request_sent_at": gen.request_sent_at if gen else "",
            "response_received_at": gen.response_received_at if gen else "",
            "api_attempts": gen.api_attempts if gen else None,
            "retry_wait_s": gen.retry_wait_s if gen else None,
            "call_wall_s": wall_s,
            # Vendor-side handles, so a row here can be traced to a request
            # in the vendor's own logs. Empty for a call that never
            # returned, and for the mock provider.
            "request_id": gen.request_id if gen else "",
            "response_id": gen.response_id if gen else "",
            "stop_reason": gen.stop_reason if gen else "error",
            "token_reporting": gen.token_reporting if gen else "estimated",
            "tokens_estimated": gen.tokens_estimated if gen else True,
            "reasoning_note": gen.reasoning_note if gen else "",
            "generation_error": gen_error,
            "extract_parse_ok": extract.parse_ok,
            "extract_error": extract.error,
            # Handles for the (to-be-restructured) quality scoring:
            # what was generated, and what it should be compared to.
            # Empty generated_gv = nothing extractable was produced.
            "generated_gv": (str(generated_path.relative_to(run_dir))
                             if generated_path else ""),
            "ground_truth_path": (str(item.ground_truth_path)
                                  if item.ground_truth_path else ""),
            # ── O6: what this row can be traced back to ──────────────────
            # The prompt that went out, the input behind it and the reply
            # verbatim are files in this run directory; these are the handles
            # onto them, plus the hashes that say which *text* those files
            # hold and the version of the template it was built from. A row
            # lifted out of the CSV therefore still identifies its own prompt,
            # and two runs can be told apart by their template version rather
            # than by the date on the folder.
            "prompt_file": prov["prompt_file"],
            "prompt_sha256": prov["prompt_sha256"],
            "prompt_template_version": prov["prompt_template_version"],
            "prompt_source": prov["prompt_source"],
            "input_file": prov["input_file"],
            "input_sha256": prov["input_sha256"],
            "raw_output_file": prov["raw_output_file"],
            # The rates this row was actually priced at, and the date of the
            # sheet they were read from. `cost_basis` says which rate applied,
            # not what it was — and a published price moves.
            "price_in_per_mtok": prov["price_in_per_mtok"],
            "price_out_per_mtok": prov["price_out_per_mtok"],
            "pricing_as_of": prov["pricing_as_of"],
        }
        # Scored here, on this worker, the moment the reply is on disk - not in
        # a pass after the last call. The metrics are pure local computation
        # over the .gv that was just written, so the only thing they compete for
        # is the interpreter, and this thread would otherwise be blocked on the
        # next HTTP round trip anyway. It also means a run stopped halfway holds
        # fully scored rows rather than rows still awaiting a verdict.
        #
        # Every parser this reaches has to tolerate being called from several
        # workers at once: quality/graph.py takes postprocess.DOT_PARSE_LOCK for
        # pydot, and quality/textsim.py guards its one-time WordNet load.
        if score_inline:
            row.update(quality_score.score_generation(
                generated_path, item.ground_truth_path,
                generation_error=gen_error))

        # An exhausted balance, dead billing or a revoked key cannot be fixed by
        # trying the next item: every remaining generation would fail the same
        # way, slowly. Handed back rather than acted on here, because ending the
        # run is the consumer's decision - what has been produced is still
        # scored and aggregated, and the manifest records it as a partial run
        # exactly like a keyboard stop.
        fatal = ""
        if gen is None and providers_mod.is_fatal_account_error(gen_error):
            fatal = f"{model.vendor}: {gen_error}"

        return _CallResult(
            row=row, tag=tag_str, item_id=item.item_id, swap=swap_seen,
            abandoned=abandoned, fatal_error=fatal,
            untracked=(gen.untracked_usage if gen is not None else ""),
        )

    # Said once, before the first call: a run that cannot be paused is one a
    # reader kills with Ctrl-C, losing the scoring and the report. Ctrl-C is
    # folded into the same stop, so it costs nothing either.
    if control is not None:
        print(control.hint())
    if concurrent_models > 1:
        print(f"  Parallel: {concurrent_models} models, one request per model at "
              f"a time. Rows carry concurrent_models={concurrent_models} - their "
              f"latency was measured at that width, not alone.")

    # Sequential runs announce a call before sending it, so a reader watching a
    # slow model knows what is on the wire. Parallel runs cannot - a dozen calls
    # leave at once - so they report on arrival instead, further down.
    announced = [0]

    def _announce(tag_str: str) -> None:
        announced[0] += 1
        print(f"  [{announced[0]}/{total}] {tag_str}")

    stop_flag = threading.Event()
    run_started = time.monotonic()

    def _heartbeat(n_in_flight: int) -> None:
        """Printed while calls are out and nothing has come back yet. Without it
        a parallel run looks frozen: the slow models take minutes, and every
        completion line waits on them."""
        elapsed = int(time.monotonic() - run_started)
        left = total - done
        print(f"      ... {done}/{total} done | {n_in_flight} in flight | "
              f"{left} left | {elapsed // 60}m{elapsed % 60:02d}s elapsed")

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_f, \
            runcontrol.stop_on_sigint(control):
        if concurrent_models > 1:
            # One sub-plan per model, each keeping the plan's own order, so a
            # model still works through repetition by repetition exactly as it
            # would sequentially.
            sub_plans: Dict[str, List] = {}
            for entry in plan:
                sub_plans.setdefault(entry[1].key, []).append(entry)
            results = _iter_per_model(sub_plans, run_one, control, stop_flag,
                                      heartbeat=_heartbeat)
        else:
            results = _iter_sequential(
                plan, lambda e: run_one(e, announce=_announce), control, stop_flag)

        for res in results:
            done += 1
            if concurrent_models > 1:
                print(f"  [{done}/{total}] {res.tag}")

            if res.abandoned:
                stopped_early = True

            if res.swap is not None and res.item_id not in substitutions:
                dropped, replacement = res.swap
                substitutions[res.item_id] = replacement
                # A completed swap is routine and stays out of the log - the
                # manifest records it. A swap with no stand-in left is not: that
                # item runs on a shorter prompt than the rest.
                if not replacement:
                    print(f"  [few-shot] item {res.item_id} is exemplar "
                          f"{dropped}, no stand-in left - running with one "
                          f"exemplar fewer")

            if res.fatal_error:
                print(f"\n  ✗  {res.fatal_error}")
                print("     Nothing further can succeed on this account — stopping the run. "
                      "Everything generated so far is kept, scored and aggregated.")
                stopped_early = True

            # A billed quantity nobody prices is the one failure mode that makes
            # the cost column quietly wrong, so it is surfaced the moment it
            # happens rather than being left for a reader of the CSV to notice.
            if res.untracked:
                print(f"      ⚠  untracked billable usage: {res.untracked} "
                      f"— cost_usd for this row is a lower bound")

            # Written before any stop takes effect, so the failure that ended
            # the run is itself part of the record.
            rows.append(res.row)
            jsonl_f.write(json.dumps(res.row, default=str) + "\n")
            jsonl_f.flush()

            if stopped_early:
                stop_flag.set()
                if concurrent_models == 1:
                    break
                # Parallel: keep draining. Each worker stops before its next
                # call, and what it already produced is written rather than
                # thrown away.

    # A stop asked for at the keyboard reaches the loop through stop_flag, which
    # both iterators set before returning.
    if stop_flag.is_set():
        stopped_early = True

    # No table is written here. `results.jsonl` — appended and flushed per call
    # above, so it survives a crash mid-run — is the run's only data file; the
    # scoring pass reads it and hands the scored frame to the report builder.

    # A stopped run is a partial run. Say so in the manifest, so a later reader
    # cannot mistake `n_generations` for the full cross product it planned.
    manifest["stopped_early"] = stopped_early
    manifest["n_generations_planned"] = total
    manifest["n_generations_completed"] = done
    # Which items ran on a swapped exemplar set (empty for the usual run where
    # no evaluated item is an exemplar). Written after the loop because it is
    # only known once every item has been through it.
    manifest["few_shot_substitutions"] = substitutions
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if stopped_early:
        print(f"\n  Stopped after {done} of {total} generation(s). "
              f"{total - done} were not run; everything completed is kept.")

    _warn_about_truncated_calls(rows, settings)
    return run_dir


def _warn_about_truncated_calls(rows, settings) -> None:
    """Say so, loudly, when a call was cut off by the output budget.

    A reasoning model spends its thinking inside the output allowance. When the
    allowance runs out first, the vendor bills the whole call and returns
    nothing extractable — the row then reads as a quality failure rather than a
    configuration one, and the difference matters: no prompt change will fix it.
    Silence here is what let one model score 0/4 across several runs.
    """
    truncated = [r for r in rows if str(r.get("stop_reason") or "") == "length"]
    if not truncated:
        return

    empty = [r for r in truncated if not str(r.get("generated_gv") or "").strip()]
    wasted = sum(float(r.get("cost_usd") or 0) for r in empty)
    print(f"\n  ⚠  {len(truncated)} call(s) hit the output cap of "
          f"{settings.max_output_tokens} tokens (stop_reason='length').")
    if empty:
        by_model: Dict[str, int] = {}
        for r in empty:
            key = str(r.get("model_key"))
            by_model[key] = by_model.get(key, 0) + 1
        listing = ", ".join(f"{k} ({n})" for k, n in sorted(by_model.items()))
        print(f"     {len(empty)} of them returned nothing usable — {listing} — "
              f"billed ${wasted:.4f} for no output.")
        print("     These are reasoning models spending the whole budget on "
              "thinking, not bad answers. Raise it: --max-output-tokens "
              f"{max(32768, settings.max_output_tokens * 2)}")
