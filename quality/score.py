"""
quality/score.py — orchestration
====================================
Runs every dimension over one generated model, and over a whole finished run.

Scoring is a **separate pass over a run directory**: it never generates, never
calls an API, and can be repeated as often as the metrics change without
spending anything. It reads `results.jsonl` and the `.gv` files in `generated/`,
and returns the scored table **in memory** — a run stores no intermediate CSV,
so no stored table can fall out of step with the metric definitions. `run.py`
calls it at the end of a run and hands the result to the report builder;
`score_run.py` re-runs it over any run directory and re-renders the report.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import graph as g
from . import pragmatic, semantic, syntactic, validity

# Identifies one generation. Unique per row in results.csv.
# `repetition` is part of the key: with --repetitions N the same
# (item, model, strategy) appears N times, and merging the quality columns back
# on the first three alone would cross-join every repetition with every other
# (N² rows). Runs written before repetitions existed have no such column and
# are keyed on the first three — every lookup below filters KEY_COLS by what
# the file actually contains.
KEY_COLS = ["item_id", "model_key", "strategy", "repetition"]

# The column namespace `score_generation` owns. Re-scoring replaces all of it,
# so a run scored under an older metric set cannot leak a column that no longer
# exists — see the `stale` list in `score_run`.
_SCORED_PREFIXES = ("val_", "syn_", "sem_", "prag_")
_SCORED_COLUMNS = ("quality_note",)


def _dimension(name: str,
               evaluate: Callable[[], Dict[str, Any]],
               empty: Callable[[], Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """One dimension's columns — or its empty column set and a note, when the
    metric code itself raises.

    A metric that raises is a defect in our code, or an input that walks into
    one. It is not a verdict on the model, and it must not become one on the
    run: this runs on a generation worker the moment a reply lands
    (`pipeline.run_one`), where an escaping exception sets the stop flag,
    stops every other worker and abandons everything still queued. Run
    20260831_182408 lost 21700 of 25740 generations that way, to a
    RecursionError in `pragmatic._bef4llm_diameter` over a single deepseek
    model — 4040 paid calls in, with no way to resume.

    The failure is per dimension, so one broken metric costs its own columns
    and nothing else: a model whose pragmatic block blows up keeps its
    syntactic, semantic and validity verdicts, and `quality_note` carries the
    exception so the row can be found again.
    """
    try:
        return evaluate(), ""
    except Exception as exc:  # noqa: BLE001 — the run outlives any one metric
        return empty(), f"{name} metrics failed: {type(exc).__name__}: {exc}"


def score_model(generated_path,
                ground_truth_path=None) -> Dict[str, Any]:
    """Score one generated model. Returns a flat dict of metric columns.

    A file that does not parse yields the `empty()` column set (all checks
    None) plus a note — never a silent zero.
    """
    gen_pg: Optional[g.ProcessGraph] = None
    parse_error = ""
    try:
        gen_pg = g.load(generated_path)
    except Exception as exc:  # noqa: BLE001 — any parser failure is a validity issue
        parse_error = str(exc)

    gt_pg: Optional[g.ProcessGraph] = None
    gt_error = ""
    if ground_truth_path:
        try:
            gt_pg = g.load(ground_truth_path)
        except Exception as exc:  # noqa: BLE001
            gt_error = str(exc)

    out: Dict[str, Any] = {}
    notes: List[str] = []
    if gen_pg is None:
        out.update(syntactic.empty())
        notes.append(f"generated model did not parse: {parse_error}")
    else:
        cols, note = _dimension("syntactic",
                                lambda: syntactic.evaluate(gen_pg),
                                syntactic.empty)
        out.update(cols)
        if note:
            notes.append(note)
        if gt_error:
            notes.append(f"ground truth did not parse: {gt_error}")

    # Claimed here and filled in at the end: a dict keeps the position a key
    # was first inserted at, and this column has sat right behind the syntactic
    # block in every table the project has written.
    out["quality_note"] = ""

    # `validity` is deliberately unimplemented and contributes no columns
    # until it is; `semantic` and `pragmatic` return their full column sets,
    # all None, when there is no model (or, for semantic, no reference) to
    # measure.
    for name, module in (("semantic", semantic), ("pragmatic", pragmatic)):
        if gen_pg is None:
            out.update(module.empty())
            continue
        cols, note = _dimension(name,
                                lambda m=module: m.evaluate(gen_pg, gt_pg),
                                module.empty)
        out.update(cols)
        if note:
            notes.append(note)
    # Validity reads the file itself: its verdict comes from Graphviz over the
    # bytes on disk, deliberately independent of whether `quality/graph.py`
    # managed to parse them. It therefore runs even when gen_pg is None — that
    # is exactly the case it is there to explain.
    cols, note = _dimension("validity",
                            lambda: validity.evaluate(gen_pg,
                                                      parse_error=parse_error,
                                                      source_path=generated_path),
                            validity.empty)
    out.update(cols)
    if note:
        notes.append(note)
    out["quality_note"] = "; ".join(notes)
    return out


def load_generation_log(run_dir) -> pd.DataFrame:
    """The run's generation records, read from `results.jsonl`.

    `results.jsonl` is the run's only tabular data file: it is written and
    flushed one line per call during generation and never rewritten afterwards.
    Everything downstream — scoring, aggregation, the HTML report — reads it.
    """
    run_dir = Path(run_dir)
    path = run_dir / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No results.jsonl in {run_dir}")
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(records)
    if "item_id" in df.columns:
        # PMo ids are zero-padded ("03"); pandas would otherwise make them ints.
        df["item_id"] = df["item_id"].astype(str)
    return df


def score_generation(generated_path, ground_truth_path=None, *,
                     generation_error: str = "") -> Dict[str, Any]:
    """Score one generation as a run records it, including the two cases that
    are not a model at all.

    Split out of `score_run` so the generation loop can score each reply on the
    worker that fetched it, while a later re-scoring pass over the same run
    still reaches exactly the same verdicts. Two callers with two copies of this
    decision would be two chances for them to drift apart.

    `generated_path` of None means nothing extractable came back. That is a
    format failure worth 0.0 - unless the call never returned at all
    (`generation_error`), which is a missing measurement and gets no verdict.
    """
    if not generated_path:
        out: Dict[str, Any] = {}
        out.update(syntactic.empty())
        out.update(semantic.empty())
        out.update(pragmatic.empty())
        # Validity is the one dimension that can still judge this: a reply that
        # arrived without a DOT block in it *is* a format failure. A call that
        # never returned is a missing measurement instead.
        if generation_error:
            out.update(validity.empty())
        else:
            out.update(validity.invalid("no DOT block in the reply"))
        out["quality_note"] = "no generated model (nothing extractable in the reply)"
        return out

    path = Path(generated_path)
    if not path.exists():
        out = {}
        out.update(syntactic.empty())
        out.update(semantic.empty())
        out.update(pragmatic.empty())
        out.update(validity.empty())
        out["quality_note"] = f"generated file missing: {path}"
        return out

    try:
        return score_model(path, ground_truth_path or None)
    except Exception as exc:  # noqa: BLE001 — a scorer must never end a run
        # `score_model` already contains every metric failure per dimension;
        # this is the net under everything else it does, because the caller is
        # a generation worker and the cost of an exception escaping here is the
        # rest of the run (see `_dimension`).
        out = {}
        out.update(syntactic.empty())
        out.update(semantic.empty())
        out.update(pragmatic.empty())
        out.update(validity.empty())
        out["quality_note"] = f"scoring failed: {type(exc).__name__}: {exc}"
        return out


def _scoring_tasks(run_dir: Path
                   ) -> Tuple[List[Dict[str, Any]],
                              List[Tuple[Optional[Path], Optional[str], str]]]:
    """`(key columns, scoring tasks)` for a run, read by streaming the log.

    Deliberately **not** via `load_generation_log`. Scoring needs four fields
    per row — the generated model, its ground truth, the generation error and
    the join key — while that function materialises the whole record: a 200 MB
    file as one string, then a list of one dict per line with every field in it,
    then a DataFrame of all of them. On run 20260831_182408 (25740 rows, 235
    columns) that is several gigabytes, and holding it *while* a process pool is
    alive is what the machine cannot afford: a 12-worker re-score was killed for
    running the system out of memory 37% in, the parent frame and 12 x ~340 MB
    of workers together exceeding what 15 GB had free.

    So the expensive structure is never built here. The parent stays small for
    the hours the pool runs, and `score_run` reads the log afterwards, when the
    workers are gone and it has the memory to itself. The peak becomes the
    larger of the two phases instead of their sum.

    One line at a time, so the 200 MB is never a single string either.
    """
    path = run_dir / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No results.jsonl in {run_dir}")

    def _field(record: Dict[str, Any], col: str) -> str:
        """Streaming sees raw JSON, so a missing field is absent or None — the
        NaN that `load_generation_log` would have introduced cannot occur."""
        value = record.get(col)
        return "" if value is None else str(value).strip()

    keys: List[Dict[str, Any]] = []
    tasks: List[Tuple[Optional[Path], Optional[str], str]] = []
    checked = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not checked:
                if "generated_gv" not in record:
                    raise ValueError(
                        f"{path} has no 'generated_gv' field — it predates the "
                        "current pipeline. Re-run to score it.")
                checked = True
            # `item_id` is str here as it is there: PMo ids are zero-padded
            # ("03") and the merge key has to match on both sides.
            keys.append({k: str(record[k]) if k == "item_id" else record[k]
                         for k in KEY_COLS if k in record})
            rel = _field(record, "generated_gv")
            tasks.append(((run_dir / rel) if rel else None,
                          _field(record, "ground_truth_path") or None,
                          _field(record, "generation_error")))
    return keys, tasks


def _score_task(task: Tuple[Optional[Path], Optional[str], str]) -> Dict[str, Any]:
    """One generation, scored in whatever process picks it up.

    Module level and arguments-by-tuple because a process pool has to pickle
    both the callable and its argument, and a closure is neither.
    """
    generated, ground_truth, generation_error = task
    return score_generation(generated, ground_truth,
                            generation_error=generation_error)


def _jsonable(value: Any) -> Any:
    """numpy scalars on the way into the checkpoint. `json` refuses them, and a
    metric computed through numpy hands back np.float64 rather than float."""
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def _cache_header(tasks_total: int) -> Dict[str, Any]:
    """What a checkpoint has to agree with before it may be resumed from.

    The metric set is in it because that is the thing most likely to have moved:
    resuming a run scored under a different `METRICS` would silently splice two
    metric sets into one table.
    """
    return {"tasks": tasks_total,
            "dot_encoding": "utf-8",
            "semantic": [k for k, _g in semantic.METRICS],
            "pragmatic": [m.key for m in pragmatic.METRICS],
            "syntactic": list(syntactic.CHECKS)}


def _load_cache(path: Path, tasks_total: int,
                verbose: bool) -> List[Dict[str, Any]]:
    """Rows already scored in an earlier attempt, or nothing.

    A mismatched or damaged checkpoint is discarded rather than repaired: it is
    an optimisation, and resuming from the wrong one would cost more than the
    hours it saves. The last line is dropped if it is not valid JSON — a kill
    can land mid-write.
    """
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    header: Optional[Dict[str, Any]] = None
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break  # truncated tail of an interrupted write
            if i == 0:
                header = record
                continue
            rows.append(record)
    if header != _cache_header(tasks_total):
        if verbose:
            print(f"  [quality] ignoring {path.name}: it was written for a "
                  f"different run or metric set", flush=True)
        return []
    rows = rows[:tasks_total]
    if verbose and rows:
        print(f"  [quality] resuming from {path.name}: {len(rows)} row(s) "
              f"already scored", flush=True)
    return rows


def _score_all(tasks: List[Tuple[Optional[Path], Optional[str], str]], *,
               workers: int = 1, verbose: bool = True,
               cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every generation's columns, in the order the tasks were given.

    Sequential by default — identical to what this function has always done.
    `workers > 1` spreads the work over a process pool, which is worth the
    plumbing only because the semantic dimension now carries the graph edit
    distance: it is bounded by a wall-clock budget per pair
    (`semantic._GED_BUDGET_S`), so on LLM-generated models it dominates
    everything else. Measured on run 20260831_182408: 4.76 s per model, i.e.
    **33 hours** for its 25 217 scorable rows in one process.

    Scoring one generation reads two files and touches nothing shared, so the
    only thing the pool has to get right is ordering — `map` preserves it. A
    small chunksize on purpose: the per-model cost ranges from milliseconds to
    the full GED budget, and large chunks would leave workers idle at the end
    while one of them grinds through a chunk of hard models.

    `cache_path` makes the work survive an interruption. Rows are appended as
    they arrive and a later call resumes after them, which matters because this
    is now a job measured in hours: the first attempt on run 20260831_182408 was
    killed by the OS for memory pressure 37% and 59 minutes in, and without a
    checkpoint that hour is simply gone. The checkpoint is only ever an
    optimisation — `_load_cache` discards one it cannot fully vouch for.
    """
    total = len(tasks)
    out: List[Dict[str, Any]] = (
        _load_cache(cache_path, total, verbose) if cache_path else [])
    remaining = tasks[len(out):]
    if not remaining:
        return out

    if workers <= 1 or len(remaining) < 2:
        scored_iter: Any = (_score_task(t) for t in remaining)
        pool_ctx: Any = None
    else:
        from concurrent.futures import ProcessPoolExecutor
        if verbose:
            print(f"  [quality] scoring {len(remaining)} generation(s) over "
                  f"{workers} process(es)", flush=True)
        pool_ctx = ProcessPoolExecutor(max_workers=workers)
        scored_iter = pool_ctx.map(_score_task, remaining, chunksize=4)

    # Opened once and kept open: one append per row, flushed, so a kill costs
    # the row in flight and nothing behind it.
    cache = None
    if cache_path is not None:
        fresh = not out
        cache = cache_path.open("w" if fresh else "a", encoding="utf-8")
        if fresh:
            cache.write(json.dumps(_cache_header(total)) + "\n")
            cache.flush()

    # `resumed` rows cost nothing now, so only the ones scored in this attempt
    # may feed the rate — otherwise a resumed job reports a throughput it never
    # had and an ETA to match.
    resumed = len(out)
    done = resumed
    step = max(1, total // 40)
    started = time.perf_counter()
    try:
        for scored in scored_iter:
            out.append(scored)
            done += 1
            if cache is not None:
                cache.write(json.dumps(scored, default=_jsonable) + "\n")
                cache.flush()
            if verbose and (done % step == 0 or done == total):
                elapsed = time.perf_counter() - started
                rate = (done - resumed) / elapsed if elapsed else 0.0
                left = (total - done) / rate if rate else 0.0
                print(f"  [quality] {done}/{total} "
                      f"({done / total:.0%}) — {elapsed / 60:.1f} min elapsed, "
                      f"~{left / 60:.1f} min left", flush=True)
    finally:
        if cache is not None:
            cache.close()
        if pool_ctx is not None:
            pool_ctx.shutdown()
    return out


def already_scored(df) -> bool:
    """True when a generation log already carries the quality columns, because
    the run scored each reply as it arrived."""
    return "syn_score" in df.columns and "quality_note" in df.columns


def scored_frame(run_dir, verbose: bool = True, workers: int = 1) -> pd.DataFrame:
    """The scored table for a finished run, computing it only if it is missing.

    A run that scored inline already holds every metric column in
    `results.jsonl`; re-deriving them would spend minutes reproducing identical
    numbers. `score_run` stays the way to apply *changed* metrics to a finished
    run - that is what `score_run.py` calls, and it deliberately recomputes.
    """
    run_dir = Path(run_dir)
    df = load_generation_log(run_dir)
    if already_scored(df):
        if verbose:
            n = int(df["syn_score"].notna().sum())
            print(f"  [quality] {len(df)} row(s) were scored during the run "
                  f"({n} with a syntactic score) — not recomputed.")
        return df
    return score_run(run_dir, verbose=verbose, workers=workers)


def score_run(run_dir, verbose: bool = True, workers: int = 1,
              cache_path: Optional[Path] = None) -> pd.DataFrame:
    """Score every generation in a finished run directory.

    Returns the generation log with the quality columns joined on — **in
    memory**. Nothing is written: a run keeps only its raw inputs
    (`manifest.json`, `results.jsonl`, `raw/`, `generated/`) and the rendered
    `results.html`, so there is no intermediate CSV to fall out of step with
    the metrics. Re-scoring after a metric change means calling this again and
    re-rendering, which costs nothing and never touches an API.
    """
    run_dir = Path(run_dir)
    # The generation log is **not** read yet, on purpose: see `_scoring_tasks`.
    keys, tasks = _scoring_tasks(run_dir)

    scored_rows = _score_all(tasks, workers=workers, verbose=verbose,
                             cache_path=cache_path)

    rows = []
    n_scored = n_unparseable = n_missing = 0
    for (generated, _gt, _err), key_cols, scored in zip(tasks, keys, scored_rows):
        row: Dict[str, Any] = dict(key_cols)
        row.update(scored)
        if (generated is None
                or scored.get("quality_note", "").startswith("generated file missing")):
            n_missing += 1
        elif row.get("syn_score") is None:
            n_unparseable += 1
        else:
            n_scored += 1
        rows.append(row)

    # Hand back what the workers produced before the generation log comes in:
    # the two are the memory peak of this function and must not coincide.
    del tasks, keys, scored_rows
    quality = pd.DataFrame(rows)
    del rows

    # **Only now** is the log read. By here the pool is closed and its processes
    # are gone, so the frame has the machine to itself — see `_scoring_tasks`.
    df = load_generation_log(run_dir)

    # Join onto the generation log. `repetition` is part of the key: without it
    # the same (item, model, strategy) appears N times and the join would cross
    # every repetition with every other.
    key = [k for k in KEY_COLS if k in df.columns]
    # Everything the scorer owns goes, not only the columns it produced this
    # time: a metric that has since been **removed** is still in the run's
    # `results.jsonl`, and keeping it would carry a stale value into the
    # re-scored frame and out into the CSV export — the deleted metric
    # reappearing in every re-scored run.
    stale = [c for c in df.columns
             if c not in key
             and (c.startswith(_SCORED_PREFIXES) or c in _SCORED_COLUMNS
                  or c in quality.columns)]
    # Rebound rather than chained, and dropped the moment the join is done.
    # `drop` and `merge` each copy, so the naive one-liner holds the log three
    # times over at once — on run 20260831_182408 that is ~3 GB before the
    # report builder has allocated anything, and the report is what the OS
    # killed this function in. The scored rows are in `quality`; the log's own
    # columns survive in `merged`; nothing needs `df` past this line.
    df = df.drop(columns=stale)
    merged = df.merge(quality, on=key, how="left")
    del df

    if verbose:
        def _mean(col: str) -> str:
            if col not in quality:
                return "n/a"
            m = quality[col].mean()
            return f"{m:.3f}" if pd.notna(m) else "n/a"

        print(f"  [quality] scored {n_scored} model(s) "
              f"({n_unparseable} unparseable, {n_missing} without output) "
              f"— mean syn_score {_mean('syn_score')}, "
              f"syn_bef_score {_mean('syn_bef_score')}, "
              f"prag_score {_mean('prag_score')}, "
              f"sem_score {_mean('sem_score')}, "
              f"val_score {_mean('val_score')}")
        # A missing Graphviz makes every val_* cell None. Say so once, here,
        # rather than letting an all-empty column read as "nothing was wrong".
        if validity.nop_executable() is None:
            print(f"  [quality] NOTE: DOT validity was not checked — "
                  f"{validity.checker_label()}. The val_* columns are empty, not zero.")
        label = syntactic.CHECK_LABELS["sequence_flow_connection_rules"]
        print(f"  [quality] {label}: mean "
              f"{_mean('syn_sequence_flow_connection_rules')}")
    return merged
