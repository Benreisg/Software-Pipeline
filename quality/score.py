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
from pathlib import Path
from typing import Any, Dict, Optional

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
    if gen_pg is None:
        out.update(syntactic.empty())
        out["quality_note"] = f"generated model did not parse: {parse_error}"
    else:
        out.update(syntactic.evaluate(gen_pg))
        out["quality_note"] = f"ground truth did not parse: {gt_error}" if gt_error else ""

    # `validity` is deliberately unimplemented and contributes no columns
    # until it is; `semantic` and `pragmatic` return their full column sets,
    # all None, when there is no model (or, for semantic, no reference) to
    # measure.
    out.update(semantic.evaluate(gen_pg, gt_pg) if gen_pg is not None
               else semantic.empty())
    out.update(pragmatic.evaluate(gen_pg, gt_pg) if gen_pg is not None
               else pragmatic.empty())
    # Validity reads the file itself: its verdict comes from Graphviz over the
    # bytes on disk, deliberately independent of whether `quality/graph.py`
    # managed to parse them. It therefore runs even when gen_pg is None — that
    # is exactly the case it is there to explain.
    out.update(validity.evaluate(gen_pg, parse_error=parse_error,
                                 source_path=generated_path))
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

    return score_model(path, ground_truth_path or None)


def already_scored(df) -> bool:
    """True when a generation log already carries the quality columns, because
    the run scored each reply as it arrived."""
    return "syn_score" in df.columns and "quality_note" in df.columns


def scored_frame(run_dir, verbose: bool = True) -> pd.DataFrame:
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
    return score_run(run_dir, verbose=verbose)


def score_run(run_dir, verbose: bool = True) -> pd.DataFrame:
    """Score every generation in a finished run directory.

    Returns the generation log with the quality columns joined on — **in
    memory**. Nothing is written: a run keeps only its raw inputs
    (`manifest.json`, `results.jsonl`, `raw/`, `generated/`) and the rendered
    `results.html`, so there is no intermediate CSV to fall out of step with
    the metrics. Re-scoring after a metric change means calling this again and
    re-rendering, which costs nothing and never touches an API.
    """
    run_dir = Path(run_dir)
    df = load_generation_log(run_dir)

    if "generated_gv" not in df.columns:
        raise ValueError(
            f"{run_dir}/results.jsonl has no 'generated_gv' field — it predates "
            "the current pipeline. Re-run to score it."
        )

    def _cell(row, col: str) -> str:
        """A missing field arrives as None or NaN; `NaN or ""` is NaN (floats
        are truthy), which would turn into the path 'nan'."""
        v = row.get(col)
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()

    rows = []
    n_scored = n_unparseable = n_missing = 0
    for _, r in df.iterrows():
        rel = _cell(r, "generated_gv")
        gt = _cell(r, "ground_truth_path")
        row: Dict[str, Any] = {k: r[k] for k in KEY_COLS if k in df.columns}

        scored = score_generation(
            (run_dir / rel) if rel else None, gt or None,
            generation_error=_cell(r, "generation_error"))
        row.update(scored)
        if not rel or scored.get("quality_note", "").startswith("generated file missing"):
            n_missing += 1
        elif row.get("syn_score") is None:
            n_unparseable += 1
        else:
            n_scored += 1
        rows.append(row)

    quality = pd.DataFrame(rows)

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
    merged = df.drop(columns=stale).merge(quality, on=key, how="left")

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
