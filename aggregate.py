"""
aggregate.py — the two aggregations the results report is built on
=====================================================================
A run answers two questions, and this module produces exactly those two tables:

    by_model(df)     one row per LLM, aggregated over all prompting strategies
    by_strategy(df)  one row per prompting strategy, aggregated over all LLMs

Both take the scored generation log (`quality.score_run`) and return a frame
whose columns are grouped into the three blocks the report can switch between —
quality, tokens & cost, latency.

Two rules are enforced here rather than left to the renderer, because getting
them wrong silently produces a misleading table:

* **No mean without its sample size.** Every row carries `n_calls`,
  `n_valid`, `n_scored` and `parse_rate`. A model with a parse rate of 0.6 and
  a high mean score is worse than one with 0.95 and a middling score, and the
  reader has to be able to see that.
* **Right statistic per quantity.** Token counts get mean *and* median, since
  reasoning models produce strongly right-skewed distributions; cost gets a sum
  as well, because the total is what the budget cares about; latency gets the
  spread, because a p-max matters operationally.

Nothing is written to disk — a run stores its raw inputs and `results.html`,
nothing in between.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

# ── metric blocks ─────────────────────────────────────────────────────────────
# (column, label, direction) — direction "up"/"down"/"none". Pragmatic quality
# is deliberately "none": it rewards smallness and carries no direction, so a
# higher value is not a better model.
# Syntactic quality is reported with BEF4LLM's published set (Table 2 / A.15),
# decided 2026-08-18: it is citable and sits in the same framework as the
# semantic and pragmatic dimensions, which come from the same paper. The eleven
# dictated pass/fail checks are still computed and still carried in the run's
# data — they say *which* rule a model broke, which the ratio score cannot —
# but they are not a second headline: the two disagree on the ranking of the
# strategies, and reporting both would mean reporting two orders.
QUALITY_METRICS: List[tuple] = [
    ("val_score", "Validity (DOT)", "up"),
    ("syn_bef_score", "Syntactic", "up"),
    # A verdict, not a score: yes only when every call behind the row produced a
    # model that satisfies all five conditions (see quality/syntactic.py). The
    # report renders it as yes/no; the mean it is aggregated from is what tells
    # the two apart, and `_count` below is what keeps a row with unparseable
    # calls from being called "yes" over the ones that did parse.
    ("syn_correct", "Syntactical Correctness", "up"),
    ("sem_score", "Semantic", "up"),
    ("prag_score", "Pragmatic", "none"),
]

COST_METRICS: List[tuple] = [
    ("input_tokens", "Input tokens (uncached)", "down"),
    ("output_tokens", "Output tokens", "down"),
    ("thinking_tokens", "of which reasoning", "none"),
    ("cached_tokens", "Cache read tokens", "none"),
    ("billable_input_tokens", "Billable input tokens", "down"),
    ("total_tokens", "Total tokens", "down"),
    ("cost_usd", "Cost USD", "down"),
]

# One column, not the three nested spans: `latency_s` is the whole provider
# call — the round trip plus any retry and its backoff. The narrower
# `api_latency_s` (answering attempt only), `call_wall_s`, `api_attempts` and
# `retry_wait_s` stay in the per-call fold-out for anyone who needs to see
# where a slow call spent its time.
LATENCY_METRICS: List[tuple] = [
    ("latency_s", "Total latency (s)", "down"),
]

BLOCKS: Dict[str, List[tuple]] = {
    "quality": QUALITY_METRICS,
    "cost": COST_METRICS,
    "latency": LATENCY_METRICS,
}

# Which statistics each block gets. Tokens are right-skewed on reasoning models,
# so the median sits next to the mean; cost also carries the run total.
STATS = {
    # `count` is not shown anywhere: it is how the report can tell "every call
    # of this row passed" from "every call that produced a model passed", which
    # a mean alone cannot say.
    "quality": ["mean", "count"],
    "cost": ["mean", "median", "sum"],
    "latency": ["mean", "median", "max"],
}


def _filled(df: pd.DataFrame, column: str) -> pd.Series:
    """Boolean mask: does this text column actually say something?

    The generation log is JSONL, where "no error" is the empty string — not
    NaN, as it was when the same field round-tripped through CSV. `notna()`
    would count every successful call as a failure, so emptiness is tested on
    the stripped text instead.
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna("").astype(str).str.strip().ne("")


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """The column as numbers, always as a Series.

    A column a run never produced must still aggregate — to NaN, which renders
    as "not computed". Passing the missing column straight to `to_numeric`
    would return a scalar, and a scalar has no `.median()`.
    """
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df), dtype="float64", index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _aggregate(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """One row per value of `group`, with every block's metrics and the sample
    sizes that make them readable."""
    if group not in df.columns:
        return pd.DataFrame()

    out_rows = []
    for key, part in df.groupby(group, dropna=False):
        row: Dict[str, object] = {group: key}

        # Sample sizes first — a mean is not reportable without them.
        n_calls = len(part)
        parsed = part.get("extract_parse_ok")
        valid = _numeric(part, "val_dot_valid")
        scored = _numeric(part, "syn_score")
        row["n_calls"] = n_calls
        row["n_parsed"] = int(parsed.fillna(False).astype(bool).sum()) if parsed is not None else 0
        row["n_valid"] = int(valid.fillna(0).sum())
        row["n_scored"] = int(scored.notna().sum())
        row["parse_rate"] = row["n_parsed"] / n_calls if n_calls else float("nan")
        row["n_errors"] = int(_filled(part, "generation_error").sum())

        for block, metrics in BLOCKS.items():
            for column, _label, _direction in metrics:
                series = _numeric(part, column)
                for stat in STATS[block]:
                    value = getattr(series, stat)() if series is not None else float("nan")
                    row[f"{column}_{stat}"] = value
        out_rows.append(row)

    result = pd.DataFrame(out_rows)
    return result.sort_values(group).reset_index(drop=True)


def by_model(df: pd.DataFrame) -> pd.DataFrame:
    """One row per LLM, aggregated over every prompting strategy."""
    return _aggregate(df, "model_key")


def by_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """One row per prompting strategy, aggregated over every LLM."""
    return _aggregate(df, "strategy")


def totals(df: pd.DataFrame) -> Dict[str, object]:
    """Run-level figures for the header band."""
    n = len(df)
    parsed = df.get("extract_parse_ok")
    valid = _numeric(df, "val_dot_valid")
    return {
        "n_calls": n,
        "n_parsed": int(parsed.fillna(False).astype(bool).sum()) if parsed is not None else 0,
        "n_valid": int(valid.fillna(0).sum()),
        "n_errors": int(_filled(df, "generation_error").sum()),
        "n_untracked": int(_filled(df, "untracked_usage").sum()),
        "cost_usd": float(_numeric(df, "cost_usd").sum()),
        "total_tokens": int(_numeric(df, "total_tokens").sum()),
        "api_seconds": float(_numeric(df, "api_latency_s").sum()),
        "n_models": int(df["model_key"].nunique()) if "model_key" in df else 0,
        "n_strategies": int(df["strategy"].nunique()) if "strategy" in df else 0,
        "n_items": int(df["item_id"].nunique()) if "item_id" in df else 0,
        "repetitions": int(_numeric(df, "repetition").max()) if "repetition" in df else 1,
    }
