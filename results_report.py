#!/usr/bin/env python3
"""
results_report.py — the one file a run produces for reading
===============================================================
    python results_report.py runs/<run_id>        # → runs/<run_id>/results.html

A run keeps its raw inputs (`manifest.json`, `results.jsonl`, `raw/`,
`generated/`) and this page. Nothing in between: no summary CSV, no pivot zoo,
nothing that can fall out of step with the metric definitions.

Structure, matching how the results are actually read:

    Navigation (top)      By LLM          — one row per model, prompts aggregated
                          By strategy     — one row per prompt, LLMs aggregated
    Selector (per view)   Quality  ·  Token Usage & Cost  ·  Latency

Three presentation rules are enforced rather than left to the eye:

* **No mean without its sample size.** Every row shows `n`, valid/total and the
  parse rate, so a high mean on a low parse rate cannot pass for a good result.
  The row head says it in words too — *(Average from all 9 API Calls)*, and
  *(From all 9 API Calls)* under latency, whose columns are a mean, a median and
  a max rather than one average — because that is what stays on screen when the
  row is folded shut, and a bare model name there reads like a single
  measurement.
* **Pragmatic quality carries no direction** — it rewards smallness. It is shown
  unshaded and marked, never coloured good/bad, and so are reasoning tokens and
  cache reads: they describe how a result came about rather than setting a
  target.
* **A metric a run did not produce reads "not computed"**, never 0 and never
  an empty cell.
* **The tokens block is one table per vendor**, because the vendors do not
  report the same quantities: a single grid could hold either every vendor or
  every figure. It holds every figure. By LLM that *is* the comparison table —
  one table per vendor, one row per model, every entry the mean over that
  model's calls; by strategy the same tables come once per strategy, each entry
  the mean over the calls that strategy made. The columns are the vendor's own
  fields, so only the cost columns can be read across two tables. A field the
  vendor names but never fills gets no column: it would be blank in every row.

The full per-call dataset is embedded as JSON so the page stays the artefact of
record: the aggregates on screen can be re-derived, and the raw numbers pulled
out, without the run directory. 
"""
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import aggregate as agg
# The one thing this page borrows from the provider layer: the rule for what
# counts as a measured quantity. Importing it costs nothing (providers keeps
# every vendor SDK behind a lazy import) and keeps one definition instead of
# two that can drift apart.
from providers import _is_billable_name

VIEWS = [("model", "By LLM", "model_key", "one row per model — every prompting strategy pooled"),
         ("strategy", "By prompt strategy", "strategy",
          "one row per strategy — every model pooled")]

# The three metric blocks, one panel each per view, reached through one flat
# selector — every button on screen at all times:
#
#     Quality | Token Usage & Cost | Latency
#
# One level, not two: tokens and latency were once nested under an "Efficiency"
# button that had to be pressed before they appeared, which hid two of the three
# readings behind a fourth button. They are three questions about the same run —
# what a model produced, what it cost, how long it took — and choosing one now
# moves nothing on the row.
TABS = [("quality", "Quality"), ("cost", "Token Usage &amp; Cost"), ("latency", "Latency")]
DEFAULT_TAB = TABS[0][0]

# Which statistic to show per block, and how to format it. The cost row is not
# rendered by any panel — that block is the per-vendor tables in both views —
# but it is the column spec a pooled tokens table would be built from, and
# `agg` computes those statistics either way.
_COLUMNS = {
    "quality": [("mean", "", 3)],
    "cost": [("mean", "mean", 1), ("median", "median", 0), ("sum", "total", 2)],
    "latency": [("mean", "mean", 2), ("median", "median", 2), ("max", "max", 2)],
}


# ── formatting ────────────────────────────────────────────────────────────────
def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


# Thousands separator: a narrow no-break space, so 1 244 reads as one number
# and cannot break across a line.
GROUP_SEP = "\u202f"


def num(value: Any, digits: int = 2) -> str:
    if missing(value):
        return '<span class="na">not computed</span>'
    if isinstance(value, (int,)) or (isinstance(value, float) and value == int(value)
                                     and abs(value) >= 10):
        return f"{int(value):,}".replace(",", GROUP_SEP)
    return f"{float(value):.{digits}f}"


def count(value: Any, digits: int = 1) -> str:
    """A token figure. A whole number prints as a count — a cache-read figure of
    zero is `0`, not `0.00` — and only a genuinely fractional value, which here
    means a mean, keeps decimals. `num` cannot do this: a quality score of 0.0
    must stay 0.000, and nothing in the value itself tells the two apart."""
    if missing(value):
        return '<span class="na">not computed</span>'
    if float(value) == int(value):
        return f"{int(value):,}".replace(",", GROUP_SEP)
    return f"{float(value):.{digits}f}"


def pct(value: Any) -> str:
    return '<span class="na">—</span>' if missing(value) else f"{float(value):.0%}"


# ── rendering ─────────────────────────────────────────────────────────────────
# How a row head names what stands behind the key. "Average" is right where
# every column of the row is one mean; the latency block is not that row — it
# puts a mean, a median and a max side by side, so calling the row an average
# would name one of its three columns and misdescribe the other two.
_ROW_LEAD = {"latency": "From"}
_DEFAULT_ROW_LEAD = "Average from"


def _rowhead(key: Any, n_calls: Any, block: str = "") -> str:
    """The head of an aggregate row: the key, and behind it what the row is.

    The figures in the row are computed over the calls that key stands for, and
    the row does not say so on its own — the `n` column gives the number but
    not what was done with those calls, so a row reads like one measurement of
    one model. Spelling it out in the head is the one place it cannot be
    missed, because the head is also what stays on screen when the row is
    folded shut.
    """
    meta = ""
    if not missing(n_calls):
        n = int(n_calls)
        lead = _ROW_LEAD.get(block, _DEFAULT_ROW_LEAD)
        meta = (f'<span class="rowmeta">({lead} 1 API Call)</span>' if n == 1
                else f'<span class="rowmeta">({lead} all {n} API Calls)</span>')
    return (f'<th class="rowhead" scope="row">'
            f'<button type="button" class="expander" aria-expanded="false">'
            f'<span class="caret" aria-hidden="true">▸</span>{esc(key)}{meta}'
            f'</button></th>')


# The sample columns a block carries beside its key. Quality and tokens are
# only readable against how many calls produced a usable model — a high mean on
# a low parse rate is not a good result, which is the rule "no mean without its
# sample size" exists for. A latency is measured on the API call itself, which
# happens whether or not the reply parsed, so `valid` and `parse rate` there
# answer a question the row is not asking. `n` stays everywhere: it is the
# sample the row's own figures are computed over.
_DEFAULT_SAMPLE_COLUMNS = [("n", "n"), ("valid", "valid"), ("parse", "parse&nbsp;rate")]
_SAMPLE_COLUMNS = {"latency": [("n", "n")]}


def _sample_cell(row: Any, key: str) -> str:
    if key == "valid":
        return f'{int(row["n_valid"])}/{int(row["n_calls"])}'
    if key == "parse":
        return pct(row["parse_rate"])
    return str(int(row["n_calls"]))


# Columns that are a verdict about a model, not a measurement of one. Per call
# the answer is yes or no; a row pools several calls, so it shows **how many of
# them said yes**, out of every call behind the row — `5/8`, read like the
# `valid` column beside it. A mean would read as a score, and this metric has no
# partial credit; a bare yes/no would hide whether a row missed by one call or
# by seven.
_VERDICT_COLUMNS = {"syn_correct"}

# Metrics the scoring pass computes, that the page does not draw. A name here is
# hidden, not deleted — it keeps being computed and keeps travelling in
# `results.jsonl`, in the CSV export and in the dataset embedded below, so the
# figure stays recoverable and everything that drew it comes back by taking the
# name out again. Two scopes are filtered against this one set: an aggregate
# column of `agg.BLOCKS` — its header, its cells, and the fold-out block it is
# the headline of — and a single metric row inside a fold-out block.
#
# Both entries are BEF4LLM syntactic metrics hidden on 2026-08-29 (author's
# instruction) — the two that measure how gateways are wired:
#
# * `syn_bef_gateway_in_out_degree` — #15/#16 as this port merges them;
# * `syn_bef_split_has_matching_join` — #9 (hidden once before, on 2026-08-28,
#   and drawn again the next day).
#
# Both are still scored and still counted in `syn_bef_score` over all eleven
# metrics; what goes is their row in the syntactic fold-out, the one place on
# the page they appeared. The Syntactical Correctness verdict was in this set
# too, until 2026-08-26.
HIDDEN_COLUMNS = {"syn_bef_gateway_in_out_degree",
                  "syn_bef_split_has_matching_join"}


def _verdict_counts(row: Any, column: str) -> Optional[tuple]:
    """`(passed, n_calls, n_scored)` for a verdict column, or None when the row
    holds no verdict at all.

    `n_scored` is how many calls produced a model to judge; the mean travels
    over those only, so the passed count is recovered from the two together. The
    denominator is `n_calls` regardless: a call that produced nothing did not
    pass, and dividing by the survivors would quietly flatter the row.
    """
    value = row.get(f"{column}_mean")
    if missing(value):
        return None
    scored = row.get(f"{column}_count")
    n_scored = 0 if missing(scored) else int(scored)
    return int(round(float(value) * n_scored)), int(row["n_calls"]), n_scored


def _verdict_cell(row: Any, column: str, value: Any, first: bool,
                  lo: float, hi: float) -> str:
    """One `passed/total` cell, with the arithmetic in the tooltip.

    Shaded like every other column: relative to the other rows of the same
    table, so the darkest cell is the model that got it right most often — never
    against an absolute scale.
    """
    edge = " group-start" if first else ""
    counts = _verdict_counts(row, column)
    if counts is None:
        return (f'<td class="num cell-flat stat-verdict{edge}" style="--w:0.000">'
                f'{num(value)}</td>')
    n_passed, n_calls, n_scored = counts
    share = n_passed / n_calls if n_calls else 0.0
    weight = 0.0 if hi == lo else (share - lo) / (hi - lo)
    title = f"{n_passed} of {n_calls} call(s) passed"
    if n_scored < n_calls:
        title += (f"; {n_calls - n_scored} produced no parseable model and have "
                  f"no verdict")
    return (f'<td class="num cell-scale stat-verdict verdict{edge}" '
            f'style="--w:{weight:.3f}" title="{esc(title)}">'
            f'{n_passed}/{n_calls}</td>')


def _metric_table(frame: pd.DataFrame, group: str, block: str) -> str:
    """One aggregate table: rows are models or strategies, columns the block's
    metrics with the statistics that suit them."""
    if frame.empty:
        return '<p class="na">No rows for this view.</p>'

    metrics = [m for m in agg.BLOCKS[block] if m[0] not in HIDDEN_COLUMNS]
    stats = _COLUMNS[block]
    sample = _SAMPLE_COLUMNS.get(block, _DEFAULT_SAMPLE_COLUMNS)

    head = ['<th class="rowhead">' + ("Model" if group == "model_key" else "Strategy") + "</th>"]
    head += [f'<th class="num sample">{label}</th>' for _key, label in sample]
    for column, label, direction in metrics:
        span = len(stats)
        # No direction marker in the head: the shading below already points at
        # the best value in each column, and `direction` still decides which
        # end that is — a metric without one simply stays unshaded.
        note = ' title="no direction: this metric rewards smallness"' if direction == "none" else ""
        head.append(f'<th class="num group" colspan="{span}"{note}>{esc(label)}</th>')
    sub = ['<th></th>'] * (1 + len(sample))
    # `group-start` on the first statistic of each metric carries the vertical
    # rule that separates one token kind (or quality dimension) from the next —
    # the group header already has it, and the body needs it too or the line
    # stops at the top of the table.
    for _column, _label, _direction in metrics:
        for i, (_stat, stat_label, _digits) in enumerate(stats):
            edge = " group-start" if i == 0 else ""
            sub.append(f'<th class="num substat stat-{_stat}{edge}">{stat_label}</th>')

    # The verdict columns are shaded against each other like every other
    # column, so the range they are normalised over is computed once here —
    # over what the cells actually show (passed / calls), not over the mean,
    # which counts only the calls that produced a model.
    verdict_range = {}
    for column, _label, _direction in metrics:
        if column in _VERDICT_COLUMNS:
            shares = [c[0] / c[1] for c in
                      (_verdict_counts(r, column) for _, r in frame.iterrows())
                      if c and c[1]]
            verdict_range[column] = (min(shares), max(shares)) if shares else (0.0, 0.0)

    body = []
    n_columns = 1 + len(sample) + len(metrics) * len(stats)
    for _, row in frame.iterrows():
        key = str(row[group])
        # The row head is a button: pressing it unfolds every call behind this
        # aggregate. The calls are built client-side from the dataset already
        # embedded in the page, so unfolding costs no extra bytes on disk.
        cells = [_rowhead(key, row["n_calls"], block)]
        cells += [f'<td class="num sample">{_sample_cell(row, key_)}</td>'
                  for key_, _label in sample]
        for column, _label, direction in metrics:
            values = [row.get(f"{column}_{stat}") for stat, _l, _d in stats]
            # Shade only where a direction exists, and only against the other
            # rows of this table — never against an absolute scale.
            for i, ((stat, _l, digits), value) in enumerate(zip(stats, values)):
                if column in _VERDICT_COLUMNS:
                    lo, hi = verdict_range[column]
                    cells.append(_verdict_cell(row, column, value, i == 0, lo, hi))
                    continue
                col_values = pd.to_numeric(frame[f"{column}_{stat}"], errors="coerce")
                lo, hi = col_values.min(), col_values.max()
                share = 0.0
                if direction != "none" and not missing(value) and hi != lo:
                    share = (value - lo) / (hi - lo)
                    if direction == "down":
                        share = 1 - share
                cls = "cell-flat" if direction == "none" else "cell-scale"
                edge = " group-start" if i == 0 else ""
                cells.append(f'<td class="num {cls} stat-{stat}{edge}" style="--w:{share:.3f}">'
                             f'{num(value, digits)}</td>')
        body.append(f'<tr class="aggrow" data-key="{esc(key)}">{"".join(cells)}</tr>'
                    f'<tr class="detailrow" hidden>'
                    f'<td colspan="{n_columns}"><div class="detail"></div></td></tr>')

    return (f'<div class="tablewrap"><table class="agg" data-group="{esc(group)}" '
            f'data-block="{esc(block)}">'
            f'<thead><tr>{"".join(head)}</tr><tr class="substats">{"".join(sub)}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def paper_metric_spec() -> List[Dict[str, Any]]:
    """The metrics a fold-out under *Quality* shows: BEF4LLM's published set,
    nothing else.

    Derived from the metric modules themselves rather than listed here, so a
    change to the ported metric set reaches the report without an edit. What is
    deliberately left out of the fold-out:

    * the eleven **dictated** syntactic checks — the author's own set, not the
      paper's;
    * `syn_extra_*` — present in BEF4LLM's code but in neither definition table;
    * `val_*` — the Graphviz DOT check, this project's own gate;
    * every `*_n_*` diagnostic counter and `*_truncated` flag — instrumentation
      this project added to explain a value, not a metric of the paper;
    * `prag_diameter` — BEF4LLM's own traversal of the diameter, kept as the
      column comparable with their published figures. The paper's metric #5 is
      in the fold-out as `prag_diameter_nogw`, which is what `prag_score` has
      scored since 2026-08-29 (author's instruction); the fold-out follows
      `pragmatic.METRICS`, so the swap reached it without an edit here;
    * `prag_group_*_score` / `sem_group_*_score` — this project's roll-up per
      group, not the paper's: it aggregates a dimension in one step. The
      grouping itself is the paper's and stays, as a label on each metric;
    * the `syn_tnn/tng/tnsf/diameter` duplicates — the same four size metrics
      already appear under `prag_`, and showing each number twice is noise.

    Tokens, cost and latency are not paper metrics either; they stay in the
    card's header line, where the per-call figures remain visible.
    """
    from quality import pragmatic, semantic, syntax_rules

    syntactic_metrics = [
        {"key": f"syn_bef_{m.key}", "label": m.label,
         # A count metric is a conforming/covered ratio in the paper; carrying
         # both makes the value legible instead of an unexplained decimal.
         "ratio": None if m.boolean else [f"syn_bef_{m.key}_conforming",
                                          f"syn_bef_{m.key}_covered"]}
        for m in syntax_rules.METRICS
    ]

    # The paper's grouping survives as a label on each metric; the per-group
    # *scores* do not appear. They are this project's own roll-up — both metric
    # modules say so — and the paper aggregates each dimension in one step.
    # On a single call they would also be nothing but the arithmetic mean of
    # the rows directly above them.
    semantic_metrics = [
        {"key": f"sem_{key}", "label": f"{key.replace('_', ' ')}  [{group.replace('_', ' ')}]"}
        for key, group in semantic.METRICS
    ]

    pragmatic_metrics = [
        # Raw value plus the paper's banded score (Eq. 1 / Eq. 2) side by side.
        {"key": f"prag_{m.key}",
         "label": f"{m.label}  [{pragmatic.GROUP_LABELS.get(m.group, m.group)}]",
         "score": f"prag_{m.key}_score"}
        for m in pragmatic.METRICS
    ]

    return [
        {"title": "Syntactic — BEF4LLM Table 2 / A.15", "headline": "syn_bef_score",
         "metrics": syntactic_metrics},
        {"title": "Semantic — Table A.17", "headline": "sem_score",
         "metrics": semantic_metrics},
        {"title": "Pragmatic — Table A.16 / A.18", "headline": "prag_score",
         "metrics": pragmatic_metrics},
    ]


# The verdict block, first in a quality fold-out: a yes/no about the model as a
# whole, with the five conditions under it so a `no` says which rule broke.
# It is not the paper's — hence its own function and its own heading, and hence
# `paper_metric_spec` above stays exactly the published set.
CORRECTNESS_BLOCK = {
    "title": "Syntactical Correctness — all five must hold",
    "headline": "syn_correct",
    "metrics": [
        {"key": "syn_correct_functions",
         "label": "functions have exactly one incoming and one outgoing arc"},
        {"key": "syn_correct_gateways",
         "label": "gateways have at least one incoming and one outgoing arc"},
        {"key": "syn_correct_start_end",
         "label": "at least one start node and one end node"},
        {"key": "syn_correct_directed", "label": "graph is directed (agisdirected)"},
        {"key": "syn_correct_connected", "label": "graph is coherent (isConnected)"},
        {"key": "syn_n_malformed_functions", "label": "functions breaking the rule"},
        {"key": "syn_n_unwired_gateways", "label": "gateways breaking the rule"},
        {"key": "syn_n_components", "label": "connected components (1 = coherent)"},
    ],
}


def quality_metric_spec() -> List[Dict[str, Any]]:
    """What a fold-out under *Quality* shows: this project's verdict on top of
    the paper's metric set, less whatever `HIDDEN_COLUMNS` names.

    Two things that are not the same kind of statement, kept apart rather than
    merged: `CORRECTNESS_BLOCK` is a yes/no this project defines, everything
    below it is BEF4LLM's published measurement.

    Hiding happens here rather than in `paper_metric_spec`, so that function
    goes on stating the published set exactly — the page is what a hidden name
    is kept out of, not the record of what BEF4LLM defines. A block whose
    headline is hidden goes whole (a fold-out block for a column the table does
    not show would be the only place on the page it appeared); a metric row
    whose key is hidden goes on its own, and takes its block with it if it was
    the last one in it.
    """
    blocks = []
    for block in [CORRECTNESS_BLOCK] + paper_metric_spec():
        if block["headline"] in HIDDEN_COLUMNS:
            continue
        metrics = [m for m in block["metrics"] if m["key"] not in HIDDEN_COLUMNS]
        if not metrics:
            continue
        blocks.append({**block, "metrics": metrics})
    return blocks


# Prefix for the per-token-class costs derived at build time. They are not in
# `results.jsonl` — the run records what the vendor billed in total, and these
# split that total across the classes that produced it.
COST_FIELD_PREFIX = "cost_of_"

# The token classes that carry their own rate. `billable_input_tokens` is their
# subtotal and `thinking_tokens` a subset of the output, so neither is priced
# again — adding either would double-count.
PRICED_TOKEN_FIELDS = ["input_tokens", "cached_tokens", "cache_write_tokens",
                       "cache_write_1h_tokens", "output_tokens"]


def add_cost_breakdown(records: List[dict]) -> None:
    """Split each call's recorded cost across the token classes that produced it.

    Rates come from `cfg.rates_for`, driven by the row's own `cost_basis`, so a
    call priced under an intro window or an off-peak hour is broken down at the
    rates it actually paid rather than today's. A row whose parts do not add up
    to the recorded total gets `cost_breakdown_note` — that means the total
    contains something the classes do not, a server-side tool being the usual
    reason, and the reader should trust the recorded total.
    """
    import config as cfg

    for record in records:
        for field in PRICED_TOKEN_FIELDS:
            record[COST_FIELD_PREFIX + field] = None
        record["cost_breakdown_note"] = ""

        # A run can hold a model key the catalogue no longer has — renamed, or
        # retired by the vendor, as Gemini 2.5 and deepseek-chat were. Such a row
        # keeps its recorded total and simply gets no breakdown.
        try:
            model = cfg.model_by_key(record.get("model_key") or "")
        except KeyError:
            record["cost_breakdown_note"] = (
                f"no rates on file for model key {record.get('model_key')!r} — "
                "the recorded total stands, but it cannot be split by token class")
            continue
        rates = cfg.rates_for(model, str(record.get("cost_basis") or ""))
        if rates is None:
            continue

        parts = 0.0
        for field in PRICED_TOKEN_FIELDS:
            tokens = record.get(field)
            if tokens is None or (isinstance(tokens, float) and pd.isna(tokens)):
                continue
            amount = float(tokens) * rates[field] / 1_000_000
            record[COST_FIELD_PREFIX + field] = amount
            parts += amount

        total = record.get("cost_usd")
        if total is not None and not (isinstance(total, float) and pd.isna(total)):
            if abs(parts - float(total)) > max(1e-9, abs(float(total)) * 1e-6):
                record["cost_breakdown_note"] = (
                    f"the classes below sum to {parts:.6f}, the recorded total is "
                    f"{float(total):.6f} — the difference is not token cost "
                    "(a per-request server tool, or a rate the breakdown cannot see)")


# ── the usage object as the vendor itself returned it ─────────────────────────
# Every live call archives its vendor usage object verbatim next to the reply
# (`raw/<tag>.usage.json`, written by pipeline.py). The tables above compare the
# normalised classes this project prices, and that normalisation is what makes
# two vendors comparable at all — but it can only keep what all of them state.
# What it drops is exactly what makes each vendor's accounting its own:
# Anthropic splits cache writes by TTL, Gemini counts thoughts and tool
# definitions in their own fields, DeepSeek reports prompt-cache hits *and*
# misses, OpenAI details cache reads and writes inside the input count, and not
# every vendor states a total at all. Read back here, that detail is shown per
# model, where it does not have to line up with any other vendor's.
USAGE_SUFFIX = ".usage.json"

# Where the flattened usage object rides on a per-call record, and therefore
# also in the dataset embedded in the page.
VENDOR_USAGE_FIELD = "vendor_usage"

def flatten_usage(value: Any, prefix: str = "") -> Dict[str, Any]:
    """A vendor's nested usage object as flat `dotted.name -> number` pairs.

    Numbers survive, and so do the fields a vendor named but left empty: those
    carry None and are reported as "not reported", never as 0 — a vendor that
    does not fill `thoughts_token_count` has not measured zero thinking. Text
    values (`service_tier`, `traffic_type`, `inference_geo`) are not token
    metrics and are dropped; the ones that matter already have a column of
    their own.

    Gemini reports its prompt breakdown as a *list* of `{modality,
    token_count}` objects. It is keyed by modality rather than by position —
    `prompt_tokens_details.TEXT` — because a name that depends on the order the
    vendor happened to send in cannot be aggregated across calls.
    """
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            out.update(flatten_usage(sub, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            label = ""
            if isinstance(entry, dict):
                for name in ("modality", "type", "kind"):
                    if isinstance(entry.get(name), str):
                        # `MediaModality.TEXT` → `TEXT`: model_dump(mode="json")
                        # resolves the enum, an older SDK stringifies it.
                        label = str(entry[name]).rsplit(".", 1)[-1]
                        break
            label = label or f"[{index}]"
            out.update(flatten_usage(entry, f"{prefix}.{label}" if prefix else label))
    elif value is None:
        out[prefix] = None
    elif isinstance(value, bool) or isinstance(value, str):
        pass
    elif isinstance(value, (int, float)):
        out[prefix] = value if math.isfinite(float(value)) else None
    return out


def usage_metrics(payload: Any) -> Dict[str, Any]:
    """A vendor usage object reduced to the quantities it states.

    A field that is empty is kept when its name says it measures something a
    vendor bills for — that Gemini offers `thoughts_token_count` and never
    fills it, or that Mistral offers `prompt_audio_seconds` and leaves it null,
    is a fact about that vendor's reporting and the whole point of this
    section. An empty field that measures nothing (`server_tool_use`,
    `traffic_type`) says nothing and is dropped.

    The test is `providers._is_billable_name`, the same one the untracked-usage
    guard applies, so a unit the guard has learned to watch for cannot be a
    unit this page silently omits.
    """
    return {name: value for name, value in flatten_usage(payload).items()
            if value is not None or _is_billable_name(name)}


def read_usage_file(path: Path) -> Dict[str, Any]:
    """One archived usage object. An unreadable archive costs its call's
    detail, never the page."""
    try:
        return usage_metrics(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def attach_vendor_usage(run_dir: Path, records: List[dict]) -> int:
    """Give every record the usage object its own call archived.

    Matched by the artefact name the pipeline wrote, which is the run's key for
    a call: item, model, strategy, and the repetition — the last only when the
    run had more than one, so both naming schemes are tried. Returns how many
    records found their archive; a run older than the archive, and every mock
    call, simply finds none and says so on the page.
    """
    archive = Path(run_dir) / "raw"
    files = ({p.name[: -len(USAGE_SUFFIX)]: p for p in archive.glob("*" + USAGE_SUFFIX)}
             if archive.is_dir() else {})
    cache: Dict[str, Dict[str, Any]] = {}

    found = 0
    for record in records:
        record[VENDOR_USAGE_FIELD] = {}
        if not files:
            continue
        name = next((n for n in call_stems(record) if n in files), None)
        if name is None:
            continue
        if name not in cache:
            cache[name] = read_usage_file(files[name])
        if cache[name]:
            record[VENDOR_USAGE_FIELD] = dict(cache[name])
            found += 1
    return found


# ── one token table per vendor ────────────────────────────────────────────────
# The comparison table cannot be one table any more. Its columns were the
# classes this project prices, which is the only vocabulary all five vendors
# share — but a vendor's own accounting is finer than that shared vocabulary
# and differs from every other vendor's, so a single grid can hold either every
# vendor or every figure, not both. It now holds every figure: one table per
# vendor, one row per LLM of that vendor, and one column per token quantity the
# run actually holds for it.
#
# The token classes this project prices or derives, in reading order. The
# `reported_*` columns and `tool_use_prompt_tokens` are deliberately absent:
# they are the vendor's own figures carried under this project's names, and the
# vendor's own figures are the other half of the same table.
# A vendor's own field name can collide with this project's name for a class
# without meaning the same thing — OpenAI reports an `input_tokens` that
# *includes* its cache reads, where this project's `input_tokens` excludes
# them. The two must never share a key, or one column would print the other's
# figure and the difference between them would be exactly what is hidden.
NATIVE_PREFIX = "native:"

# The classes this project prices. They are **not** the columns of a vendor
# table: a table lists what that vendor's own API reported, and for four of the
# five vendors these classes are the same numbers again under this project's
# names (verified against a live run: only Anthropic's derived total and
# Gemini's thought-inclusive output differ, and only once caching or reasoning
# is in play). They stay here for the one case where nothing else exists — a
# mock run, or a run made before the usage archive — where they are the only
# token figures the run holds. Cost is still computed from them; the per-call
# breakdown in a fold-out is what shows that.
PROJECT_TOKEN_FIELDS = [
    ("input_tokens", "input (uncached)", "down"),
    ("cached_tokens", "cache read", "none"),
    ("cache_write_tokens", "cache write", "none"),
    ("cache_write_1h_tokens", "cache write 1h", "none"),
    ("output_tokens", "output", "down"),
    ("thinking_tokens", "of which reasoning", "none"),
    ("billable_input_tokens", "billable input", "down"),
    ("total_tokens", "total", "down"),
]


def _vendor_label(vendor: str) -> str:
    """The vendor's own spelling of its name, for a heading."""
    return (TRACKABLE_VENDORS.get(vendor)
            or UNTRACKABLE_VENDORS.get(vendor) or vendor or "unknown vendor")


def money(value: Any) -> str:
    """A cost, at the precision it actually has. Costs here run from cents to
    millionths of a cent, so a fixed number of decimals either hides the small
    ones or pads the large ones with noise. Mirrors `usd()` in the page's
    script, which formats the same figures in the fold-outs."""
    if missing(value):
        return '<span class="na">not computed</span>'
    amount = float(value)
    if amount == 0:
        return "$0"
    if amount < 0.000001:
        return "&lt;$0.000001"
    return f"${amount:.6f}" if amount < 0.01 else f"${amount:.4f}"


def vendor_token_columns(records: List[dict]) -> Dict[str, Dict[str, Any]]:
    """Which token columns this run holds, per vendor.

    A column appears when at least one call of that vendor carried a value for
    it. A class the vendor never reports would otherwise be a column of "not
    computed" in every row — and columns differing per vendor is the whole
    reason the table is split by vendor in the first place.

    Computed over the *whole* run, so a vendor's table has the same columns in
    every strategy: a column set that changed with the rows would make two
    tables of the same vendor unreadable side by side.
    """
    vendors: Dict[str, Dict[str, Any]] = {}
    for record in records:
        vendor = str(record.get("vendor") or "")
        entry = vendors.setdefault(vendor, {"project": set(), "native": {}, "models": {}})
        for key, _label, _direction in PROJECT_TOKEN_FIELDS:
            if not missing(record.get(key)):
                entry["project"].add(key)
        for name, value in (record.get(VENDOR_USAGE_FIELD) or {}).items():
            if value is not None:
                entry["native"].setdefault(name, None)
        entry["models"].setdefault(str(record.get("model_key")),
                                   str(record.get("model_id") or ""))

    return {vendor: {"project": [field for field in PROJECT_TOKEN_FIELDS
                                 if field[0] in entry["project"]],
                     "native": list(entry["native"]),
                     "models": entry["models"]}
            for vendor, entry in sorted(vendors.items())}


def vendor_token_rows(records: List[dict], vendor: str, columns: Dict[str, Any],
                      filter_column: Optional[str] = None,
                      filter_value: Optional[str] = None) -> List[Dict[str, Any]]:
    """One row per LLM of `vendor`: every column's mean over that model's calls.

    The mean is over the calls that *have* the figure, and `n` is beside it, so
    a column only some calls reported cannot pass for a full sample. A model
    with no call in scope is dropped rather than shown as a row of blanks.
    """
    rows = []
    for model_key in sorted(columns["models"]):
        calls = [r for r in records
                 if str(r.get("model_key")) == model_key
                 and str(r.get("vendor") or "") == vendor
                 and (filter_column is None
                      or str(r.get(filter_column)) == str(filter_value))]
        if not calls:
            continue
        values: Dict[str, Any] = {}
        readers = [(key, lambda call, key=key: call.get(key))
                   for key, _label, _direction in columns["project"]]
        readers += [(NATIVE_PREFIX + name,
                     lambda call, name=name: (call.get(VENDOR_USAGE_FIELD) or {}).get(name))
                    for name in columns["native"]]
        for key, read in readers:
            numbers = [float(read(call)) for call in calls if not missing(read(call))]
            values[key] = statistics.fmean(numbers) if numbers else None
        costs = [float(c["cost_usd"]) for c in calls if not missing(c.get("cost_usd"))]
        values["n_calls"] = len(calls)
        values["n_usage"] = sum(1 for c in calls if c.get(VENDOR_USAGE_FIELD))
        values["cost_mean"] = statistics.fmean(costs) if costs else None
        values["cost_sum"] = sum(costs) if costs else None
        rows.append({"model_key": model_key,
                     "model_id": columns["models"][model_key],
                     "values": values})
    return rows


def _column_groups(vendor: str, columns: Dict[str, Any]) -> List[tuple]:
    """The table's columns, grouped the way the header row groups them:
    (group title, [(key, label, direction, format)])."""
    sample = [("n_calls", "n", "none", "int")]
    if columns["native"]:
        sample.append(("n_usage", "usage", "none", "int"))
    groups = [("calls", sample)]
    if columns["native"]:
        # The vendor's own figures, and nothing beside them: this project's
        # classes would print most of the same numbers a second time under
        # different names.
        groups.append((f"as {_vendor_label(vendor)} reported it",
                       [(NATIVE_PREFIX + name, name, "none", "count")
                        for name in columns["native"]]))
    elif columns["project"]:
        # No usage object was archived for this vendor — a mock run, or a run
        # older than the archive. The classes this project records are then the
        # only token figures that exist, so they stand in rather than leaving a
        # table of nothing but call counts.
        groups.append(("token classes recorded (no usage object archived)",
                       [(key, label, direction, "count")
                        for key, label, direction in columns["project"]]))
    groups.append(("cost", [("cost_mean", "$ / call", "down", "money"),
                            ("cost_sum", "$ total", "down", "money")]))
    return groups


def _vendor_token_table(vendor: str, columns: Dict[str, Any], rows: List[Dict[str, Any]],
                        filter_column: Optional[str] = None,
                        filter_value: Optional[str] = None) -> str:
    """One vendor's token table: LLMs down the side, its own quantities across.

    Rows unfold into their calls like every other aggregate row. Where the
    table is one strategy's, the fold-out has to filter on that strategy as
    well, which the two `data-filter-*` attributes tell the script to do.
    """
    if not rows:
        return ""
    groups = _column_groups(vendor, columns)

    head = ['<th class="rowhead">Model</th>']
    sub_head = ['<th></th>']
    for title, fields in groups:
        head.append(f'<th class="num group" colspan="{len(fields)}">{esc(title)}</th>')
        for index, (key, label, _direction, _fmt) in enumerate(fields):
            edge = " group-start" if index == 0 else ""
            # A vendor's field name is a long identifier and has to be allowed
            # to wrap, or one column stretches the table past the screen.
            kind = " fieldhead" if key.startswith(NATIVE_PREFIX) else ""
            # `stat-sum` is what the stylesheet accents: in a tokens table the
            # run total is the figure that gets quoted.
            kind += " stat-sum" if key == "cost_sum" else ""
            sub_head.append(f'<th class="num substat{kind}{edge}">{esc(label)}</th>')

    n_columns = 1 + sum(len(fields) for _title, fields in groups)
    body = []
    for row in rows:
        cells = [_rowhead(row["model_key"], row["values"].get("n_calls"), "cost")]
        for _title, fields in groups:
            for index, (key, _label, direction, fmt) in enumerate(fields):
                value = row["values"].get(key)
                column = [other["values"].get(key) for other in rows]
                share = _share(column, value, direction)
                text = (f'{int(value)}' if fmt == "int" and not missing(value)
                        else money(value) if fmt == "money" else count(value, 1))
                klass = ("sample" if fmt == "int" else
                         "cell-scale" if share is not None else "cell-flat")
                edge = " group-start" if index == 0 else ""
                edge += " stat-sum" if key == "cost_sum" else ""
                style = f' style="--w:{share:.3f}"' if share is not None else ""
                cells.append(f'<td class="num {klass}{edge}"{style}>{text}</td>')
        body.append(f'<tr class="aggrow" data-key="{esc(row["model_key"])}">'
                    f'{"".join(cells)}</tr>'
                    f'<tr class="detailrow" hidden>'
                    f'<td colspan="{n_columns}"><div class="detail"></div></td></tr>')

    filters = ""
    if filter_column is not None:
        filters = (f' data-filter-group="{esc(filter_column)}" '
                   f'data-filter-key="{esc(filter_value)}"')
    return (f'<div class="tablewrap"><table class="agg vendortable" '
            f'data-group="model_key" data-block="cost"{filters}>'
            f'<thead><tr>{"".join(head)}</tr>'
            f'<tr class="substats">{"".join(sub_head)}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _share(column: List[Any], value: Any, direction: str) -> Optional[float]:
    """How dark this cell is: 0 = worst value in its column, 1 = best.

    Only where the metric has a direction, and only against the other rows of
    the same table — never an absolute scale, and never for a vendor's own
    field, whose direction this project has no basis to declare.
    """
    numbers = [float(v) for v in column if not missing(v)]
    if direction == "none" or missing(value) or len(numbers) < 2:
        return None
    low, high = min(numbers), max(numbers)
    if high == low:
        return None
    share = (float(value) - low) / (high - low)
    return 1 - share if direction == "down" else share


def _vendor_sections(records: List[dict], columns_by_vendor: Dict[str, Dict[str, Any]],
                     tag: str = "h5", filter_column: Optional[str] = None,
                     filter_value: Optional[str] = None) -> str:
    """A table per vendor, each under its own heading."""
    out = []
    for vendor, columns in columns_by_vendor.items():
        rows = vendor_token_rows(records, vendor, columns, filter_column, filter_value)
        table = _vendor_token_table(vendor, columns, rows, filter_column, filter_value)
        if not table:
            continue
        calls = sum(row["values"]["n_calls"] for row in rows)
        meta = f"{len(rows)} model(s) · {calls} call(s)"
        out.append(f'<div class="vendorsection"><{tag}>{esc(_vendor_label(vendor))}'
                   f'<span class="usagemeta">{esc(meta)}</span></{tag}>{table}</div>')
    return "".join(out)


def _vendor_tables_by_model(records: List[dict]) -> str:
    """The by-LLM tokens table, split into one table per vendor."""
    columns_by_vendor = vendor_token_columns(records)
    sections = _vendor_sections(records, columns_by_vendor)
    if not sections:
        return '<p class="na">No rows for this view.</p>'
    note = ('<p class="note usagenote"><b>One table per vendor, one row per LLM.</b> '
            'The columns are the token figures <b>that vendor\'s own API returned</b>, '
            'under its own names — that is why this is one table each and not one grid: '
            'the five APIs do not report the same quantities, and nothing here lines up '
            'with the table above or below it. Every entry is the mean over all of that '
            'model\'s calls in this run; the <code>n</code> column beside the name is the '
            'sample it is a mean of, and <code>usage</code> how many of those calls '
            'carried a usage object at all. A field the vendor never filled has no column '
            '— it would be blank in every row. Only <b>cost</b> is shaded: whether '
            'more of a vendor-specific quantity is better or worse is not something '
            'this project can declare. Cost is computed from the token classes this '
            'project prices, not from these columns; the breakdown of a call into '
            'those classes is in its fold-out.</p>')
    return (f'<div class="vendorblock">{note}{sections}</div>')


def _strategy_total_cost(records: List[dict], strategy: str) -> str:
    """What this prompting strategy cost over the whole run, in one figure.

    The tables under the heading are means per model, one vendor at a time — no
    cell anywhere adds a strategy up, and doing it by eye across vendors is
    exactly what a reader would get wrong. A call the run could not price (an
    unverified rate, a failed request) is not counted as zero: the heading says
    how many such calls there are, because the sum is then a lower bound and
    reading it as the price of the strategy would understate it.
    """
    costs = [r.get("cost_usd") for r in records if str(r.get("strategy")) == strategy]
    priced = [float(c) for c in costs if not missing(c)]
    if not priced:
        return ('<span class="strategycost">(Total Cost: '
                '<span class="na">not computed</span>)</span>')
    unpriced = len(costs) - len(priced)
    note = f" — {unpriced} of {len(costs)} calls carry no price" if unpriced else ""
    return (f'<span class="strategycost">(Total Cost: {money(sum(priced))}'
            f'{esc(note)})</span>')


def _vendor_tables_by_strategy(records: List[dict]) -> str:
    """The same tables once per prompting strategy: what one prompt costs at
    one vendor, model by model, and what the strategy cost in total."""
    columns_by_vendor = vendor_token_columns(records)
    strategies = sorted({str(r.get("strategy")) for r in records if r.get("strategy")})
    blocks = []
    for strategy in strategies:
        sections = _vendor_sections(records, columns_by_vendor, tag="h6",
                                    filter_column="strategy", filter_value=strategy)
        if sections:
            blocks.append(f'<div class="strategygroup"><h5 class="strategyhead">'
                          f'{esc(strategy)}'
                          f'{_strategy_total_cost(records, strategy)}</h5>'
                          f'{sections}</div>')
    if not blocks:
        return ""
    note = ('<p class="note usagenote"><b>Each strategy, vendor by vendor.</b> '
            'One block per prompting strategy, and inside it one table per vendor: '
            'the columns are the token figures that '
            'one vendor reports, the rows are that vendor\'s models, and every entry is '
            'the mean over the calls this strategy made with that model. The figure '
            'beside the strategy name is the other direction: <b>every call that '
            'strategy made in this run, added up</b> — no table here holds that '
            'sum. Reading one '
            'strategy\'s table against the same vendor\'s table under another strategy '
            'is what shows what a prompt costs at that vendor. Across two vendors only '
            'the cost columns can be read against each other — the token columns are the '
            'vendors\' own and describe different quantities.</p>')
    return f'<div class="vendorblock">{note}{"".join(blocks)}</div>'


def json_safe(records: List[dict]) -> None:
    """Replace every non-finite float with None, in place.

    `json.dumps` writes NaN and Infinity as bare `NaN` / `Infinity` literals,
    which Python reads back happily and **no browser will parse** — JSON has no
    such values. One of them in the embedded dataset makes `JSON.parse` throw at
    page load, which aborts the whole script: the fold-outs then do nothing at
    all, silently, on an otherwise perfect-looking page.

    They are unavoidable upstream: `DataFrame.where(..., None)` leaves NaN in a
    float column, because assigning None to one coerces straight back to NaN.
    """
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None


def efficiency_metric_spec() -> Dict[str, List[dict]]:
    """What an unfolded call shows under *Token Usage & Cost* and under *Latency*.

    An unfolded row belongs to the table it was opened from, so it shows that
    table's subject and nothing else — quality metrics under a cost table are
    noise the reader has to look past. Both blocks go beyond their aggregate
    columns, because per call the provenance of a figure matters: a cost is only
    interpretable next to the price basis that produced it, and a latency only
    next to the number of attempts it took.
    """
    return {
        "cost": [
            # `money` turns this block into three columns: label, token count,
            # and what those tokens cost. A class with no rate of its own —
            # a subtotal, or a subset of another class — carries no cost cell
            # rather than a number that would be counted twice.
            {"title": "Tokens", "headline": "total_tokens", "money": True,
             "total": {"label": "Total", "key": "total_tokens", "cost": "cost_usd"},
             "metrics": [
                {"key": "input_tokens", "label": "input (uncached)",
                 "cost": COST_FIELD_PREFIX + "input_tokens"},
                {"key": "cached_tokens", "label": "cache read",
                 "cost": COST_FIELD_PREFIX + "cached_tokens"},
                {"key": "cache_write_tokens", "label": "cache write",
                 "cost": COST_FIELD_PREFIX + "cache_write_tokens"},
                {"key": "cache_write_1h_tokens", "label": "cache write 1h",
                 "cost": COST_FIELD_PREFIX + "cache_write_1h_tokens"},
                {"key": "output_tokens", "label": "output",
                 "cost": COST_FIELD_PREFIX + "output_tokens"},
                # Subtotals, subsets and raw vendor figures — priced above, or
                # not a priced class at all. No cost cell, so nothing is
                # counted twice.
                {"key": "reported_input_tokens", "label": "input as the vendor reported it"},
                {"key": "reported_total_tokens", "label": "total as the vendor reported it"},
                {"key": "reported_cache_creation_tokens",
                 "label": "cache writes as the vendor reported them"},
                {"key": "reported_output_tokens", "label": "output as the vendor reported it"},
                {"key": "tool_use_prompt_tokens", "label": "tool-use prompts (Gemini)"},
                {"key": "billable_input_tokens", "label": "billable input (subtotal)"},
                {"key": "thinking_tokens", "label": "of which reasoning"},
             ]},
            {"title": "Cost", "headline": "cost_usd", "metrics": [
                {"key": "cost_usd", "label": "cost USD"},
                {"key": "cost_basis", "label": "price basis"},
                {"key": "service_tier", "label": "service tier"},
                {"key": "web_search_requests", "label": "web search requests"},
                {"key": "audio_input_seconds", "label": "audio input seconds"},
                {"key": "token_reporting", "label": "token reporting"},
                # The guard that makes the cost figure trustworthy: anything the
                # vendor billed but the model does not price shows up here.
                # Both of these are sentences, not numbers — `prose` gives them
                # the full width of the card instead of the narrow value column,
                # where a long string wraps one character per line.
                {"key": "untracked_usage", "label": "untracked usage", "prose": True},
                {"key": "reasoning_note", "label": "reasoning", "prose": True},
            ]},
        ],
        "latency": [
            {"title": "Latency", "headline": "latency_s", "metrics": [
                {"key": "latency_s", "label": "total (s)"},
                {"key": "api_latency_s", "label": "answering attempt (s)"},
                {"key": "retry_wait_s", "label": "spent in backoff (s)"},
                {"key": "call_wall_s", "label": "wall clock (s)"},
                {"key": "api_attempts", "label": "attempts"},
                {"key": "stop_reason", "label": "stop reason"},
            ]},
        ],
    }


# Vendors whose API returns an identifier this project can record. DeepSeek was
# listed as untrackable while it was served by the mock provider; since it went
# live on 2026-08-18 it returns a completion id on every call like the rest, so
# leaving it labelled "not trackable" would hide data the run actually holds.
# UNTRACKABLE_VENDORS stays in place for the next vendor that does not.
TRACKABLE_VENDORS = {"anthropic": "Anthropic", "deepseek": "DeepSeek",
                     "google": "Google", "mistral": "Mistral", "openai": "OpenAI"}
UNTRACKABLE_VENDORS: Dict[str, str] = {}

REQUEST_IDS_FILE = "request_ids.html"


# ── O6: the conditions one call was produced under ────────────────────────────
# The request-id table is the run's index of vendor-side calls, which makes it
# the place to answer "what produced this row?". Every entry unfolds into the
# record `pipeline._write_provenance` wrote beside the reply: the input, the
# prompt and its template version, the model, the generation parameters, the
# timestamps, the reply verbatim, the dated price list the cost was computed
# with, and the measurements derived from all of it.
#
# What that buys is repeatability of the *conditions*: the same configuration
# can be sent again exactly as it was. It is not a promise of the same text —
# generation is stochastic, and the same prompt to the same model at the same
# settings may answer differently. The page says so where the fold-outs start.
#
# Long texts are embedded up to a limit and linked in full. A run of a few
# hundred calls carries several megabytes of prompt and reply, and this page is
# read in a browser; the complete text is one click away in the run directory,
# which is where it has to be anyway.
PROMPTS_DIR = "prompts"
DETAIL_INPUT_LIMIT = 4000
DETAIL_TURN_LIMIT = 2500
# The exemplar turns of a few-shot prompt are the same text in every call of
# that strategy, so they are abbreviated harder than the turn that carries the
# item: repeating two reference models per call is what would make this page
# tens of megabytes, and they are one click away in the record itself.
DETAIL_EXEMPLAR_LIMIT = 700
DETAIL_RAW_LIMIT = 6000


def call_stems(record: Dict[str, Any]) -> List[str]:
    """The artefact name(s) the pipeline may have written this call under.

    The repetition is part of the name only when the run had more than one, so
    both schemes are tried — the repetition-bearing one first, because a run
    with repetitions also holds rows whose bare name belongs to another call.
    """
    from pipeline import _safe as safe_name   # the function that named the files

    base = (f"{record.get('item_id')}__{record.get('model_key')}"
            f"__{record.get('strategy')}")
    stems = []
    repetition = record.get("repetition")
    if not missing(repetition):
        stems.append(safe_name(f"{base}__r{int(repetition):02d}"))
    stems.append(safe_name(base))
    return stems


def read_provenance(run_dir: Path, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The provenance record written for one call, or None if it has none.

    None is a fact about the run, not an error: runs made before that record
    existed archive their reply and their vendor usage but not the prompt that
    produced them, and the fold-out says so rather than showing empty fields.
    """
    named = str(record.get("prompt_file") or "").strip()
    candidates = [Path(run_dir) / named] if named else []
    candidates += [Path(run_dir) / PROMPTS_DIR / f"{stem}.json"
                   for stem in call_stems(record)]
    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


def read_raw_output(run_dir: Path, record: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """`(text, relative path)` of the reply archived for one call.

    The text is None when no file was found; the path comes back whenever one
    was, so the fold-out can link a file it could not read.
    """
    named = str(record.get("raw_output_file") or "").strip()
    candidates = [Path(run_dir) / named] if named else []
    candidates += [Path(run_dir) / "raw" / f"{stem}.txt" for stem in call_stems(record)]
    for path in candidates:
        if path.is_file():
            rel = path.relative_to(Path(run_dir)).as_posix()
            try:
                return path.read_text(encoding="utf-8"), rel
            except OSError:
                return None, rel
    return None, ""


def _rel_link(rel: Optional[str], label: Optional[str] = None) -> str:
    """A link to a file inside the run directory. This page sits in that
    directory, so the relative path is also the href."""
    if not rel:
        return ""
    href = str(rel).replace("\\", "/")
    return f'<a class="file" href="{esc(href)}">{esc(label or href)}</a>'


def _mono(value: Any) -> str:
    return f'<span class="mono">{esc(value)}</span>' if value else ""


def _facts(pairs: List[tuple]) -> str:
    """A fact list. A value of None is left out — this run cannot have it — and
    an empty one reads as unrecorded rather than as a blank line. Values are
    HTML: the caller escapes."""
    items = []
    for label, value in pairs:
        if value is None:
            continue
        body = value if str(value).strip() else '<span class="na">not recorded</span>'
        items.append(f"<dt>{esc(label)}</dt><dd>{body}</dd>")
    return f'<dl class="facts">{"".join(items)}</dl>' if items else ""


def _grouped(value: Any) -> str:
    return f"{int(value):,}".replace(",", GROUP_SEP)


def _text_block(text: Optional[str], limit: int, link: str = "",
                empty: str = "not recorded", show_link: bool = True) -> str:
    """A verbatim text, capped at `limit` characters and linked in full.

    `show_link=False` keeps the link for the case that needs it — the text was
    cut and the reader has to go somewhere for the rest — and drops it when the
    text is complete, so a prompt whose file is already named once above does
    not repeat that line under every turn.
    """
    if text is None:
        return f'<p class="na">{esc(empty)}</p>'
    if not text.strip():
        return '<p class="na">empty — the file holds no text</p>'
    shown, note = text, ""
    if len(text) > limit:
        shown = text[:limit]
        note = (f'<p class="cut">first {_grouped(limit)} of {_grouped(len(text))} '
                f'characters shown'
                + (f' — full text in {_rel_link(link)}' if link else "") + "</p>")
    elif link and show_link:
        note = f'<p class="cut">file: {_rel_link(link)}</p>'
    return f'<pre class="text">{esc(shown)}</pre>{note}'


def _yesno(value: Any, yes: str = "yes", no: str = "no") -> str:
    if missing(value) or value == "":
        return ""
    if isinstance(value, str):
        return yes if value.strip().lower() in ("true", "yes", "1") else no
    return yes if value else no


def _pricing_info(record: Dict[str, Any], prov: Optional[Dict[str, Any]],
                  manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The price list this one call was costed against.

    Read from the call's own record first, because the rate that applied is a
    property of when the call was made — an introductory window that has since
    closed, an off-peak hour — and only falls back to the run manifest for a run
    made before that record existed. The per-class rates are re-derived from the
    basis the row itself carries, never from today's catalogue.
    """
    pricing = dict((prov or {}).get("pricing") or {})
    run_pricing = (manifest or {}).get("pricing") or {}
    if not pricing:
        applied = (run_pricing.get("rates_applied") or {}).get(
            str(record.get("model_key")), {})
        sheet = (run_pricing.get("sheets") or {}).get(str(record.get("vendor")), {})
        pricing = {
            "input_per_mtok": (record.get("price_in_per_mtok")
                               if not missing(record.get("price_in_per_mtok"))
                               else applied.get("input_per_mtok")),
            "output_per_mtok": (record.get("price_out_per_mtok")
                                if not missing(record.get("price_out_per_mtok"))
                                else applied.get("output_per_mtok")),
            "basis": record.get("cost_basis") or applied.get("basis") or "",
            "price_verified": record.get("price_verified"),
            "sheet_as_of": record.get("pricing_as_of") or sheet.get("as_of"),
            "sheet_source": sheet.get("source") or run_pricing.get("source"),
            "priced_on": run_pricing.get("priced_on"),
        }
    if not pricing.get("rates_per_mtok"):
        import config as cfg
        try:
            model = cfg.model_by_key(str(record.get("model_key") or ""))
        except KeyError:
            model = None
        if model is not None:
            pricing["rates_per_mtok"] = cfg.rates_for(
                model, str(record.get("cost_basis") or pricing.get("basis") or ""))
    return pricing


def _price_table(record: Dict[str, Any], pricing: Dict[str, Any]) -> str:
    """Tokens × rate = cost, one line per priced class, as the cost was computed.

    The rates are the ones the price list above resolved; the amounts are the
    breakdown `add_cost_breakdown` already attached to the record, so this shows
    what was charged rather than a second calculation of it.
    """
    rates = pricing.get("rates_per_mtok") or {}
    labels = {"input_tokens": "input", "cached_tokens": "cache read",
              "cache_write_tokens": "cache write",
              "cache_write_1h_tokens": "cache write (1h)",
              "output_tokens": "output"}
    lines = []
    for field in PRICED_TOKEN_FIELDS:
        tokens = record.get(field)
        rate = rates.get(field)
        amount = record.get(COST_FIELD_PREFIX + field)
        if missing(tokens) and missing(rate):
            continue
        rate_cell = "" if missing(rate) else f"${float(rate):.4f}"
        lines.append(
            f"<tr><td>{esc(labels.get(field, field))}</td>"
            f'<td class="figure">{count(tokens)}</td>'
            f'<td class="figure">{rate_cell}</td>'
            f'<td class="figure">{money(amount)}</td></tr>')
    if not lines:
        return ""
    # A mock call was never billed, so its recorded total is 0 and the classes
    # above cannot match it. That is not the mismatch `add_cost_breakdown` warns
    # about — it is the offline stub — and saying so is clearer than letting the
    # generic note blame a server-side tool.
    note = ("mock provider — nothing was billed; the lines above show what these "
            "rates would have cost" if record.get("is_mock")
            else str(record.get("cost_breakdown_note") or ""))
    total = ('<tr class="sumline"><td>total</td><td class="figure"></td>'
             '<td class="figure"></td>'
             f'<td class="figure">{money(record.get("cost_usd"))}</td></tr>')
    return ('<table class="mini"><thead><tr><th>token class</th>'
            '<th class="figure">tokens</th><th class="figure">$/Mtok</th>'
            '<th class="figure">cost</th></tr></thead>'
            f'<tbody>{"".join(lines)}{total}</tbody></table>'
            + (f'<p class="cut">{esc(note)}</p>' if note else ""))


def _detail_panel(run_dir: Path, record: Dict[str, Any],
                  prov: Optional[Dict[str, Any]], manifest: Dict[str, Any]) -> str:
    """Everything O6 asks a run to store about one generation, in that order:
    the input, the prompt and its version, the model, the generation parameters,
    the timestamps, the raw output, the dated price list, and the measurements
    derived from them."""
    cards = []
    prompt = (prov or {}).get("prompt") or {}
    inp = (prov or {}).get("input") or {}
    params = (prov or {}).get("parameters") or {}
    settings = manifest.get("settings") or {}

    if prov is None:
        cards.append(
            '<section class="card wide missing"><h4>Prompt and input not archived</h4>'
            '<p>This run was made before the pipeline wrote a record per generation, '
            'so the prompt that went out and the description it was built from are not '
            'in the run directory. Everything below comes from the result row, the '
            'reply archive and the run manifest.</p></section>')

    # 1 — the input
    # A run made before the record existed archived no description, but it did
    # record the reference model it was scored against, and the description is
    # that file's pair in the dataset. Read back rather than left blank — and
    # labelled, because it is the file as it stands today, not a copy taken when
    # the call was made.
    input_text = inp.get("text")
    recovered = ""
    if input_text is None and record.get("ground_truth_path"):
        from dataset import description_path_for, read_description
        try:
            path = description_path_for(Path(str(record["ground_truth_path"])))
            if path.is_file():
                input_text = read_description(path)
                recovered = ('<p class="cut">read back from the dataset just now — '
                             'this run archived no copy, so this is the file as it '
                             'stands today</p>')
        except (OSError, ValueError):
            pass
    cards.append(
        '<section class="card wide"><h4>Input</h4>'
        + _facts([
            ("item", esc(record.get("item_id"))),
            ("sha-256", _mono(inp.get("sha256") or record.get("input_sha256") or "")),
        ])
        + _text_block(input_text, DETAIL_INPUT_LIMIT,
                      empty="the description text was not archived with this run")
        + recovered
        + "</section>")

    # 2 — the prompt, and the version of the template behind it
    turns = []
    if prompt.get("system"):
        turns.append('<div class="turn"><span class="role">system</span>'
                     + _text_block(prompt.get("system"), DETAIL_TURN_LIMIT)
                     + "</div>")
    messages = prompt.get("messages") or []
    for i, message in enumerate(messages, start=1):
        # The last turn is the item this call was made for; everything before it
        # is exemplar material shared with every other call of this strategy.
        limit = DETAIL_TURN_LIMIT if i == len(messages) else DETAIL_EXEMPLAR_LIMIT
        turns.append(f'<div class="turn"><span class="role">{esc(message.get("role"))}'
                     f' · turn {i}{"" if i == len(messages) else " · exemplar"}</span>'
                     + _text_block(message.get("content"), limit,
                                   link=str(record.get("prompt_file") or ""),
                                   show_link=False)
                     + "</div>")
    cards.append(
        '<section class="card wide"><h4>Prompt</h4>'
        + _facts([
            ("strategy", esc(record.get("strategy"))),
            ("template version", _mono(prompt.get("template_version")
                                       or record.get("prompt_template_version") or "")),
            ("prompt sha-256", _mono(prompt.get("sha256")
                                     or record.get("prompt_sha256") or "")),
            ("turns", "" if missing(prompt.get("n_turns")) else str(prompt.get("n_turns"))),
            ("record", (_rel_link(record.get("prompt_file"))
                        if record.get("prompt_file") else None)),
        ])
        + ("".join(turns) if turns
           else '<p class="na">the prompt was not archived with this run</p>')
        + "</section>")

    # 3 — the model identifier, and 4 — the parameters it ran under
    temperature = params.get("temperature", settings.get("temperature"))
    max_output = params.get("max_output_tokens", settings.get("max_output_tokens"))
    n_few_shot = params.get("n_few_shot", settings.get("n_few_shot"))
    cards.append(
        '<section class="card"><h4>Model and parameters</h4>'
        + _facts([
            ("vendor", esc(record.get("vendor"))),
            ("model id", _mono(record.get("model_id"))),
            ("model key", _mono(record.get("model_key"))),
            ("tier", esc(record.get("model_tier"))),
            ("served by", esc(record.get("provider"))
             + (' <span class="warn">mock — not a measurement</span>'
                if record.get("is_mock") else "")),
            ("temperature", ("not supported by this model"
                             if params.get("temperature_supported") is False
                             else ("vendor default" if missing(temperature)
                                   else esc(temperature)))),
            ("max output tokens", "" if missing(max_output) else _grouped(max_output)),
            ("reasoning", _yesno(params.get("thinking"), "requested", "not requested")
             + (f' · effort {esc(params.get("effort"))}' if params.get("effort") else "")
             + (" · switched off" if params.get("reasoning_off") else "")),
            ("few-shot exemplars", "" if missing(n_few_shot) else str(n_few_shot)),
            ("repetition", "" if missing(record.get("repetition"))
             else f'{int(record.get("repetition"))} of '
                  f'{esc(manifest.get("repetitions", "?"))}'),
            ("models in flight", "" if missing(record.get("concurrent_models"))
             else str(int(record.get("concurrent_models")))),
            ("service tier", esc(record.get("service_tier") or "")),
        ])
        + "</section>")

    # 5 — when it happened
    cards.append(
        '<section class="card"><h4>Timestamps</h4>'
        + _facts([
            ("request sent", _mono(record.get("request_sent_at") or "")),
            ("response received", _mono(record.get("response_received_at") or "")),
            ("round trip", "" if missing(record.get("api_latency_s"))
             else f'{num(record.get("api_latency_s"), 3)} s'),
            ("incl. retries", "" if missing(record.get("latency_s"))
             else f'{num(record.get("latency_s"), 3)} s'),
            ("attempts", "" if missing(record.get("api_attempts"))
             else str(int(record.get("api_attempts")))),
            ("spent in backoff", "" if missing(record.get("retry_wait_s"))
             else f'{num(record.get("retry_wait_s"), 3)} s'),
            ("wall clock", "" if missing(record.get("call_wall_s"))
             else f'{num(record.get("call_wall_s"), 3)} s'),
            ("stop reason", esc(record.get("stop_reason") or "")),
            ("run started", esc(manifest.get("created_at") or "")),
            ("error", (f'<span class="warn">{esc(record.get("generation_error"))}</span>'
                       if record.get("generation_error") else None)),
        ])
        + "</section>")

    # 6 — what came back, verbatim
    raw_text, raw_rel = read_raw_output(run_dir, record)
    usage_rel = ((prov or {}).get("artefacts") or {}).get("vendor_usage")
    cards.append(
        '<section class="card wide"><h4>Raw output</h4>'
        + _facts([
            ("reply archive", _rel_link(raw_rel) if raw_rel else ""),
            ("DOT block extracted", _yesno(record.get("extract_parse_ok"))),
            ("extracted model", (_rel_link(record.get("generated_gv"))
                                 if record.get("generated_gv") else "")),
            ("extraction error", (f'<span class="warn">{esc(record.get("extract_error"))}</span>'
                                  if record.get("extract_error") else None)),
            ("vendor usage", (_rel_link(usage_rel)
                              if usage_rel and (Path(run_dir) / usage_rel).is_file()
                              else None)),
        ])
        + _text_block(raw_text, DETAIL_RAW_LIMIT, link=raw_rel,
                      empty="no reply was archived for this call")
        + "</section>")

    # 7 — the dated price list the cost was computed with
    pricing = _pricing_info(record, prov, manifest)
    cards.append(
        '<section class="card"><h4>Price list</h4>'
        + _facts([
            ("sheet dated", esc(pricing.get("sheet_as_of") or "")),
            ("sheet source", esc(pricing.get("sheet_source") or "")),
            ("priced on", esc(pricing.get("priced_on") or "")),
            ("basis", esc(pricing.get("basis") or record.get("cost_basis") or "")),
            ("rates confirmed", _yesno(pricing.get("price_verified")
                                       if pricing.get("price_verified") is not None
                                       else record.get("price_verified"))),
        ])
        + _price_table(record, pricing)
        + "</section>")

    # 8 — what was derived from all of the above
    cards.append(
        '<section class="card"><h4>Derived measurements</h4>'
        + _facts([
            ("input tokens (uncached)", count(record.get("input_tokens"))),
            ("billable input", count(record.get("billable_input_tokens"))),
            ("output tokens", count(record.get("output_tokens"))),
            ("of which reasoning", count(record.get("thinking_tokens"))),
            ("total tokens", count(record.get("total_tokens"))),
            ("cost", money(record.get("cost_usd"))),
            ("valid DOT", "" if missing(record.get("val_score"))
             else _yesno(float(record.get("val_score")) >= 1.0)),
            ("syntactic (11 checks)", num(record.get("syn_score"), 4)),
            ("syntactic (BEF4LLM)", num(record.get("syn_bef_score"), 4)),
            ("semantic", num(record.get("sem_score"), 4)),
            ("pragmatic", num(record.get("prag_score"), 4)),
            ("note", esc(record.get("quality_note"))
             if record.get("quality_note") else None),
        ])
        + "</section>")

    return f'<div class="o6">{"".join(cards)}</div>'


def _vendors_used(df: pd.DataFrame) -> List[str]:
    if "vendor" not in df.columns:
        return []
    return sorted({str(v) for v in df["vendor"].dropna().unique()})


def _request_id_links(df: pd.DataFrame) -> str:
    """The footer list: one entry per vendor the run actually used."""
    entries = []
    for vendor in _vendors_used(df):
        if vendor in TRACKABLE_VENDORS:
            entries.append(f'<a href="{REQUEST_IDS_FILE}#{esc(vendor)}">'
                           f'{esc(TRACKABLE_VENDORS[vendor])}</a>')
        elif vendor in UNTRACKABLE_VENDORS:
            entries.append(f'<span class="untrackable">'
                           f'{esc(UNTRACKABLE_VENDORS[vendor])} (not trackable)</span>')
        else:
            entries.append(f'<span class="untrackable">{esc(vendor)} (not trackable)</span>')
    if not entries:
        return ""
    return ('<section class="reqids"><h3>Request-IDs</h3><p>'
            + " · ".join(entries) + "</p></section>")


def build_request_ids(run_dir: Path, df: pd.DataFrame,
                      out_path: Optional[Path] = None) -> Optional[Path]:
    """The companion page: every call's vendor-side identifiers and timestamps,
    each id unfolding into the conditions that produced it.

    One section per vendor used, so the anchors the results footer links to
    resolve. A call whose id was never recorded — an older run, a failed
    request, the mock provider — says so rather than showing a blank cell, and
    still unfolds: the conditions are on file whether or not the vendor named
    the call.
    """
    out_path = out_path or (run_dir / REQUEST_IDS_FILE)
    vendors = [v for v in _vendors_used(df) if v in TRACKABLE_VENDORS]
    if not vendors:
        return None

    manifest: Dict[str, Any] = {}
    manifest_path = Path(run_dir) / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}

    sections = []
    call_no = 0
    for vendor in vendors:
        part = df[df["vendor"].astype(str) == vendor]
        order = [c for c in ("request_sent_at", "item_id", "strategy", "repetition")
                 if c in part.columns]
        if order:
            part = part.sort_values(order)
        # The same per-call records the results page is built from, so a
        # fold-out shows the charge per token class as it was computed rather
        # than computing it a second time.
        records = part.where(pd.notna(part), None).to_dict(orient="records")
        add_cost_breakdown(records)

        rows = []
        for record in records:
            call_no += 1
            panel_id = f"c{call_no}"

            def cell(column: str, mono: bool = False) -> str:
                value = record.get(column)
                if missing(value) or str(value).strip() == "":
                    return '<td class="na">not recorded</td>'
                klass = ' class="mono"' if mono else ""
                return f"<td{klass}>{esc(value)}</td>"

            request_id = record.get("request_id")
            shown = ('<span class="na">not recorded</span>'
                     if missing(request_id) or not str(request_id).strip()
                     else f'<span class="mono">{esc(request_id)}</span>')
            call_label = (f'{esc(record.get("item_id"))} · {esc(record.get("strategy"))}'
                          + (f' · r{esc(record.get("repetition"))}'
                             if not missing(record.get("repetition")) else ""))
            panel = _detail_panel(Path(run_dir), record,
                                  read_provenance(Path(run_dir), record), manifest)
            rows.append(
                '<tbody class="call"><tr class="sum">'
                + cell("request_sent_at", mono=True)
                + '<td class="ridcell"><button type="button" class="rid" '
                  f'aria-expanded="false" aria-controls="{panel_id}">'
                  '<span class="caret" aria-hidden="true">&#9656;</span>'
                  f'{shown}</button></td>'
                + cell("response_id", mono=True)
                + cell("model_id", mono=True)
                + f'<td>{call_label}</td></tr>'
                f'<tr class="det" id="{panel_id}" hidden>'
                f'<td colspan="5">{panel}</td></tr></tbody>')

        recorded = int((part.get("request_id", pd.Series(dtype=object))
                        .fillna("").astype(str).str.strip() != "").sum()) \
            if "request_id" in part.columns else 0
        sections.append(
            f'<section id="{esc(vendor)}">'
            f'<h2>{esc(TRACKABLE_VENDORS[vendor])}</h2>'
            f'<p class="count">{len(part)} call(s), {recorded} with a recorded request id</p>'
            f'<div class="tablewrap"><table>'
            f'<thead><tr><th>Request sent</th><th>Request id</th><th>Response id</th>'
            f'<th>Model</th><th>Call</th></tr></thead>'
            f'{"".join(rows)}</table></div></section>')

    untrackable = [UNTRACKABLE_VENDORS.get(v, v) for v in _vendors_used(df)
                   if v not in TRACKABLE_VENDORS]
    note = ""
    if untrackable:
        note = (f'<p class="note">{esc(", ".join(untrackable))}: the API returns no '
                f'identifier this pipeline can record, so those calls are not listed here.</p>')

    out_path.write_text(_REQUEST_IDS_PAGE.format(
        run_id=esc(run_dir.name), sections="".join(sections), note=note,
        generated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")), encoding="utf-8")
    return out_path


def _stats(totals: Dict[str, Any]) -> str:
    cards = [
        ("Calls", f"{totals['n_calls']}", f"{totals['n_errors']} failed"),
        ("Valid DOT", f"{totals['n_valid']}/{totals['n_calls']}",
         "Graphviz accepted the artefact"),
        ("Extractable", f"{totals['n_parsed']}/{totals['n_calls']}",
         "a DOT block came out of the reply"),
        ("Cost", f"${totals['cost_usd']:.4f}", "billable classes included"),
        ("Tokens", f"{totals['total_tokens']:,}".replace(",", GROUP_SEP),
         "billable input + output"),
        ("API time", f"{totals['api_seconds']:.0f}s", "sum of the round trips"),
    ]
    return "".join(
        f'<div class="stat"><span class="statlabel">{esc(label)}</span>'
        f'<span class="statvalue">{esc(value)}</span>'
        f'<span class="stathint">{esc(hint)}</span></div>'
        for label, value, hint in cards)


def build(run_dir: Path, scored: Optional[pd.DataFrame] = None,
          out_path: Optional[Path] = None) -> Path:
    """Render `results.html` for a run.

    `scored` is the frame from `quality.score_run`; when omitted the run is
    scored here, so the script also works standalone on an older directory.
    """
    run_dir = Path(run_dir)
    out_path = out_path or (run_dir / "results.html")

    if scored is None:
        import quality
        scored = quality.score_run(run_dir, verbose=False)

    totals = agg.totals(scored)

    frames = {"model": agg.by_model(scored), "strategy": agg.by_strategy(scored)}

    # The per-call rows travel with the page so the aggregates can be re-derived.
    # Built before the panels, because the by-LLM tokens panel is built from
    # them: the vendor's own usage fields are not in `results.jsonl`, they are
    # read back from the run's archive and ride along from here on.
    embedded = scored.where(pd.notna(scored), None).to_dict(orient="records")
    add_cost_breakdown(embedded)
    attach_vendor_usage(run_dir, embedded)
    json_safe(embedded)

    panels = []
    for view_id, view_title, group, _hint in VIEWS:
        for block, block_label in TABS:
            # Tokens are the one block that cannot be a single grid: the
            # vendors do not report the same quantities, so this block is the
            # per-vendor tables in both views — by LLM one row per model, by
            # strategy the same tables once per strategy. Every model of the run
            # has a row in one of them.
            if block == "cost" and view_id == "model":
                table = _vendor_tables_by_model(embedded)
            elif block == "cost":
                table = _vendor_tables_by_strategy(embedded)
            else:
                table = _metric_table(frames[view_id], group, block)
            # The heading is invisible on screen (the nav already says which
            # view you are in) and visible in print, where all six tables are
            # laid out one after another and each needs its own label.
            panels.append(
                f'<section class="panel" data-view="{view_id}" data-tab="{block}" hidden>'
                f'<h3 class="panelhead">{view_title} · {block_label}</h3>'
                f'{table}</section>')

    nav = "".join(
        f'<button type="button" class="navbtn" data-view="{v}"'
        f'{" aria-current=\"page\"" if i == 0 else ""}>{t}</button>'
        for i, (v, t, _g, _h) in enumerate(VIEWS))
    hints = "".join(f'<p class="viewhint" data-view="{v}" hidden>{esc(h)}</p>'
                    for v, _t, _g, h in VIEWS)
    # Every block is its own button, side by side and always on screen:
    # nothing is revealed or hidden by choosing one, only the panel below
    # changes.
    tabs = "".join(
        f'<button type="button" class="tabbtn" data-tab="{b}"'
        f'{" aria-pressed=\"true\"" if b == DEFAULT_TAB else ""}>{l}</button>'
        for b, l in TABS)

    build_request_ids(run_dir, scored)

    page = _PAGE.format(
        run_id=esc(run_dir.name), stats=_stats(totals),
        nav=nav, hints=hints, tabs=tabs, panels="".join(panels),
        data=json.dumps(embedded, default=str, allow_nan=False),
        spec=json.dumps(quality_metric_spec()),
        effspec=json.dumps(efficiency_metric_spec()),
        n_calls=totals["n_calls"], reqids=_request_id_links(scored),
        default_tab=DEFAULT_TAB,
        generated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )
    out_path.write_text(page, encoding="utf-8")
    return out_path


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Results {run_id}</title>
<style>
:root {{
  --ground:#f4f6f8; --panel:#fff; --sunken:#eef1f5;
  --ink:#141a21; --ink-soft:#4a5867; --ink-faint:#75838f;
  --line:#d8dfe6; --line-soft:#e8edf2;
  --accent:#0f6b7b; --accent-soft:#d6ebee;
  --accent2:#b4530d; --accent2-soft:#fbe6d3;
  --sans:ui-sans-serif,"Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;
  --mono:ui-monospace,"Cascadia Mono",Consolas,"SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#11161c; --panel:#181f27; --sunken:#141a21;
    --ink:#e7ecf2; --ink-soft:#a9b6c3; --ink-faint:#7d8b99;
    --line:#2a333d; --line-soft:#222a33;
    --accent:#4fb3c4; --accent-soft:#123840;
    --accent2:#f0954a; --accent2-soft:#3a2413;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#11161c; --panel:#181f27; --sunken:#141a21;
  --ink:#e7ecf2; --ink-soft:#a9b6c3; --ink-faint:#7d8b99;
  --line:#2a333d; --line-soft:#222a33;
  --accent:#4fb3c4; --accent-soft:#123840;
  --accent2:#f0954a; --accent2-soft:#3a2413;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
       font:15px/1.55 var(--sans); -webkit-font-smoothing:antialiased; }}
main {{ max-width:1400px; margin:0 auto; padding:36px 26px 80px; }}
h1 {{ font-size:29px; margin:0 0 26px; letter-spacing:-.02em; }}
h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.1em;
     color:var(--ink-soft); margin:34px 0 12px; font-weight:650; }}
.na {{ color:var(--ink-faint); font-style:italic; }}
code,.num,.statvalue {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}

.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
        padding:13px 15px; display:flex; flex-direction:column; gap:3px; }}
.statlabel {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--ink-faint); }}
.statvalue {{ font-size:22px; font-weight:600; }}
.stathint {{ font-size:12px; color:var(--ink-soft); }}

/* navigation */
.nav {{ position:sticky; top:0; z-index:10; display:flex; flex-wrap:wrap; gap:8px;
       align-items:center; padding:12px 0; margin-bottom:6px;
       background:color-mix(in srgb, var(--ground) 92%, transparent); backdrop-filter:blur(8px);
       border-bottom:1px solid var(--line); }}
.navbtn {{ font:inherit; font-size:14px; font-weight:600; cursor:pointer;
          background:none; border:none; border-bottom:2px solid transparent;
          color:var(--ink-soft); padding:8px 4px; margin-right:14px; }}
.navbtn[aria-current] {{ color:var(--accent); border-bottom-color:var(--accent); }}
.navbtn:hover {{ color:var(--ink); }}
/* The buttons live on one line at the right end of the nav, always visible:
   there is no second level to reveal, so nothing on this row ever moves. */
.tabs {{ display:flex; gap:8px; margin-left:auto; align-items:center; }}
.tabbtn {{ font:inherit; font-size:13px; cursor:pointer; color:var(--ink-soft);
          background:var(--panel); border:1px solid var(--line); border-radius:999px;
          padding:6px 14px; }}
.tabbtn:hover {{ color:var(--ink); border-color:var(--accent); }}
.tabbtn[aria-pressed="true"] {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.viewhint {{ color:var(--ink-soft); font-size:13px; margin:10px 0 0; }}

.tablewrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line);
             border-radius:10px; margin-top:14px; }}
table.agg {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
.agg th, .agg td {{ padding:8px 11px; border-bottom:1px solid var(--line-soft); text-align:left; }}
.agg thead th {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em;
                color:var(--ink-faint); font-weight:650; border-bottom:1px solid var(--line); }}
/* One continuous vertical rule per metric group, from the group header all the
   way down the body — so the three statistics of "Input tokens" read as one
   block and cannot be mistaken for the next token kind's. */
.agg th.group {{ text-align:center; border-left:1px solid var(--line); }}
.agg td.group-start, .agg th.substat.group-start {{ border-left:1px solid var(--line); }}
.agg .substats th {{ font-size:10px; padding-top:0; border-bottom:1px solid var(--line);
                    color:var(--ink-faint); text-transform:none; letter-spacing:0; }}
.agg td.num, .agg th.num {{ text-align:right; white-space:nowrap; font-family:var(--mono);
                           font-variant-numeric:tabular-nums; }}
.agg th.rowhead {{ font-weight:600; white-space:nowrap; }}
.agg .sample {{ color:var(--ink-soft); background:var(--sunken); }}
.cell-scale {{ position:relative; }}
.cell-scale::before {{ content:""; position:absolute; inset:2px 3px; border-radius:4px;
  background:var(--accent); opacity:calc(.06 + .32 * var(--w,0)); }}
.cell-scale > * {{ position:relative; }}
/* In the tokens & cost table the run total is the figure that gets quoted, and
   mean/median are the context around it — so the total carries the accent and
   the weight, the other two step back a size. Scoped to that table, because
   latency's third column is a max, not a total. The number itself keeps full ink strength — the shading behind it is
   already an accent wash, and accent on accent stops being readable. */
table.agg[data-block="cost"] .stat-mean,
table.agg[data-block="cost"] .stat-median {{ font-size:11.5px; color:var(--ink-soft); }}
table.agg[data-block="cost"] th.stat-sum {{ color:var(--accent); font-weight:700; }}
table.agg[data-block="cost"] td.stat-sum {{ font-weight:650; color:var(--ink);
  box-shadow:inset 2px 0 0 color-mix(in srgb, var(--accent) 45%, transparent); }}
.cell-flat {{ color:var(--ink-soft); }}
/* `5/8` among the scores: a count, so it keeps the mono figures of the
   columns around it, and carries a little more weight than a plain mean —
   it is the one column here that is not a score. */
td.verdict {{ font-weight:650; }}

/* expandable rows */
.expander {{ font:inherit; font-weight:600; color:inherit; background:none; border:none;
            cursor:pointer; padding:0; display:inline-flex; align-items:center; gap:7px; }}
.expander:hover {{ color:var(--accent); }}
/* What the row is, beside what it is called: lighter and a size down, so the
   key stays the thing you read and the sentence behind it does not compete
   with it. `.expander` is an inline flex row, so its 7px gap does the
   spacing. */
.rowmeta {{ font-weight:400; font-size:11.5px; color:var(--ink-faint);
           white-space:nowrap; }}
.caret {{ display:inline-block; transition:transform .12s ease; color:var(--ink-faint); }}
.expander[aria-expanded="true"] {{ color:var(--accent); }}
.expander[aria-expanded="true"] .caret {{ transform:rotate(90deg); color:var(--accent); }}
.aggrow:has(.expander[aria-expanded="true"]) {{ background:var(--sunken); }}
.detailrow > td {{ padding:0; background:var(--sunken);
                  border-bottom:2px solid var(--line); }}
.detail {{ display:flex; flex-direction:column; gap:10px; padding:14px 16px; }}
.callcard {{ background:var(--panel); border:1px solid var(--line-soft); border-radius:9px; }}
.callhead {{ display:flex; flex-wrap:wrap; gap:6px 16px; align-items:baseline;
            padding:9px 13px; border-bottom:1px solid var(--line-soft); }}
.callid {{ font-family:var(--mono); font-weight:650; font-size:13px; }}
.callchips {{ display:flex; flex-wrap:wrap; gap:6px 12px; font-size:12px; color:var(--ink-soft);
             margin-left:auto; font-family:var(--mono); }}
.callchips b {{ color:var(--ink); font-weight:600; }}
.callblocks {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
              gap:0 18px; padding:11px 13px 13px; }}
.callblock h4 {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.09em;
                color:var(--ink-faint); margin:10px 0 5px; font-weight:650;
                display:flex; gap:8px; align-items:baseline; }}
.blockscore {{ margin-left:auto; font-family:var(--mono); font-size:12px;
              letter-spacing:0; color:var(--accent); font-weight:700; }}
.aside {{ margin-left:7px; color:var(--ink-faint); font-size:11px; }}
.kv {{ display:grid; grid-template-columns:1fr auto; gap:1px 12px; font-size:12.5px; }}
/* Three columns where a block prices its rows: label, count, cost. */
.kv.money {{ grid-template-columns:1fr auto auto; }}
.kv dt {{ color:var(--ink-soft); overflow-wrap:anywhere; }}
/* A vendor's own field name is an identifier, not prose — set like the values
   it labels, so `input_tokens_details.cached_tokens` reads as one token. */
.kv dt.native {{ font-family:var(--mono); font-size:11.5px; }}
/* `min-width` is load-bearing. Without it an `auto` column holding a long
   string collapses to its smallest possible width and `overflow-wrap:anywhere`
   then breaks the text after every character — the value renders vertically.
   Sentences avoid the column entirely via `.prose` below; this is the guard for
   anything long that is still a value. */
.kv dd {{ margin:0; min-width:4.5em; text-align:right; font-family:var(--mono);
         font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
.kv dd.money {{ color:var(--accent2); min-width:5.5em; }}
.kv dd.nocost {{ color:var(--ink-faint); }}
.kv dd.miss {{ color:var(--ink-faint); font-style:italic; font-family:var(--sans); }}
/* A sentence takes the whole card width, under its own label, and wraps on word
   boundaries like prose rather than mid-word like an identifier. */
.kv dt.prose {{ grid-column:1 / -1; margin-top:6px; }}
.kv dd.prose {{ grid-column:1 / -1; min-width:0; text-align:left;
               font-family:var(--sans); color:var(--ink-soft);
               overflow-wrap:break-word; margin-bottom:3px; }}
/* The figure the reader came for. */
.kv dt.grandtotal, .kv dd.grandtotal {{ margin-top:7px; padding-top:6px;
  border-top:1px solid var(--line); font-size:14.5px; font-weight:700;
  color:var(--ink); }}
.kv dd.grandtotal.money {{ color:var(--accent2); }}
.emptydetail {{ color:var(--ink-faint); font-style:italic; padding:12px 4px; }}
.note {{ background:var(--sunken); border-left:3px solid var(--line); padding:10px 13px;
        border-radius:7px; font-size:13px; color:var(--ink-soft); margin-top:14px; }}

/* One token table per vendor. The heading carries the vendor, so the tables
   read as one series rather than as unrelated tables that happen to follow
   each other. */
.vendorblock {{ margin-top:14px; }}
.vendorsection {{ margin-top:18px; }}
.vendorsection h5, .vendorsection h6 {{ margin:0; font-size:14px; font-weight:650;
  display:flex; flex-wrap:wrap; gap:3px 10px; align-items:baseline; }}
.vendorsection h6 {{ font-size:13px; color:var(--ink-soft); }}
.vendorsection .tablewrap {{ margin-top:7px; }}
/* One strategy's vendors form a group, set off by a rule so the eye can find
   where the next strategy starts in a long stack of tables. */
.strategygroup {{ margin-top:26px; padding-top:12px; border-top:2px solid var(--line); }}
.strategyhead {{ margin:0; font-size:12px; text-transform:uppercase; letter-spacing:.1em;
                color:var(--accent2); font-weight:700; font-family:var(--mono);
                display:flex; flex-wrap:wrap; gap:2px 12px; align-items:baseline; }}
/* The one figure on the page that adds a strategy up, so it takes the accent
   rather than the strategy's own colour — and no letter-spacing or capitals,
   which pull a currency amount apart. */
.strategycost {{ color:var(--accent); font-weight:650; letter-spacing:0;
                text-transform:none; }}
/* A vendor's field name is a long identifier. It gets the mono face, a size
   down, and permission to wrap — without it one column stretches the table
   past any screen. */
table.vendortable th.substat.fieldhead {{ white-space:normal; word-break:break-word;
  font-family:var(--mono); font-size:9.5px; max-width:8.5em; line-height:1.25; }}
table.vendortable th.substat {{ vertical-align:bottom; }}
table.vendortable td.num {{ font-size:12.5px; }}

/* The note that introduces the per-vendor tables, and the meta line beside a
   vendor's heading. */
.usagenote {{ margin-top:8px; }}
.usagemeta {{ font-family:var(--mono); font-size:11px; color:var(--ink-faint);
             letter-spacing:0; }}
.reqids {{ margin-top:34px; font-size:11.5px; color:var(--ink-faint); }}
.reqids h3 {{ font-size:11px; text-transform:uppercase; letter-spacing:.09em;
             margin:0 0 4px; font-weight:650; color:var(--ink-faint); }}
.reqids p {{ margin:0; }}
.reqids a {{ color:var(--accent); text-decoration:none; border-bottom:1px solid currentColor; }}
.reqids a:hover {{ color:var(--ink); }}
.untrackable {{ color:var(--ink-faint); }}
footer {{ color:var(--ink-faint); font-size:12.5px; border-top:1px solid var(--line);
         padding-top:15px; margin-top:16px; }}
.panelhead {{ display:none; }}
/* On paper every table goes in, not just the tab that happens to be open. */
@media print {{
  .nav, .viewhint {{ display:none; }}
  .panel[hidden] {{ display:block !important; }}
  .panelhead {{ display:block; font-size:14px; margin:22px 0 0; page-break-after:avoid; }}
  .panel {{ break-inside:auto; }}
  .tablewrap {{ overflow:visible; border-radius:0; }}
  table.agg {{ font-size:9.5pt; }}
  .detailrow {{ display:none; }}
  .callcard, tr {{ break-inside:avoid; }}
  .vendorsection, .strategygroup {{ break-inside:auto; }}
  .vendorsection h5, .vendorsection h6, .strategyhead {{ page-break-after:avoid; }}
  body {{ background:#fff; }}
  main {{ padding:0; max-width:none; }}
  a[href^="request_ids.html"]::after {{ content:" (see request_ids.html)"; color:#666; }}
}}
@page {{ size:A4 landscape; margin:14mm 12mm; }}
</style></head>
<body><main>

<h1>Results {run_id}</h1>
<h2>Totals</h2>
<div class="stats">{stats}</div>

<nav class="nav" aria-label="Aggregation">
  {nav}
  <div class="tabs" role="group" aria-label="Metric block">{tabs}</div>
</nav>
{hints}

{panels}

<p class="note"><b>Reading the tables.</b> The darkest cell in a column is its best value —
the largest where more is better, the smallest where less is. Shading is relative to the other
rows of the same table, never an absolute scale. <b>Pragmatic quality is left unshaded</b>: it
rewards smallness, so a higher value is not a better model and there is no “best” to point at.
Reasoning tokens and cache reads are unshaded for the same reason — they describe how a result
came about rather than setting a target. Token counts carry a median next to the mean because
reasoning models produce strongly right-skewed distributions. <b>Tokens are the one block
without a single table</b>: each vendor reports its own quantities, so that block is one
table per vendor, listing that vendor’s own figures — across two of those tables only the
cost columns can be read against each other. A metric a run did not produce
reads “not computed”, never zero. In an unfolded call, <b>→</b> separates a pragmatic metric's
raw value from the paper's banded score, and the last block of a token fold-out is that
call's vendor usage object, field for field.</p>

{reqids}

<footer>Generated {generated} by <code>results_report.py</code> · {n_calls} calls ·
self-contained, no network needed.</footer>

<script id="calls" type="application/json">{data}</script>
<script id="paperspec" type="application/json">{spec}</script>
<script id="effspec" type="application/json">{effspec}</script>
</main><script>
(function () {{
  // One level of selector: every block is on screen at all times, so the
  // state is just which of the buttons is pressed.
  var view = "model", tab = "{default_tab}";
  function apply() {{
    document.querySelectorAll(".panel").forEach(function (p) {{
      p.hidden = !(p.dataset.view === view && p.dataset.tab === tab);
    }});
    document.querySelectorAll(".viewhint").forEach(function (h) {{
      h.hidden = h.dataset.view !== view;
    }});
    document.querySelectorAll(".navbtn").forEach(function (b) {{
      if (b.dataset.view === view) b.setAttribute("aria-current", "page");
      else b.removeAttribute("aria-current");
    }});
    document.querySelectorAll(".tabbtn").forEach(function (b) {{
      b.setAttribute("aria-pressed", String(b.dataset.tab === tab));
    }});
  }}
  document.querySelectorAll(".navbtn").forEach(function (b) {{
    b.addEventListener("click", function () {{ view = b.dataset.view; apply(); }});
  }});
  document.querySelectorAll(".tabbtn").forEach(function (b) {{
    b.addEventListener("click", function () {{ tab = b.dataset.tab; apply(); }});
  }});

  // ── expandable rows: every call behind an aggregate ──────────────────────
  // Built from the dataset already embedded in the page, on first open — so
  // unfolding adds nothing to the file size. Values are written with
  // textContent, never innerHTML: these strings come from model output and
  // error messages and must not be able to inject markup.
  var CALLS = JSON.parse(document.getElementById("calls").textContent);
  var SPEC = JSON.parse(document.getElementById("paperspec").textContent);
  var EFFSPEC = JSON.parse(document.getElementById("effspec").textContent);

  // An unfolded call shows the subject of the table it was opened from. The
  // block is read off the table, not off the current selector state: every
  // block has its own panel with its own rows, so a fold-out built once stays
  // correct no matter which tab is open later.
  function specFor(block) {{ return block === "quality" ? SPEC : (EFFSPEC[block] || SPEC); }}

  var CHIPS = {{
    quality: [["syn", "syn_score"], ["sem", "sem_score"], ["prag", "prag_score"]],
    cost: [["tokens", "total_tokens"], ["USD", "cost_usd"]],
    latency: [["s", "latency_s"], ["attempts", "api_attempts"]]
  }};

  function show(value) {{
    if (value === null || value === undefined || value === "") return null;
    // A verdict, not a measurement: `true` printed as 1 reads as a score of
    // one, which is the one thing this metric never is.
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (typeof value === "number") {{
      if (!isFinite(value)) return null;
      if (Number.isInteger(value)) return String(value);
      return value.toFixed(Math.abs(value) < 1 ? 4 : 3);
    }}
    return String(value);
  }}

  // Costs run from cents to millionths of a cent, so a fixed number of decimals
  // either hides the small ones or pads the large ones with noise.
  function usd(amount) {{
    var v = Number(amount);
    if (!isFinite(v)) return "–";
    if (v === 0) return "$0";
    if (v < 0.000001) return "<$0.000001";
    return "$" + v.toFixed(v < 0.01 ? 6 : 4);
  }}

  function card(call, block) {{
    var wrap = document.createElement("article");
    wrap.className = "callcard";

    var head = document.createElement("div");
    head.className = "callhead";
    var id = document.createElement("span");
    id.className = "callid";
    id.textContent = [call.item_id, call.model_key, call.strategy,
                      call.repetition ? "r" + call.repetition : null]
                     .filter(Boolean).join("  \\u00b7  ");
    head.appendChild(id);

    var chips = document.createElement("span");
    chips.className = "callchips";
    (CHIPS[block] || CHIPS.quality).forEach(function (pair) {{
        var value = call[pair[1]];
        var text = show(value);
        if (text === null) return;
        var chip = document.createElement("span");
        var strong = document.createElement("b");
        strong.textContent = pair[0] === "USD" ? Number(value).toFixed(5) : text;
        chip.appendChild(strong);
        chip.appendChild(document.createTextNode(" " + pair[0]));
        chips.appendChild(chip);
      }});
    head.appendChild(chips);
    wrap.appendChild(head);

    var blocks = document.createElement("div");
    blocks.className = "callblocks";
    specFor(block).forEach(function (part) {{
      var section = document.createElement("section");
      section.className = "callblock";

      var title = document.createElement("h4");
      title.textContent = part.title;
      var headline = show(call[part.headline]);
      if (headline !== null) {{
        var badge = document.createElement("span");
        badge.className = "blockscore";
        badge.textContent = headline;
        title.appendChild(badge);
      }}
      section.appendChild(title);

      var list = document.createElement("dl");
      list.className = part.money ? "kv money" : "kv";
      part.metrics.forEach(function (metric) {{
        var term = document.createElement("dt");
        term.textContent = metric.label;
        var def = document.createElement("dd");
        // A sentence, not a number: given the full width of the card, because
        // in the narrow value column a long string wraps one character per line.
        if (metric.prose) {{
          term.className = "prose";
          def.className = "prose";
        }}
        var text = show(call[metric.key]);
        if (text === null) {{
          def.className += " miss";
          def.textContent = "not computed";
        }} else {{
          def.textContent = text;
          // A count metric is a conforming/covered ratio in the paper, and a
          // pragmatic metric carries the banded score of Eq. 1 / Eq. 2 — both
          // shown beside the value so the number can be read, not just seen.
          var aside = null;
          if (metric.ratio) {{
            var conforming = call[metric.ratio[0]], covered = call[metric.ratio[1]];
            if (conforming !== null && conforming !== undefined
                && covered !== null && covered !== undefined) {{
              aside = conforming + "/" + covered;
            }}
          }} else if (metric.score) {{
            var scored = show(call[metric.score]);
            if (scored !== null) aside = "\\u2192 " + scored;
          }}
          if (aside) {{
            var small = document.createElement("span");
            small.className = "aside";
            small.textContent = aside;
            def.appendChild(small);
          }}
        }}
        list.appendChild(term);
        list.appendChild(def);
        // Third column: what these tokens cost, at the rates this call was
        // actually billed at. A class without its own rate — a subtotal, or a
        // subset of another class — gets a dash rather than a repeated figure.
        if (part.money) {{
          var money = document.createElement("dd");
          money.className = "money";
          var amount = metric.cost ? call[metric.cost] : null;
          if (amount === null || amount === undefined) {{
            money.className += " nocost";
            money.textContent = "–";
          }} else {{
            money.textContent = usd(amount);
          }}
          list.appendChild(money);
        }}
      }});

      if (part.total) {{
        var tTerm = document.createElement("dt");
        tTerm.className = "grandtotal";
        tTerm.textContent = part.total.label;
        var tCount = document.createElement("dd");
        tCount.className = "grandtotal";
        tCount.textContent = show(call[part.total.key]) || "–";
        var tCost = document.createElement("dd");
        tCost.className = "grandtotal money";
        var totalAmount = call[part.total.cost];
        tCost.textContent = (totalAmount === null || totalAmount === undefined)
          ? "–" : usd(totalAmount);
        list.appendChild(tTerm);
        list.appendChild(tCount);
        list.appendChild(tCost);

        // Only when the parts do not add up to the recorded total.
        var note = call.cost_breakdown_note;
        if (note) {{
          var nTerm = document.createElement("dt");
          nTerm.className = "prose";
          nTerm.textContent = "note";
          var nDef = document.createElement("dd");
          nDef.className = "prose";
          nDef.textContent = note;
          list.appendChild(nTerm);
          list.appendChild(nDef);
        }}
      }}
      section.appendChild(list);
      blocks.appendChild(section);
    }});

    // The vendor's own usage object for this one call, verbatim. It sits with
    // the tokens, where the reader is already asking what a call cost: the
    // block above is what this project prices, this is what the API said. On
    // one call there is only one vendor, so nothing has to be reconciled.
    var native = (block === "cost" && call.vendor_usage) || null;
    if (native && Object.keys(native).length) {{
      var raw = document.createElement("section");
      raw.className = "callblock";
      var rawTitle = document.createElement("h4");
      rawTitle.textContent = "As " + (call.vendor || "the vendor") + " reported it";
      raw.appendChild(rawTitle);
      var rawList = document.createElement("dl");
      rawList.className = "kv";
      Object.keys(native).forEach(function (name) {{
        var term = document.createElement("dt");
        term.className = "native";
        term.textContent = name;
        var def = document.createElement("dd");
        var text = show(native[name]);
        if (text === null) {{
          def.className = "miss";
          // The vendor named the field and sent nothing in it — which is not
          // the same as a zero, and must not be shown as one.
          def.textContent = "not reported";
        }} else {{
          def.textContent = text;
        }}
        rawList.appendChild(term);
        rawList.appendChild(def);
      }});
      raw.appendChild(rawList);
      blocks.appendChild(raw);
    }}

    wrap.appendChild(blocks);
    return wrap;
  }}

  // `filterGroup`/`filterKey` are the second key a table can carry: the
  // per-strategy vendor tables are one strategy each, so a row there means
  // this model *within that strategy*, not the model's whole run.
  function fill(container, group, key, block, filterGroup, filterKey) {{
    var mine = CALLS.filter(function (c) {{
      return String(c[group]) === key
        && (!filterGroup || String(c[filterGroup]) === filterKey);
    }});
    if (!mine.length) {{
      var empty = document.createElement("p");
      empty.className = "emptydetail";
      empty.textContent = "No calls recorded for this row.";
      container.appendChild(empty);
      return;
    }}
    mine.sort(function (a, b) {{
      return String(a.item_id).localeCompare(String(b.item_id))
          || String(a.strategy).localeCompare(String(b.strategy))
          || String(a.model_key).localeCompare(String(b.model_key))
          || (a.repetition || 0) - (b.repetition || 0);
    }});
    mine.forEach(function (c) {{ container.appendChild(card(c, block)); }});
  }}

  document.querySelectorAll(".expander").forEach(function (button) {{
    button.addEventListener("click", function () {{
      var row = button.closest("tr");
      var detailRow = row.nextElementSibling;
      var container = detailRow.querySelector(".detail");
      var open = button.getAttribute("aria-expanded") === "true";
      if (!open && !container.dataset.filled) {{
        var host = row.closest("table");
        fill(container, host.dataset.group, row.dataset.key, host.dataset.block,
             host.dataset.filterGroup, host.dataset.filterKey);
        container.dataset.filled = "1";
      }}
      button.setAttribute("aria-expanded", String(!open));
      detailRow.hidden = open;
    }});
  }});

  apply();
}})();
</script></body></html>
"""


_REQUEST_IDS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Request-IDs {run_id}</title>
<style>
:root {{
  --ground:#f4f6f8; --panel:#fff; --sunken:#eef1f5;
  --ink:#141a21; --ink-soft:#4a5867; --ink-faint:#75838f;
  --line:#d8dfe6; --line-soft:#e8edf2; --accent:#0f6b7b; --warn:#a53f22;
  --sans:ui-sans-serif,"Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;
  --mono:ui-monospace,"Cascadia Mono",Consolas,"SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#11161c; --panel:#181f27; --sunken:#141a21;
    --ink:#e7ecf2; --ink-soft:#a9b6c3; --ink-faint:#7d8b99;
    --line:#2a333d; --line-soft:#222a33; --accent:#4fb3c4; --warn:#e08a6b;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#11161c; --panel:#181f27; --sunken:#141a21;
  --ink:#e7ecf2; --ink-soft:#a9b6c3; --ink-faint:#7d8b99;
  --line:#2a333d; --line-soft:#222a33; --accent:#4fb3c4; --warn:#e08a6b;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
       font:14px/1.5 var(--sans); -webkit-font-smoothing:antialiased; }}
main {{ max-width:1200px; margin:0 auto; padding:34px 24px 70px; }}
h1 {{ font-size:25px; margin:0 0 4px; letter-spacing:-.02em; }}
h2 {{ font-size:15px; margin:30px 0 3px; scroll-margin-top:16px; }}
.back {{ color:var(--accent); font-size:13px; text-decoration:none;
        border-bottom:1px solid currentColor; }}
.lead {{ color:var(--ink-soft); font-size:13px; max-width:74ch; margin:14px 0 0; }}
.count {{ color:var(--ink-faint); font-size:12px; margin:0 0 9px; }}
.tablewrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line);
             border-radius:9px; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th, td {{ text-align:left; padding:6px 11px; border-bottom:1px solid var(--line-soft);
         white-space:nowrap; }}
thead th {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
           color:var(--ink-faint); border-bottom:1px solid var(--line); background:var(--sunken); }}
td.mono, .mono {{ font-family:var(--mono); }}
td.na, .na {{ color:var(--ink-faint); font-style:italic; }}
.note {{ color:var(--ink-faint); font-size:12.5px; margin-top:26px; }}
footer {{ color:var(--ink-faint); font-size:12px; border-top:1px solid var(--line);
         padding-top:14px; margin-top:34px; }}

/* ── the fold-out: click a request id, get the call's conditions ──────────── */
.toolbar {{ margin:16px 0 0; }}
.toolbar button {{ font:12px var(--sans); color:var(--ink-soft); cursor:pointer;
                  background:var(--panel); border:1px solid var(--line);
                  border-radius:7px; padding:5px 11px; }}
.toolbar button:hover {{ border-color:var(--accent); color:var(--accent); }}
td.ridcell {{ padding:0; }}
button.rid {{ display:flex; align-items:center; gap:7px; width:100%; margin:0;
             padding:6px 11px; background:none; border:0; cursor:pointer;
             color:inherit; font:inherit; text-align:left; }}
button.rid:hover {{ background:var(--sunken); }}
button.rid:focus-visible {{ outline:2px solid var(--accent); outline-offset:-2px; }}
button.rid[aria-expanded="true"] {{ background:var(--sunken); }}
.caret {{ color:var(--ink-faint); font-size:10px; }}
button.rid[aria-expanded="true"] .caret {{ color:var(--accent); }}
tr.det > td {{ padding:2px 11px 14px; white-space:normal; background:var(--sunken);
              border-bottom:1px solid var(--line); }}
.o6 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
        padding:11px 13px; min-width:0; }}
.card.wide {{ grid-column:1 / -1; }}
.card.missing {{ border-style:dashed; color:var(--ink-soft); font-size:12.5px; }}
.card h4 {{ margin:0 0 8px; font-size:10.5px; text-transform:uppercase;
           letter-spacing:.07em; color:var(--ink-faint); }}
.card p {{ margin:0; }}
dl.facts {{ display:grid; grid-template-columns:max-content minmax(0,1fr);
           gap:3px 14px; margin:0; font-size:12.5px; }}
dl.facts dt {{ color:var(--ink-faint); }}
dl.facts dd {{ margin:0; overflow-wrap:anywhere; }}
pre.text {{ margin:8px 0 0; padding:9px 11px; background:var(--sunken);
           border:1px solid var(--line-soft); border-radius:6px;
           font:12px/1.55 var(--mono); white-space:pre-wrap; overflow-wrap:anywhere;
           max-height:330px; overflow:auto; }}
.turn {{ margin-top:10px; }}
.role {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
        color:var(--accent); }}
.cut {{ margin:5px 0 0 !important; font-size:11.5px; color:var(--ink-faint); }}
.warn {{ color:var(--warn); }}
a.file {{ color:var(--accent); font-family:var(--mono); font-size:12px; }}
table.mini {{ margin-top:9px; }}
table.mini th, table.mini td {{ padding:3px 7px; font-size:12px; border-bottom:0; }}
table.mini thead th {{ background:none; border-bottom:1px solid var(--line-soft); }}
.figure {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr.sumline td {{ border-top:1px solid var(--line); font-weight:600; }}
@media (max-width:860px) {{ .o6 {{ grid-template-columns:1fr; }} }}
@media print {{
  tr.det[hidden] {{ display:table-row; }}
  .toolbar, button.rid .caret {{ display:none; }}
  pre.text {{ max-height:none; }}
}}
</style></head>
<body><main>
<h1>Request-IDs</h1>
<p><a class="back" href="results.html">&larr; back to the results of {run_id}</a></p>
<p class="lead">Every request id opens. Click one for the conditions that produced that
call: the input, the prompt and the version of the template it was built from, the model,
the generation parameters, the timestamps, the reply verbatim, the dated price list the
cost was computed with, and the measurements derived from them. What that makes
repeatable is the <em>experimental conditions</em> — the same configuration can be sent
again exactly as it was. Generation is stochastic: the same conditions need not return
the same text.</p>
<div class="toolbar"><button type="button" id="expandall">Expand all</button></div>
{sections}
{note}
<footer>Generated {generated} by <code>results_report.py</code>. Request ids come from the
vendor's response header, response ids from the payload — quote them when asking a vendor
about a specific call. The fold-outs read <code>prompts/</code>, <code>raw/</code> and
<code>manifest.json</code> in this run directory; long texts are shown up to a limit and
linked in full.</footer>
</main>
<script>
(function () {{
  var main = document.querySelector('main');
  var all = document.getElementById('expandall');

  function set(button, open) {{
    button.setAttribute('aria-expanded', String(open));
    button.querySelector('.caret').innerHTML = open ? '&#9662;' : '&#9656;';
    var panel = document.getElementById(button.getAttribute('aria-controls'));
    if (panel) panel.hidden = !open;
  }}

  main.addEventListener('click', function (event) {{
    var button = event.target.closest('button.rid');
    if (button) set(button, button.getAttribute('aria-expanded') !== 'true');
  }});

  all.addEventListener('click', function () {{
    var open = all.dataset.open !== 'true';
    Array.prototype.forEach.call(document.querySelectorAll('button.rid'),
                                 function (b) {{ set(b, open); }});
    all.dataset.open = String(open);
    all.textContent = open ? 'Collapse all' : 'Expand all';
  }});
}})();
</script>
</body></html>
"""


RUNS_DIR = Path(__file__).resolve().parent / "runs"


def latest_run(runs_dir: Optional[Path] = None) -> Optional[Path]:
    """The newest run directory, by the timestamp in its name.

    Sorting by name rather than mtime on purpose: re-rendering a report touches
    a directory's files, and mtime would then make an old run look like the
    newest one.
    """
    runs_dir = runs_dir or RUNS_DIR
    if not runs_dir.is_dir():
        return None
    candidates = [d for d in runs_dir.iterdir()
                  if d.is_dir() and (d / "results.jsonl").exists()]
    return max(candidates, key=lambda d: d.name) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="?", default=None,
                        help="a run directory, e.g. runs/20260817_002523 "
                             "(omit to take the newest one)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run()
    if run_dir is None:
        sys.exit("No run directory found under runs/ — nothing to report on.")
    if not (run_dir / "results.jsonl").exists():
        sys.exit(f"No results.jsonl in {run_dir} — is that a run directory?")
    if args.run_dir is None:
        print(f"  latest run: {run_dir.name}")
    out = build(run_dir, out_path=args.out)
    print(f"  wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
