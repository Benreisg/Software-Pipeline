"""
csv_export.py — a finished run as CSV, one table per file
=============================================================
The run's own artefacts stay what they are: `results.jsonl` is the generation
log, `results.html` is the report, and neither is a table anyone can open in R,
pandas or Excel without unpacking it first. This module writes that third form
— a `csv/` folder inside the run directory, one file per table:

    csv/
    ├── README.md                     what every file below holds
    ├── run_info.csv                  the run's settings and its totals
    ├── models.csv                    one row per LLM in the run
    ├── strategies.csv                one row per prompting strategy
    ├── by_model_quality.csv          ┐
    ├── by_model_tokens_cost.csv      │ the report's six panels — two views
    ├── by_model_latency.csv          │ (by LLM, by strategy) times three
    ├── by_strategy_quality.csv       │ blocks (quality, tokens & cost,
    ├── by_strategy_tokens_cost.csv   │ latency)
    ├── by_strategy_latency.csv       ┘
    ├── calls.csv                     one row per generation — the spine
    ├── calls_tokens_cost.csv         its tokens, its cost, class by class
    ├── calls_latency.csv             its timings and attempts
    ├── quality_validity.csv          ┐
    ├── quality_syntactic.csv         │ one file per quality dimension, all
    ├── quality_syntactic_bef4llm.csv │ of them keyed on the same four
    ├── quality_semantic.csv          │ columns as calls.csv
    ├── quality_pragmatic.csv         ┘
    ├── vendor_usage_<vendor>.csv     the vendor's own usage object, per vendor
    └── metrics_legend.csv            column → the label the report prints

Three properties make this an export rather than a second source of truth,
which is what the run directory has so far deliberately avoided:

* **Derived, never read back.** Every file is rewritten from the scoring pass
  each time a run is scored or its report rebuilt. Nothing in this project ever
  reads them, so no stored table can drift out of step with the metric
  definitions — the reason there was no `results.csv` to begin with.
* **Split on the seams the report already has.** The two aggregate views and
  the three metric blocks are the report's own structure, and the per-call
  files carry the same key, so any two of them join on
  `item_id, model_key, strategy, repetition`.
* **Full precision, no fabricated values.** Values are written exactly as the
  metric modules computed them (rounding is the report's business, see
  `quality/__init__.py`), and a metric that was not measured is an **empty
  cell** — never a zero.

    python csv_export.py runs/<run_id>       # one run
    python csv_export.py --all               # every run under runs/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import aggregate as agg
import results_report as report

# The folder written inside the run directory.
DIR_NAME = "csv"

# The key of one generation, and the join key of every per-call file below.
# `repetition` is part of it: with --repetitions N the same (item, model,
# strategy) appears N times (see quality/score.py).
KEY_COLS = ["item_id", "model_key", "strategy", "repetition"]

# The per-call cost breakdown `results_report.add_cost_breakdown` derives at
# build time. Not in `results.jsonl` — the run records what the vendor billed
# in total, and these split that total across the classes that produced it.
COST_BREAKDOWN_COLUMNS = (
    [report.COST_FIELD_PREFIX + field for field in report.PRICED_TOKEN_FIELDS]
    + ["cost_breakdown_note"])

# ── the per-call files ────────────────────────────────────────────────────────
# calls.csv is the spine: what was asked of whom, what came back, and one
# headline figure per dimension. Everything else is a detail file keyed on the
# same four columns, so nothing here has to be repeated there.
CALL_COLUMNS = [
    "item_id", "vendor", "model_key", "model_id", "model_tier", "strategy",
    "repetition", "provider", "is_mock",
    # what came back
    "generation_error", "extract_parse_ok", "extract_error", "stop_reason",
    "quality_note",
    # the five headline scores — the columns the report's quality table shows
    "val_score", "syn_bef_score", "syn_correct", "sem_score", "prag_score",
    # the author's own syntactic ratio, kept beside the paper's (README:
    # "Which of the two syntactic scorings goes in the thesis?")
    "syn_score",
    # one figure per efficiency block; both are detailed in their own file
    "total_tokens", "cost_usd", "latency_s",
    # provenance — enough to find the call again at the vendor and on disk
    "request_id", "response_id", "request_sent_at", "response_received_at",
    "generated_gv", "ground_truth_path",
    # …and enough to say what produced it: the record holding the prompt that
    # went out, the hashes of that prompt and of the input, the version of the
    # template behind the wording, and the reply verbatim (O6).
    "prompt_file", "prompt_sha256", "prompt_template_version", "prompt_source",
    "input_file", "input_sha256", "raw_output_file",
]

TOKEN_COST_COLUMNS = (
    # the classes this project prices and compares across vendors
    ["input_tokens", "cached_tokens", "cache_write_tokens",
     "cache_write_1h_tokens", "output_tokens", "thinking_tokens",
     "billable_input_tokens", "total_tokens"]
    # the same call as the vendor itself counted it
    + ["reported_input_tokens", "reported_output_tokens",
       "reported_total_tokens", "reported_cache_creation_tokens",
       "tool_use_prompt_tokens"]
    # what it cost, and what each class contributed to that total
    + ["cost_usd"] + COST_BREAKDOWN_COLUMNS
    # the rates it was actually charged at, and the date of the sheet they came
    # from — `cost_basis` says which rate applied, not what it was
    + ["price_in_per_mtok", "price_out_per_mtok", "pricing_as_of"]
    + ["cost_basis", "price_verified", "service_tier", "web_search_requests",
       "audio_input_seconds",
       "token_reporting", "tokens_estimated", "untracked_usage",
       "reasoning_note"])

LATENCY_COLUMNS = ["latency_s", "api_latency_s", "retry_wait_s", "call_wall_s",
                   "api_attempts", "request_sent_at", "response_received_at",
                   "stop_reason"]

# ── the quality dimensions ────────────────────────────────────────────────────
# Prefix rules rather than lists: the metric modules own their column sets, and
# a metric added there has to reach this export without an edit here. The one
# split that is not a prefix is inside `syn_`: BEF4LLM's rule catalogue
# (`syn_bef_*`, plus the two `syn_extra_*` rules its code has and its tables do
# not) is a different measurement from the eleven dictated checks, and the two
# disagree on the ranking of the strategies — so they get a file each.
BEF_PREFIXES = ("syn_bef_", "syn_extra_")

# (file, prefixes it takes, prefixes it leaves to another file). Between them
# the five files hold every `val_`/`syn_`/`sem_`/`prag_` column of the run, each
# in exactly one place.
QUALITY_FILES: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = [
    ("quality_validity", ("val_",), ()),
    ("quality_syntactic", ("syn_",), BEF_PREFIXES),
    ("quality_syntactic_bef4llm", BEF_PREFIXES, ()),
    ("quality_semantic", ("sem_",), ()),
    ("quality_pragmatic", ("prag_",), ()),
]

# Carried into every quality file: it is what says why a row is empty.
QUALITY_CONTEXT = ["vendor", "quality_note"]

# ── the aggregate files ───────────────────────────────────────────────────────
# The sample sizes lead every aggregate file. A mean without them is not
# reportable (see aggregate.py): a model with parse_rate 0.6 and a high mean
# score is worse than one with 0.95 and a middling score.
SAMPLE_COLUMNS = ["n_calls", "n_parsed", "n_valid", "n_scored", "parse_rate",
                  "n_errors"]

BLOCK_FILES = {"quality": "quality", "cost": "tokens_cost", "latency": "latency"}

# Count columns whose name does not say so on its own — the size metrics above
# all, which are counts of nodes, gateways, flows and hops.
COUNT_COLUMNS = {
    "repetition", "api_attempts", "web_search_requests",
    "syn_checks_passed", "syn_checks_total",
    "syn_bef_metrics_total", "syn_bef_metrics_perfect",
    "syn_bef_metrics_applicable",
    "syn_tnn", "syn_tng", "syn_tnsf", "syn_diameter", "syn_diameter_nogw",
    "prag_tnn", "prag_tng", "prag_tnsf", "prag_diameter",
    "prag_diameter_nogw", "prag_depth", "prag_cfc",
    "sem_ged_operations",
}


def _present(frame: pd.DataFrame, columns: List[str]) -> List[str]:
    """Those of `columns` the frame actually has, in the order given.

    A run older than a column simply exports without it, rather than failing.
    """
    seen, out = set(), []
    for column in columns:
        if column in frame.columns and column not in seen:
            seen.add(column)
            out.append(column)
    return out


def _is_count(name: str) -> bool:
    """Does this column count things, rather than measure them?

    Names, not values: a score that happens to be 1.0 in every row of a small
    run is still a score, and must not be written as `1`. `cost_of_*` is the
    trap in the other direction — `cost_of_cached_tokens` ends in `_tokens` and
    is an amount of money, which stays a decimal even when it is 0 throughout.
    """
    if name.startswith(report.COST_FIELD_PREFIX):
        return False
    return (name.endswith(("_tokens", "_conforming", "_covered"))
            or name.startswith("n_") or "_n_" in name
            or name in COUNT_COLUMNS)


def _whole_numbers(frame: pd.DataFrame) -> pd.DataFrame:
    """Counts as whole numbers.

    A count column with a gap in it — one call that never returned — becomes
    float64 in pandas, and 1880 is then written `1880.0`, which in a table of
    token counts is noise. The gap itself stays a gap: `Int64` is the nullable
    integer, so "not measured" remains an empty cell.
    """
    out = frame
    for column in frame.columns:
        series = frame[column]
        if series.dtype != "float64" or not _is_count(str(column)):
            continue
        values = series.dropna()
        if values.empty or (values % 1 == 0).all():
            if out is frame:
                out = frame.copy()
            out[column] = series.astype("Int64")
    return out


def _write(frame: pd.DataFrame, path: Path) -> int:
    """One CSV: comma-separated, UTF-8, LF, empty cell for "not measured".

    No `float_format`: values go out at the precision the metric modules
    computed them at. Rounding belongs to whatever displays them.
    """
    _whole_numbers(frame).to_csv(path, index=False, encoding="utf-8",
                                 lineterminator="\n")
    return len(frame)


def _sorted_calls(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-call rows in a stable order, so two exports of the same run differ
    only where the run does."""
    key = _present(frame, KEY_COLS)
    return frame.sort_values(key).reset_index(drop=True) if key else frame


def _quality_frame(calls: pd.DataFrame, take: Tuple[str, ...],
                   leave: Tuple[str, ...]) -> pd.DataFrame:
    """The columns of one dimension, keyed like `calls.csv`."""
    columns = [c for c in calls.columns
               if c.startswith(take) and not (leave and c.startswith(leave))]
    front = _present(calls, KEY_COLS + QUALITY_CONTEXT)
    return calls[front + columns]


def _block_frame(frame: pd.DataFrame, group: str, block: str) -> pd.DataFrame:
    """One aggregate view restricted to one metric block.

    The columns are `<metric>_<stat>` for exactly the statistics `aggregate`
    computes for that block — mean and n for quality, mean/median/total for
    tokens and cost, mean/median/max for latency.
    """
    if frame.empty:
        return frame
    metrics = [f"{column}_{stat}"
               for column, _label, _direction in agg.BLOCKS[block]
               for stat in agg.STATS[block]]
    return frame[_present(frame, [group] + SAMPLE_COLUMNS + metrics)]


def _manifest(run_dir: Path) -> Dict[str, Any]:
    """The run's manifest, or an empty dict — a run without one still exports."""
    try:
        return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _run_info(run_dir: Path, manifest: Dict[str, Any],
              totals: Dict[str, Any]) -> pd.DataFrame:
    """`field,value` rather than one wide row: the settings are nested, they are
    read one at a time, and a long table takes a new setting without a new
    column."""
    settings = manifest.get("settings") or {}
    pricing = manifest.get("pricing") or {}

    def join(value: Any) -> Any:
        return ", ".join(str(v) for v in value) if isinstance(value, list) else value

    fields = [
        ("run_id", run_dir.name),
        ("exported_at", pd.Timestamp.now().isoformat(timespec="seconds")),
        ("created_at", manifest.get("created_at")),
        ("mode", manifest.get("mode")),
        ("input_source", manifest.get("input_source")),
        ("dataset_dir", manifest.get("dataset_dir")),
        ("vendors_used", join(manifest.get("vendors_used"))),
        ("strategies", join(manifest.get("strategies"))),
        ("n_items", manifest.get("n_items")),
        ("repetitions", manifest.get("repetitions")),
        ("temperature", settings.get("temperature")),
        ("max_output_tokens", settings.get("max_output_tokens")),
        ("n_few_shot", settings.get("n_few_shot")),
        ("n_generations_planned", manifest.get("n_generations_planned")),
        ("n_generations_completed", manifest.get("n_generations_completed")),
        ("stopped_early", manifest.get("stopped_early")),
        ("pricing_as_of", pricing.get("as_of")),
        ("priced_on", pricing.get("priced_on")),
        # …and what the run actually produced, the header band of the report.
        ("n_calls", totals.get("n_calls")),
        ("n_parsed", totals.get("n_parsed")),
        ("n_valid", totals.get("n_valid")),
        ("n_errors", totals.get("n_errors")),
        ("n_untracked", totals.get("n_untracked")),
        ("n_models", totals.get("n_models")),
        ("n_strategies", totals.get("n_strategies")),
        ("n_items_scored", totals.get("n_items")),
        ("cost_usd", totals.get("cost_usd")),
        ("total_tokens", totals.get("total_tokens")),
        ("api_seconds", totals.get("api_seconds")),
    ]
    return pd.DataFrame([{"field": name, "value": value} for name, value in fields])


def _models_frame(manifest: Dict[str, Any], calls: pd.DataFrame) -> pd.DataFrame:
    """One row per LLM: the catalogue entry the run resolved, plus how many
    calls it actually made. `price_verified = False` marks a model whose rates
    were never confirmed — its `cost_usd` cells are empty, not guessed."""
    models = manifest.get("models") or []
    if not models and "model_key" in calls.columns:
        models = [{"key": key} for key in sorted(calls["model_key"].dropna().unique())]
    counts = (calls["model_key"].value_counts().to_dict()
              if "model_key" in calls.columns else {})
    rows = []
    for model in models:
        row = dict(model)
        row["n_calls"] = counts.get(model.get("key"), 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _strategies_frame(manifest: Dict[str, Any], calls: pd.DataFrame) -> pd.DataFrame:
    """One row per prompting strategy, with the citation the manifest records
    for its prompt — a strategy in a thesis table needs its source next to it."""
    sources = manifest.get("prompt_source") or {}
    strategies = manifest.get("strategies") or []
    if not strategies and "strategy" in calls.columns:
        strategies = sorted(calls["strategy"].dropna().unique())
    counts = (calls["strategy"].value_counts().to_dict()
              if "strategy" in calls.columns else {})
    return pd.DataFrame([{"strategy": name,
                          "prompt_source": sources.get(name),
                          "n_calls": counts.get(name, 0)}
                         for name in strategies])


def _vendor_usage_frames(records: List[dict]) -> Dict[str, pd.DataFrame]:
    """The vendors' own usage objects, one file per vendor.

    Not one table: the five APIs do not describe the same quantities, so a
    single grid could hold either every vendor or every figure. Fields keep the
    vendor's own name behind `native:` — OpenAI's `input_tokens` *includes* its
    cache reads where this project's excludes them, and the two must never
    share a column.
    """
    per_vendor: Dict[str, List[dict]] = {}
    for record in records:
        usage = record.get(report.VENDOR_USAGE_FIELD) or {}
        if not usage:
            continue
        row = {key: record.get(key) for key in KEY_COLS}
        row["model_id"] = record.get("model_id")
        for name, value in usage.items():
            row[report.NATIVE_PREFIX + name] = value
        per_vendor.setdefault(str(record.get("vendor") or "unknown"), []).append(row)

    frames = {}
    for vendor, rows in per_vendor.items():
        frame = pd.DataFrame(rows)
        front = _present(frame, KEY_COLS + ["model_id"])
        # A field the vendor named and never filled gets no column, exactly as
        # in the report's per-vendor table: it would be empty in every row, and
        # an empty column in a CSV reads as a measurement that came out blank.
        native = sorted(c for c in frame.columns
                        if c not in front and frame[c].notna().any())
        frames[vendor] = _sorted_calls(frame[front + native])
    return frames


def _legend_frame() -> pd.DataFrame:
    """`column → the label the report prints`, built from the same specs the
    page is built from — so a raw column name in any file above can be looked
    up without reading the source."""
    rows: List[Dict[str, Any]] = []
    for block, metrics in agg.BLOCKS.items():
        section = f"aggregate — {BLOCK_FILES[block]}"
        for column, label, direction in metrics:
            for stat in agg.STATS[block]:
                # `count` is the sample size behind the mean, not a measurement
                # of the model — it carries no direction.
                rows.append({"section": section, "column": f"{column}_{stat}",
                             "label": f"{label} ({stat})", "also_columns": "",
                             "better": "" if stat == "count" else direction})
    for spec in report.quality_metric_spec():
        for metric in spec["metrics"]:
            companions = [metric[name] for name in ("score",) if metric.get(name)]
            companions += list(metric.get("ratio") or [])
            rows.append({"section": spec["title"], "column": metric["key"],
                         "label": metric["label"],
                         "also_columns": ", ".join(companions), "better": ""})
        rows.append({"section": spec["title"], "column": spec["headline"],
                     "label": "— headline of this block —", "also_columns": "",
                     "better": "up"})
    return pd.DataFrame(rows)


_README_HEAD = """# `{run_id}` — results as CSV

Written by `csv_export.py`, and rewritten every time the run is scored or its
report rebuilt:

```bash
python csv_export.py {run_dir}
```

Nothing in the pipeline reads these files back — they are an export of the
scoring pass, not an input to it. The run's own data (`manifest.json`,
`results.jsonl`, `raw/`, `generated/`) is untouched by it.

## Conventions

* **Join key** — every per-call file is keyed on {key}.
  `calls.csv` is the spine; the others add columns to it and repeat nothing
  else.
* **An empty cell means "not measured", never 0.** A metric that could not be
  computed (nothing parsed, no ground truth, a vendor field the API never
  filled) is left empty on purpose.
* **Full precision.** Values are written exactly as computed; round when you
  display them. Counts are whole numbers, scores are decimals.
* **`item_id` is text**, zero-padded the way PMo writes it (`03`). Read it as
  one — `dtype={{"item_id": str}}` — or a reader turns `03` into `3`.
* Comma-separated, UTF-8, LF line endings.
* `metrics_legend.csv` maps every metric column to the label the HTML report
  prints for it.

## Files

| file | rows | what it holds |
|---|---|---|
"""


def _readme(run_dir: Path, written: List[Tuple[str, int, str]]) -> str:
    body = "".join(f"| `{name}` | {rows} | {what} |\n" for name, rows, what in written)
    return _README_HEAD.format(run_id=run_dir.name, run_dir=f"runs/{run_dir.name}",
                               key=", ".join(f"`{c}`" for c in KEY_COLS)) + body


def build(run_dir, scored: Optional[pd.DataFrame] = None,
          out_dir: Optional[Path] = None) -> Path:
    """Write the run's `csv/` folder. Returns the folder.

    `scored` is the frame from `quality.score_run`; when omitted the run is
    scored here, so this also works standalone on an older directory.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / DIR_NAME

    if scored is None:
        import quality
        scored = quality.score_run(run_dir, verbose=False)

    manifest = _manifest(run_dir)
    totals = agg.totals(scored)

    # The cost breakdown and the archived usage objects are derived the same way
    # the report derives them — one implementation, so the CSV and the page can
    # never state different numbers for the same call.
    records = scored.where(pd.notna(scored), None).to_dict(orient="records")
    report.add_cost_breakdown(records)
    report.attach_vendor_usage(run_dir, records)
    # Joined onto the scored frame rather than rebuilt from the records, which
    # would turn every integer column that has a gap into floats.
    derived = pd.DataFrame([{c: r.get(c) for c in COST_BREAKDOWN_COLUMNS}
                            for r in records], index=scored.index)
    calls = _sorted_calls(pd.concat([scored, derived], axis=1))

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Tuple[str, int, str]] = []

    def emit(name: str, frame: pd.DataFrame, what: str) -> None:
        written.append((name, _write(frame, out_dir / name), what))

    emit("run_info.csv", _run_info(run_dir, manifest, totals),
         "the run's settings and its totals, one `field,value` pair per row")
    emit("models.csv", _models_frame(manifest, calls),
         "one row per LLM: catalogue entry, rates applied, calls made")
    emit("strategies.csv", _strategies_frame(manifest, calls),
         "one row per prompting strategy, with the source its prompt is cited from")

    frames = {"model": agg.by_model(scored), "strategy": agg.by_strategy(scored)}
    for view, group in (("model", "model_key"), ("strategy", "strategy")):
        subject = "LLM" if view == "model" else "strategy"
        pooled = "strategy" if view == "model" else "LLM"
        for block, label in BLOCK_FILES.items():
            what = "quality scores" if block == "quality" else label.replace("_", " & ")
            emit(f"by_{view}_{label}.csv", _block_frame(frames[view], group, block),
                 f"one row per {subject} — {what}, pooled over every {pooled}")

    emit("calls.csv", calls[_present(calls, CALL_COLUMNS)],
         "one row per generation — what was asked, what came back, one headline "
         "figure per dimension")
    emit("calls_tokens_cost.csv",
         calls[_present(calls, KEY_COLS + ["vendor"] + TOKEN_COST_COLUMNS)],
         "per call: every token class, and the cost each of them contributed")
    emit("calls_latency.csv",
         calls[_present(calls, KEY_COLS + ["vendor"] + LATENCY_COLUMNS)],
         "per call: total latency, the answering attempt, backoff and attempts")

    dimension_note = {
        "quality_validity": "per call: the Graphviz DOT check — this project's format gate",
        "quality_syntactic": "per call: the eleven dictated checks, the size metrics, "
                             "the Syntactical Correctness verdict and its counters",
        "quality_syntactic_bef4llm": "per call: BEF4LLM's published rule catalogue "
                                     "(Table 2 / A.15) plus the two rules only its code has",
        "quality_semantic": "per call: the seven semantic metrics (Table A.17), against "
                            "the ground truth",
        "quality_pragmatic": "per call: the fourteen pragmatic metrics (Table A.16 / A.18), "
                             "raw value and banded score",
    }
    for name, take, leave in QUALITY_FILES:
        frame = _quality_frame(calls, take, leave)
        if len(frame.columns) <= len(_present(calls, KEY_COLS + QUALITY_CONTEXT)):
            continue  # the dimension contributed no columns to this run
        emit(f"{name}.csv", frame, dimension_note[name])

    for vendor, frame in sorted(_vendor_usage_frames(records).items()):
        emit(f"vendor_usage_{vendor}.csv", frame,
             f"per call: the usage object {report._vendor_label(vendor)}'s own API "
             f"returned, field for field")

    emit("metrics_legend.csv", _legend_frame(),
         "every metric column and the label the HTML report prints for it")

    (out_dir / "README.md").write_text(_readme(run_dir, written), encoding="utf-8")
    return out_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="?", default=None,
                        help="Run directory; default is the most recent one")
    parser.add_argument("--all", action="store_true",
                        help="Export every run under runs/ that holds a results.jsonl")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the CSVs here instead of <run_dir>/csv")
    args = parser.parse_args(argv)

    if args.all:
        if args.out:
            sys.exit("--out writes one folder; it cannot be combined with --all.")
        runs = sorted(p for p in Path("runs").glob("*")
                      if (p / "results.jsonl").exists())
        if not runs:
            sys.exit("No runs with a results.jsonl under runs/.")
    else:
        run_dir = args.run_dir or report.latest_run()
        if run_dir is None:
            sys.exit("No run directory given and none found under runs/.")
        if not Path(run_dir).is_dir():
            sys.exit(f"Not a directory: {run_dir}")
        if not (Path(run_dir) / "results.jsonl").exists():
            sys.exit(f"No results.jsonl in {run_dir} — is that a run directory?")
        runs = [Path(run_dir)]

    for run in runs:
        out = build(run, out_dir=args.out)
        n = len(list(out.glob("*.csv")))
        print(f"  [csv] {run.name}: wrote {n} files to {out}")


if __name__ == "__main__":
    main()
