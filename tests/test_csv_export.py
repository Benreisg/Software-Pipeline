"""The `csv/` export of a finished run.

Builds from the synthetic run directory in `conftest.py` (`fake_run`/`scored`),
so the suite never depends on an API call having happened. What is asserted here
is what makes the export usable as a set of tables rather than a dump: every
metric column lands in exactly one file, the per-call files join, and a metric
that was never measured stays empty.
"""
from __future__ import annotations

import csv

import pandas as pd
import pytest

import csv_export


QUALITY_PREFIXES = ("val_", "syn_", "sem_", "prag_")


@pytest.fixture
def exported(fake_run, scored):
    return csv_export.build(fake_run, scored=scored)


def _read(folder, name):
    return pd.read_csv(folder / name)


def test_writes_one_file_per_table(exported):
    names = {p.name for p in exported.glob("*.csv")}
    assert {"run_info.csv", "models.csv", "strategies.csv",
            "by_model_quality.csv", "by_model_tokens_cost.csv",
            "by_model_latency.csv", "by_strategy_quality.csv",
            "by_strategy_tokens_cost.csv", "by_strategy_latency.csv",
            "calls.csv", "calls_tokens_cost.csv", "calls_latency.csv",
            "metrics_legend.csv"} <= names
    # The folder documents itself: a reader who opens it cold gets the file
    # list and the conventions without the source.
    readme = (exported / "README.md").read_text(encoding="utf-8")
    assert "calls.csv" in readme and "metrics_legend.csv" in readme


def test_every_quality_column_lands_in_exactly_one_file(exported, scored):
    """The split is by prefix, and BEF4LLM's rule catalogue is carved out of
    `syn_` — so it is easy for a column to be dropped or duplicated silently."""
    expected = {c for c in scored.columns if c.startswith(QUALITY_PREFIXES)}
    seen: dict = {}
    for name, _take, _leave in csv_export.QUALITY_FILES:
        for column in _read(exported, f"{name}.csv").columns:
            if column.startswith(QUALITY_PREFIXES):
                seen.setdefault(column, []).append(name)

    assert not expected - set(seen), f"dropped: {sorted(expected - set(seen))}"
    duplicated = {c: f for c, f in seen.items() if len(f) > 1}
    assert not duplicated, f"in more than one file: {duplicated}"


def test_per_call_files_join_on_the_key(exported):
    spine = _read(exported, "calls.csv")
    key = csv_export.KEY_COLS
    for name in ("calls_tokens_cost.csv", "calls_latency.csv",
                 "quality_syntactic.csv", "quality_pragmatic.csv"):
        frame = _read(exported, name)
        assert list(frame.columns[:len(key)]) == key
        assert len(frame) == len(spine)
        assert (frame[key].astype(str).agg("|".join, axis=1).tolist()
                == spine[key].astype(str).agg("|".join, axis=1).tolist())


def test_a_metric_that_was_not_measured_is_empty_not_zero(exported):
    """The failed call produced no model and cost nothing that was measured.
    Writing 0 there would put a data point where there is none — the whole
    reason the scoring pass distinguishes the two."""
    calls = _read(exported, "calls.csv").set_index("model_key")
    failed = calls.loc["beta"]
    failed = failed[failed["strategy"] == "few_shot"].iloc[0]
    for column in ("val_score", "syn_bef_score", "sem_score", "prag_score",
                   "cost_usd", "total_tokens"):
        assert pd.isna(failed[column]), f"{column} should be empty, is {failed[column]!r}"


def test_counts_are_whole_numbers_but_scores_stay_decimals(exported):
    """A column with a gap in it goes float in pandas, and a token count then
    reads `300.0`. Converting back is by name, not by value — a run that
    happens to score 1.0 throughout must not turn its score into an int."""
    rows = list(csv.DictReader((exported / "calls.csv").open(encoding="utf-8")))
    scored_rows = [row for row in rows if row["total_tokens"]]
    assert scored_rows
    assert all(row["total_tokens"] == "300" for row in scored_rows)
    assert all("." in row["val_score"] for row in scored_rows)


def test_a_cost_is_never_mistaken_for_a_count():
    """`cost_of_cached_tokens` ends in `_tokens` and is money: a cost of $0
    stays `0.0`, or the column reads as a count of something."""
    assert not csv_export._is_count("cost_of_cached_tokens")
    assert csv_export._is_count("cached_tokens")
    assert not csv_export._is_count("sem_score")


def test_item_ids_keep_their_padding(exported):
    """PMo ids are zero-padded text. `3` would not find `03.dot` again."""
    rows = list(csv.DictReader((exported / "calls.csv").open(encoding="utf-8")))
    assert {row["item_id"] for row in rows} == {"01"}


def test_aggregates_carry_their_sample_sizes(exported):
    """A mean without its n is not reportable (aggregate.py). Every aggregate
    file leads with the sample sizes, in both views."""
    for name in ("by_model_quality.csv", "by_model_tokens_cost.csv",
                 "by_model_latency.csv", "by_strategy_quality.csv",
                 "by_strategy_tokens_cost.csv", "by_strategy_latency.csv"):
        frame = _read(exported, name)
        assert set(csv_export.SAMPLE_COLUMNS) <= set(frame.columns), name


def test_aggregates_state_the_same_numbers_as_the_calls(exported):
    """The export must not become a second, disagreeing computation: the
    aggregate files are the same means the page shows, over the same rows."""
    calls = _read(exported, "calls.csv")
    by_model = _read(exported, "by_model_quality.csv").set_index("model_key")
    assert by_model.loc["alpha", "n_calls"] == (calls["model_key"] == "alpha").sum()

    tokens = _read(exported, "by_model_tokens_cost.csv").set_index("model_key")
    per_call = _read(exported, "calls_tokens_cost.csv")
    alpha = per_call[per_call["model_key"] == "alpha"]["output_tokens"]
    assert tokens.loc["alpha", "output_tokens_sum"] == pytest.approx(alpha.sum())


def test_run_info_reports_what_the_run_actually_did(exported):
    info = _read(exported, "run_info.csv").set_index("field")["value"]
    assert int(info["n_calls"]) == len(_read(exported, "calls.csv"))
    assert int(info["n_errors"]) == 1
    assert info["run_id"] == exported.parent.name


def test_re_exporting_rewrites_the_same_folder(fake_run, scored):
    """Every score and every report build rewrites the export, so it has to
    land in the same place with the same content — not accumulate a second set
    of files beside the first."""
    first = csv_export.build(fake_run, scored=scored)
    names = sorted(p.name for p in first.glob("*"))
    before = _read(first, "calls.csv")

    second = csv_export.build(fake_run, scored=scored)
    assert second == first
    assert sorted(p.name for p in second.glob("*")) == names
    assert _read(second, "calls.csv").equals(before)
