"""The run's raw archive — what lands on disk beside each reply.

`run_experiment` is driven with a stub provider and a redirected `cfg.RUNS_DIR`,
so the loop under test is the one a live run executes without a run directory
appearing in the project or a token being spent.
"""
from __future__ import annotations

import json

import pytest

import config as cfg
import pipeline
from dataset import ProcessItem
from providers import GenerationResult, Provider

RAW_USAGE = json.dumps({"prompt_token_count": 184,
                        "candidates_token_count": 59,
                        "prompt_tokens_details": [{"modality": "TEXT",
                                                   "token_count": 184}]})


class _StubProvider(Provider):
    """Returns one fixed generation, carrying a vendor usage object."""

    name = "stub"

    def __init__(self, raw_usage: str = RAW_USAGE):
        self._raw_usage = raw_usage

    def generate(self, model, system, messages, settings) -> GenerationResult:
        return GenerationResult(
            text="Output:\ndigraph g { a -> b }\n",
            reported_input_tokens=184, reported_total_tokens=243,
            reported_output_tokens=59, tool_use_prompt_tokens=12,
            input_tokens=184, output_tokens=59, billable_input_tokens=184,
            total_tokens=243, cost_usd=0.0, cost_basis="list",
            raw_usage=self._raw_usage, stop_reason="end_turn",
        )

    def check_connection(self, model_id):
        return True, "ok"


def _run(tmp_path, monkeypatch, provider):
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    return pipeline.run_experiment(
        providers={"mock": provider},
        mode="mock",
        models=[cfg.model_by_key("google_advanced")],
        strategies=["zero_shot"],
        items=[ProcessItem(item_id="01", description="A does X, then B does Y.")],
        exemplar_files=[],
        settings=cfg.RunSettings(temperature=None, max_output_tokens=4096, n_few_shot=2),
        dataset_dir=tmp_path,
        tag=None,
        input_source="test",
        repetitions=1,
    )


def test_the_vendor_usage_object_is_archived_beside_the_reply(tmp_path, monkeypatch):
    """The results table carries the classes this project prices; the archive
    carries everything the API said, so an audit does not depend on the schema
    that was current when the run happened."""
    run_dir = _run(tmp_path, monkeypatch, _StubProvider())

    raw = list((run_dir / "raw").glob("*.txt"))
    archived = list((run_dir / "raw").glob("*.usage.json"))
    assert len(raw) == 1 and len(archived) == 1
    # One archive per reply, named after it.
    assert archived[0].name == raw[0].name.replace(".txt", ".usage.json")

    payload = json.loads(archived[0].read_text(encoding="utf-8"))
    assert payload["prompt_token_count"] == 184
    # The modality breakdown, which no column in the results table holds.
    assert payload["prompt_tokens_details"][0]["modality"] == "TEXT"


def test_the_raw_columns_reach_the_results_row(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, monkeypatch, _StubProvider())
    row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["reported_input_tokens"] == 184
    assert row["reported_total_tokens"] == 243
    assert row["reported_output_tokens"] == 59
    assert row["tool_use_prompt_tokens"] == 12


def test_a_provider_without_a_usage_object_writes_no_archive(tmp_path, monkeypatch):
    """The mock provider calls no API, so there is nothing to archive — an empty
    file would claim the opposite."""
    run_dir = _run(tmp_path, monkeypatch, _StubProvider(raw_usage=""))
    assert list((run_dir / "raw").glob("*.txt"))
    assert not list((run_dir / "raw").glob("*.usage.json"))


# ── the conditions each generation ran under (O6) ─────────────────────────────
class _FailingProvider(Provider):
    """Raises instead of answering — the call the vendor never completed."""

    name = "stub"

    def generate(self, model, system, messages, settings) -> GenerationResult:
        raise RuntimeError("Error code: 500 - fixture failure")

    def check_connection(self, model_id):
        return True, "ok"


def test_each_generation_archives_the_conditions_it_ran_under(tmp_path, monkeypatch):
    """results.jsonl holds the measurements and manifest.json the run-wide
    configuration; neither held the prompt that actually went out, so a row
    could be re-run but not re-derived."""
    import prompts

    run_dir = _run(tmp_path, monkeypatch, _StubProvider())

    records = list((run_dir / "prompts").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))

    assert record["input"]["text"] == "A does X, then B does Y."
    assert record["input"]["chars"] == len("A does X, then B does Y.")
    assert record["prompt"]["messages"][-1]["content"].endswith("A does X, then B does Y.")
    # The version of the wording, not of this one prompt: it moves when the
    # template is edited and stays put when the item changes.
    assert record["prompt"]["template_version"] == prompts.template_version("zero_shot")
    assert record["parameters"]["max_output_tokens"] == 4096
    assert record["pricing"]["basis"] == "list"
    assert record["call"]["model_id"] == cfg.model_by_key("google_advanced").model_id
    assert record["artefacts"]["raw_output"].endswith(".txt")


def test_the_row_names_the_conditions_it_was_produced_under(tmp_path, monkeypatch):
    """A row lifted out of the CSV still has to identify its own prompt."""
    run_dir = _run(tmp_path, monkeypatch, _StubProvider())
    row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert (run_dir / row["prompt_file"]).is_file()
    assert (run_dir / row["raw_output_file"]).is_file()
    record = json.loads((run_dir / row["prompt_file"]).read_text(encoding="utf-8"))
    assert row["prompt_sha256"] == record["prompt"]["sha256"]
    assert row["input_sha256"] == record["input"]["sha256"]
    assert row["prompt_template_version"] == record["prompt"]["template_version"]
    # The rate that applied, not just the basis that selected it.
    assert row["pricing_as_of"] == cfg.pricing_sheet("google")["as_of"]


def test_a_call_that_never_returned_still_records_its_conditions(tmp_path, monkeypatch):
    """The conditions are what they were; the failure is part of what they
    produced, and a run cannot be repeated from a gap."""
    run_dir = _run(tmp_path, monkeypatch, _FailingProvider())

    records = list((run_dir / "prompts").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert "fixture failure" in record["call"]["generation_error"]
    assert record["prompt"]["messages"]
