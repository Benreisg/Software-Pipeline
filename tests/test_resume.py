"""Continuing an interrupted run.

A run is a cross product of (repetition × model × strategy × item) with no
random element in it, so what a run directory is missing is computable: the
cells whose key is not in `results.jsonl`. These tests drive `run_experiment`
with a counting stub provider and a redirected `cfg.RUNS_DIR`, so what is
asserted is how many calls a resumed run actually makes.
"""
from __future__ import annotations

import json

import pytest

import config as cfg
import pipeline
from dataset import ProcessItem
from providers import GenerationResult, Provider

ITEMS = [ProcessItem(item_id="01", description="A does X, then B does Y."),
         ProcessItem(item_id="02", description="B does Y, then C does Z.")]
MODELS = ["google_advanced", "anthropic_advanced"]


class _CountingProvider(Provider):
    """One fixed generation per call, and a tally of how many were asked for."""

    name = "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, model, system, messages, settings) -> GenerationResult:
        self.calls += 1
        return GenerationResult(
            text="Output:\ndigraph g { a -> b }\n",
            reported_input_tokens=10, reported_total_tokens=20,
            reported_output_tokens=10, tool_use_prompt_tokens=0,
            input_tokens=10, output_tokens=10, billable_input_tokens=10,
            total_tokens=20, cost_usd=0.0, cost_basis="list",
            raw_usage="", stop_reason="end_turn",
        )

    def check_connection(self, model_id):
        return True, "ok"


def _run(tmp_path, monkeypatch, provider, *, items=None, models=None,
         strategies=None, repetitions=2, **kwargs):
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    return pipeline.run_experiment(
        providers={"mock": provider},
        mode="mock",
        models=[cfg.model_by_key(k) for k in (models or MODELS)],
        strategies=strategies or ["zero_shot"],
        items=items or ITEMS,
        exemplar_files=[],
        settings=cfg.RunSettings(temperature=None, max_output_tokens=4096,
                                 n_few_shot=2),
        dataset_dir=tmp_path,
        tag=None,
        input_source="test",
        repetitions=repetitions,
        score_inline=False,
        **kwargs,
    )


def _rows(run_dir):
    return [json.loads(line) for line
            in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _keys(run_dir):
    return [(r["item_id"], r["model_key"], r["strategy"], r["repetition"])
            for r in _rows(run_dir)]


def _truncate(run_dir, keep: int):
    """Leave `keep` rows and strip the closing manifest — a run killed mid-flight
    never gets to write one."""
    lines = [line for line
             in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()][:keep]
    (run_dir / "results.jsonl").write_text("".join(l + "\n" for l in lines),
                                           encoding="utf-8")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for key in ("stopped_early", "n_generations_planned",
                "n_generations_completed", "few_shot_substitutions"):
        manifest.pop(key, None)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")


def test_resume_calls_only_the_missing_cells(tmp_path, monkeypatch):
    provider = _CountingProvider()
    run_dir = _run(tmp_path, monkeypatch, provider)
    assert provider.calls == 8          # 2 items × 2 models × 1 strategy × 2 reps
    _truncate(run_dir, 3)

    resumed = _CountingProvider()
    again = _run(tmp_path, monkeypatch, resumed, resume_dir=run_dir)

    assert again == run_dir             # the same directory, not a new one
    assert resumed.calls == 5           # only what was missing was paid for
    keys = _keys(run_dir)
    assert len(keys) == 8 and len(set(keys)) == 8


def test_resume_keeps_the_run_identity(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, monkeypatch, _CountingProvider())
    created = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["created_at"]
    _truncate(run_dir, 3)

    _run(tmp_path, monkeypatch, _CountingProvider(), resume_dir=run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    # One run, two sittings: the date it started stays, and the second sitting
    # is accounted for rather than overwriting the first.
    assert manifest["created_at"] == created
    assert len(manifest["resumed_at"]) == 1
    assert manifest["n_generations_planned"] == 8
    assert manifest["n_generations_completed"] == 8
    segment = manifest["segments"][-1]
    assert (segment["already_present"], segment["planned"], segment["completed"]) == (3, 5, 5)


def test_resume_refuses_a_wider_plan(tmp_path, monkeypatch):
    """The dangerous mismatch: every recorded row still fits, so only the plan
    size gives it away — and the run would buy cells nobody planned for."""
    run_dir = _run(tmp_path, monkeypatch, _CountingProvider())
    _truncate(run_dir, 3)

    provider = _CountingProvider()
    with pytest.raises(ValueError, match="planned 8 generation"):
        _run(tmp_path, monkeypatch, provider, resume_dir=run_dir,
             items=ITEMS + [ProcessItem(item_id="03", description="C does Z.")])
    assert provider.calls == 0


def test_rewind_requeues_the_last_call_of_every_model(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, monkeypatch, _CountingProvider())
    before = _rows(run_dir)
    last_per_model = {r["model_key"]: r for r in before}   # file order: last wins

    provider = _CountingProvider()
    _run(tmp_path, monkeypatch, provider, resume_dir=run_dir, resume_rewind=1)

    # One call per model, and no cell recorded twice.
    assert provider.calls == len(MODELS)
    keys = _keys(run_dir)
    assert len(keys) == 8 and len(set(keys)) == 8

    # The rewound rows are moved, not dropped.
    rewound = [json.loads(line) for line
               in (run_dir / "rewound.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert {(r["item_id"], r["model_key"], r["strategy"], r["repetition"])
            for r in rewound} == {
        (r["item_id"], r["model_key"], r["strategy"], r["repetition"])
        for r in last_per_model.values()}
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")
                      )["segments"][-1]["rewound"] == len(MODELS)


def test_rewind_of_zero_changes_nothing(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, monkeypatch, _CountingProvider())
    _truncate(run_dir, 5)
    before = _keys(run_dir)

    provider = _CountingProvider()
    _run(tmp_path, monkeypatch, provider, resume_dir=run_dir, resume_rewind=0)

    assert provider.calls == 3
    assert _keys(run_dir)[:5] == before
    assert not (run_dir / "rewound.jsonl").exists()
