"""Parallel execution — one worker per model, one request per model at a time.

The point of the arrangement is a guarantee, not a speed-up: a model must never
have two calls in flight, because a model measured against copies of itself
reports a latency that belongs to the load rather than to the model. These tests
pin that guarantee, and the two things that follow from it — that the models do
run beside each other, and that every row records how wide the run was.

`run_experiment` is driven with stub providers and a redirected `cfg.RUNS_DIR`,
so the loop under test is the one a live run executes, without a run directory
appearing in the project or a token being spent.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

import config as cfg
import pipeline
import postprocess
from dataset import ProcessItem
from providers import GenerationResult, Provider

MODELS = [cfg.model_by_key(k) for k in
          ("anthropic_efficient", "openai_efficient", "google_efficient")]
ITEMS = [ProcessItem(item_id=f"{i:02d}", description=f"Step {i} happens.")
         for i in range(1, 5)]
SETTINGS = cfg.RunSettings(temperature=None, max_output_tokens=4096, n_few_shot=2)


class _WatchingProvider(Provider):
    """Records how many calls are in flight, in total and per model.

    The counters are the assertion: `peak_per_model` is what the one-slot-per-
    model rule promises to hold at 1, and `peak_total` is the evidence that the
    models really did overlap rather than the run quietly serialising.
    """

    name = "stub"

    def __init__(self, delay: float = 0.02):
        self._delay = delay
        self._lock = threading.Lock()
        self._in_flight: dict[str, int] = {}
        self.peak_per_model = 0
        self.peak_total = 0
        self.calls = 0

    def generate(self, model, system, messages, settings) -> GenerationResult:
        with self._lock:
            self.calls += 1
            self._in_flight[model.key] = self._in_flight.get(model.key, 0) + 1
            self.peak_per_model = max(self.peak_per_model, self._in_flight[model.key])
            self.peak_total = max(self.peak_total, sum(self._in_flight.values()))
        try:
            # Long enough that a second call for the same model would overlap
            # this one if the rule were broken.
            time.sleep(self._delay)
            return GenerationResult(
                text="digraph g { a -> b }", reported_input_tokens=10,
                reported_total_tokens=20, reported_output_tokens=10,
                input_tokens=10, output_tokens=10, billable_input_tokens=10,
                total_tokens=20, cost_usd=0.0, cost_basis="list",
                stop_reason="end_turn",
            )
        finally:
            with self._lock:
                self._in_flight[model.key] -= 1

    def check_connection(self, model_id):
        return True, "ok"


def _run(tmp_path, monkeypatch, provider, *, parallel, models=None, items=None):
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    return pipeline.run_experiment(
        providers={v: provider for v in cfg.VENDORS} | {"mock": provider},
        mode="mock",
        models=models if models is not None else MODELS,
        strategies=["zero_shot"],
        items=items if items is not None else ITEMS,
        exemplar_files=[],
        settings=SETTINGS,
        dataset_dir=tmp_path,
        tag=None,
        input_source="test",
        repetitions=1,
        parallel_models=parallel,
    )


def _rows(run_dir):
    return [json.loads(l) for l in
            (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def _cells(rows):
    return {(r["item_id"], r["model_key"], r["strategy"], r["repetition"])
            for r in rows}


# ── the guarantee ─────────────────────────────────────────────────────────────
def test_only_one_call_per_model_is_ever_in_flight(tmp_path, monkeypatch):
    """The rule the whole arrangement exists for. Break it and every latency a
    parallel run reports is a measurement of self-inflicted load."""
    provider = _WatchingProvider()
    _run(tmp_path, monkeypatch, provider, parallel=True)
    assert provider.peak_per_model == 1
    assert provider.calls == len(MODELS) * len(ITEMS)


def test_the_models_do_run_beside_each_other(tmp_path, monkeypatch):
    """The counterpart: one slot *per model* must still mean all models at once,
    or the flag buys nothing and the run is merely sequential with extra steps."""
    provider = _WatchingProvider()
    _run(tmp_path, monkeypatch, provider, parallel=True)
    assert provider.peak_total > 1


def test_a_sequential_run_never_overlaps_anything(tmp_path, monkeypatch):
    provider = _WatchingProvider()
    _run(tmp_path, monkeypatch, provider, parallel=False)
    assert provider.peak_total == 1


# ── results are the same either way ───────────────────────────────────────────
def test_both_regimes_produce_exactly_the_same_cells(tmp_path, monkeypatch):
    """Parallel execution may reorder the rows — it must not lose, duplicate or
    alter one."""
    seq = _rows(_run(tmp_path / "seq", monkeypatch, _WatchingProvider(), parallel=False))
    par = _rows(_run(tmp_path / "par", monkeypatch, _WatchingProvider(), parallel=True))
    assert len(seq) == len(par) == len(MODELS) * len(ITEMS)
    assert _cells(seq) == _cells(par)
    assert all(r["extract_parse_ok"] for r in par)


def test_every_artefact_is_written_in_both_regimes(tmp_path, monkeypatch):
    for name, parallel in (("seq", False), ("par", True)):
        run_dir = _run(tmp_path / name, monkeypatch, _WatchingProvider(), parallel=parallel)
        n = len(MODELS) * len(ITEMS)
        assert len(list((run_dir / "raw").glob("*.txt"))) == n, name
        assert len(list((run_dir / "generated").glob("*.gv"))) == n, name


# ── the width is recorded ─────────────────────────────────────────────────────
def test_every_row_records_the_width_it_ran_at(tmp_path, monkeypatch):
    """Without this column a latency read back later cannot be told apart from a
    sequential one, and the two would be pooled."""
    par = _rows(_run(tmp_path / "par", monkeypatch, _WatchingProvider(), parallel=True))
    seq = _rows(_run(tmp_path / "seq", monkeypatch, _WatchingProvider(), parallel=False))
    assert {r["concurrent_models"] for r in par} == {len(MODELS)}
    assert {r["concurrent_models"] for r in seq} == {1}


def test_the_manifest_records_the_regime(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, monkeypatch, _WatchingProvider(), parallel=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parallel_models"] is True
    assert manifest["concurrent_models"] == len(MODELS)
    assert manifest["n_generations_completed"] == len(MODELS) * len(ITEMS)
    assert manifest["stopped_early"] is False


def test_one_model_is_not_a_parallel_run(tmp_path, monkeypatch):
    """Asking for parallelism with a single model would otherwise label the run
    parallel while running exactly as it always did."""
    run_dir = _run(tmp_path, monkeypatch, _WatchingProvider(), parallel=True,
                   models=MODELS[:1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["concurrent_models"] == 1
    assert manifest["parallel_models"] is False
    assert {r["concurrent_models"] for r in _rows(run_dir)} == {1}


# ── the hazard parallel execution uncovered ───────────────────────────────────
def test_dot_parsing_survives_concurrent_use():
    """pydot binds parse actions to module-level pyparsing objects, so two
    threads parsing at once corrupt each other and valid DOT comes back as
    `parse error: push_ID() missing 1 required positional argument`. Without the
    lock in postprocess this fails in the dozens; the generation loop hits it
    because every worker parses its own reply.
    """
    dot = "digraph g { " + " ".join(f'n{i} -> n{i+1};' for i in range(30)) + " }"
    failures = []

    def parse():
        for _ in range(20):
            result = postprocess.extract_and_validate(dot)
            if not result.parse_ok:
                failures.append(result.error)

    threads = [threading.Thread(target=parse) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failures == []


# ── the run must not look frozen while it waits ───────────────────────────────
def test_the_heartbeat_ticks_while_calls_are_still_out():
    """Nothing completes while the calls are on the wire, and on the slow models
    that silence lasts minutes. Without a tick in that gap a running parallel
    sweep is indistinguishable from a hung one."""
    ticks = []

    def slow(entry):
        time.sleep(0.5)
        return entry

    sub_plans = {f"m{i}": [f"m{i}-c0"] for i in range(3)}
    consumed = list(pipeline._iter_per_model(
        sub_plans, slow, None, threading.Event(),
        heartbeat=lambda n: ticks.append(n), heartbeat_every=0.1))

    assert sorted(consumed) == ["m0-c0", "m1-c0", "m2-c0"]
    assert ticks, "no heartbeat fired while three calls were in flight"
    # Every tick happened while all three were still out, so it must report
    # them — a heartbeat that always said 0 would be worse than none.
    assert max(ticks) == 3


def test_no_heartbeat_is_required(tmp_path, monkeypatch):
    """The tick is an optional courtesy; leaving it out must not change what the
    iterator yields."""
    sub_plans = {"m0": ["a", "b"], "m1": ["c"]}
    consumed = list(pipeline._iter_per_model(
        sub_plans, lambda e: e, None, threading.Event(), heartbeat_every=0.05))
    assert sorted(consumed) == ["a", "b", "c"]


# ── the wizard offers the choice, and only when it is one ─────────────────────
def test_the_wizard_only_asks_when_there_is_more_than_one_model(monkeypatch):
    """With a single model the two regimes are the same run. Asking anyway would
    imply a difference that does not exist — and would let someone label a
    one-model run 'parallel'."""
    import run as runmod

    asked = []
    monkeypatch.setattr(runmod, "_ask",
                        lambda prompt, default="": asked.append(prompt) or "y")

    assert runmod._select_execution(MODELS[:1]) is False
    assert asked == []

    assert runmod._select_execution(MODELS) is True
    assert len(asked) == 1


def test_the_wizard_takes_no_for_an_answer(monkeypatch):
    import run as runmod

    monkeypatch.setattr(runmod, "_ask", lambda prompt, default="": "n")
    assert runmod._select_execution(MODELS) is False


# ── scoring on the worker that fetched the reply ──────────────────────────────
def test_rows_carry_their_metrics_the_moment_they_are_written(tmp_path, monkeypatch):
    """The whole point of scoring inline: a row is complete when it lands, so a
    run stopped halfway holds finished rows rather than rows awaiting a verdict."""
    run_dir = _run(tmp_path, monkeypatch, _WatchingProvider(), parallel=True)
    rows = _rows(run_dir)
    assert all("syn_score" in r and "quality_note" in r for r in rows)
    assert all(r["syn_score"] is not None for r in rows)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scored_inline"] is True


def test_inline_scores_match_a_separate_scoring_pass(tmp_path, monkeypatch):
    """Scoring on thirteen workers must reach the same verdicts as one pass over
    the finished directory. It is the same code either way — what this pins is
    that running it concurrently does not change its answers, which is exactly
    what the pydot and WordNet races did before they were locked.
    """
    from quality import score

    run_dir = _run(tmp_path, monkeypatch, _WatchingProvider(), parallel=True)
    inline = score.load_generation_log(run_dir)
    recomputed = score.score_run(run_dir, verbose=False)

    cols = [c for c in inline.columns
            if c.startswith(("syn_", "sem_", "prag_", "val_")) or c == "quality_note"]
    assert cols, "no metric columns were written"

    key = [k for k in score.KEY_COLS if k in inline.columns]
    a = inline.set_index(key)[cols].sort_index()
    b = recomputed.set_index(key)[cols].sort_index()
    for col in cols:
        assert a[col].astype(str).tolist() == b[col].astype(str).tolist(), col


def test_scoring_can_be_switched_off(tmp_path, monkeypatch):
    """`--no-score` has to mean no metrics at all, not 'metrics anyway, just not
    a second time'."""
    provider = _WatchingProvider()
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    run_dir = pipeline.run_experiment(
        providers={v: provider for v in cfg.VENDORS} | {"mock": provider},
        mode="mock", models=MODELS, strategies=["zero_shot"], items=ITEMS[:1],
        exemplar_files=[], settings=SETTINGS, dataset_dir=tmp_path, tag=None,
        input_source="test", repetitions=1, parallel_models=True,
        score_inline=False,
    )
    rows = _rows(run_dir)
    assert all("syn_score" not in r for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scored_inline"] is False


# ── a broken worker must be loud ──────────────────────────────────────────────
def test_a_crashing_worker_is_not_a_finished_model(tmp_path, monkeypatch):
    """Regression. A worker that raised still ran its `finally`, posted the
    done-sentinel and left the run claiming a clean finish — observed as a run
    reporting `stopped_early: false` with 3 rows of a planned 39, and nothing
    anywhere saying why. Anything `run_one` does not handle is a defect, and a
    defect has to reach the surface.
    """
    def explode(*args, **kwargs):
        raise RuntimeError("scoring is broken")

    monkeypatch.setattr(pipeline.quality_score, "score_generation", explode)
    with pytest.raises(RuntimeError, match="scoring is broken"):
        _run(tmp_path, monkeypatch, _WatchingProvider(), parallel=True)


def test_a_crash_stops_the_other_workers(tmp_path, monkeypatch):
    """Whatever broke one worker will break the rest, one paid call at a time."""
    provider = _WatchingProvider(delay=0.05)
    calls = []

    def explode(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline.quality_score, "score_generation", explode)
    with pytest.raises(RuntimeError):
        _run(tmp_path, monkeypatch, provider, parallel=True)
    # Far below the full plan: the run stops instead of failing its way through
    # every remaining call.
    assert provider.calls < len(MODELS) * len(ITEMS)
