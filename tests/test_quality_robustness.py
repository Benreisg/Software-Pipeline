"""Scoring survives a model that breaks a metric.

Two failures found together in run 20260831_182408, where a single generated
model raised a RecursionError inside `pragmatic._bef4llm_diameter` and took the
whole run down after 4040 of 25740 paid calls:

* the traversal itself had to terminate on a cycle that runs through an end
  event, which is a shape a generated model reaches on its own; and
* no single metric may end a run, because scoring happens on the generation
  worker while the remaining calls are still queued.
"""
from __future__ import annotations

import pytest

from quality import graph as g, pragmatic, score


# The shape that broke it: an end event and its plaintext label joined by
# invisible edges in *both* directions — "for proper ranking", as the model
# that produced it put it — which is a two-node cycle through an end event.
CYCLE_THROUGH_END = """
digraph G {
    start [shape=circle];
    collect [shape=box, label="Collect data"];
    end_stop [shape=circle];
    end_stop_label [shape=plaintext, label="End (Machine Stopped)"];
    start -> collect;
    collect -> end_stop;
    end_stop_label -> end_stop [style=invis];
    end_stop -> end_stop_label [style=invis];
}
"""

STRAIGHT_LINE = """
digraph G {
    start [shape=circle];
    collect [shape=box, label="Collect data"];
    end [shape=doublecircle];
    start -> collect;
    collect -> end;
}
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_diameter_terminates_on_a_cycle_through_an_end_event(tmp_path):
    pg = g.load(_write(tmp_path, "cycle.gv", CYCLE_THROUGH_END))
    assert "end_stop" in pg.ends
    # Without the `ends_seen` guard this recurses until CPython's stack limit,
    # because reaching an end event clears the `visited` cycle counter.
    assert isinstance(pragmatic._bef4llm_diameter(pragmatic._View(pg)), int)


def test_scoring_a_cycle_through_an_end_event_returns_columns(tmp_path):
    out = score.score_model(_write(tmp_path, "cycle.gv", CYCLE_THROUGH_END))
    assert out["prag_diameter"] is not None
    assert out["quality_note"] == ""


def test_a_failing_metric_costs_only_its_own_columns(tmp_path, monkeypatch):
    real = pragmatic.evaluate

    def boom(generated, ground_truth=None):
        # Only for a real model: `pragmatic.empty()` derives its column list by
        # calling `evaluate` on a blank graph, and that call has to keep working
        # — it is what the failure path returns.
        if generated.nodes:
            raise RecursionError("maximum recursion depth exceeded")
        return real(generated, ground_truth)

    monkeypatch.setattr(score.pragmatic, "evaluate", boom)
    out = score.score_model(_write(tmp_path, "line.gv", STRAIGHT_LINE))

    # The run keeps every other verdict, and the row says what broke.
    assert out["syn_score"] is not None
    assert out["sem_score"] is None or isinstance(out["sem_score"], float)
    assert out["prag_score"] is None
    assert "pragmatic metrics failed: RecursionError" in out["quality_note"]
    # Every pragmatic column is still present, so the frame keeps its shape.
    assert set(pragmatic.empty()) <= set(out)
    # Validity is judged from the file itself and is unaffected.
    assert "val_score" in out


def test_score_generation_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "score_model",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = score.score_generation(_write(tmp_path, "line.gv", STRAIGHT_LINE))
    assert out["quality_note"].startswith("scoring failed: RuntimeError")
    assert out["syn_score"] is None
