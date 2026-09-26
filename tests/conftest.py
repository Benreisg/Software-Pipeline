"""Shared test fixtures.

Every test in this suite is offline: no test may make a network call or spend
money. Vendor SDKs are exercised by stubbing the client object and feeding the
provider the SDK's own response types, so the code under test is the same code a
live run executes — only the transport is replaced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402  (import after the path fix, deliberately)


@pytest.fixture
def settings() -> "cfg.RunSettings":
    return cfg.RunSettings(temperature=None, max_output_tokens=4096, n_few_shot=2)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


# ── a run directory, without a run ────────────────────────────────────────────
# Shared by every test that needs a scored run: the report, the CSV export, and
# anything else built on `quality.score_run`.
@pytest.fixture
def fake_run(tmp_path):
    """A minimal but realistic run: two models × two strategies, one call of
    which failed outright — the case the page must render without crashing."""
    run_dir = tmp_path / "20260101_000000_fixture"
    (run_dir / "generated").mkdir(parents=True)
    (run_dir / "raw").mkdir()

    (run_dir / "manifest.json").write_text(json.dumps({
        "created_at": "2026-01-01T00:00:00", "mode": "live",
        "n_items": 1, "repetitions": 1,
        "strategies": ["zero_shot", "few_shot"],
        "input_source": "fixture",
        "models": [{"key": "alpha", "model_id": "fixture-alpha"},
                   {"key": "beta", "model_id": "fixture-beta"}],
        "settings": {"max_output_tokens": 4096},
        "n_generations_planned": 4, "n_generations_completed": 4,
        "stopped_early": False,
        "pricing": {"as_of": "fixture 2026-01-01"},
    }), encoding="utf-8")

    def record(model, strategy, ok=True):
        base = {"item_id": "01", "model_key": model, "strategy": strategy,
                "repetition": 1, "vendor": "fixture", "is_mock": False,
                "ground_truth_path": "", "generation_error": None}
        if not ok:
            # A call that never returned: an error text, no artefact, no tokens.
            base.update({"generation_error": "Error code: 500 - fixture failure",
                         "generated_gv": "", "extract_parse_ok": False,
                         "input_tokens": None, "output_tokens": None,
                         "total_tokens": None, "cost_usd": None,
                         "api_latency_s": None, "tokens_estimated": False})
            return base
        name = f"01__{model}__{strategy}.gv"
        (run_dir / "generated" / name).write_text(
            'digraph G { s [shape=circle label=""]; s -> "do it"; '
            '"do it" -> e; e [shape=doublecircle label=""] }', encoding="utf-8")
        base.update({"generated_gv": f"generated/{name}", "extract_parse_ok": True,
                     "input_tokens": 100, "output_tokens": 200, "total_tokens": 300,
                     "billable_input_tokens": 100, "cost_usd": 0.001,
                     "api_latency_s": 1.5, "latency_s": 1.5, "api_attempts": 1,
                     "tokens_estimated": False})
        return base

    rows = [record("alpha", "zero_shot"), record("alpha", "few_shot"),
            record("beta", "zero_shot"), record("beta", "few_shot", ok=False)]
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return run_dir


@pytest.fixture
def scored(fake_run):
    import quality
    return quality.score_run(fake_run, verbose=False)
