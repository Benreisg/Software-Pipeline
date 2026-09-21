"""Pausing and stopping a run from the console.

The mechanism is worth a test for one reason: it is the only thing standing
between a reader and Ctrl-C on a run that is spending money. Every command is
checked through `RunControl.handle`, which is the listener thread's whole body —
so the vocabulary is tested without a terminal, and without a thread reading
stdin under pytest.
"""
from __future__ import annotations

import signal
import threading

import pytest

import runcontrol
from runcontrol import RunControl, stop_on_sigint


@pytest.fixture
def control():
    """Keyboard half off, as it is on a piped run — the flags below are the
    half that must work regardless."""
    return RunControl(enabled=False)


# ── stopping ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("word", sorted(runcontrol.STOP_WORDS))
def test_every_stop_word_stops_the_run(control, word):
    """`end` is the documented one. The others are what a person types when the
    run is burning money and the printed word has scrolled away — a run that
    ignores 'stop' because it wanted 'end' is the failure this prevents."""
    assert control.handle(word + "\n") is False       # nothing left to listen for
    assert control.stopped
    assert control.wait_if_paused() is False          # the loop's read of it


def test_a_stop_is_honoured_even_without_the_keyboard_listener():
    """`enabled` covers reading stdin, not the flag. Ctrl-C sets the flag on a
    piped run too, and a loop that ignored it there would run to the end."""
    control = RunControl(enabled=False)
    assert not control.enabled
    control.request_stop()
    assert control.wait_if_paused() is False


def test_the_first_reason_is_the_one_reported(control, capsys):
    """Ctrl-C landing on an already-stopping run must not tell the story a
    second time."""
    control.request_stop("first")
    control.request_stop("second")
    out = capsys.readouterr().out
    assert "first" in out and "second" not in out


# ── pausing ───────────────────────────────────────────────────────────────────
def test_enter_toggles_pause_and_resume(control):
    control.handle("\n")
    assert control.paused
    control.handle("\n")
    assert not control.paused


def test_the_words_say_what_the_toggle_does(control):
    """Same two states, reachable by name: a reader who does not remember that
    Enter is a toggle types what they mean instead."""
    control.handle("pause\n")
    assert control.paused
    control.handle("resume\n")
    assert not control.paused
    control.handle("p\n")
    assert control.paused


def test_an_unknown_word_neither_pauses_nor_stops(control, capsys):
    assert control.handle("stahp\n") is True
    assert not control.paused and not control.stopped
    assert "Unknown input" in capsys.readouterr().out


def test_the_help_command_prints_the_controls(control, capsys):
    """`?` is for the reader whose banner has scrolled away, so it prints the
    keys that actually work here — with a listener, that includes the words."""
    control.enabled = True                            # as if stdin were a tty
    control.handle("?\n")
    out = capsys.readouterr().out
    assert runcontrol.STOP_WORD in out and "[Enter]" in out
    assert not control.paused and not control.stopped


def test_the_hint_names_the_keys_that_work(control):
    """Printed once before the first call. With no keyboard listener it must
    promise only what is left — Ctrl-C."""
    assert "Ctrl-C" in control.hint()
    assert "[Enter]" not in control.hint()
    enabled = RunControl(enabled=True)
    enabled.enabled = True                            # as if stdin were a tty
    assert "[Enter]" in enabled.hint() and runcontrol.STOP_WORD in enabled.hint()


# ── Ctrl-C ────────────────────────────────────────────────────────────────────
def test_the_first_ctrl_c_stops_the_run_and_the_second_kills_it(control):
    """One press is `end`: the call in flight finishes and the run still scores
    and reports. Two presses mean it — the default handler is back, so a run
    wedged in a network read stays killable."""
    previous = signal.getsignal(signal.SIGINT)
    with stop_on_sigint(control):
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not previous
        handler(signal.SIGINT, None)                  # the first Ctrl-C
        assert control.stopped
        assert signal.getsignal(signal.SIGINT) is previous
    assert signal.getsignal(signal.SIGINT) is previous


def test_the_handler_is_restored_even_when_the_run_raises(control):
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError):
        with stop_on_sigint(control):
            raise RuntimeError("the run failed")
    assert signal.getsignal(signal.SIGINT) is previous


def test_without_a_control_object_ctrl_c_is_left_alone():
    """A caller that passed no control has no flag to set, so the default
    KeyboardInterrupt must survive rather than being swallowed."""
    previous = signal.getsignal(signal.SIGINT)
    with stop_on_sigint(None):
        assert signal.getsignal(signal.SIGINT) is previous


# ── the loop actually honours it ──────────────────────────────────────────────
def test_a_stop_mid_run_keeps_everything_generated_so_far(tmp_path, monkeypatch):
    """The end of the story: a stop asked for during call 2 ends the run before
    call 3, and what was generated is still written, counted and declared
    partial in the manifest. That is the whole difference to Ctrl-C killing the
    process."""
    import json

    import config as cfg
    import pipeline
    from dataset import ProcessItem
    from providers import GenerationResult, Provider

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    control = RunControl(enabled=False)
    calls = []

    class _Stub(Provider):
        name = "stub"

        def generate(self, model, system, messages, settings):
            calls.append(1)
            if len(calls) == 2:
                control.request_stop()        # as if 'end' were typed mid-call
            return GenerationResult(
                text="Output:\ndigraph g { a -> b }\n",
                input_tokens=10, output_tokens=5, billable_input_tokens=10,
                total_tokens=15, cost_usd=0.0, cost_basis="list",
                stop_reason="end_turn",
            )

        def check_connection(self, model_id):
            return True, "ok"

    run_dir = pipeline.run_experiment(
        providers={"mock": _Stub()},
        mode="mock",
        models=[cfg.model_by_key("google_advanced")],
        strategies=["zero_shot"],
        items=[ProcessItem(item_id=str(i), description="A does X, then B does Y.")
               for i in ("01", "02", "03")],
        exemplar_files=[],
        settings=cfg.RunSettings(temperature=None, max_output_tokens=4096, n_few_shot=2),
        dataset_dir=tmp_path,
        tag=None,
        input_source="test",
        repetitions=1,
        control=control,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stopped_early"] is True
    assert (manifest["n_generations_completed"],
            manifest["n_generations_planned"]) == (2, 3)
    rows = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2                     # both finished calls, kept
    assert len(calls) == 2                    # and the third was never made


# ── a stop drops the call in flight ───────────────────────────────────────────
def _slow_provider(started, release, control=None, stop_after_start=False):
    """A provider whose call blocks until `release` is set — a request on the
    wire that has not answered yet."""
    from providers import GenerationResult, Provider

    class _Slow(Provider):
        name = "slow"

        def generate(self, model, system, messages, settings):
            started.set()
            if stop_after_start and control is not None:
                control.request_stop()            # as if 'end' were typed now
            release.wait(5)
            return GenerationResult(text="digraph g { a -> b }", input_tokens=1,
                                    output_tokens=1, total_tokens=2, cost_usd=0.0,
                                    cost_basis="list", stop_reason="end_turn")

        def check_connection(self, model_id):
            return True, "ok"

    return _Slow()


def test_a_stop_drops_the_call_in_flight_instead_of_waiting():
    """`end` must not sit out a reasoning call that takes minutes: the loop gets
    control back at once, and the row says the reply was never waited for."""
    import time

    import config as cfg
    import pipeline

    started, release = threading.Event(), threading.Event()
    control = RunControl(enabled=False)
    provider = _slow_provider(started, release, control, stop_after_start=True)

    t0 = time.monotonic()
    gen, error, abandoned = pipeline._generate(
        provider, cfg.model_by_key("google_advanced"), None, [], cfg.DEFAULTS, control)
    waited = time.monotonic() - t0

    assert started.is_set()                       # the request did go out
    assert abandoned is True and gen is None
    assert error == pipeline.ABANDONED_ERROR
    assert waited < 2                             # ... and was not waited out
    release.set()


def test_a_call_that_answers_is_never_abandoned():
    import config as cfg
    import pipeline

    started, release = threading.Event(), threading.Event()
    release.set()
    control = RunControl(enabled=False)
    gen, error, abandoned = pipeline._generate(
        _slow_provider(started, release), cfg.model_by_key("google_advanced"),
        None, [], cfg.DEFAULTS, control)
    assert abandoned is False and error == "" and gen is not None


def test_an_abandoned_call_still_lands_in_the_run_as_a_row(tmp_path, monkeypatch):
    """The run stops with a record of what happened: a row marked abandoned, a
    manifest that says the run is partial, and no reply invented for it."""
    import json

    import config as cfg
    import pipeline
    from dataset import ProcessItem

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    started, release = threading.Event(), threading.Event()
    control = RunControl(enabled=False)

    run_dir = pipeline.run_experiment(
        providers={"mock": _slow_provider(started, release, control,
                                          stop_after_start=True)},
        mode="mock",
        models=[cfg.model_by_key("google_advanced")],
        strategies=["zero_shot"],
        items=[ProcessItem(item_id="01", description="A does X."),
               ProcessItem(item_id="02", description="B does Y.")],
        exemplar_files=[],
        settings=cfg.RunSettings(temperature=None, max_output_tokens=4096, n_few_shot=2),
        dataset_dir=tmp_path,
        tag=None,
        input_source="test",
        repetitions=1,
        control=control,
    )
    release.set()

    rows = [json.loads(line) for line in
            (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert len(rows) == 1                         # the abandoned call itself
    assert rows[0]["generation_error"] == pipeline.ABANDONED_ERROR
    assert not rows[0].get("extract_parse_ok")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stopped_early"] is True
