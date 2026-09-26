"""Every dataset item is evaluated — exemplars included — without leakage.

The run used to hold `cfg.FEW_SHOT_IDS` out of the evaluated set, so a run
configured for "all" covered 53 of PMo's 55 items. Now all 55 run, and the rule
that a model is never shown the answer to the item it is generating for is
enforced on the one prompt where it can be violated: the exemplar that *is* the
target item is swapped for a stand-in, leaving the other exemplars and the
exemplar count untouched.

The items nearest the exemplar set (cfg.FEW_SHOT_ALT_FOR_IDS) skip that swap
and run on a complete alternate set (cfg.FEW_SHOT_ALT_IDS) instead, so they all
see the same exemplars rather than a differently-swapped prompt each.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import config as cfg
import prompts
import run as run_mod
from dataset import load_dataset
from prompts import Exemplar, exemplars_for_item

EX = [Exemplar("01", "desc 1", "digraph g { one }"),
      Exemplar("02", "desc 2", "digraph g { two }")]
RESERVE = [Exemplar("03", "desc 3", "digraph g { three }"),
           Exemplar("04", "desc 4", "digraph g { four }")]
ALT = [Exemplar("40", "desc 40", "digraph g { forty }"),
       Exemplar("41", "desc 41", "digraph g { fortyone }")]


def _ids(exemplars):
    return [e.item_id for e in exemplars]


def test_item_that_is_not_an_exemplar_sees_the_configured_set():
    shown, swap = exemplars_for_item(EX, "07", RESERVE)
    assert _ids(shown) == ["01", "02"]
    assert swap is None


@pytest.mark.parametrize("item_id, expected", [("01", ["03", "02"]),
                                               ("02", ["01", "03"])])
def test_target_item_is_swapped_out_of_its_own_prompt(item_id, expected):
    shown, swap = exemplars_for_item(EX, item_id, RESERVE)
    assert _ids(shown) == expected          # position kept, count kept
    assert swap == (item_id, "03")
    assert item_id not in _ids(shown)


def test_stand_in_is_never_the_item_being_generated_for():
    # Reserve id 03 is unusable here because 03 is the target; the next one is.
    shown, _ = exemplars_for_item([Exemplar("03", "d", "g"), EX[1]], "03", RESERVE)
    assert _ids(shown) == ["04", "02"]


def test_without_a_reserve_the_exemplar_is_dropped_not_shown():
    shown, swap = exemplars_for_item(EX, "01", [])
    assert _ids(shown) == ["02"]            # one fewer, never its own answer
    assert swap == ("01", "")


def test_the_swapped_prompt_does_not_contain_the_targets_own_model():
    shown, _ = exemplars_for_item(EX, "01", RESERVE)
    _, messages = prompts.build_messages("few_shot", "desc 1", exemplars=shown)
    blob = "\n".join(m["content"] for m in messages)
    assert EX[0].dot not in blob
    assert RESERVE[0].dot in blob
    assert len(messages) == 2 * len(shown) + 1   # exemplar turns + the target


def test_defaults_do_not_overlap():
    assert not set(cfg.FEW_SHOT_IDS) & set(cfg.FEW_SHOT_FALLBACK_IDS)


# ── the run itself ──────────────────────────────────────────────────────────

def _write_pmo(tmp_path: Path, n: int) -> Path:
    root = tmp_path / "pmo-dataset"
    (root / "descriptions").mkdir(parents=True)
    (root / "graphviz").mkdir(parents=True)
    for i in range(1, n + 1):
        (root / "descriptions" / f"{i:02d}.txt").write_text(f"process {i}", encoding="utf-8")
        (root / "graphviz" / f"{i:02d}.dot").write_text(
            f"digraph g {{ a{i} -> b{i} }}", encoding="utf-8")
    return root


def test_no_exemplar_is_dropped_from_the_evaluated_set(tmp_path):
    root = _write_pmo(tmp_path, 8)
    items = load_dataset(root, exclude_ids=set())
    assert [i.item_id for i in items] == [f"{i:02d}" for i in range(1, 9)]
    assert set(cfg.FEW_SHOT_IDS) <= {i.item_id for i in items}


def test_hold_out_flag_still_removes_them(tmp_path):
    root = _write_pmo(tmp_path, 8)
    items = load_dataset(root, exclude_ids=set(cfg.FEW_SHOT_IDS))
    assert len(items) == 8 - len(cfg.FEW_SHOT_IDS)


def test_the_flag_defaults_to_off():
    args = run_mod._parse_args(["--provider", "mock"])
    assert args.hold_out_few_shot_items is False
    assert args.few_shot_fallback_ids == ",".join(cfg.FEW_SHOT_FALLBACK_IDS)
    assert args.few_shot_alt_ids == ",".join(cfg.FEW_SHOT_ALT_IDS)
    assert args.few_shot_alt_for_ids == ",".join(cfg.FEW_SHOT_ALT_FOR_IDS)


# ── the complete alternate set ─────────────────────────────────────

@pytest.mark.parametrize("item_id", ["01", "02", "03"])
def test_alternates_replace_the_whole_set_not_one_slot(item_id):
    shown, swap = exemplars_for_item(EX, item_id, RESERVE, alternates=ALT)
    assert _ids(shown) == ["40", "41"]      # the same pair for all three
    assert swap == ("01,02", "40,41")       # the manifest records the whole set


def test_alternates_leave_the_per_slot_swap_alone_when_not_passed():
    shown, swap = exemplars_for_item(EX, "01", RESERVE, alternates=None)
    assert _ids(shown) == ["03", "02"]
    assert swap == ("01", "03")


def test_an_alternate_is_still_never_the_item_being_generated_for():
    # Only reachable through a mis-configured FEW_SHOT_ALT_IDS; the guarantee
    # holds however the exemplar set was chosen.
    shown, _ = exemplars_for_item(EX, "40", RESERVE, alternates=ALT)
    assert _ids(shown) == ["41"]


def test_alternates_of_only_the_item_itself_fall_through_to_the_swap():
    shown, swap = exemplars_for_item(EX, "01", RESERVE, alternates=[EX[0]])
    assert _ids(shown) == ["03", "02"]
    assert swap == ("01", "03")


def test_the_alternate_prompt_holds_neither_the_answer_nor_the_stand_in():
    shown, _ = exemplars_for_item(EX, "01", RESERVE, alternates=ALT)
    _, messages = prompts.build_messages("few_shot", "desc 1", exemplars=shown)
    blob = "\n".join(m["content"] for m in messages)
    assert EX[0].dot not in blob            # never its own answer
    assert RESERVE[0].dot not in blob       # and not the stand-in either
    assert ALT[0].dot in blob and ALT[1].dot in blob
    assert len(messages) == 2 * len(ALT) + 1


def test_alt_defaults_are_disjoint_and_cover_every_collision():
    alt, alt_for = set(cfg.FEW_SHOT_ALT_IDS), set(cfg.FEW_SHOT_ALT_FOR_IDS)
    assert not alt & alt_for
    assert not alt & set(cfg.FEW_SHOT_IDS)
    assert not alt & set(cfg.FEW_SHOT_FALLBACK_IDS)
    # Every item that would otherwise be shown its own answer gets the set.
    assert set(cfg.FEW_SHOT_IDS) <= alt_for
    assert len(cfg.FEW_SHOT_ALT_IDS) >= cfg.RunSettings().n_few_shot
