"""
score_pmo_pairs.py — the semantic dimension over manipulated/original pairs
==========================================================================
`score_pmo_dataset.py` compares every model with itself, which makes the
semantic columns 1.0 by construction — a wiring check, not a measurement. This
scores each **manipulated** model against its PMo original, so the semantic
metrics have something to measure and the two pipelines can be compared where
they actually differ.

    python score_pmo_pairs.py                    # → pmo_manipulated_pairs.csv
    python score_pmo_pairs.py --both-directions  # each pair also the other way

**The pairing.** Item *i* of `PMo_manipulated/Graphviz` is scored as the
*generated* model against item *i* of `pmo-dataset/graphviz` as the *ground
truth*. The manipulated dataset carries the same process changed in known ways
— renamed tasks, a removed or added edge, see its `manipulations.json` — in
**both** representations, and both carry the identical manipulation. So each
pipeline reads its own format of the same pair and nothing is converted:

    python score_pmo_pairs.py
    .venv-bef4llm/Scripts/python score_pmo_bef4llm.py         --pairs pmo_manipulated_pairs_list.csv         --bpmn-dir pmo-dataset/bpmn --generated-dir PMo_manipulated/BPMN         --out pmo_manipulated_pairs_bef4llm.csv

**Only this pairing, since 2026-08-30** (author's instruction). Two others
existed and are gone: an *offset* pairing (item *i* against item *i + 27*,
wrapping) that put unrelated processes side by side at the bottom of the
similarity scale, and `--basic-variations`, which paired the simplified models
under `variations/graphviz_basic` with their full counterparts at the top. What
their artefacts showed is worth keeping in view: on the manipulated pairing both
label metrics reproduce BEF4LLM's code exactly (55/55 with the id spaces
separated), while the offset pairing — nearly no label overlap, so their
stemming, synonym and deleting-overlap paths all run — had SimSyn at 51/55 and
**SimSem at 23/55**. This pairing does not exercise that, so it cannot show it.

**What this pairing has settled.** The semantic dimension is BEF4LLM's own
three natural-language metrics, and this pairing is where each was checked
against their code over the same models in BPMN (`--pairs`, then
`prefix_bpmn_ids.py` for the id-disjoint side). Both label similarities
reproduce exactly, 55 of 55. The **context similarity**, added on 2026-08-30,
reaches 44 of 55 (max |d| 0.167, mean |d| 0.015). The 11 that are left are
theirs, not the port's: their greedy matching breaks a tie by the order the
nodes stand in the file, and a tie is common — every unlabelled event scores 0
against everything, and "Order was shipped" scores 1.0 against "Shipping the
order". The two label metrics never notice, since the *value* is the same
whichever partner wins; the context similarity resolves neighbourhoods
*through* that partner and does. Their order is the BPMN document's, this
port's the sorted DOT id, and nothing in a .dot file carries the former.
Verified node by node on all 11: the neighbourhoods agree everywhere, and
every node bound differently is bound at an equal similarity.

**Direction matters.** The metrics match the nodes of one model onto the other,
and that is not symmetric: `sem(a, b)` need not equal `sem(b, a)`. The column
pair `item_generated` / `item_reference` records which was which, and
`--both-directions` scores each pair twice so the asymmetry is visible instead
of hidden.

**The counterpart.** `pmo_manipulated_pairs_list.csv` lists exactly the pairs
that were scored, for `score_pmo_bef4llm.py --pairs`, which runs the authors'
own semantic check over the same pairs in their native BPMN. Both sides key on
`item_generated` / `item_reference`, so the two CSVs join on those two columns.
`prefix_bpmn_ids.py` produces the id-disjoint variant of their side, which is
what the `sem_*_disjoint` columns are to be held against.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config

# A legacy console code page (cp1252) cannot encode the arrows this script
# prints, and an UnicodeEncodeError there would kill it *after* the scoring is
# done. Same guard `score_pmo_dataset.py` carries.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
from quality import columns
from quality.score import score_model

DEFAULT_OUT = Path(__file__).resolve().parent / "pmo_semantic_pairs.csv"
DEFAULT_MANIPULATED = Path(__file__).resolve().parent / "PMo_manipulated"
DEFAULT_MANIPULATED_OUT = (Path(__file__).resolve().parent
                           / "pmo_manipulated_pairs.csv")

# The columns that make the pair readable, before the metric block.
_IDENTITY = ["item_generated", "item_reference", "direction",
             "generated_path", "reference_path"]


def score_pairs(dot_dir: Path, pairs: List[Tuple[str, str, str]],
                reference_dir: Optional[Path] = None,
                verbose: bool = True) -> pd.DataFrame:
    """Score every pair with `quality.score.score_model`.

    `reference_dir` defaults to `dot_dir`; it is separate so the same function
    can pair a simplified model with its full counterpart, which live in two
    directories.
    """
    reference_dir = reference_dir or dot_dir
    rows: List[Dict[str, Any]] = []
    for i, (generated, reference, direction) in enumerate(pairs, 1):
        gen_path = dot_dir / f"{generated}.dot"
        ref_path = reference_dir / f"{reference}.dot"
        row: Dict[str, Any] = {
            "item_generated": generated,
            "item_reference": reference,
            "direction": direction,
            "generated_path": str(gen_path),
            "reference_path": str(ref_path),
        }
        row.update(score_model(gen_path, ref_path))
        rows.append(row)
        if verbose:
            print(f"  [{i:>3}/{len(pairs)}] {generated} vs {reference}  "
                  f"sem {row.get('sem_score', float('nan')):.4f}", flush=True)
    frame = pd.DataFrame(rows)
    lead = [c for c in _IDENTITY if c in frame.columns]
    lead += [c for c in frame.columns if c.startswith("sem_")]
    return frame[lead + [c for c in frame.columns if c not in lead]]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score the semantic dimension over manipulated/original pairs.")
    ap.add_argument("--dataset-dir", default=str(config.PMO_DATASET_DIR),
                    help="PMo dataset root, the originals "
                         "(default: config.PMO_DATASET_DIR)")
    ap.add_argument("--manipulated-dir", default=str(DEFAULT_MANIPULATED),
                    help=f"the manipulated dataset "
                         f"(default: {DEFAULT_MANIPULATED.name})")
    ap.add_argument("--both-directions", action="store_true",
                    help="also score every pair the other way round, with the "
                         "original as the generated model")
    ap.add_argument("--out", default=str(DEFAULT_MANIPULATED_OUT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset_dir)
    dot_dir = dataset / "graphviz"
    items = sorted(p.stem for p in dot_dir.glob("*.dot"))
    if not items:
        sys.exit(f"no .dot files under {dot_dir}")

    manipulated = Path(args.manipulated_dir)
    # The dataset writes its folders capitalised; accept either spelling so a
    # renamed copy still works.
    gen_dir = next((manipulated / name for name in ("Graphviz", "graphviz")
                    if (manipulated / name).is_dir()), None)
    if gen_dir is None:
        sys.exit(f"no Graphviz/ folder under {manipulated}")
    manipulated_items = sorted(q.stem for q in gen_dir.glob("*.dot"))
    shared = [i for i in manipulated_items if i in items]
    if not shared:
        sys.exit(f"no item under {gen_dir} has an original in {dot_dir}")
    missing = [i for i in manipulated_items if i not in items]
    if missing:
        print(f"  ! no PMo original for {len(missing)} manipulated model(s): "
              f"{', '.join(missing[:5])}", file=sys.stderr)

    pairs = [(i, i, "manipulated-vs-original") for i in shared]
    if args.both_directions:
        pairs += [(i, i, "original-vs-manipulated") for i in shared]
        forward = score_pairs(gen_dir, pairs[:len(shared)], reference_dir=dot_dir,
                              verbose=not args.quiet)
        reverse = score_pairs(dot_dir, pairs[len(shared):], reference_dir=gen_dir,
                              verbose=not args.quiet)
        frame = pd.concat([forward, reverse], ignore_index=True)
    else:
        frame = score_pairs(gen_dir, pairs, reference_dir=dot_dir,
                            verbose=not args.quiet)
    print(f"Paired {len(shared)} manipulated models with their PMo originals")

    out = Path(args.out)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} rows × {len(frame.columns)} columns → {out}")

    # The pair list, for the BEF4LLM side to score exactly the same pairs.
    listing = out.with_name(f"{out.stem}_list.csv")
    frame[["item_generated", "item_reference", "direction"]].to_csv(listing,
                                                                    index=False)
    print(f"Wrote the pair list ({len(frame)} rows) → {listing}")

    legend = columns.legend_dataframe(
        [c for c in frame.columns if c not in _IDENTITY])
    legend_path = out.with_name(f"{out.stem}_columns.csv")
    legend.to_csv(legend_path, index=False)
    print(f"Wrote the column legend ({len(legend)} rows) → {legend_path}")


if __name__ == "__main__":
    main()
