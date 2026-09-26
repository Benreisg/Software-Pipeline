"""
pmo_common_edge.py — common nodes and edges over manipulated/original pairs
==========================================================================
Metric 5 of the BEF4LLM semantic suite (Table A.17), scored for each
**manipulated** PMo model against its original, with the node half and the edge
half reported separately.

    python pmo_common_edge.py                  # → pmo_common_edge.csv
    python pmo_common_edge.py --out foo.csv --dataset-dir <path>

**Why this exists.** The metric was dropped from `quality/semantic.py` on
2026-08-29: BEF4LLM's own `common_percentage_similarity` is called with
`edges=False` and its matching binds every node at `threshold=0.0`, so their
published value is **1.0 for any two models** — verified on 155 pairs across
three pairings (55 unrelated offset pairs, 45 basic-vs-full, 55
manipulated-vs-original, all constant). A column that cannot separate two
unrelated processes from two nearly identical ones carries no information, and
with it went the only thing this port's own reading could have been validated
against.

This script is the other way round at that wall. `PMo_manipulated` postdates the
decision by a day, and its `manipulations.json` records **what was changed in
each model** — so the reading can be checked against the known change instead of
against their constant:

    19 items  rename only              → the edge half must stay exactly 1.0
    18 items  add_edge/remove_edge only → the node half must stay exactly 1.0
     9 items  add_edge + rename
     9 items  remove_edge + rename

That is the point of splitting the halves out. A rename-only item that loses
edge similarity, or an edge-only item that loses node similarity, is a defect in
the implementation — provable without asking BEF4LLM anything. `--check` runs
exactly that assertion and reports the violations.

── What the first run of it actually showed (2026-09-09) ────────────────────
**Only one of the two directions is evidence.** The 19 rename-only items do keep
the edge half at exactly 1.0, which is a real check and it passes: renames do not
leak into the edge term.

The other direction is **vacuous on this dataset**. `common_nodes` came back
**1.0 on all 55 items**, with zero unmatched nodes anywhere, so "the node half
must be 1.0" is satisfied by every item whatever the metric does. Two reasons,
both structural rather than a defect: `manipulations.json` only renames labels
and rewires edges, so it never changes a node count (|FOc| = |FOg| in all 55),
and the metric counts a node as matched or not — never *how well* — so a rename
that leaves any word overlap ("Place order" → "Submit order") keeps its node
bound. Consequence: the 19 rename-only items score exactly 1.0 on the **whole**
metric, and every distinct value in the corpus comes from the edge half alone.

Exercising the node half needs manipulations this dataset does not contain:
adding or deleting a flow object, or renaming a label to one with no word in
common. `report()` flags any half that never varies, so this stays visible
instead of reading as 55 passes.

Be clear about what this can and cannot show: it is a **plausibility check
against constructed changes, not a reproduction proof against the paper.** It
cannot restore the "reproduced 55/55" standard the other metrics are held to,
because there is no informative reference column to reproduce.

**The metric.** Nodes are the compared nodes of `quality/semantic.py` (non-gateway
flow objects), edges the contracted skeleton (gateways skipped over), and the
node correspondence is M^opt_Sem — the optimal matching under the semantic label
similarity, the same one the dimension uses everywhere else:

    1 - (unmatched nodes + unmatched edges) / (|FOc| + |FOg| + |Fc| + |Fg|)

which is `(2·matched nodes + 2·matched edges) / (that same denominator)`. The
halves are the same quotient over their own terms. **The combined value is not
the mean of the two halves** — it weighs each by how many elements it ranges
over, so an edge-rich model moves it more through its edges.

**The metric lives here, not in `quality/semantic.py`.** It was deleted from the
library deliberately; restoring it there would put it back into `sem_score` and
move every published semantic figure. Only the shared preparation is imported —
the contraction, the label representations and the matching — so this cannot
drift away from how the rest of the dimension reads a model.

Direction matters: the matching is not symmetric, so `m(a, b)` need not equal
`m(b, a)`. This scores manipulated-as-candidate against original-as-reference,
which is the direction `score_pmo_pairs.py` uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config

# A legacy console code page (cp1252) cannot encode the arrows this script
# prints, and an UnicodeEncodeError there would kill it *after* the scoring is
# done. Same guard `score_pmo_pairs.py` carries.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from quality import graph as g
from quality.semantic import (_common_nodes_edges, _optimal_matching,
                              _pair_matrix, _sim_sem, _Side)

DEFAULT_MANIPULATED = Path(__file__).resolve().parent / "PMo_manipulated"
DEFAULT_OUT = Path(__file__).resolve().parent / "pmo_common_edge.csv"
DEFAULT_BEF4LLM = (Path(__file__).resolve().parent
                   / "pmo_manipulated_pairs_bef4llm.csv")

_LEGEND: List[Tuple[str, str]] = [
    ("item", "PMo item id — the same file stem on both sides"),
    ("profile", "What manipulations.json changed in this item: 'rename only', "
                "'edges only', 'mixed' or 'none'"),
    ("n_rename", "Label changes applied to this item"),
    ("n_add_edge", "Sequence flows added to this item"),
    ("n_remove_edge", "Sequence flows removed from this item"),
    ("common_nodes_edges", "The metric: 1 - (unmatched nodes + unmatched edges) "
                           "/ (|FOc| + |FOg| + |Fc| + |Fg|). 1.0 = every node "
                           "and every edge matched."),
    ("common_nodes", "The node half alone: 2*matched nodes / (|FOc| + |FOg|). "
                     "Must be 1.0 on an item whose only changes are edges."),
    ("common_edges", "The edge half alone: 2*matched edges / (|Fc| + |Fg|). "
                     "Must be 1.0 on an item whose only changes are renames."),
    ("expected_full", "Which half the manipulation profile says must be 1.0 — "
                      "'common_edges', 'common_nodes', 'both' or '' when the "
                      "item was changed in both dimensions"),
    ("check", "ok / VIOLATION — whether expected_full actually is 1.0"),
    ("n_nodes_manipulated", "Compared nodes (non-gateway flow objects), "
                            "manipulated side"),
    ("n_nodes_original", "Compared nodes, original side"),
    ("n_nodes_matched", "|M^opt_Sem| — node pairs bound by the optimal matching"),
    ("n_edges_manipulated", "Contracted skeleton edges, manipulated side"),
    ("n_edges_original", "Contracted skeleton edges, original side"),
    ("n_edges_matched", "Candidate edges whose mapped endpoints and flow type "
                        "exist in the original"),
    ("n_unmatched_nodes", "Numerator term: nodes of both sides left unmatched"),
    ("n_unmatched_edges", "Numerator term: edges of both sides left unmatched"),
    ("bef_common_percentage", "BEF4LLM's own column for the same pair, joined "
                              "from their run when available. Constant 1.0 by "
                              "construction — this is the reference that could "
                              "not validate anything."),
    ("manipulated_path", "The scored manipulated .dot"),
    ("original_path", "The PMo original .dot it was compared against"),
    ("note", "Why a row has no value — a model that does not parse, or no "
             "original for the item"),
]


def common_nodes_and_edges(cand: _Side, ref: _Side) -> Dict[str, Any]:
    """The metric and every count it is built from, for one prepared pair."""
    n_c, n_r = len(cand.nodes), len(ref.nodes)

    # M^opt_Sem — the same node correspondence the rest of the dimension uses.
    # Zero-similarity pairs are dropped by `_optimal_matching`: they contribute
    # nothing to any sum and must not count as "the same node" structurally.
    matrix = _pair_matrix(
        n_c, n_r,
        lambda i, j: _sim_sem(cand.rep[i], cand.stems[i], cand.syn_text[i],
                              ref.rep[j], ref.stems[j], ref.syn_text[j],
                              cand.kind[i] == ref.kind[j]))
    matching = _optimal_matching(matrix)
    mapped = {cand.nodes[i]: ref.nodes[j] for (i, j) in matching}
    n_matched = len(matching)

    # An edge "is in" the other side iff both endpoints are matched and the
    # mapped edge exists there with the same flow type. The node mapping is
    # injective, so this count is symmetric between the sides.
    f_c, f_r = len(cand.skel), len(ref.skel)
    edges_matched = sum(
        1 for (src, dst, kind) in cand.skel
        if (mapped.get(src), mapped.get(dst), kind) in ref.skel)

    # The formula itself lives in `quality/semantic.py`, where the metric is a
    # live column again since 2026-09-09 — this script must not carry a second
    # copy of it that could drift.
    combined, nodes, edges = _common_nodes_edges(
        n_c, n_r, n_matched, f_c, f_r, edges_matched)

    return {
        # Full precision — see the precision policy in quality/__init__.
        "common_nodes_edges": combined,
        "common_nodes": nodes,
        "common_edges": edges,
        "n_nodes_manipulated": n_c,
        "n_nodes_original": n_r,
        "n_nodes_matched": n_matched,
        "n_edges_manipulated": f_c,
        "n_edges_original": f_r,
        "n_edges_matched": edges_matched,
        "n_unmatched_nodes": n_c + n_r - 2 * n_matched,
        "n_unmatched_edges": f_c + f_r - 2 * edges_matched,
    }


def load_manipulations(path: Path) -> Dict[str, Dict[str, int]]:
    """Per item, how many of each operation `manipulations.json` applied.

    Returns an empty dict when the file is absent: the metric is still
    computable then, only the cross-check over the profiles is not.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    counts: Dict[str, Dict[str, int]] = {}
    for item, changes in raw.items():
        per_item = {"rename": 0, "add_edge": 0, "remove_edge": 0}
        for change in changes:
            op = change.get("op", "")
            if op in per_item:
                per_item[op] += 1
        counts[str(item)] = per_item
    return counts


def profile_of(counts: Optional[Dict[str, int]]) -> Tuple[str, str]:
    """`(profile, expected_full)` for one item's manipulation counts.

    The second element is the cross-test: an item changed only by renames must
    leave the edge half untouched, and one changed only by edge operations must
    leave the node half untouched. An item changed in both dimensions predicts
    nothing, and an unmanipulated one predicts both.
    """
    if counts is None:
        return "unknown", ""
    renames = counts["rename"]
    edges = counts["add_edge"] + counts["remove_edge"]
    if renames and edges:
        return "mixed", ""
    if renames:
        return "rename only", "common_edges"
    if edges:
        return "edges only", "common_nodes"
    return "none", "both"


def score(manipulated_dir: Path, dataset_dir: Path,
          manipulations: Dict[str, Dict[str, int]],
          verbose: bool = True) -> pd.DataFrame:
    """One row per manipulated model, paired with its PMo original."""
    original_dir = dataset_dir / "graphviz"
    items = sorted(p.stem for p in manipulated_dir.glob("*.dot"))
    rows: List[Dict[str, Any]] = []

    for i, item in enumerate(items, 1):
        manipulated = manipulated_dir / f"{item}.dot"
        original = original_dir / f"{item}.dot"
        counts = manipulations.get(item)
        profile, expected = profile_of(counts)
        row: Dict[str, Any] = {
            "item": item,
            "profile": profile,
            "n_rename": counts["rename"] if counts else None,
            "n_add_edge": counts["add_edge"] if counts else None,
            "n_remove_edge": counts["remove_edge"] if counts else None,
            "expected_full": expected,
            "manipulated_path": str(manipulated),
            "original_path": str(original) if original.exists() else "",
            "note": "",
        }
        if not original.exists():
            row["note"] = f"no PMo original for item {item}"
            rows.append(row)
            if verbose:
                print(f"  [{i:>3}/{len(items)}] {item}: {row['note']}", flush=True)
            continue
        try:
            cand = _Side(g.load(manipulated))
            ref = _Side(g.load(original))
        except Exception as exc:  # noqa: BLE001 — a model that does not parse is a result
            row["note"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if verbose:
                print(f"  [{i:>3}/{len(items)}] {item}: {exc}", flush=True)
            continue

        row.update(common_nodes_and_edges(cand, ref))
        row["check"] = _verdict(row)
        rows.append(row)
        if verbose:
            flag = "" if row["check"] in ("ok", "") else f"  <- {row['check']}"
            print(f"  [{i:>3}/{len(items)}] item {item} ({profile}): "
                  f"metric {row['common_nodes_edges']:.4f}  "
                  f"nodes {row['common_nodes']:.4f}  "
                  f"edges {row['common_edges']:.4f}{flag}", flush=True)

    frame = pd.DataFrame(rows)
    order = [name for name, _ in _LEGEND if name in frame.columns]
    return frame[order + [c for c in frame.columns if c not in order]]


def _verdict(row: Dict[str, Any]) -> str:
    """Does the half the manipulation profile predicts to be untouched actually
    come back 1.0? Empty when the profile predicts nothing."""
    expected = row.get("expected_full", "")
    if not expected:
        return ""
    halves = ["common_nodes", "common_edges"] if expected == "both" else [expected]
    broken = [h for h in halves if row.get(h) != 1.0]
    if not broken:
        return "ok"
    return "VIOLATION: " + ", ".join(
        f"{h}={row[h]:.4f}" for h in broken)


def join_bef4llm(frame: pd.DataFrame, path: Path,
                 verbose: bool = True) -> pd.DataFrame:
    """Add BEF4LLM's own column for the same pairs, when their run is present.

    Not a validation — their `common_percentage_similarity` is 1.0 for any two
    models — but the contrast is the whole reason this script scores the halves
    itself, so it belongs in the same table when it can be had.
    """
    if not path.exists():
        if verbose:
            print(f"  ! no BEF4LLM run at {path.name} — their column is left empty")
        return frame
    theirs = pd.read_csv(path)
    if "bef_sem_common_percentage" not in theirs.columns:
        if verbose:
            print(f"  ! {path.name} has no bef_sem_common_percentage column")
        return frame
    theirs = theirs.copy()
    theirs["item"] = theirs["item_generated"].astype(str).str.zfill(2)
    merged = frame.merge(
        theirs[["item", "bef_sem_common_percentage"]].rename(
            columns={"bef_sem_common_percentage": "bef_common_percentage"}),
        on="item", how="left")
    order = [name for name, _ in _LEGEND if name in merged.columns]
    return merged[order + [c for c in merged.columns if c not in order]]


def report(frame: pd.DataFrame) -> None:
    """The cross-test, per manipulation profile."""
    scored = frame[frame["common_nodes_edges"].notna()]
    if scored.empty:
        print("\nNothing was scored.")
        return

    print(f"\n{'profile':<14} {'n':>3}  {'metric':>8} {'nodes':>8} {'edges':>8}"
          f"   what must be 1.0")
    print("-" * 68)
    for profile in ("rename only", "edges only", "mixed", "none", "unknown"):
        rows = scored[scored["profile"] == profile]
        if rows.empty:
            continue
        expected = rows["expected_full"].iloc[0] or "—"
        print(f"{profile:<14} {len(rows):>3}  "
              f"{rows['common_nodes_edges'].mean():>8.4f} "
              f"{rows['common_nodes'].mean():>8.4f} "
              f"{rows['common_edges'].mean():>8.4f}   {expected}")

    violations = scored[scored["check"].fillna("").str.startswith("VIOLATION")]
    checked = scored[scored["expected_full"].fillna("") != ""]
    print(f"\nCross-check: {len(checked) - len(violations)} of {len(checked)} "
          f"items behaved as their manipulation profile predicts.")
    for _, row in violations.iterrows():
        print(f"  item {row['item']} ({row['profile']}): {row['check']}")

    # A half that takes the same value on every model cannot be evidence for
    # anything: every item predicted to leave it at 1.0 does so whatever the
    # metric computes. Name the items that were therefore checked against
    # nothing, rather than letting them count as passes.
    for half in ("common_nodes", "common_edges"):
        if scored[half].nunique() > 1:
            continue
        vacuous = checked[checked["expected_full"].isin((half, "both"))]
        print(f"  ! {half} is {scored[half].iloc[0]:g} on all {len(scored)} "
              f"items — it does not vary, so the {len(vacuous)} item(s) "
              f"predicting it to be 1.0 are checked against nothing. This "
              f"dataset cannot exercise that half.")

    if "bef_common_percentage" in scored.columns:
        theirs = scored["bef_common_percentage"].dropna()
        if not theirs.empty:
            print(f"\nBEF4LLM's own column over the same {len(theirs)} pairs: "
                  f"min {theirs.min()}, max {theirs.max()} — "
                  f"{theirs.nunique()} distinct value(s). This port's: "
                  f"{scored['common_nodes_edges'].nunique()} distinct.")


def write_workbook(frame: pd.DataFrame, out: Path) -> None:
    """`common_edge` with the data, `columns` with the legend, notes on the
    headers — the same shape `score_graph_edit_distance.py` writes.

    The one addition: the `check` column is coloured, because a VIOLATION is the
    single thing in this table anyone opens it to look for.
    """
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    legend = pd.DataFrame(
        [{"column": name, "description": text} for name, text in _LEGEND
         if name in frame.columns],
        columns=["column", "description"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="common_edge", index=False)
        legend.to_excel(writer, sheet_name="columns", index=False)

        data, cols = writer.sheets["common_edge"], writer.sheets["columns"]
        header = Font(bold=True)
        described = dict(_LEGEND)
        data.freeze_panes = "B2"
        data.auto_filter.ref = data.dimensions
        for i, name in enumerate(frame.columns, 1):
            cell = data.cell(row=1, column=i)
            cell.font = header
            cell.alignment = Alignment(vertical="top")
            if name in described:
                cell.comment = Comment(described[name], "pmo_common_edge.py",
                                       height=130, width=340)
            data.column_dimensions[get_column_letter(i)].width = max(
                12, min(len(str(name)) + 2, 34))

        if "check" in frame.columns:
            column = list(frame.columns).index("check") + 1
            ok = PatternFill("solid", fgColor="E8F5E9")
            broken = PatternFill("solid", fgColor="FFCDD2")
            for row, value in enumerate(frame["check"], start=2):
                text = "" if pd.isna(value) else str(value)
                if not text:
                    continue
                data.cell(row=row, column=column).fill = (
                    broken if text.startswith("VIOLATION") else ok)

        cols.freeze_panes = "A2"
        for i in (1, 2):
            cols.cell(row=1, column=i).font = header
        cols.column_dimensions["A"].width = 24
        cols.column_dimensions["B"].width = 96
        for row in cols.iter_rows(min_row=2, min_col=2, max_col=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Common nodes and edges between each PMo model and its "
                    "manipulated counterpart.")
    ap.add_argument("--dataset-dir", default=str(config.PMO_DATASET_DIR),
                    help="the PMo dataset; its graphviz/ holds the originals")
    ap.add_argument("--manipulated-dir", default=str(DEFAULT_MANIPULATED),
                    help="PMo_manipulated, with Graphviz/ and manipulations.json")
    ap.add_argument("--bef4llm-csv", default=str(DEFAULT_BEF4LLM),
                    help="a BEF4LLM run over the same pairs, for the contrast "
                         "column; skipped when absent")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero when an item contradicts its "
                         "manipulation profile")
    ap.add_argument("--no-xlsx", action="store_true",
                    help="skip the workbook beside the CSV (needs openpyxl)")
    ap.add_argument("--quiet", action="store_true",
                    help="write the CSV, print no per-item line")
    args = ap.parse_args()

    manipulated = Path(args.manipulated_dir)
    # The dataset writes its folders capitalised; accept either spelling so a
    # renamed copy still works.
    gen_dir = next((manipulated / name for name in ("Graphviz", "graphviz")
                    if (manipulated / name).is_dir()), None)
    if gen_dir is None:
        sys.exit(f"no Graphviz/ folder under {manipulated}")
    if not sorted(gen_dir.glob("*.dot")):
        sys.exit(f"no .dot files under {gen_dir}")

    dataset = Path(args.dataset_dir)
    if not (dataset / "graphviz").is_dir():
        sys.exit(f"no graphviz/ folder under {dataset}")

    manipulations = load_manipulations(manipulated / "manipulations.json")
    if not manipulations:
        print(f"  ! no manipulations.json under {manipulated} — the metric is "
              f"still scored, but the cross-check over the profiles is not",
              file=sys.stderr)

    print(f"Scoring {gen_dir} against {dataset / 'graphviz'}")
    frame = score(gen_dir, dataset, manipulations, verbose=not args.quiet)
    frame = join_bef4llm(frame, Path(args.bef4llm_csv), verbose=not args.quiet)

    out = Path(args.out)
    frame.to_csv(out, index=False)
    print(f"\nWrote {len(frame)} rows x {len(frame.columns)} columns -> {out}")

    legend = pd.DataFrame(
        [{"column": name, "description": text} for name, text in _LEGEND
         if name in frame.columns],
        columns=["column", "description"])
    legend_path = out.with_name(f"{out.stem}_columns.csv")
    legend.to_csv(legend_path, index=False)
    print(f"Wrote the column legend ({len(legend)} rows) -> {legend_path}")

    if not args.no_xlsx:
        workbook = out.with_suffix(".xlsx")
        try:
            write_workbook(frame, workbook)
            print(f"Wrote the workbook -> {workbook}")
        except ImportError:
            print("  ! openpyxl is not installed — no workbook written "
                  "(pip install openpyxl, or pass --no-xlsx)", file=sys.stderr)
        except OSError as exc:
            # Almost always the workbook being open in Excel, which holds an
            # exclusive lock on Windows. The CSVs are already written by now, so
            # losing the workbook must not lose the run with it.
            print(f"  ! could not write {workbook.name}: {exc}\n"
                  f"    (close it if it is open in Excel; the CSVs are written)",
                  file=sys.stderr)

    if not args.quiet:
        report(frame)

    if args.check:
        violations = frame["check"].fillna("").str.startswith("VIOLATION").sum()
        if violations:
            sys.exit(f"{violations} item(s) contradict their manipulation profile")


if __name__ == "__main__":
    main()
