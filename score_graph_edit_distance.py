"""
score_graph_edit_distance.py — GED between each PMo model and its generation
============================================================================
One row per PMo item: the ground-truth `.dot` against the model an LLM
generated from that item's process description, scored with the graph edit
distance `quality/semantic.py` uses for metric 4.

    python score_graph_edit_distance.py \\
        --generated-dir "runs/20260824_235554/generated" \\
        --out "C:/Users/breis/Desktop/Graph_Edit_Distance.xlsx"

**What is compared.** Both sides are reduced the way the semantic dimension
reduces them: the flow objects without gateways, joined by the contracted
skeleton, so a gateway between two tasks becomes a direct edge and the distance
measures the process rather than its connector plumbing. Two nodes are
interchangeable when their label *and* their element kind agree, two edges when
the flow type does.

**The two numbers.** `ged_operations` is the raw edit distance — how many node
and edge insertions, deletions and substitutions turn one model into the other.
`ged_similarity` normalises it against the trivial edit script (delete every
node and edge of one, insert every node and edge of the other), so it lands in
[0, 1] and 1.0 means the two models are the same graph:

    ged_similarity = 1 − ged_operations / (|V_gen| + |V_ref| + |E_gen| + |E_ref|)

**It is an upper bound.** The exact distance is exponential to compute and does
not finish on two models this size, so this reads networkx's
`optimize_edit_paths`, which yields successively cheaper edit paths and cuts the
search on its own `timeout` — `quality/semantic.py::_GED_BUDGET_S`, five seconds
by default. The cheapest path found by then is an upper bound on the distance,
so `ged_similarity` is a *lower* bound on how alike the two models are, and
`truncated` records that it happened. A truncated value depends on how fast the
machine was and is not reproducible; an untruncated one is exact.

On real generations truncation is the common case, not the exception: on
`runs/20260824_235554`, 43 of the 55 pairs were cut at five seconds. Raising the
budget does not buy much — at 60 seconds half of a 12-pair sample was still cut,
for eight times the wall time.

Pairing is by the item id the generated file name starts with (`07__…gv` → item
`07`), which is also what `results.jsonl` records, so a run that generated
several models per item still lines up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import config

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from quality import graph as g
from quality import semantic

DEFAULT_GENERATED = Path("runs/20260824_235554/generated")
DEFAULT_OUT = Path.home() / "Desktop" / "Graph_Edit_Distance.xlsx"

_LEGEND = [
    ("item_id", "PMo item the pair belongs to"),
    ("generated_file", "The generated model's file name"),
    ("ground_truth_file", "The PMo ground-truth .dot"),
    ("ged_operations", "Raw graph edit distance: node and edge insertions, "
                       "deletions and substitutions between the two contracted "
                       "graphs. Lower is closer."),
    ("ged_similarity", "1 − ged_operations / (|V1|+|V2|+|E1|+|E2|). 1.0 = same "
                       "graph, 0.0 = nothing in common."),
    ("truncated", "True when the five-second search budget ran out before the "
                  "third edit path — the distance is then a cruder upper bound."),
    ("nodes_generated", "Flow objects without gateways, generated side"),
    ("nodes_ground_truth", "Flow objects without gateways, ground truth"),
    ("edges_generated", "Contracted skeleton edges, generated side"),
    ("edges_ground_truth", "Contracted skeleton edges, ground truth"),
    ("note", "Why a row has no distance — a model that does not parse, or no "
             "ground truth for the item"),
]


def score(generated_dir: Path, dataset_dir: Path,
          verbose: bool = True) -> pd.DataFrame:
    """One row per generated model, paired with its PMo ground truth."""
    rows: List[Dict[str, Any]] = []
    files = sorted(generated_dir.glob("*.gv")) + sorted(generated_dir.glob("*.dot"))
    for i, path in enumerate(files, 1):
        item = path.stem.split("__")[0]
        reference = dataset_dir / "graphviz" / f"{item}.dot"
        row: Dict[str, Any] = {
            "item_id": item,
            "generated_file": path.name,
            "ground_truth_file": reference.name if reference.exists() else "",
            "note": "",
        }
        if not reference.exists():
            row["note"] = f"no ground truth for item {item}"
            rows.append(row)
            continue
        try:
            gen_graph, ref_graph = g.load(path), g.load(reference)
        except Exception as exc:  # noqa: BLE001 — a model that does not parse is a result
            row["note"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if verbose:
                print(f"  [{i:>3}/{len(files)}] {path.name}: {exc}", flush=True)
            continue

        cand, ref = semantic._Side(gen_graph), semantic._Side(ref_graph)
        similarity, operations, truncated = semantic._graph_edit_distance(cand, ref)
        row.update({
            "ged_operations": operations,
            "ged_similarity": similarity,
            "truncated": truncated,
            "nodes_generated": len(cand.nodes),
            "nodes_ground_truth": len(ref.nodes),
            "edges_generated": len(cand.skel),
            "edges_ground_truth": len(ref.skel),
        })
        rows.append(row)
        if verbose:
            print(f"  [{i:>3}/{len(files)}] item {item}: "
                  f"{operations:.0f} operations, similarity {similarity:.4f}"
                  + ("  (truncated)" if truncated else ""), flush=True)

    frame = pd.DataFrame(rows)
    order = [name for name, _ in _LEGEND if name in frame.columns]
    return frame[order + [c for c in frame.columns if c not in order]]


def write_workbook(frame: pd.DataFrame, out: Path) -> None:
    """`distances` with the data, `columns` with the legend, notes on the
    headers — the same shape `score_pmo_dataset.py` writes."""
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    legend = pd.DataFrame(
        [{"column": name, "description": text} for name, text in _LEGEND
         if name in frame.columns])
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="distances", index=False)
        legend.to_excel(writer, sheet_name="columns", index=False)

        data, cols = writer.sheets["distances"], writer.sheets["columns"]
        header = Font(bold=True)
        described = dict(_LEGEND)
        data.freeze_panes = "B2"
        data.auto_filter.ref = data.dimensions
        for i, name in enumerate(frame.columns, 1):
            cell = data.cell(row=1, column=i)
            cell.font = header
            cell.alignment = Alignment(vertical="top")
            if name in described:
                cell.comment = Comment(described[name],
                                       "score_graph_edit_distance.py",
                                       height=130, width=340)
            data.column_dimensions[get_column_letter(i)].width = max(
                12, min(len(str(name)) + 2, 34))
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
        description="Graph edit distance between PMo models and their generations.")
    ap.add_argument("--generated-dir", default=str(DEFAULT_GENERATED))
    ap.add_argument("--dataset-dir", default=str(config.PMO_DATASET_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    generated = Path(args.generated_dir)
    if not generated.is_dir():
        sys.exit(f"no such directory: {generated}")
    print(f"Scoring the generations in {generated}", flush=True)

    frame = score(generated, Path(args.dataset_dir), verbose=not args.quiet)
    out = Path(args.out)
    write_workbook(frame, out)

    scored = frame["ged_similarity"].notna().sum() if "ged_similarity" in frame else 0
    print(f"\nWrote {len(frame)} rows x {len(frame.columns)} columns -> {out}")
    if scored:
        print(f"  {scored} pair(s) scored, mean similarity "
              f"{frame['ged_similarity'].mean():.4f}, "
              f"mean operations {frame['ged_operations'].mean():.1f}")
    unscored = len(frame) - scored
    if unscored:
        print(f"  {unscored} row(s) without a distance — see the note column")


if __name__ == "__main__":
    main()
