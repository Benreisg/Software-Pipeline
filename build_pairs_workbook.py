#!/usr/bin/env python3
"""
build_pairs_workbook.py — the pair scoring as one readable workbook
===================================================================
`score_pmo_pairs.py` writes CSVs; this puts them side by side in Excel, one
column group per metric, one row per scored pair:

    python build_pairs_workbook.py             # → pmo_manipulated_pairs.xlsx
    python build_pairs_workbook.py --offset    # the older offset-pair artefacts
    python build_pairs_workbook.py --basic     # the basic-variation artefacts

Four readings of the same metric can stand next to each other, and which ones
appear depends on what has been produced:

    port         `pmo_semantic_pairs.csv` — this project's port of Table A.17
    repro        the `sem_*_bef` columns from the same file: BEF4LLM's own
                 algorithm, reproduced on this project's DOT graphs
    original     `*_bef4llm.csv` — their code over PMo's own BPMN
    disjoint     `*_bef4llm_disjoint.csv` — their code over the *same* graphs
                 converted from DOT, with the generated side's ids prefixed so
                 their id-keyed matching cannot collapse entries across the two
                 files (`dot_to_bpmn.py --id-prefix`)

The last two are the reason this exists. A gap between *repro* and *original*
can be the algorithm or it can be the serialisation; *disjoint* removes the
serialisation and the id collision from the comparison, so what remains is the
algorithm alone — and where `|repro − disjoint|` is 0, the reproduction is
exact.

Two sheets: `pairs` with the data, `columns` with one line per column. The
header of every data column carries its legend entry as a hover note.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
KEY = ["item_generated", "item_reference"]

# (label, port column, their column) — the reproduction column is the port one
# with `_bef` appended, and is picked up when it is there.
METRICS: List[Tuple[str, str, str]] = [
    ("SimSyn", "sem_label_sim_syntactic", "bef_sem_label_sim_syntactic"),
    ("SimSem", "sem_label_sim_semantic", "bef_sem_label_sim_semantic"),
    ("SimCon", "sem_label_sim_context", "bef_sem_label_sim_context"),
    # The graph-edit distance and common nodes/edges were dropped from the
    # pipeline on 2026-08-29; their columns stay listed so a re-run over older
    # CSVs still lays them out, and are skipped when the port no longer has
    # them (see `build`).
    ("GED", "sem_graph_edit_distance", "bef_sem_graph_edit_distance"),
    ("Qsem", "sem_score", "bef_sem_score"),
]

NOTES = {
    "port": "This project's port of the metric as Table A.17 defines it.",
    "repro": "The same algorithm with the two id spaces separated, so their "
             "matching cannot collapse entries (the sem_*_disjoint columns). "
             "Never part of sem_score; compare against a BEF4LLM run over "
             "prefix_bpmn_ids.py output.",
    "original": "Their code over PMo's own BPMN — a different serialisation, "
                "and their id-keyed matching collapses entries where the two "
                "files share a node id.",
    "disjoint": "Their code over the same graphs converted from DOT, with the "
                "generated side's ids prefixed: identical models, disjoint id "
                "spaces, so only the algorithm is left to differ.",
    "repro_disjoint": "The same reproduction with the two id spaces separated, "
                      "so their matching cannot collapse entries. This is the "
                      "column that belongs next to a BEF4LLM run over "
                      "prefix_bpmn_ids.py output.",
    "|repro - disjoint|": "How far the reproduction is from their algorithm "
                          "once serialisation and id collisions are out of the "
                          "way. 0 means exact.",
}


def _read(path: Path, what: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  ! missing {what}: {path.name}", file=sys.stderr)
        return None
    frame = pd.read_csv(path)
    for key in KEY:
        if key not in frame.columns:
            print(f"  ! {path.name} has no {key} column", file=sys.stderr)
            return None
        frame[key] = frame[key].astype(str).str.zfill(2)
    return frame


def build(port_csv: Path, original_csv: Path, disjoint_csv: Path,
          out: Path) -> Optional[Path]:
    port = _read(port_csv, "the port's pair scores")
    if port is None:
        return None
    sides: Dict[str, pd.DataFrame] = {}
    for name, path in (("original", original_csv), ("disjoint", disjoint_csv)):
        frame = _read(path, f"their {name} run")
        if frame is not None:
            sides[name] = frame

    data: Dict[str, List] = {k: list(port[k]) for k in KEY}
    docs: List[Tuple[str, str]] = [
        ("item_generated", "PMo item scored as the generated model."),
        ("item_reference", "PMo item scored as the ground truth."),
    ]
    for label, ours, theirs in METRICS:
        if ours not in port.columns:
            continue
        column = f"{label} — port"
        data[column] = list(port[ours])
        docs.append((column, f"{NOTES['port']}  ({ours})"))

        bef = f"{ours}_disjoint"
        repro = list(port[bef]) if bef in port.columns else None
        if repro is not None:
            column = f"{label} — repro"
            data[column] = repro
            docs.append((column, f"{NOTES['repro']}  ({bef})"))

        bef_d, repro_d = bef, repro
        if repro_d is not None:
            column = f"{label} — repro (ids getrennt)"
            data[column] = repro_d
            docs.append((column, f"{NOTES['repro_disjoint']}  ({bef_d})"))

        for name, frame in sides.items():
            if theirs not in frame.columns:
                continue
            merged = port[KEY].merge(frame[KEY + [theirs]], on=KEY, how="left")
            column = f"{label} — {name}"
            data[column] = list(merged[theirs])
            docs.append((column, f"{NOTES[name]}  ({theirs})"))

        base = repro_d if repro_d is not None else repro
        if base is not None and "disjoint" in sides and theirs in sides["disjoint"]:
            repro = base
            merged = port[KEY].merge(sides["disjoint"][KEY + [theirs]],
                                     on=KEY, how="left")
            column = f"{label} — |repro − disjoint|"
            data[column] = (pd.Series(repro) - merged[theirs]).abs()
            docs.append((column, NOTES["|repro - disjoint|"]))

    frame = pd.DataFrame(data)
    legend = pd.DataFrame(docs, columns=["column", "what it holds"])
    return _write(frame, legend, out)


def _write(frame: pd.DataFrame, legend: pd.DataFrame, out: Path) -> Optional[Path]:
    """Both sheets, header frozen and filterable, legend on the header cells."""
    try:
        from openpyxl.comments import Comment
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("  [xlsx] openpyxl is not installed — skipped", file=sys.stderr)
        return None

    notes = dict(zip(legend["column"], legend["what it holds"]))
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="pairs", index=False)
        legend.to_excel(writer, sheet_name="columns", index=False)
        sheet = writer.sheets["pairs"]
        sheet.freeze_panes = "C2"
        sheet.auto_filter.ref = sheet.dimensions
        for i, name in enumerate(frame.columns, start=1):
            cell = sheet.cell(row=1, column=i)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if name in notes:
                cell.comment = Comment(notes[name], "quality/columns.py")
            sheet.column_dimensions[get_column_letter(i)].width = \
                14 if i > 2 else 16
        cols = writer.sheets["columns"]
        cols.freeze_panes = "A2"
        cols.auto_filter.ref = cols.dimensions
        cols.column_dimensions["A"].width = 30
        cols.column_dimensions["B"].width = 110
    print(f"Wrote {len(frame)} pairs x {len(frame.columns)} columns -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basic", action="store_true",
                    help="use the basic-variation pair files instead of the offset ones")
    ap.add_argument("--offset", action="store_true",
                    help="use the offset pair files (pmo_semantic_pairs*) — "
                         "artefacts of the pairing score_pmo_pairs.py dropped "
                         "on 2026-08-30")
    ap.add_argument("--port", default=None)
    ap.add_argument("--original", default=None)
    ap.add_argument("--disjoint", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # `score_pmo_pairs.py` produces only the manipulated pairing since
    # 2026-08-30; the other two stems still work on artefacts from before it.
    if args.basic:
        stem = "pmo_semantic_basic_pairs"
    elif args.offset:
        stem = "pmo_semantic_pairs"
    else:
        stem = "pmo_manipulated_pairs"
    port = Path(args.port or PROJECT_DIR / f"{stem}.csv")
    original = Path(args.original or PROJECT_DIR / f"{stem}_bef4llm.csv")
    disjoint = Path(args.disjoint or PROJECT_DIR / f"{stem}_bef4llm_disjoint.csv")
    out = Path(args.out or PROJECT_DIR / f"{stem}.xlsx")
    if build(port, original, disjoint, out) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
