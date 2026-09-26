#!/usr/bin/env python3
"""
build_run_pipeline_comparison.py — one workbook, both pipelines, one run
========================================================================
Scores the generated models of a run against the PMo ground truth with **this
project's** semantic metrics and with **BEF4LLM's own implementation**, and
writes the two side by side.

    python build_run_pipeline_comparison.py runs/<run_id>

── What it reads ────────────────────────────────────────────────────────────
    <run>/csv/quality_semantic.csv          this pipeline (DOT vs DOT)
    <run>/bef4llm_semantic_native.csv       theirs, PMo's own .bpmn as truth
    <run>/bef4llm_semantic_converted.csv    theirs, the truth converted from
                                            the same .dot this pipeline reads
    <run>/bef4llm_semantic_disjoint.csv     theirs, native truth, generated
                                            model written with disjoint ids

The last three come from `score_pmo_bef4llm.py --pairs`, over BPMN produced by
`dot_to_bpmn.py`. Three variants, because one cannot answer the question on its
own: a difference in the *native* run is a difference in the metric **or** in
the serialisation. The *converted* run puts the same graphs on both sides. The
*disjoint* run additionally makes the two id spaces disjoint (`--id-prefix`) —
their node matching is a dict keyed by node id holding entries for both models,
so ids shared between the two files collapse into one entry, and two BPMN files
written to the same convention collide on `Task_1`, `Task_2`, … by construction
(12.7 ids per pair here).

── The two sheets ───────────────────────────────────────────────────────────
    comparison        every semantic metric, both pipelines, per item
    columns           what each column is

The workbook had a third sheet built around the dependency-graph overlap; that
metric was removed from this pipeline on 2026-08-26, so their two behavioural
columns now stand alone, with nothing on this side to compare them against.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# (our column, their column, label) — None where a side has no counterpart.
METRIC_MAP = [
    ("sem_label_sim_syntactic", "bef_sem_label_sim_syntactic",
     "Syntactic label similarity"),
    ("sem_label_sim_semantic", "bef_sem_label_sim_semantic",
     "Semantic label similarity"),
    (None, "bef_sem_label_sim_context", "Context similarity"),
    ("sem_graph_edit_distance", "bef_sem_graph_edit_distance",
     "Graph-edit distance"),
    ("sem_common_nodes_edges", "bef_sem_common_percentage",
     "Common nodes and edges"),
    (None, "bef_sem_causal_footprint", "Causal-footprint overlap"),
    (None, "bef_sem_dependency_graph", "Dependency-graph overlap"),
    ("sem_score", "bef_sem_score", "Qsem (headline)"),
]

COLUMN_NOTES = {
    "item_id": "PMo item the model was generated for.",
    "generated": "File stem in <run>/generated.",
}


def _read(path: Path, what: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  ! missing {what}: {path}", file=sys.stderr)
        return None
    frame = pd.read_csv(path)
    if "item_reference" in frame.columns:
        frame["item_id"] = frame.item_reference.astype(str).str.zfill(2)
    elif "item_id" in frame.columns:
        frame["item_id"] = frame.item_id.astype(str).str.zfill(2)
    return frame


def build(run_dir: Path, out: Path) -> None:
    ours = _read(run_dir / "csv" / "quality_semantic.csv", "this pipeline's scores")
    if ours is None:
        sys.exit("nothing to compare — run `python score_run.py <run>` first")
    native = _read(run_dir / "bef4llm_semantic_native.csv", "their native run")
    converted = _read(run_dir / "bef4llm_semantic_converted.csv",
                      "their identical-input run")
    disjoint = _read(run_dir / "bef4llm_semantic_disjoint.csv",
                     "their disjoint-id run")

    frame = ours.copy()
    if "item_generated" in (native.columns if native is not None else []):
        frame = frame.merge(
            native[["item_id", "item_generated"]].rename(
                columns={"item_generated": "generated"}),
            on="item_id", how="left")

    # ── sheet 1: every metric, both pipelines ──
    sheet: Dict[str, pd.Series] = {"item_id": frame.item_id}
    if "generated" in frame.columns:
        sheet["generated"] = frame.generated
    for our_col, their_col, label in METRIC_MAP:
        if our_col and our_col in frame.columns:
            sheet[f"{label} — port"] = frame[our_col]
        for source, tag in ((native, "original"),
                            (converted, "original, identical input"),
                            (disjoint, "original, disjoint ids")):
            if source is not None and their_col in source.columns:
                merged = frame[["item_id"]].merge(
                    source[["item_id", their_col]], on="item_id", how="left")
                sheet[f"{label} — {tag}"] = merged[their_col].values
    comparison = pd.DataFrame(sheet)

    # ── sheet 2: the legend ──
    notes = [{"column": c,
              "meaning": COLUMN_NOTES.get(c, _metric_note(c))}
             for c in comparison.columns]

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="comparison", index=False)
        pd.DataFrame(notes).to_excel(writer, sheet_name="columns", index=False)
        for sheet_name, data in (("comparison", comparison),
                                 ("columns", pd.DataFrame(notes))):
            worksheet = writer.sheets[sheet_name]
            for i, column in enumerate(data.columns, 1):
                width = max(len(str(column)), 12)
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=i).column_letter].width = min(
                        width + 2, 60)

    print(f"Wrote {len(comparison)} items x {len(comparison.columns)} columns "
          f"-> {out}")


def _metric_note(column: str) -> str:
    if column.endswith("— port"):
        return "This project's metric, DOT against the PMo ground-truth DOT."
    if column.endswith("— original, identical input"):
        return ("BEF4LLM's implementation on the ground truth converted from "
                "the same DOT.")
    if column.endswith("— original, disjoint ids"):
        return ("BEF4LLM's implementation with the two id spaces made "
                "disjoint, so their id-keyed matching cannot collide.")
    if column.endswith("— original"):
        return "BEF4LLM's implementation, PMo's own BPMN as the ground truth."
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare both pipelines on one run's generated models.")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "pipeline_comparison.xlsx"
    build(run_dir, out)


if __name__ == "__main__":
    main()
