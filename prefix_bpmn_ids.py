#!/usr/bin/env python3
"""
prefix_bpmn_ids.py — the same BPMN model, under its own id space
=================================================================
Copies a folder of BPMN 2.0 files and puts a prefix in front of every element
id and every reference to one. Nothing else changes: element types, names,
flows, lanes, pools and the diagram stay byte-for-byte what they were.

    python prefix_bpmn_ids.py pmo-dataset/bpmn --out pmo-dataset/bpmn_prefixed
    python prefix_bpmn_ids.py PMo_manipulated/BPMN --out PMo_manipulated/BPMN_prefixed \\
        --prefix GEN_

**Why this exists.** BEF4LLM's `create_euquivalence_mapping_for_nodes` fills one
dict with entries for *both* models and keys it by node id, so an id present in
both collapses into a single entry and their similarity sums lose half their
terms. Two BPMN files written to one convention share ids by construction —
`Task_1`, `Task_2`, … — and on a dataset that deliberately keeps ids stable
across a manipulation, *every* id collides. Their number then depends on the
file's id scheme rather than on the models.

Prefixing one side removes that dependency without touching the models, so a
comparison measures the metric instead of the naming. It is the counterpart of
this project's `sem_*_bef_disjoint` columns, which take the same cut on the DOT
side — run both, and the two must agree.

Compare with `dot_to_bpmn.py --id-prefix`, which is for something else: that one
*converts* a generated DOT model into BPMN so their code can read it at all, and
its output is a different serialisation of the model. This script never leaves
BPMN, so nothing about the model can be lost on the way.

── What is rewritten ────────────────────────────────────────────────────────
    id="…"                     every element's own id
    sourceRef / targetRef      sequence and message flow endpoints
    processRef                 a participant's process
    bpmnElement               the diagram's link back to the model
    <incoming> / <outgoing>    the flow ids listed inside a node

`targetNamespace` and every attribute that is not an id are left alone, and so
is `id="Definitions_1"` — prefixing the document element would be harmless but
noisy. Ids are rewritten only where they are *known* ids of this file, so a
value that merely looks like one is never touched.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Set

# Attributes whose value is an id (or a reference to one).
_REF_ATTRS = ("sourceRef", "targetRef", "processRef", "bpmnElement",
              "attachedToRef", "default", "dataObjectRef", "categoryValueRef")
# Elements whose text content is a bare id.
_REF_TEXT = ("incoming", "outgoing", "dataInputRefs", "dataOutputRefs",
             "flowNodeRef", "sourceRef", "targetRef")

_ID_ATTR = re.compile(r'\bid="([^"]+)"')


def collect_ids(text: str) -> Set[str]:
    """Every id this file declares, minus the document element's own."""
    return {i for i in _ID_ATTR.findall(text) if not i.startswith("Definitions")}


def prefix_text(text: str, prefix: str) -> str:
    """The file with every known id and every reference to one prefixed."""
    ids = collect_ids(text)
    if not ids:
        return text
    # Longest first, so `Task_1` cannot be rewritten inside `Task_12`.
    alternatives = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))

    def attr(match: re.Match) -> str:
        return f'{match.group(1)}="{prefix}{match.group(2)}"'

    attrs = "|".join(("id",) + _REF_ATTRS)
    text = re.sub(rf'\b({attrs})="({alternatives})"', attr, text)

    def element(match: re.Match) -> str:
        return f"{match.group(1)}{prefix}{match.group(2)}{match.group(3)}"

    tags = "|".join(_REF_TEXT)
    text = re.sub(rf'(<(?:\w+:)?(?:{tags})>)\s*({alternatives})\s*(</)',
                  element, text)
    return text


def convert(source: Path, out_dir: Path, prefix: str, quiet: bool = False) -> int:
    files = sorted(source.glob("*.bpmn")) if source.is_dir() else [source]
    if not files:
        sys.exit(f"no .bpmn files under {source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        text = path.read_text(encoding="utf-8")
        n_ids = len(collect_ids(text))
        rewritten = prefix_text(text, prefix)
        (out_dir / path.name).write_text(rewritten, encoding="utf-8")
        if not quiet:
            print(f"  {path.name}: {n_ids} ids -> {prefix}*", flush=True)
    print(f"Wrote {len(files)} of {len(files)} models -> {out_dir}")
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a .bpmn file or a directory of them")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--prefix", default="GEN_",
                    help="what to put in front of every id (default: GEN_)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    convert(Path(args.source), Path(args.out), args.prefix, args.quiet)


if __name__ == "__main__":
    main()
