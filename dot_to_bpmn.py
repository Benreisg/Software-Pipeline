#!/usr/bin/env python3
"""
dot_to_bpmn.py — export a DOT process model as BPMN 2.0 XML
===========================================================
BEF4LLM's implementation reads BPMN; this project reads DOT. Every comparison
between the two therefore either scores two *serialisations* of one model — and
cannot tell an implementation difference from a parser difference — or converts
one side. This converts.

    python dot_to_bpmn.py runs/<run_id>/generated --out <dir>
    python dot_to_bpmn.py model.gv --out <dir>

What comes out is what `quality/graph.py` read, written in their vocabulary:
the same nodes, the same classification (`kind_of`, `resolve_gateway_type`),
the same edges. Nothing is inferred that the DOT did not state, and nothing the
DOT stated is dropped — a node this project could not classify has no BPMN
element to map to and is reported rather than guessed at.

── Two decisions the format forces, both made to match PMo's own BPMN ───────
**Ids.** DOT identifies a node by its id, which in PMo (and in what the models
generate) is usually the label itself. BPMN separates the two, so ids are
assigned per kind in declaration order — `Task_1`, `ExclusiveGateway_2`,
`StartEvent_1` — exactly PMo's convention. This matters more than it looks:
BEF4LLM's node matching is greedy and keeps the *first* partner of equal
similarity, so the id scheme and the element order are part of their numbers.

**Element order.** Tasks, then gateways, then events, then the flows — the
order PMo's `bpmn/` files use, for the same reason.

**Line breaks.** A DOT label wraps with a literal `\\n`; the BPMN carries the
plain text, so the escape becomes a space. Without this the word counts of
every label similarity would differ from what their pipeline sees.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape, quoteattr

from quality import graph as g

# kind_of / resolve_gateway_type → BPMN element name
_EVENT_TAGS = {
    "start": "startEvent",
    "end": "endEvent",
    "event": "intermediateCatchEvent",
}
_GATEWAY_TAGS = {
    "exclusive": "exclusiveGateway",
    "parallel": "parallelGateway",
    "inclusive": "inclusiveGateway",
    "eventbased": "eventBasedGateway",
}
_ID_PREFIX = {
    "task": "Task",
    "startEvent": "StartEvent",
    "endEvent": "EndEvent",
    "intermediateCatchEvent": "IntermediateCatchEvent",
    "exclusiveGateway": "ExclusiveGateway",
    "parallelGateway": "ParallelGateway",
    "inclusiveGateway": "InclusiveGateway",
    "eventBasedGateway": "EventBasedGateway",
}
_NS = ('xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" '
       'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')


def _label(pg: g.ProcessGraph, node: str) -> str:
    """The text a reader sees, with the DOT line-break escape resolved."""
    return pg.label_of(node).replace("\\n", " ").replace("\\l", " ").strip()


def _tag_of(pg: g.ProcessGraph, node: str) -> str:
    """The BPMN element name for a node, or "" when it has none."""
    kind = pg.kind_of(node)
    if kind == "task":
        return "task"
    if kind in _EVENT_TAGS:
        return _EVENT_TAGS[kind]
    if kind == "gateway":
        return _GATEWAY_TAGS[g.resolve_gateway_type(pg, node)]
    return ""                      # pool anchors, unclassifiable nodes


def convert(pg: g.ProcessGraph, name: str = "",
            id_prefix: str = "") -> Tuple[str, List[str]]:
    """Return (BPMN 2.0 XML, list of nodes that had no element to map to).

    `id_prefix` prepends a string to every element id. BEF4LLM's node matching
    is a dict keyed by node id holding entries for *both* models, so an id that
    occurs in both files collapses into one entry — and two BPMN files written
    to the same convention collide on `Task_1`, `Task_2`, … by construction
    (12.7 ids per pair on this dataset). A prefix makes the two id spaces
    disjoint, which is the only way to see what their metric does without that
    accident.
    """
    # ── element ids, per kind, in declaration order ──
    tags: Dict[str, str] = {}
    skipped: List[str] = []
    for node in pg.nodes:
        tag = _tag_of(pg, node)
        if tag:
            tags[node] = tag
        elif node not in pg.markers:
            skipped.append(node)

    counters: Dict[str, int] = {}
    ids: Dict[str, str] = {}
    # Tasks, gateways, events — PMo's own element order.
    groups = (["task"],
              ["exclusiveGateway", "parallelGateway", "inclusiveGateway",
               "eventBasedGateway"],
              ["startEvent", "endEvent", "intermediateCatchEvent"])
    ordered: List[str] = []
    for group in groups:
        for node in pg.nodes:
            if tags.get(node) in group:
                tag = tags[node]
                counters[tag] = counters.get(tag, 0) + 1
                ids[node] = f"{id_prefix}{_ID_PREFIX[tag]}_{counters[tag]}"
                ordered.append(node)

    # ── flows ──
    sequence: List[Tuple[str, str, str]] = []      # (id, source, target)
    message: List[Tuple[str, str, str]] = []
    seen = set()
    for edge in pg.edges:
        source, target = edge
        if edge in seen or source not in ids or target not in ids:
            continue                # an edge to a pool anchor has no BPMN flow
        seen.add(edge)
        if pg.is_message_flow(edge):
            message.append((f"MessageFlow_{len(message) + 1}", source, target))
        else:
            sequence.append((f"SequenceFlow_{len(sequence) + 1}", source, target))

    incoming: Dict[str, List[str]] = {n: [] for n in ids}
    outgoing: Dict[str, List[str]] = {n: [] for n in ids}
    for flow_id, source, target in sequence:
        outgoing[source].append(flow_id)
        incoming[target].append(flow_id)

    # ── the document ──
    lines = ['<?xml version="1.0" ?>',
             f'<definitions {_NS} id="Definitions_1" '
             f'targetNamespace="http://www.omg.org/spec/BPMN/20100524/MODEL">']

    if message:
        # A message flow is only legal inside a collaboration; one participant
        # is enough to hold the process the flows belong to.
        lines.append('  <collaboration id="Collaboration_1">')
        lines.append('    <participant id="Participant_1" processRef="Process_1"/>')
        for flow_id, source, target in message:
            lines.append(f'    <messageFlow id="{flow_id}" '
                         f'sourceRef="{ids[source]}" targetRef="{ids[target]}"/>')
        lines.append('  </collaboration>')

    lines.append(f'  <process id="Process_1" name={quoteattr(name)} '
                 f'isExecutable="false">')
    for node in ordered:
        tag, node_id = tags[node], ids[node]
        lines.append(f'    <{tag} id="{node_id}" '
                     f'name={quoteattr(_label(pg, node))}>')
        for flow_id in incoming[node]:
            lines.append(f'      <incoming>{escape(flow_id)}</incoming>')
        for flow_id in outgoing[node]:
            lines.append(f'      <outgoing>{escape(flow_id)}</outgoing>')
        lines.append(f'    </{tag}>')
    for flow_id, source, target in sequence:
        lines.append(f'    <sequenceFlow id="{flow_id}" '
                     f'sourceRef="{ids[source]}" targetRef="{ids[target]}"/>')
    lines.append('  </process>')
    lines.append('</definitions>')
    return "\n".join(lines) + "\n", skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export DOT process models as BPMN 2.0 XML.")
    ap.add_argument("source", help="a .dot/.gv file or a directory of them")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--id-prefix", default="",
                    help="prepend this to every element id, so two converted "
                         "models cannot share one (their node matching is "
                         "keyed by id)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    source = Path(args.source)
    paths = (sorted(p for p in source.iterdir() if p.suffix in (".dot", ".gv"))
             if source.is_dir() else [source])
    if not paths:
        sys.exit(f"no .dot/.gv files under {source}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in paths:
        try:
            pg = g.load(path)
            xml, skipped = convert(pg, name=path.stem,
                                   id_prefix=args.id_prefix)
        except Exception as exc:                    # noqa: BLE001
            failed += 1
            print(f"  ! {path.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        (out_dir / f"{path.stem}.bpmn").write_text(xml, encoding="utf-8")
        if not args.quiet:
            note = f"   skipped {len(skipped)} unclassified" if skipped else ""
            print(f"  {path.name} -> {path.stem}.bpmn"
                  f"  ({len(pg.nodes)} nodes, {len(pg.edges)} edges){note}")
    print(f"Wrote {len(paths) - failed} of {len(paths)} models -> {out_dir}"
          + (f"  ({failed} failed)" if failed else ""))


if __name__ == "__main__":
    main()
