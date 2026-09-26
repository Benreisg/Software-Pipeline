"""
quality/pragmatic.py — Pragmatic quality (BEF4LLM's published metric set)
============================================================================
Can a human read and use the model? Comprehensibility rather than correctness.

The pragmatic dimension of BEF4LLM **as the paper defines it** — §4.2, the
formulas in Table A.16 and the thresholds in Table A.18 — ported to this
project's DOT-based `ProcessGraph`. Where the paper's definition and the
supplied code (`bef4llm/pragmatic_quality/`) disagree, this follows the paper,
and says so at the metric.

    group                 metrics
    ────────────────────  ──────────────────────────────────────────────────
    size                  TNN, TNG, TNSF, diameter
    density               density, average gateway degree, connectivity coeff.
    connector_interplay   gateway heterogeneity, control-flow complexity,
                          cross-connectivity
    partitionability      sequentiality, separability, depth
    cyclicity             — see below
    concurrency           token split
    other                 — (no metric in the published set)

**Cyclicity** is a category the paper names but deliberately leaves empty
(§4.2): "Cyclicity, frequently employed as a pragmatic measure, is not included
in the BEF4LLM framework because existing research does not provide multiple
thresholds for cyclicity metrics, which prevents categorization of these
metrics in a manner consistent with the other metrics used in the framework."
Not an oversight in this port.

**TNMF** (total number of message flows), the paper's fourth size metric, is
left out at the author's instruction — as it is in `quality/syntactic.py`, and
since 2026-08-16 no message-flow count is reported anywhere, not even as a
diagnostic. Message flows are perfectly expressible in DOT (PMo writes them as
dashed edges and `graph.py` reads all 20, agreeing with BEF4LLM's BPMN count on
55/55 models), but **no prompt template prescribes that notation**, so a
generated model cannot produce one deliberately and any count would measure the
prompt rather than the model. Only PMo items 23, 24 and 38 carry message flows
at all. Fourteen of the paper's fifteen metrics therefore remain, and
`prag_score` divides by 14.

Message-flow *detection* is untouched and load-bearing: it is what keeps these
edges out of TNSF, out of every control-flow degree, and out of the
cross-connectivity graph.

All fourteen are **model-internal**: no ground truth is consulted, so every
generation is scorable, including CSV generation-only runs. `ground_truth` is
accepted (and ignored) to keep the dimension signature uniform — see
*Comparing against the reference* at the bottom of this docstring.

── Scoring, exactly as the paper defines it ─────────────────────────────────
Each metric is banded into five groups against four empirically validated
thresholds (§4.2, Table A.18) by `normdesc` (lower is better) or `normasc`
(higher is better) in `quality/normalize.py`, then aggregated by the
**arithmetic mean over the metric set, equally weighted** (§4.5): "we sum the
individual metric scores and divide by the number of metrics". The divisor is
therefore the full metric count, not the count of metrics that happened to be
computable — a metric with nothing to measure (no gateways to average) scores
1.0 rather than dropping out, so two models are always scored on the same
scale.

── Three misprints in the paper, corrected here ─────────────────────────────
1. **Cross-connectivity's formula** is printed in Table A.16 as
   `normasc len(longest_loop)` — a cycle-length measure, which cannot be the
   metric: the row cites [53], Vanderfeesten et al., *On a quest for good
   process models: the cross-connectivity metric*, and the paper's §4.2 rules
   cyclicity out of the framework entirely. Implemented per [53].
2. **Sequentiality's formula** is printed as `normdesc Σ_{g∈G}(|in(g)|+|out(g)|)/|G|`
   — that is AGD's formula, repeated from three rows above. Implemented per
   Mendling (2008): arcs between two non-connector nodes over all arcs. Its
   thresholds are listed under "higher is better", which the AGD formula would
   contradict, so the misprint is unambiguous.
3. **§4.2's examples of "greater is better"** name *token split* and
   *connectivity coefficient*. Both are complexity measures — more concurrent
   tokens and more arcs per node make a model harder to read, not easier — and
   Table A.16 marks both `normdesc`. The prose is what is wrong; the table,
   the thresholds and the semantics agree with each other.

── Columns ──────────────────────────────────────────────────────────────────
  prag_<metric>            raw value           e.g. prag_cfc = 18
  prag_<metric>_score      0.0–1.0 band        e.g. prag_cfc_score = 0.75
  prag_group_<group>_score mean over that group's metrics
  prag_score               Qprag — the mean over all 14 metric scores
  prag_n_*                 diagnostics, so a band can be explained without
                           reopening the file

── Three things to know before reporting these numbers ──────────────────────
1. **This is not a correctness measure**, and the paper says so itself: "its
   score decreases as a process model becomes larger and more complex.
   However, 'simpler' is not always better". Eleven of the fourteen metrics
   score smallness or simplicity, so a degenerate model scores high — an empty
   graph gets `prag_score` 1.0. Pragmatic quality is only meaningful *next to*
   `syn_score`, never instead of it.
2. **The thresholds are calibrated for smaller models than PMo contains.**
   TNG's t4 = 6.49 while the reference models average 8.4 gateways, so most of
   them score 0.0 on it. The calibration table in the README gives the
   reference models' own scores — report generated models against *that*
   baseline, not against 1.0.
3. **Size is reported twice.** The paper files TNN/TNG/TNSF/diameter under
   pragmatic quality; this project implemented them earlier in
   `quality/syntactic.py` as `syn_tnn`/`syn_tng`/`syn_tnsf`/`syn_diameter`.
   Both come from the same `ProcessGraph` primitives and agree by construction
   (`prag_tnn == syn_tnn`, and so on) — the same four numbers under two
   prefixes. Which one survives into the thesis tables is an open decision —
   see the README.

── Comparing against the reference ──────────────────────────────────────────
BEF4LLM's pragmatic check takes one model and no reference, and so does
`evaluate()`. A deviation-style variant (|generated − reference| per metric) is
one call away — run `evaluate()` on the ground-truth `.dot` and subtract — but
it is deliberately not built in here, because it would double every column and
change what `prag_score` means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .graph import ProcessGraph, resolve_gateway_type
from .normalize import (AGD_THRESHOLDS, CFC_THRESHOLDS, CNC_THRESHOLDS,
                        CROSS_CONNECTIVITY_THRESHOLDS, DENSITY_THRESHOLDS,
                        DEPTH_THRESHOLDS, DIAMETER_THRESHOLDS, GH_THRESHOLDS,
                        SEPARABILITY_THRESHOLDS, SEQUENTIALITY_THRESHOLDS,
                        TNG_THRESHOLDS, TNN_THRESHOLDS,
                        TNSF_THRESHOLDS, TOKEN_SPLIT_THRESHOLDS, Thresholds,
                        normasc, normdesc)

# Expansion cap for the two simple-path searches (depth). Same value and same
# reason as `ProcessGraph.diameter`: longest-simple-path is NP-hard, so a
# pathological generated model must not hang the run. Never reached on PMo.
_PATH_BUDGET = 200_000

# The gateway types `graph.resolve_gateway_type` can return. Reported as
# diagnostic counts; PMo contains genuine event-based gateways — item 22 writes
# `"EventBasedGateway" [label="E", shape=diamond]`.
GATEWAY_TYPES = ("exclusive", "parallel", "inclusive", "eventbased")

# The paper's connector vocabulary: `p(l)` is the "gateway-type share for
# l ∈ {AND, XOR, OR}" (notation, Table A.16), and the same three sets index the
# control-flow complexity and the token split. There is no fourth share for an
# event-based gateway, and it needs none: it routes the token to exactly one of
# several catching events, i.e. it is XOR in its semantics — which is also how
# BEF4LLM's code treats it in the control-flow complexity.
CONNECTOR_CLASSES = ("AND", "XOR", "OR")
_CONNECTOR_CLASS = {
    "parallel": "AND",
    "exclusive": "XOR",
    "eventbased": "XOR",
    "inclusive": "OR",
}

GROUPS = ("size", "density", "connector_interplay", "partitionability",
          "cyclicity", "concurrency", "other")


@dataclass(frozen=True)
class MetricSpec:
    """One pragmatic metric: what it is called, where it belongs, how it scores."""
    key: str                                  # column suffix: prag_<key>
    label: str                                # human-readable name
    group: str
    thresholds: Optional[Thresholds]          # None → reported raw, not scored
    higher_is_better: bool = False            # True → normasc, else normdesc


# Order is the reporting order; the group column order follows GROUPS.
METRICS: Tuple[MetricSpec, ...] = (
    # ── size ──
    MetricSpec("tnn", "TNN (total number of nodes)", "size", TNN_THRESHOLDS),
    MetricSpec("tng", "TNG (total number of gateways)", "size", TNG_THRESHOLDS),
    MetricSpec("tnsf", "TNSF (total number of sequence flows)", "size", TNSF_THRESHOLDS),
    # **The scored diameter is the gateway-skipping longest path since
    # 2026-08-29** (author's instruction). Three readings of the same metric are
    # computed on every model and all three are reported; this is the one that
    # enters `prag_score`:
    #
    #   `prag_diameter_nogw`  longest simple start-to-end path, activities and
    #                         events counted, gateways not — BEF4LLM's own
    #                         counting convention, honestly maximised. Scored.
    #   `prag_diameter`       `_bef4llm_diameter`, their traversal reproduced
    #                         defects and all. Kept as the column that is
    #                         comparable with their published figures, and no
    #                         longer scored: its early `return`s make it report
    #                         the first backward path rather than the longest,
    #                         which on a generated model read a diameter of 1
    #                         where the longest path holds nine flow objects —
    #                         and since the band rewards smallness, that defect
    #                         flattered the model with a 1.00.
    #   `syn_diameter`        every flow object on the path, gateways included,
    #                         which is Table A.16's `max{|p|}` read literally.
    #
    # Swapping the key below is what selects the scored one; all three raw
    # values and all three bands are produced either way.
    MetricSpec("diameter_nogw",
               "Diameter (longest start-to-end path, gateways not counted)",
               "size", DIAMETER_THRESHOLDS),
    # ── density ──
    MetricSpec("density", "Density", "density", DENSITY_THRESHOLDS),
    MetricSpec("agd", "AGD (average gateway degree)", "density", AGD_THRESHOLDS),
    MetricSpec("cnc", "CNC (connectivity coefficient)", "density", CNC_THRESHOLDS),
    # ── connector interplay ──
    MetricSpec("gh", "GH (gateway heterogeneity)", "connector_interplay", GH_THRESHOLDS),
    MetricSpec("cfc", "CFC (control-flow complexity)", "connector_interplay", CFC_THRESHOLDS),
    # Labelled after what it measures here: BEF4LLM's implementation, which is
    # a connectedness ratio and not [53]'s cross-connectivity. See
    # `cross_connectivity` and the README section on it. The thresholds are the
    # published ones and are calibrated for [53], so this metric now scores 1.0
    # on every model of the reference set.
    MetricSpec("cc", "CC (cross-connectivity, BEF4LLM's reading)",
               "connector_interplay",
               CROSS_CONNECTIVITY_THRESHOLDS, higher_is_better=True),
    # ── partitionability ──
    MetricSpec("sequentiality", "Sequentiality", "partitionability",
               SEQUENTIALITY_THRESHOLDS, higher_is_better=True),
    MetricSpec("separability", "Separability", "partitionability",
               SEPARABILITY_THRESHOLDS, higher_is_better=True),
    MetricSpec("depth", "Depth", "partitionability", DEPTH_THRESHOLDS),
    # ── cyclicity: BEF4LLM defines the group but no metric in it, and publishes
    #    no thresholds — `compute_cyclicity_metrics` is a bare `pass`. Kept as an
    #    empty group rather than invented here, so the structure stays the
    #    suite's. Mendling's cyclicity (nodes on a cycle / all nodes) is the
    #    obvious thing to add if a threshold band is ever settled on.
    # ── concurrency ──
    MetricSpec("token_split", "Token split", "concurrency", TOKEN_SPLIT_THRESHOLDS),
    # ── other: empty in BEF4LLM too (`compute_other_metrics` is a bare `pass`).
)

METRIC_LABELS: Dict[str, str] = {m.key: m.label for m in METRICS}
GROUP_LABELS: Dict[str, str] = {
    "size": "Size",
    "density": "Density",
    "connector_interplay": "Connector interplay",
    "partitionability": "Partitionability",   # BEF4LLM spells it "partionability"
    "cyclicity": "Cyclicity",
    "concurrency": "Concurrency",
    "other": "Other metrics",
}


# ── the control-flow view every metric is computed on ───────────────────────
class _View:
    """The process model as these metrics need to see it.

    One graph, built once per `evaluate()`, so fifteen metrics cannot drift
    apart on what "a node" or "a flow" means:

    * **Nodes** are the flow objects — events, tasks, gateways. Pool/lane
      anchors (`markers`) and nodes of unreadable type are excluded, exactly as
      they are from `syn_tnn`.
    * **Arcs** are the deduplicated *sequence* flows between two flow objects.
      Message flows are not control flow, and BEF4LLM's metrics read
      `bpmn:incoming`/`bpmn:outgoing`, which in BPMN reference sequence flows
      only. A sequence flow with an endpoint outside the flow objects is
      dropped from this graph and counted in `n_flows_outside_fo` so the
      exclusion stays auditable — on PMo that count is 0.
    * **Degrees** are counted on those arcs. This is deliberately *not*
      `ProcessGraph.in_degree`, which counts every edge including message
      flows: the syntactic checks want "is this node wired correctly at all",
      these metrics want the control-flow degree.

    `tnsf` is counted on the full edge set, not on the restricted arc set — it
    is a size count of what the model contains.
    """

    def __init__(self, pg: ProcessGraph):
        self.pg = pg
        self.nodes: Set[str] = set(pg.flow_objects())
        all_sequence_flows = pg.sequence_flows()
        self.arcs: Set[Tuple[str, str]] = {
            (s, d) for s, d in all_sequence_flows
            if s in self.nodes and d in self.nodes and s != d
        }
        self.n_flows_outside_fo = sum(
            1 for s, d in all_sequence_flows
            if s not in self.nodes or d not in self.nodes
        )
        # Self-loops are dropped from the arc set too — they carry no path
        # information and would break the degree-based weights — but stay in
        # the TNSF count, which is a count of what the file contains.
        self.n_self_loops = sum(1 for s, d in all_sequence_flows if s == d)
        self.tnsf = len(all_sequence_flows)

        self.succ: Dict[str, List[str]] = {}
        self.pred: Dict[str, List[str]] = {}
        self._indeg: Dict[str, int] = {}
        self._outdeg: Dict[str, int] = {}
        for s, d in self.arcs:
            self.succ.setdefault(s, []).append(d)
            self.pred.setdefault(d, []).append(s)
            self._outdeg[s] = self._outdeg.get(s, 0) + 1
            self._indeg[d] = self._indeg.get(d, 0) + 1

        self.gateways: Set[str] = set(pg.gateways) & self.nodes
        self.splits: Set[str] = set(pg.splits()) & self.nodes
        self.joins: Set[str] = set(pg.joins()) & self.nodes
        self.gateway_type: Dict[str, str] = {
            g: resolve_gateway_type(pg, g) for g in self.gateways
        }
        # Gateways whose type the notation never stated and that fell back to
        # the BPMN default reading (see `graph.resolve_gateway_type`).
        self.n_type_defaulted = sum(
            1 for g in self.gateways if not pg.gateway_types.get(g)
            and self.gateway_type[g] == "exclusive"
        )

    def indeg(self, node: str) -> int:
        return self._indeg.get(node, 0)

    def outdeg(self, node: str) -> int:
        return self._outdeg.get(node, 0)

    def degree(self, node: str) -> int:
        return self.indeg(node) + self.outdeg(node)


# ── size ────────────────────────────────────────────────────────────────────
def _bef4llm_diameter(view: _View) -> int:
    """`pragmatic_quality_metrics.py::diameter`, reproduced (author's decision,
    2026-08-24). Verified against their output on 55 of 55 reference models.

    Their `countpaths` walks **backwards from the end events**, counting 1 for
    every node that is not a gateway and 0 for every gateway, and returns
    `maxcounter` — but three of its exits `return` from inside the loop over the
    node list, so the maximum over the alternatives is often never taken:

        for n in nodeslist:
            if n in longest_path_dict:   return longest_path_dict[n]
            if visited_pre.get(n, 0) > 2: return 0
            ...
            if predecessors == []:       return counter

    On top of that `longest_path_dict[n]` stores the result of *one* backward
    branch and is served to every later path through the same node. What comes
    out is the length of the first backward path that reaches a source, not the
    longest one: PMo's item 01 scores 4 where the longest gateway-free path is
    9 and the longest path over all flow objects is 15. The `> 2` counter also
    lets a cycle be walked twice, which is why items 14 and 50 come out *above*
    the true longest path.

    Kept as their traversal because this column is meant to be the comparable
    one; `diameter_nogw` and `syn_diameter` carry the two honest readings.
    """
    sequence_flows = view.pg.sequence_flows()
    predecessors: Dict[str, List[str]] = {}
    for source, target in view.pg.edges:          # file order, as theirs is
        if (source, target) in sequence_flows:
            row = predecessors.setdefault(target, [])
            if source not in row:
                row.append(source)
    ends = [n for n in view.pg.nodes if n in view.pg.ends]
    memo: Dict[str, int] = {}
    # How often each end event has been entered, kept *outside* the `visited`
    # reset below — the one addition to their traversal, and a termination
    # guard rather than a change of reading.
    #
    # Their only cycle guard is `visited[n] > 2`, and they clear `visited`
    # whenever the walk reaches an end event. A cycle that runs *through* an
    # end event therefore clears its own guard on every turn and recurses until
    # CPython's stack limit: run 20260831_182408 item 40 (deepseek) drew the
    # invisible label edge in both directions, `end_stop -> end_stop_label` and
    # back, and took the whole run down with a RecursionError. `memo` is no
    # help — it is written *after* the recursive call, so a node still on the
    # stack is not in it yet.
    #
    # This counter is the same `> 2` allowance applied to the reset itself. On
    # a model without such a cycle every end event is entered exactly once,
    # from the top-level call, so the walk is bit-for-bit what it was: the 55
    # reference figures are unchanged (checked over PMo and over the 4032
    # generations of that run).
    ends_seen: Dict[str, int] = {}

    def countpaths(nodes: List[str], visited: Dict[str, int]) -> int:
        maxcounter = 1
        for node in nodes:
            if node in memo:
                return memo[node]
            if visited.get(node, 0) > 2:
                return 0
            if node in view.pg.ends:
                if ends_seen.get(node, 0) > 2:
                    return 0
                ends_seen[node] = ends_seen.get(node, 0) + 1
                visited = {}
            counter = 0 if node in view.pg.gateways else 1
            before = predecessors.get(node, [])
            if not before:
                return counter
            for p in before:
                visited[p] = visited.get(p, 0) + 1
            longest = countpaths(before, visited)
            memo[node] = longest
            counter += longest
            maxcounter = max(maxcounter, counter)
        return maxcounter

    return countpaths(ends, {})


def size_metrics(view: _View) -> Dict[str, Any]:
    """TNN, TNG, TNSF, diameter.

    Identical by construction to `syntactic.size_metrics` — both read the same
    `ProcessGraph` primitives. BEF4LLM's fifth size metric, TNMF, is left out
    here as it is there; see the module docstring.

    **Diameter is BEF4LLM's own traversal** since 2026-08-24 (author's
    decision) — `_bef4llm_diameter` below, which reproduces their published
    figures on 55 of 55 reference models. `diameter_nogw` keeps the *longest*
    path over the non-gateway flow objects, and `syn_diameter` in the syntactic
    block keeps the longest path over every flow object, which is Table A.16's
    `max{|p|}` over start-to-end node sequences. The three now differ by
    construction — on PMo they average 9.02, 10.93 and 18.80 — so which one is
    quoted has to be stated. Since 2026-08-29 the scored one is `diameter_nogw`
    (see `METRICS`); `diameter` and `syn_diameter` ride along raw plus their own
    bands.
    """
    diameter = _bef4llm_diameter(view)
    _, truncated = view.pg.diameter(budget=_PATH_BUDGET)
    diameter_nogw, truncated_nogw = view.pg.diameter(budget=_PATH_BUDGET,
                                                     count_gateways=False)
    return {
        "tnn": len(view.nodes),
        "tng": len(view.gateways),
        "tnsf": view.tnsf,
        "diameter": diameter,
        "diameter_nogw": diameter_nogw,
        "_diameter_truncated": truncated or truncated_nogw,
    }


# ── density ─────────────────────────────────────────────────────────────────
def density(view: _View, tnn: int, tnsf: int) -> Optional[float]:
    """Δ = |F^S| / (|N| · (|N| − 1)) — arcs against all arcs that could exist.

    Undefined for fewer than two nodes; `None` rather than a division by zero
    (BEF4LLM raises there).
    """
    if tnn < 2:
        return None
    return tnsf / (tnn * (tnn - 1))


def average_gateway_degree(view: _View) -> Optional[float]:
    """AGD = Σ_{g ∈ G} (|in(g)| + |out(g)|) / |G|   (Table A.16, #7).

    `None` when the model has no gateways — the average of an empty set is not
    0, and reporting a raw 0 (as BEF4LLM's code does) would state a measurement
    that was never made. The *score* is 1.0 either way: the aggregation treats
    a metric with nothing to measure as unpenalised, and `normdesc(0)` against
    thresholds starting at 3.67 is 1.0 as well.
    """
    if not view.gateways:
        return None
    return sum(view.degree(g) for g in view.gateways) / len(view.gateways)


def connectivity_coefficient(view: _View, tnn: int, tnsf: int) -> Optional[float]:
    """CNC = |F^S| / |N| — arcs per node."""
    if tnn == 0:
        return None
    return tnsf / tnn


# ── connector interplay ─────────────────────────────────────────────────────
def gateway_heterogeneity(view: _View) -> Optional[float]:
    """GH — **BEF4LLM's own reading** (author's decision, 2026-08-24):

        GH = −Σ_{l ∈ {AND, XOR, OR, event-based}} p(l) · log₃ p(l)

    over the gateways that fan out or in, `0.0` when there are none.

    Two departures from Table A.16, both taken from their code:

    * **Event-based gateways are a fourth class.** The paper's `p(l)` runs over
      `l ∈ {AND, XOR, OR}` — three classes, which is also why the logarithm is
      base 3 — and an event-based gateway is a decision, so it belongs to the
      XOR share. `gateway_heterogeneity()` over there sums `parallel_val`,
      `exlusive_val`, `event_val` and `inclusive_val`, keeping base 3 for four
      classes. PMo's item 22 is where it shows: two exclusive gateways and one
      event-based are one class here (entropy 0) and two classes there (0.5794).
    * **A gateway that neither fans out nor in is not counted.**
      `get_number_gateway_types` files a gateway under `split` when `out > 1`,
      else under `join` when `in > 1`, and drops it otherwise — so a 1:1
      pass-through leaves the distribution. No reference model has one.

    What this cannot reproduce is their treatment of a **complex gateway**:
    their type table has four keys, so a `complexGateway` is skipped and their
    divisor shrinks. In DOT a complex gateway is a bare diamond — item 37's
    `"First parts arrived"` is one — with nothing to tell it apart from an
    exclusive one, so it stays in the distribution here and that model reads
    0.2773 against their 0.2959.
    """
    counted = [g for g in view.gateways
               if view.outdeg(g) > 1 or view.indeg(g) > 1]
    if not counted:
        return 0.0
    total = len(counted)
    result = 0.0
    for gateway_type in ("parallel", "exclusive", "inclusive", "eventbased"):
        n = sum(1 for g in counted if view.gateway_type[g] == gateway_type)
        if n:
            p = n / total
            result += p * math.log(p, 3)
    # Clamped at 0: a single-type model gives -0.0, which is mathematically
    # the same number but prints as "-0.0" in every table.
    return max(0.0, -result)


def control_flow_complexity(view: _View) -> int:
    """CFC = Σ over the split gateways of their branching factor:

        AND-split  → 1                    (one outcome: all branches)
        XOR-split  → |out(c)|             (one outcome per branch)
        OR-split   → 2^|out(c)| − 1       (one per non-empty branch subset)

    Event-based splits count as XOR, as in BEF4LLM. A sum over an empty set is
    genuinely 0, so a model without gateways scores CFC 0 rather than `None`.

    Which gateways are splits comes from `graph.py::_gateway_role` — declared
    intent first, topology second — not from the out-degree alone; see the note
    there. BEF4LLM instead calls anything with out > 1 a split, which would
    count a malformed `AND_SPLIT_1` wired 2→1 as a join.
    """
    total = 0
    for g in view.splits:
        out = view.outdeg(g)
        t = view.gateway_type.get(g, "exclusive")
        if t == "parallel":
            total += 1
        elif t == "inclusive":
            total += (2 ** out) - 1
        else:                                    # exclusive, event-based
            total += out
    return total


def _node_weight(view: _View, node: str) -> float:
    """The cross-connectivity weight of a node (Vanderfeesten et al., 2008).

    **Unused since 2026-08-24** — `cross_connectivity` below reproduces
    BEF4LLM's implementation, which never applies these weights. Kept because
    it is the published definition and switching back needs it.

    A node's weight is the probability that the flow actually continues through
    any one of its arcs: an XOR gateway with degree d passes 1/d of it, an AND
    gateway passes all of it, an activity or event is a straight-through 1.
    """
    if node not in view.gateways:
        return 1.0
    d = view.degree(node)
    if d == 0:
        return 1.0
    t = view.gateway_type.get(node, "exclusive")
    if t == "parallel":
        return 1.0
    if t == "inclusive":
        # 1/(2^d − 1) + ((2^d − 2)/(2^d − 1)) · (1/d)
        span = (2 ** d) - 1
        return 1 / span + ((span - 1) / span) * (1 / d)
    return 1 / d                                  # exclusive, event-based


def cross_connectivity(view: _View, tnn: int) -> Optional[float]:
    """CC as **BEF4LLM's own implementation** computes it (author's decision,
    2026-08-24):

        CC = (|{(a, b) : a ≠ b, b reachable from a}| + |N|) / (|N| · (|N| − 1))

    which is the share of ordered node pairs joined by a directed path, with
    every node's pair with itself counted as 1.

    **This is not Vanderfeesten's metric, and not by accident on our side.**
    `bef4llm/pragmatic_quality/pragmatic_quality_metrics.py::cross_connectivity`
    builds its arc weights like this:

        edges = graph.edges(data='id')      # 3-tuples (u, v, id)
        for tuple in edges:
            if len(tuple) == 2:             # never true
                ...
                edge_dict[edge_id] = weight_edge

    `edge_dict` therefore stays empty for every model, and the path value
    `max_weight_paths` computes multiplies nothing: every path is worth 1. The
    node weights above it — 1/d for an XOR gateway, 1 for an AND, the
    `1/(2^d−1) + …` term for an OR — are computed and never used. Since
    networkx 3.3 `all_simple_paths(g, n, n)` yields the trivial path `[n]`, so
    each node's pair with itself contributes 1 as well; that is the `+ |N|`.

    Verified against their output on the reference set: this expression
    reproduces `bef_prag_cross_connectivity` to 1e-9 on **52 of 55** models.
    The three that differ (items 23, 24, 38) are the multi-pool models, where
    their node set and ours differ over message-flow endpoints — the same
    serialisation difference that shows up in the other size metrics.

    Reproduced here rather than corrected because this column is meant to be
    the comparable one. What Vanderfeesten (2008) defines — the strongest path
    between two nodes, a path's value being the product of its arc weights and
    an arc's weight the product of its endpoint weights — is a different
    number: 0.111 on average over PMo against 0.566 for this one. `_node_weight`
    below is kept for it, unused by this function, so the switch back is two
    lines.

    More is better either way, so the banding stays `normasc`. `None` for fewer
    than two nodes.
    """
    if tnn < 2:
        return None

    succ: Dict[str, List[str]] = {}
    for source, target in view.arcs:
        succ.setdefault(source, []).append(target)

    connected = 0
    for source in view.nodes:
        seen: Set[str] = set()
        stack = list(succ.get(source, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(succ.get(node, ()))
        # A path back to the start is not the start's pair with itself: that
        # one is counted below for every node, cycle or not, exactly as the
        # trivial path `[n]` does it over there.
        connected += sum(1 for n in seen if n != source)

    return (connected + len(view.nodes)) / (tnn * (tnn - 1))


# ── partitionability ────────────────────────────────────────────────────────
def sequentiality(view: _View) -> Optional[float]:
    """Ξ as **BEF4LLM's own implementation** computes it (author's decision,
    2026-08-24):

        Ξ = |arcs between two non-connector nodes| / |arcs|

    where *arcs* are all edges of the process graph between two flow
    objects — **message flows included**, in the numerator as in the
    denominator.

    That is not what its own comment says. `pragmatic_quality_metrics.py`
    reads

        # message flows are not taken into account, only sequence flows!
        for edge in model.sequence_flow_id_edge_mapping:

    but `map_edges_to_id` builds that mapping from *every* edge of the graph:

        self.sequence_flow_id_edge_mapping = {
            data["id"]: (u, v) for u, v, data in self.process_graph.edges(data=True)}

    so a message flow between two tasks counts as a sequential arc, and the
    divisor is the edge count of the whole graph. Only message flows whose
    endpoints are both flow objects are in it — one that ends at a collapsed
    black-box pool has no node on the other side, here as there.

    On the reference set the difference shows only in the multi-pool models:
    item 23 is 9/19 instead of 6/16, item 24 is 33/37 instead of 24/28. The
    other 53 have no message flows and are unaffected.

    Also note that the paper misprints the formula for this row — Table A.16
    repeats AGD's `normdesc Σ_{g∈G}(|in(g)|+|out(g)|)/|G|` from three rows
    above. Mendling (2008) is what both implementations follow, thresholds
    included: more sequence is better, so it is banded with `normasc`.

    `None` when the model has no arcs between flow objects at all.
    """
    arcs = {(source, target) for source, target in view.pg.unique_edges
            if source in view.nodes and target in view.nodes and source != target}
    if not arcs:
        return None
    plain = sum(1 for source, target in arcs
                if source not in view.gateways and target not in view.gateways)
    return plain / len(arcs)


def _articulation_points(nodes: Iterable[str],
                         adj: Dict[str, Set[str]]) -> Set[str]:
    """Cut vertices of the undirected control-flow graph — Tarjan, iterative.

    Iterative rather than recursive because a generated model can be a long
    chain and CPython's recursion limit is not a modelling property.
    Implemented here rather than via `networkx.articulation_points` to keep the
    project's dependencies at pydot + pandas, which is the only reason the
    metric layer was rebuilt in the first place.
    """
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {}
    points: Set[str] = set()
    timer = 0

    for root in sorted(nodes):
        if root in disc:
            continue
        disc[root] = low[root] = timer
        timer += 1
        parent[root] = None
        root_children = 0
        stack: List[Tuple[str, Any]] = [(root, iter(sorted(adj.get(root, ()))))]
        while stack:
            node, neighbours = stack[-1]
            descended = False
            for nb in neighbours:
                if nb == node:
                    continue
                if nb not in disc:
                    parent[nb] = node
                    disc[nb] = low[nb] = timer
                    timer += 1
                    if node == root:
                        root_children += 1
                    stack.append((nb, iter(sorted(adj.get(nb, ())))))
                    descended = True
                    break
                if nb != parent[node]:
                    low[node] = min(low[node], disc[nb])
            if not descended:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[node])
                    # The root is judged by its child count, not by this test.
                    if parent[p] is not None and low[node] >= disc[p]:
                        points.add(p)
        if root_children > 1:
            points.add(root)
    return points


def separability(view: _View, tnn: int) -> Tuple[Optional[float], int]:
    """Π = |cut vertices| / (|N| − 2), plus the cut-vertex count itself.

    A cut vertex is a node whose removal disconnects the model; the more of
    them, the more the model reads as a chain of independent parts rather than
    one tangle. More is better.

    The two nodes subtracted are the start and end event, which are cut
    vertices in any connected model and would otherwise be counted as free
    structure. `None` for fewer than three nodes.

    Computed on the *sequence-flow* graph, undirected. BEF4LLM runs it over the
    whole graph including message flows, which in a multi-pool model merges two
    otherwise separate participants into one component and hides cut vertices
    on both sides.
    """
    if tnn < 3:
        return None, 0
    adj: Dict[str, Set[str]] = {n: set() for n in view.nodes}
    for s, d in view.arcs:
        adj[s].add(d)
        adj[d].add(s)
    cut = _articulation_points(view.nodes, adj)
    return len(cut) / (tnn - 2), len(cut)


def depth(view: _View) -> Tuple[int, bool]:
    """Λ — **BEF4LLM's own `depth()`**, reproduced (author's decision,
    2026-08-24). Returns (depth, truncated); `truncated` is always False, this
    walk has no budget to exhaust.

        Λ = min( max over the start events of walk(+1 per split, −1 per join),
                 max over the start events of walk(+1 per join, −1 per split) )

    Four things in that walk are theirs, and all four are why the number comes
    out roughly twice Mendling's on the reference set (3.91 against 1.73 on
    average, higher on 43 of the 55 models):

    * **A join cancels nothing.** `maxcounter` starts at 0 and is only ever
      raised — `if counter > maxcounter` — so the −1 a join contributes can
      never take effect. What is counted is the number of splits *along* a
      path, not how many of them are open at once.
    * **Subtrees are added on.** `new_max = maxcounter + walk(succ)` puts the
      subtree's value on top of the running count, so two blocks in sequence,
      each opened and closed again, come out as 2.
    * **The out-depth walks forward** from the start events, not backward from
      the ends: it counts joins on a forward path rather than joins still to be
      closed.
    * **One `visited` set for the whole traversal** — a node reached again on
      another branch returns 0, so the result depends on the order the
      successors come out of the graph.

    Their split/join lists are `get_gateways_by_types`': a gateway is a split
    when `out > 1`, else a join when `in > 1`, so one that fans both ways is
    only a split. Reproduced as such, which is why this function does not use
    `view.splits`.

    Mendling's reading — per node the in-depth (open splits on arrival, closed
    again by joins) against the out-depth backward from the ends, `min` of the
    two, maximum over the nodes, simple paths only — is no longer computed for
    this column.
    """
    successors: Dict[str, List[str]] = {}
    for source, target in view.pg.edges:          # file order, as theirs is
        row = successors.setdefault(source, [])
        if target not in row:
            row.append(target)

    splits = {g for g in view.pg.gateways if view.pg.out_degree(g) > 1}
    joins = {g for g in view.pg.gateways
             if g not in splits and view.pg.in_degree(g) > 1}
    starts = [n for n in view.pg.nodes if n in view.pg.starts]

    def walk(node: str, visited: Set[str],
             opening: Set[str], closing: Set[str]) -> int:
        if node in visited:
            return 0
        visited.add(node)
        counter = 0
        if node in opening:
            counter += 1
        if node in closing:
            counter -= 1
        maxcounter = max(0, counter)
        for nxt in successors.get(node, ()):
            reached = maxcounter + walk(nxt, visited, opening, closing)
            if reached > maxcounter:
                maxcounter = reached
        return maxcounter

    in_depth = max((walk(s, set(), splits, joins) for s in starts), default=0)
    out_depth = max((walk(s, set(), joins, splits) for s in starts), default=0)
    return min(in_depth, out_depth), False


# ── concurrency ─────────────────────────────────────────────────────────────
def token_split(view: _View) -> int:
    """TS — **BEF4LLM's own `token_splits()`**, reproduced (author's decision,
    2026-08-24):

        TS = 2 · Σ_{g ∈ AND-splits} (|out(g)| − 1)

    Two departures from Table A.16, both theirs:

    * **The sum is taken twice.** Their function carries the same block twice,
      a copy-paste, so every parallel split contributes its fan-out twice:

          for conn in gateways[parallelGateway]["split"]:
              result = result + (len(graph.nodes[conn]["outgoing"]) - 1)

          for conn in gateways[parallelGateway]["split"]:   # again
              result = result + (len(graph.nodes[conn]["outgoing"]) - 1)

      On 52 of the 55 reference models their figure is therefore exactly twice
      the published formula's.
    * **OR-splits are not counted.** Table A.16 sums over `G^S_AND ∪ G^S_OR`;
      their loop only walks the parallel gateways. Items 21, 28 and 32 each
      draw one inclusive gateway, which is why the doubling does not come out
      even there.

    Splits are theirs as well (`get_gateways_by_types`: `out > 1`), not
    `view.splits`. XOR and event-based splits pass the single token on and add
    nothing under either reading.

    Note §4.2's prose names token split as an example of a metric where
    "greater is better" — Table A.16 marks it `normdesc`, its thresholds
    ascend, and more concurrency plainly makes a model harder to read, so the
    prose is the part that is wrong. The banding is unaffected by this change.
    """
    total = 0
    for gateway in view.gateways:
        if view.gateway_type.get(gateway) != "parallel":
            continue
        if view.pg.out_degree(gateway) <= 1:
            continue
        total += max(view.outdeg(gateway) - 1, 0)
    return 2 * total


# ── scoring ─────────────────────────────────────────────────────────────────
# What a metric scores when its defining set is empty — no gateways to average,
# fewer than two nodes to connect. The paper leaves the case open; 1.0 ("nothing
# to measure, nothing held against the model") keeps the divisor at the full
# metric count, which is what §4.5 requires, and matches the reference
# implementation, whose 0 for AGD and GH bands to 1.0 anyway. The *raw* value
# stays `None` so the column never states a measurement that was not made.
_UNMEASURABLE_SCORE = 1.0


def _score_of(spec: MetricSpec, raw: Optional[float]) -> Optional[float]:
    """Band `raw` for one metric, or `None` when it is not measurable."""
    if raw is None or spec.thresholds is None:
        return None
    return (normasc if spec.higher_is_better else normdesc)(raw, spec.thresholds)


def evaluate(generated: ProcessGraph,
             ground_truth: Optional[ProcessGraph] = None) -> Dict[str, Any]:
    """Return the `prag_`-prefixed pragmatic-quality columns for one model.

    `ground_truth` is unused: BEF4LLM's pragmatic check is model-internal, so
    every generation is scorable whether or not a reference exists. The
    parameter stays for signature symmetry with `semantic.evaluate`.
    """
    view = _View(generated)

    size = size_metrics(view)
    tnn, tnsf = size["tnn"], size["tnsf"]
    sep, n_cut = separability(view, tnn)
    nesting, depth_truncated = depth(view)

    raw: Dict[str, Optional[float]] = {
        "tnn": tnn,
        "tng": size["tng"],
        "tnsf": tnsf,
        "diameter_nogw": size["diameter_nogw"],
        "density": density(view, tnn, tnsf),
        "agd": average_gateway_degree(view),
        "cnc": connectivity_coefficient(view, tnn, tnsf),
        "gh": gateway_heterogeneity(view),
        "cfc": control_flow_complexity(view),
        "cc": cross_connectivity(view, tnn),
        "sequentiality": sequentiality(view),
        "separability": sep,
        "depth": nesting,
        "token_split": token_split(view),
    }

    out: Dict[str, Any] = {}
    scores: Dict[str, float] = {}      # every metric, unmeasurable ones at 1.0
    n_measured = 0
    for spec in METRICS:
        value = raw[spec.key]
        # Stored at full precision — see the precision policy in quality/__init__.
        out[f"prag_{spec.key}"] = value
        score = _score_of(spec, value)
        out[f"prag_{spec.key}_score"] = score
        if score is None:
            scores[spec.key] = _UNMEASURABLE_SCORE
        else:
            scores[spec.key] = score
            n_measured += 1

    # §4.5: "we sum the individual metric scores and divide by the number of
    # metrics" — the full metric set, equally weighted, so that two models are
    # always scored on the same scale. Per group likewise, over that group's
    # metrics; the group scores are this project's addition, the paper
    # aggregates the dimension in one step.
    for group in GROUPS:
        in_group = [scores[m.key] for m in METRICS if m.group == group]
        out[f"prag_group_{group}_score"] = (
            sum(in_group) / len(in_group) if in_group else None
        )
    out["prag_score"] = sum(scores.values()) / len(METRICS)
    # How many of the 14 were actually measurable on this model; the rest
    # entered the mean at 1.0. Not a divisor — see above.
    out["prag_n_metrics_measured"] = n_measured
    out["prag_n_metrics_total"] = len(METRICS)

    # ── diagnostics ──
    type_counts = {t: sum(1 for g in view.gateways if view.gateway_type[g] == t)
                   for t in GATEWAY_TYPES}
    out.update({
        "prag_n_splits": len(view.splits),
        "prag_n_joins": len(view.joins),
        "prag_n_gw_exclusive": type_counts["exclusive"],
        "prag_n_gw_parallel": type_counts["parallel"],
        "prag_n_gw_inclusive": type_counts["inclusive"],
        "prag_n_gw_eventbased": type_counts["eventbased"],
        # How many gateways carried no type marker and were read as exclusive
        # (the BPMN default) — the assumption behind GH, CFC and CC.
        "prag_n_gateways_type_defaulted": view.n_type_defaulted,
        "prag_n_cut_vertices": n_cut,
        # Sequence flows dropped from the control-flow graph because an
        # endpoint is a pool anchor or a node of unreadable type (0 on PMo),
        # and because they are self-loops.
        "prag_n_flows_outside_fo": view.n_flows_outside_fo,
        "prag_n_self_loops": view.n_self_loops,
        # BEF4LLM's own traversal of the same longest path, reproduced with its
        # defects so this column stays comparable with their published figures.
        # Reported raw plus its band, deliberately outside `prag_score` since
        # 2026-08-29 (see the note on the diameter MetricSpec).
        "prag_diameter": size["diameter"],
        "prag_diameter_score": normdesc(size["diameter"], DIAMETER_THRESHOLDS),
        # True when a path search hit `_PATH_BUDGET`, so the value is a lower
        # bound rather than the maximum.
        "prag_diameter_truncated": size["_diameter_truncated"],
        "prag_depth_truncated": depth_truncated,
    })
    return out


def empty(note: str = "") -> Dict[str, Any]:
    """Column set for a generation that could not be parsed at all.

    Every metric is `None`, never 0: producing no model is a *validity*
    failure, and a 0 would be read as a measured value. Derived from
    `evaluate()` on an empty graph rather than listed again, so a newly added
    diagnostic cannot go missing for unparseable rows — same reason as
    `syntactic.empty`.
    """
    out: Dict[str, Any] = {k: None for k in evaluate(ProcessGraph())}
    out["prag_n_metrics_total"] = len(METRICS)
    return out
