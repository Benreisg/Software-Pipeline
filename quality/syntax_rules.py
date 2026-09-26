"""
quality/syntax_rules.py — Syntactic quality, the BEF4LLM metric set
=======================================================================
The syntactic dimension of BEF4LLM as the **paper** defines it — Table 2 and
the formulas in Table A.15 — ported to DOT. Columns are prefixed `syn_bef_`,
the headline is `syn_bef_score`.

── Why the paper and not the supplied code ──────────────────────────────────
The two disagree, and it matters. `bef4llm/synactic_quality/
synactic_quality_check.py` implements a metric set that is *not* the one
Table 2/A.15 defines:

* Table 2 has **two** gateway-degree metrics (split `in=1 ∧ out>1` over |GS|,
  join `in>1 ∧ out=1` over |GJ|); the code has one combined rule. This port
  also has one (`gateway_in_out_degree`, author's instruction 2026-08-24) and
  now uses the supplied code's predicate and denominator for that metric.
* Table 2 has an **exception-event** metric (#14); the code folds boundary
  events into the intermediate-event rule.
* The code adds `connected_nodes` and `event_gateway_predecessor_successor`,
  which appear in **no** definition table.
* The code's start/end degree rules are whole-model 0/1 measures reading
  `out ≥ 1` / `in ≥ 1`; Table A.15 defines them as **ratios** over |ES| / |EE|
  with `out = 1` / `in = 1` exactly.
* The code's "one start event per process" divides by the number of **start
  events**; Table A.15 divides by the number of **processes**, which is also
  the only version that can notice a process with *no* start event.

The paper's own result tables (C.21/C.22) report the code's set, so the
published Qsyn figures are not comparable with this module. The definition
tables are what a thesis cites, so those are what this implements.

── The metric set ───────────────────────────────────────────────────────────
Ten of the paper's sixteen, #15 and #16 counted as one, #5 present in an
adapted form, and #9 removed from this project's score at the user's request
on 2026-09-24.

Metric **6 (message-flow connection rules)** is excluded because its formula
quantifies over BPMN element types DOT cannot express — it needs message
start/intermediate/end events (`E_MS`, `E_MI`, `E_ME`), i.e. event
*definitions*, which have no Graphviz notation at all. Metric **5
(sequence-flow connection rules)** was excluded for the same reason until
2026-08-24 and is now scored in the adapted form the notation does support:
**no flow into a start event, no flow out of an end event**, 1 or 0
(`adapted_sequence_flow_connection_rules`).

Metrics **9**, **10 (exactly one process per pool)**, **13 (non-exception
intermediate event)** and **14 (exception event)** are not scored. #10/#13/#14
were removed at the author's instruction on 2026-08-21; #9 was removed at the
user's request on 2026-09-24. **#15 and #16 are merged into one gateway-degree
rule** (`gateway_in_out_degree`). State these departures wherever Qsyn is
reported: the divisor is 10, not 14, so this score is not comparable with one
computed over the paper's full set.

    Qsyn = Σ_m score(m) / 10

The separate dictated check `syn_split_has_matching_join` is still computed;
it is independent of the removed BEF4LLM metric. Metrics 15 and 16 still check
the degrees of every gateway.

── The scoring principles, all from the paper ───────────────────────────────
1. **Boolean vs. counting metrics** (§4.1). Three metrics are Boolean and score
   0 or 1: the existence of a start event, the existence of an end event, and
   the adapted sequence-flow rule. The other seven are counting metrics:
   *conforming elements / elements the rule covers*. The
   paper's own example — "a BPMN model with 8 labeled activities out of 10
   would evaluate to 0.8".
2. **Arithmetic mean, equal weights** (§4.5, Eq. 4): "we sum the individual
   metric scores and divide by the number of metrics", with no weighting,
   because there is no empirical evidence that any metric should count more.
3. **A rule with nothing to measure scores 1.0.** The paper does not define the
   empty-denominator case; scoring it 1.0 (rather than 0.0, which would read as
   a measured failure) matches the reference implementation and keeps the
   divisor at the full metric count.
4. **`in(x)` and `out(x)` are sequence flows.** Stated in the paper's notation:
   "in(x), out(x) incoming/outgoing **sequence flows** of x". A task that
   receives a message still has in-degree 1.

── Two extra columns, deliberately outside the score ────────────────────────
`syn_extra_connected_nodes` and `syn_extra_event_gateway_predecessor_successor`
are the two rules the reference implementation measures and the paper's
definition tables do not contain. They are reported because they are
informative — connectedness in particular catches genuinely broken models —
but they are **not** part of `syn_bef_score`, because they are not part of the
published metric set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from .graph import ProcessGraph, resolve_gateway_type

# Labels that make a task a *receiving* task. Only used by the extra
# event-gateway column, never by a scored metric.
_RECEIVE_WORDS = ("receive", "await", "wait for", "erhalt", "empfang", "warte")


@dataclass(frozen=True)
class MetricSpec:
    """One syntactic metric of the paper's Table 2 / Table A.15."""
    key: str
    paper_no: int        # its number in Table 2, for traceability
    label: str           # the paper's own metric description
    boolean: bool = False


# In the paper's order, skipping 5 and 6 (see the module docstring).
METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("existence_start_event", 1, "Existence of a start event", boolean=True),
    MetricSpec("existence_end_event", 2, "Existence of an end event", boolean=True),
    MetricSpec("one_start_event_per_process", 3, "One start event per process"),
    MetricSpec("one_end_event_per_process", 4, "One end event per process"),
    MetricSpec("adapted_sequence_flow_connection_rules", 5,
               "Adapted Sequence-flow connection rules", boolean=True),
    MetricSpec("start_event_in_out_degree", 7, "Start event: in = 0, out = 1"),
    MetricSpec("end_event_in_out_degree", 8, "End event: in = 1, out = 0"),
    MetricSpec("labeled_tasks", 11, "Each observable task has a label"),
    MetricSpec("task_in_out_degree", 12, "Task: in = 1, out = 1"),
    # #15 and #16 merged into one rule at the author's instruction, 2026-08-24
    # — see `gateway_in_out_degree`. The column keeps the reference
    # implementation's name for the same rule, under this module's `syn_bef_`
    # prefix: `syn_bef_gateway_in_out_degree`.
    MetricSpec("gateway_in_out_degree", 15, "Gateway in/out degree"),
)

METRIC_LABELS: Dict[str, str] = {m.key: m.label for m in METRICS}

# Removed metrics are not computed, not scored, and no longer columns of a row
# — kept as data so the divisor of Qsyn can be traced back to a decision rather
# than looking like an oversight. #13 and #14
# were all but vacuous in DOT (over a 45-model run they measured something on
# one model and on none); #10 was not — it scored 0.944 on average there, so
# what leaves with it is real signal, not only empty denominators.
REMOVED_BY_DECISION: Dict[int, str] = {
    9: "Split gateway has matching join gateway",
    10: "Exactly one process per pool",
    13: "Non-exception intermediate event: in = 1, out = 1",
    14: "Exception event: in = 0, out = 1",
}

# The paper's metrics that this module does not compute, and why. Kept as data
# so the omission is visible from the code, not only from the docstring.
NOT_COMPUTABLE_IN_DOT: Dict[int, str] = {
    5: "Sequence-flow connection rules — the published formula, which quantifies "
       "over BPMN element types DOT does not carry. The two prohibitions that do "
       "survive the notation are scored as `adapted_sequence_flow_connection_rules`",
    6: "Message-flow connection rules — the formula quantifies over message "
       "start/intermediate/end events, i.e. BPMN event definitions, which have "
       "no Graphviz notation",
}


class _Check:
    """Computes every metric over one model as `(conforming, covered)`."""

    def __init__(self, pg: ProcessGraph):
        self.pg = pg
        self.flows: Set[Tuple[str, str]] = pg.sequence_flows()

        # Principle 4: degrees count sequence flows only.
        self._ind: Dict[str, int] = {}
        self._outd: Dict[str, int] = {}
        self._succ: Dict[str, List[str]] = {}
        self._pred: Dict[str, List[str]] = {}
        for s, d in self.flows:
            self._outd[s] = self._outd.get(s, 0) + 1
            self._ind[d] = self._ind.get(d, 0) + 1
            self._succ.setdefault(s, []).append(d)
            self._pred.setdefault(d, []).append(s)

        self.starts = sorted(pg.starts)
        self.ends = sorted(pg.ends)
        self.tasks = sorted(pg.tasks)
        self.splits = sorted(pg.splits())
        # P — the processes, read as the **weakly-connected components of the
        # sequence-flow graph**, not as the DOT clusters.
        #
        # BPMN forbids a sequence flow from crossing a pool boundary, so a
        # connected control-flow region *is* a process; and lanes, which flows
        # cross freely, do not split one. Reading the clusters instead would
        # mistake a lane for a pool: a generated model drawing
        # `cluster_marketing` / `cluster_tracking` as two lanes of one process
        # would score 1/3 on "one start event per process" for a notation
        # choice, not a modelling error. On the 55 PMo reference models the two
        # readings agree exactly (items 23/24 → 2 and 3 processes); across the
        # generated corpus they differ on the lane-as-cluster models.
        #
        # A collapsed black-box pool contributes no flow objects and therefore
        # no process, which is correct — it has no process inside.
        self.processes: List[Set[str]] = self._components()

    def ind(self, node: str) -> int:
        return self._ind.get(node, 0)

    def outd(self, node: str) -> int:
        return self._outd.get(node, 0)

    def _components(self) -> List[Set[str]]:
        """Weakly-connected components of the sequence-flow graph over the
        flow objects. An isolated node is its own component — and therefore a
        process without a start event, which is what it is."""
        fo = self.pg.flow_objects()
        adjacency: Dict[str, Set[str]] = {n: set() for n in fo}
        for s, d in self.flows:
            if s in fo and d in fo:
                adjacency[s].add(d)
                adjacency[d].add(s)
        seen: Set[str] = set()
        components: List[Set[str]] = []
        for node in sorted(fo):
            if node in seen:
                continue
            stack, component = [node], set()
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                component.add(cur)
                stack.extend(adjacency[cur] - seen)
            components.append(component)
        return components

    def _in_process(self, nodes: List[str], process: Set[str]) -> int:
        return sum(1 for n in nodes if n in process)

    # ── the metrics, in the paper's order ──
    def existence_start_event(self) -> Tuple[int, int]:
        """#1  ∃e ∈ ES — Boolean."""
        return (1 if self.starts else 0), 1

    def existence_end_event(self) -> Tuple[int, int]:
        """#2  ∃e ∈ EE — Boolean."""
        return (1 if self.ends else 0), 1

    def one_start_event_per_process(self) -> Tuple[int, int]:
        """#3  |{p ∈ P : |ES_p| = 1}| / |P|.

        Counted over **processes**, not over start events, which is the only
        form that notices a process with *no* start event at all — the reference
        implementation iterates the start events and therefore cannot.
        """
        return sum(1 for p in self.processes
                   if self._in_process(self.starts, p) == 1), len(self.processes)

    def one_end_event_per_process(self) -> Tuple[int, int]:
        """#4, counted the way the reference implementation counts it (author's
        instruction, 2026-08-24):

            1 − |end events in an already-counted process| / |EE|

        which is `|{p ∈ P : |EE_p| ≥ 1}| / |EE|` — the first end event of a
        process is the one it is allowed, every further one in the same process
        is the mistake, and the divisor is the number of end events. Their code:

            for sink in self.sinks:
                total += 1
                if graph.nodes[sink]['process'] in process_ids:
                    mistakes += 1
                else:
                    process_ids.append(graph.nodes[sink]['process'])

        The paper prints `|{p ∈ P : |EE_p| = 1}| / |P|`, over the processes.
        PMo's item 22 is where the two part: one process with four end events is
        0/1 = 0.0 by the paper and 1 − 3/4 = 0.25 here. The other 54 models
        agree.

        **What this cannot see** is a process with *no* end event: the loop
        walks the end events that exist, so an empty process is never reached.
        #2 catches only the model that has none at all. `one_start_event_per_process`
        above is still counted over the processes and does notice it — the two
        are deliberately not symmetric.
        """
        with_end = sum(1 for p in self.processes if self._in_process(self.ends, p) >= 1)
        return with_end, len(self.ends)

    def adapted_sequence_flow_connection_rules(self) -> Tuple[int, int]:
        """#5, adapted to what DOT can express (author's instruction,
        2026-08-24) — Boolean, 1 or 0:

            no sequence flow **into** a start event
            ∧ no sequence flow **out of** an end event

        The paper's #5 quantifies over BPMN element types a DOT file does not
        carry, which is why it was excluded; these two prohibitions are the part
        of it that survives the notation, and they are the two that a generated
        model actually gets wrong. One violation of either kind fails the model:
        it is a rule about the model, not a rate over its flows.

        Counted over **sequence flows only** (principle 4). A message flow into
        a start event is not control flow and must not fail the rule — PMo's
        item 23 draws exactly that, a message start event receiving from another
        pool.

        The dictated check set scores the same two prohibitions as a *ratio* of
        valid flows (`syntactic.sequence_flow_connection_rules`), which is where
        to look for how badly a model breaks them; this one answers whether it
        does at all.
        """
        into_start = any(target in self.pg.starts for _, target in self.flows)
        out_of_end = any(source in self.pg.ends for source, _ in self.flows)
        return (0 if into_start or out_of_end else 1), 1

    def start_event_in_out_degree(self) -> Tuple[int, int]:
        """#7  |{e ∈ ES : |in(e)| = 0 ∧ |out(e)| = 1}| / |ES|.

        `out = 1` **exactly**: a start event that fans out into two flows
        violates the rule. The reference implementation reads `out ≥ 1` and
        scores the model as a whole rather than per event.
        """
        return sum(1 for e in self.starts
                   if self.ind(e) == 0 and self.outd(e) == 1), len(self.starts)

    def end_event_in_out_degree(self) -> Tuple[int, int]:
        """#8  |{e ∈ EE : |in(e)| ≥ 1 ∧ |out(e)| = 0}| / |EE|.

        **`in ≥ 1`, not the printed `in = 1`** (author's instruction,
        2026-08-24). Table A.15 prints `= 1`, mirroring #7's `out = 1` for the
        start event, but the reference implementation asks

            if outdegree > 0 or indegree < 1:   # → mistake

        i.e. only that something flows into the end event, and this follows it.
        What separates the two readings is the **implicit merge**: two sequence
        flows running into one end event without a gateway between them. PMo
        draws that in items 35 (`End`, in = 2) and 42 (`Repair process over`,
        in = 2), which scored 0.0 under `= 1` and are the only two models where
        this metric disagreed with theirs. With `≥ 1` all 55 agree.

        The **shape** still differs from theirs and deliberately so: this is a
        ratio over the end events, as Table A.15 defines it, while their rule is
        a whole-model 0/1 that stops at the first offender. On a model with five
        end events, one of them malformed, this scores 0.8 and theirs 0. On the
        reference set the two shapes cannot be told apart, because no model has
        a malformed end event left.
        """
        return sum(1 for e in self.ends
                   if self.ind(e) >= 1 and self.outd(e) == 0), len(self.ends)

    def labeled_tasks(self) -> Tuple[int, int]:
        """#11  |{t ∈ T : label(t) ≠ ε}| / |T|.

        A DOT node without a `label` attribute renders as its own id, so that
        id *is* the label — "unlabelled" means the node renders as nothing.
        A label that exists but names nothing (`Task_1`) counts as labelled
        here; `syn_n_placeholder_labels` reports those separately.
        """
        return sum(1 for t in self.tasks if self.pg.label_of(t).strip()), len(self.tasks)

    def task_in_out_degree(self) -> Tuple[int, int]:
        """#12  |{t ∈ T : |in(t)| = 1 ∧ |out(t)| = 1}| / |T|."""
        return sum(1 for t in self.tasks
                   if self.ind(t) == 1 and self.outd(t) == 1), len(self.tasks)

    def gateway_in_out_degree(self) -> Tuple[int, int]:
        """#15 and #16 are merged into one metric (author's instruction,
        2026-08-24), scored with the supplied BEF4LLM code's rule:

            1 − |{g ∈ G : (in > 1 ∧ out > 1) ∨ (in ≤ 1 ∧ out ≤ 1)}| / |G|

        The paper defines two separate metrics — #15 over GS, #16 over GJ —
        while the original code counts errors across every gateway. Its
        predicate accepts a gateway whenever exactly one of its degrees is
        greater than one, even when the other degree is zero; this port keeps
        that behavior for parity.

        Use the supplied BEF4LLM implementation's predicate exactly:
        a gateway is a mistake when
        `(in > 1 ∧ out > 1) ∨ (in ≤ 1 ∧ out ≤ 1)`.
        Equivalently, it conforms when exactly one degree is greater than 1.
        This also accepts `in = 0, out > 1` and `in > 1, out = 0`; those are
        consequences of the original predicate, retained here for parity.

        As in the original implementation, the denominator is every gateway,
        including gateways that do not form a well-shaped split or join. The
        original class stores the number of mistakes and the aggregate subtracts
        its rate from 1; returning conforming/covered here gives that same score.

        The merge takes the metric set from twelve to **eleven**, so
        `syn_bef_score` divides by 11 — see the module docstring. (It read
        "eleven to ten … divides by 10" until 2026-08-29: true for the hour
        between the merge and #5 being added back the same day, stale ever
        since.)
        """
        conforming = sum(
            1 for g in self.pg.gateways
            if not ((self.ind(g) > 1 and self.outd(g) > 1)
                    or (self.ind(g) <= 1 and self.outd(g) <= 1))
        )
        return conforming, len(self.pg.gateways)

    # ── the two extras, reported but never scored ──
    def connected_nodes(self) -> Tuple[int, int]:
        """Every node lies on a path from a start event to an end event.

        Not in Table 2/A.15 — this is the reference implementation's
        `connected_nodes`, kept as a diagnostic outside `syn_bef_score`.
        """
        forward = self._reachable(set(self.pg.starts))
        backward = self._reachable(set(self.pg.ends), backwards=True)
        conforming = 0
        for node in sorted(self.pg.flow_objects()):
            kind = self.pg.kind_of(node)
            if kind == "start":
                ok = self.outd(node) >= 1
            elif kind == "end":
                ok = self.ind(node) >= 1
            elif self.ind(node) == 0 or self.outd(node) == 0:
                ok = False
            elif not self.pg.starts or not self.pg.ends:
                ok = False
            else:
                ok = node in forward and node in backward
            conforming += ok
        return conforming, len(self.pg.flow_objects())

    def event_gateway_predecessor_successor(self) -> Tuple[int, int]:
        """An event-based gateway is followed by events, or preceded by a
        receive task.

        Not in Table 2/A.15 either — the reference implementation's rule, minus
        the `messageEventDefinition`/`timerEventDefinition` test on the
        successors, which DOT cannot express. Outside `syn_bef_score`.
        """
        conforming = covered = 0
        for gateway in self.splits:
            if resolve_gateway_type(self.pg, gateway) != "eventbased":
                continue
            covered += 1
            event_successor = any(
                self.pg.kind_of(s) in ("event", "end", "start")
                for s in self._succ.get(gateway, ())
            )
            receive_predecessor = any(
                self.pg.kind_of(p) == "task"
                and any(w in self.pg.label_of(p).lower() for w in _RECEIVE_WORDS)
                for p in self._pred.get(gateway, ())
            )
            conforming += event_successor or receive_predecessor
        return conforming, covered

    # ── helper ──
    def _reachable(self, seeds: Set[str], backwards: bool = False) -> Set[str]:
        adjacency = self._pred if backwards else self._succ
        seen: Set[str] = set()
        stack = list(seeds)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
        return seen


def processes(pg: ProcessGraph) -> List[Set[str]]:
    """The model's processes: the weakly-connected components of the
    sequence-flow graph over the flow objects.

    Public because `syntactic.py` needs the same decomposition for its own
    check 3 — "one start event per process" is a statement about processes, and
    the two scorings must not disagree on what a process is. See `_Check` for
    why the components and not the DOT clusters.
    """
    return _Check(pg).processes


def evaluate(pg: ProcessGraph) -> Dict[str, Any]:
    """Return the `syn_bef_`-prefixed columns for one model.

    Per metric: its score, and — for the seven counting metrics — the two
    counts behind it. The counts are there because a mean of per-model ratios
    is *not* the pooled element-level error rate: averaging
    `syn_bef_task_in_out_degree` over a run weights a two-task model like a
    forty-task one, while `Σ conforming / Σ covered` answers "what share of all
    tasks was well-formed". Both are legitimate; the columns let the thesis
    pick one knowingly.
    """
    check = _Check(pg)

    out: Dict[str, Any] = {}
    total_score = 0.0
    perfect = applicable = 0
    for metric in METRICS:
        conforming, covered = getattr(check, metric.key)()
        # Principle 3: a metric with nothing to measure scores 1.0.
        score = (conforming / covered) if covered else 1.0
        total_score += score
        if covered:
            applicable += 1
        if score == 1.0:
            perfect += 1
        # Full precision — see the precision policy in quality/__init__.
        out[f"syn_bef_{metric.key}"] = score
        if not metric.boolean:
            out[f"syn_bef_{metric.key}_conforming"] = conforming
            out[f"syn_bef_{metric.key}_covered"] = covered

    # Eq. 4: the arithmetic mean over the metric set, equally weighted.
    out["syn_bef_score"] = total_score / len(METRICS)
    out["syn_bef_metrics_total"] = len(METRICS)
    out["syn_bef_metrics_perfect"] = perfect
    # How many metrics the model actually triggered. Not a divisor — principle 2
    # keeps that at the full metric count — but it explains a high score on a
    # small model.
    out["syn_bef_metrics_applicable"] = applicable
    out["syn_bef_n_processes"] = len(check.processes)
    out["syn_bef_n_pools"] = len(pg.pool_ids())

    # Outside the score — see the module docstring.
    for extra in ("connected_nodes", "event_gateway_predecessor_successor"):
        conforming, covered = getattr(check, extra)()
        out[f"syn_extra_{extra}"] = (conforming / covered) if covered else 1.0
        out[f"syn_extra_{extra}_conforming"] = conforming
        out[f"syn_extra_{extra}_covered"] = covered
    return out
