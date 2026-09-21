"""
quality/syntactic.py — Syntactic quality
============================================
Does the generated model obey the structural rules of the notation? Model-
internal only: no ground truth is consulted, so this is computable for every
generation, including CSV generation-only runs.

Eleven checks, exactly as specified:

  1. has_start_event         Existence of a start event
  2. has_end_event           Existence of an end event
  3. single_start_event      One start event per process
  4. single_end_event        One end event per process
  5. start_in0_out1          Start event: in = 0, out = 1
  6. tasks_labeled           Each observable task has a label
  7. task_in1_out1           Task: in = 1, out = 1
  8. split_in1_outN          Split gateway: in = 1, out > 1
  9. join_inN_out1           Join gateway: in > 1, out = 1
 10. split_has_matching_join Split gateway has a matching join gateway
                              (event-based gateways exempt)
 11. sequence_flow_connection_rules   "Sequence-flow connection rules"

Every check yields a **score in 0.0–1.0**. Checks 1–10 are pass/fail and score
1.0 or 0.0; check 11 is a *ratio* (valid sequence flows / all sequence flows)
per the adapted BEF4LLM definition, so it scores continuously. `syn_score` is
the mean of all eleven scores, and `syn_checks_passed` counts the checks that
score a full 1.0.

Checks 1/3 and 2/4 overlap by design — a model with two start events fails 3 but
passes 1 — so the score distinguishes "no start at all" from "too many".

Check 5 counts **sequence flows only**: a message flow attached to a start event
is not control flow and must not make the rule fail (author's instruction,
2026-08-23). Checks 7–9 still count every edge, so on a model that draws message
flows the two conventions can disagree — see the note at check 5.

Checks 8–10 depend on telling a split gateway from a join gateway. That is
decided in `graph.py::_gateway_role` by *declared intent* (name/label) first,
because classifying by degree and then checking the degree would make 8 and 9
tautological. See the note there.

Check 10 does **not apply to event-based gateways** (author's instruction,
2026-08-23): an event-based split is resolved by whichever event fires first,
and BPMN does not require it to reconverge at a join of the same type. Such
splits are skipped by the check — never counted as unmatched, never counted as
matched — and `syn_n_event_based_splits` says how many were skipped. Checks 8
and 9 are unchanged: an event-based gateway's degrees are still checked.

Columns are prefixed `syn_`.

── The second scoring in this dimension ─────────────────────────────────────
`quality/syntax_rules.py` holds **BEF4LLM's published syntactic metric set**
(paper Table 2 / Table A.15) ported to DOT: eleven metrics scored as
*conforming elements / covered elements* rather than as verdicts, with
`syn_bef_score` as its headline. Its columns are merged in by `evaluate()`
below, so both scorings appear in the same row and can be compared per model.
Eleven of the paper's sixteen, not fourteen: #5 and #6 have no DOT notation,
and #10, #13 and #14 were removed at the author's instruction on 2026-08-21
(`syntax_rules.REMOVED_BY_DECISION`), which moves the Qsyn divisor from 14 to
11. Ten of the eleven checks here map onto one of its metrics — all but
*Sequence-flow connection rules* — and it covers one they do not: #8, the end
event's degree. See the README for the mapping and for which one to report.
"""
from __future__ import annotations

from typing import Any, Dict

from . import syntax_rules
from .graph import ProcessGraph, fan_maps, matching_joins, resolve_gateway_type
from .normalize import (DIAMETER_THRESHOLDS, TNG_THRESHOLDS, TNN_THRESHOLDS,
                        TNSF_THRESHOLDS, normdesc)

CHECKS = [
    "has_start_event",
    "has_end_event",
    "single_start_event",
    "single_end_event",
    "start_in0_out1",
    "tasks_labeled",
    "task_in1_out1",
    "split_in1_outN",
    "join_inN_out1",
    "split_has_matching_join",
    "sequence_flow_connection_rules",
]

# Human-readable name of each check, for any output a person reads. The
# sequence-flow entry carries its specified name verbatim.
CHECK_LABELS = {
    "has_start_event": "Existence of a start event",
    "has_end_event": "Existence of an end event",
    "single_start_event": "One start event per process",
    "single_end_event": "One end event per process",
    "start_in0_out1": "Start event: in = 0, out = 1",
    "tasks_labeled": "Each observable task has a label",
    "task_in1_out1": "Task: in = 1, out = 1",
    "split_in1_outN": "Split gateway: in = 1, out > 1",
    "join_inN_out1": "Join gateway: in > 1, out = 1",
    "split_has_matching_join": "Split gateway has matching join gateway",
    "sequence_flow_connection_rules": "Sequence-flow connection rules",
}

# ── Sequence-flow connection rules (adapted BEF4LLM) ────────────────────────
# Every directed DOT edge is one sequence flow. A flow is valid when its source
# is a StartEvent, Task, ExclusiveGateway or ParallelGateway, and its target is
# an EndEvent, Task, ExclusiveGateway or ParallelGateway — which reduces to the
# two prohibitions below, exactly as specified:
#
#   Start -> Task/Gateway/End   valid          End  -> Task/Gateway   invalid
#   Task  -> Task/Gateway/End   valid          Task -> Start          invalid
#   Gateway -> Task/Gateway/End valid          Gateway -> Start       invalid
#
# The score is valid flows / total flows, not a pass/fail.
_SOURCE_FORBIDDEN = {"end"}     # an EndEvent must never be the source of an edge
_TARGET_FORBIDDEN = {"start"}   # a StartEvent must never be the target of an edge


def sequence_flow_connection_rules(pg: ProcessGraph) -> Dict[str, Any]:
    """The "Sequence-flow connection rules" check.

    Returns the ratio plus the counts needed to explain it. Edges are counted
    deduplicated, consistent with how node degrees are computed everywhere else
    in this module (the same flow written twice is one flow); the duplicate
    count is reported so that choice stays auditable.

    Endpoints that could not be classified (an intermediate event, an unknown
    shape) are *not* treated as violations: the specification reduces the rule
    to two prohibitions, and punishing a model for a gap in this scorer's
    classification would measure the scorer. `n_flows_unclassified_endpoint`
    keeps them visible.
    """
    flows = sorted(pg.unique_edges)
    from_end = [(s, d) for s, d in flows if pg.kind_of(s) in _SOURCE_FORBIDDEN]
    into_start = [(s, d) for s, d in flows if pg.kind_of(d) in _TARGET_FORBIDDEN]
    invalid = set(from_end) | set(into_start)

    total = len(flows)
    n_valid = total - len(invalid)
    # No flows at all is not a process; score it 0.0 rather than dividing by
    # zero or crediting an empty model with a perfect ratio.
    score = (n_valid / total) if total else 0.0

    unclassified_endpoint = sum(
        1 for s, d in flows
        if pg.kind_of(s) in ("event", "unknown") or pg.kind_of(d) in ("event", "unknown")
    )
    return {
        "score": score,
        "n_sequence_flows": total,
        "n_valid_sequence_flows": n_valid,
        "n_invalid_sequence_flows": len(invalid),
        "n_flows_from_end_event": len(from_end),
        "n_flows_into_start_event": len(into_start),
        "n_duplicate_flows": len(pg.edges) - total,
        "n_flows_unclassified_endpoint": unclassified_endpoint,
    }


# ── BEF4LLM size metrics ────────────────────────────────────────────────────
# Four size measures, each reported as a raw value *and* a BEF4LLM score via
# the descending normalisation in normalize.py.
#
#   TNN       total number of nodes     |FO|   flow objects: events + tasks + gateways
#   TNG       total number of gateways  |G|    any gateway type
#   TNSF      total number of sequence flows |F^S|  message flows excluded
#   Diameter  max{|p|}                         longest start-to-end path, in flow objects
#
# BEF4LLM's fifth size metric, TNMF (total number of message flows), is **not
# reported at all** — neither as a score nor as a diagnostic count. Message
# flows *are* expressible in DOT (PMo writes `[style=dashed, arrowhead=open]`,
# and `graph.py` reads all 20 of them, matching BEF4LLM's BPMN count on 55/55
# models), but **no prompt template in prompts.py prescribes that notation**.
# A generated model therefore has no way to produce a message flow on purpose,
# so any count over the generated corpus would measure the prompt rather than
# the model. Removed at the author's instruction on 2026-08-16.
#
# The *detection* stays and is load-bearing: `graph.is_message_flow` is what
# keeps message flows out of |F^S|, out of the control-flow degrees and out of
# the cross-connectivity graph. Deleting it would silently inflate TNSF (item
# 23: 22 flows instead of 16).
#
# Counting is done on the semantic element types resolved in graph.py, never on
# labels or presentation: pools/lanes (`markers`) are excluded from |FO|, and
# message flows (dashed edges) are excluded from |F^S|. No DI/layout data —
# shapes, bounds, positions — takes part in any count; `shape` is read only as
# the element-type carrier the DOT notation provides, and `style=invis` only to
# recognise a pool anchor as a non-element.
SIZE_METRICS = ["tnn", "tng", "tnsf", "diameter"]

SIZE_LABELS = {
    "tnn": "TNN (total number of nodes)",
    "tng": "TNG (total number of gateways)",
    "tnsf": "TNSF (total number of sequence flows)",
    "diameter": "Diameter (longest start-to-end path)",
}

_SIZE_THRESHOLDS = {
    "tnn": TNN_THRESHOLDS,
    "tng": TNG_THRESHOLDS,
    "tnsf": TNSF_THRESHOLDS,
    "diameter": DIAMETER_THRESHOLDS,
}


def size_metrics(pg: ProcessGraph) -> Dict[str, Any]:
    """The four BEF4LLM size metrics, raw value plus normdesc score each.

    Reported as their own group: they measure *size*, not correctness, so they
    stay out of `CHECKS` and out of `syn_score` — see the note in `evaluate`.

    `syn_diameter_nogw` carries the second counting convention (gateways not
    counted) that `quality/pragmatic.py` documents, so the two prefixes stay in
    step. It is not part of `syn_size_score`.
    """
    diameter, truncated = pg.diameter()
    diameter_nogw, truncated_nogw = pg.diameter(count_gateways=False)
    raw = {
        "tnn": len(pg.flow_objects()),
        "tng": len(pg.gateways),
        "tnsf": len(pg.sequence_flows()),
        "diameter": diameter,
    }

    out: Dict[str, Any] = {}
    for name in SIZE_METRICS:
        out[f"syn_{name}"] = raw[name]
        out[f"syn_{name}_score"] = normdesc(raw[name], _SIZE_THRESHOLDS[name])

    scores = [out[f"syn_{n}_score"] for n in SIZE_METRICS]
    out["syn_size_score"] = sum(scores) / len(scores)
    # The gateway-skipping convention, outside `syn_size_score`.
    out["syn_diameter_nogw"] = diameter_nogw
    out["syn_diameter_nogw_score"] = normdesc(diameter_nogw, DIAMETER_THRESHOLDS)
    # True when the longest-path search hit its expansion cap, so `syn_diameter`
    # is a lower bound rather than the maximum.
    out["syn_diameter_truncated"] = truncated or truncated_nogw
    return out


# ── Syntactical correctness (verdict) ───────────────────────────────────────
# One yes/no about the whole model, next to the graded scores. Five conditions,
# all of which must hold:
#
#   1. every function (task) has exactly one incoming and one outgoing arc;
#   2. every gateway has at least one incoming and one outgoing arc;
#   3. there is at least one start node and at least one end node;
#   4. the graph is directed   — Graphviz `int agisdirected(Agraph_t *g)`;
#   5. the graph is coherent   — Graphviz `int isConnected(Agraph_t *g)`.
#
# 4 and 5 are graph-level, and both are answered exactly as the two C functions
# answer them (`graph.py::is_directed` / `is_connected`, which document the
# correspondence). A `graph` instead of a `digraph` has no arrow direction, so
# nothing the other conditions check means what it says; a model in two pieces
# is two processes, not one, whatever the pieces look like. `components()` was
# checked against Graphviz's own `ccomps` over 200 files — the 45 generated
# models of a live run and all 155 PMo dataset models — and agrees on every one.
#
# It is a **verdict, not a score**: a model either is well formed by these rules
# or it is not, and there is no partial credit. It therefore stays out of
# `CHECKS`, out of `syn_score` and out of `syn_checks_passed` — folding a yes/no
# into a mean would silently redefine a figure already in use, and it overlaps
# checks 1/2/7 by design (condition 1 *is* check 7; conditions 2 and 3 are
# looser than checks 8/9 and equal to 1/2).
#
# Degrees are counted over **sequence flows only** (`seq_in_degree` /
# `seq_out_degree`, deduplicated): a message flow is not an arc of the control
# flow, and an activity that also sends a message must not fail for it. Check 5
# counts the same way since 2026-08-23; checks 7/8/9 still count every edge, so
# the verdict and check 7 can disagree on a model that draws message flows. Conditions 4 and 5 are unaffected: `isConnected()` is a
# statement about the file's graph, message flows included, and answering it any
# other way would stop being a port of the C function (it would also declare
# every two-pool model incoherent, since message flows are exactly what joins
# pools). Empty sets do not pass
# vacuously: a model with no tasks and no start node is not a correct process,
# so conditions 1 and 3 require at least one of each. Condition 2 is different:
# a purely sequential process legitimately has no gateway, and that passes.
def syntactical_correctness(pg: ProcessGraph) -> Dict[str, Any]:
    """The five conditions and the verdict they add up to."""
    malformed_functions = sorted(
        t for t in pg.tasks
        if not (pg.seq_in_degree(t) == 1 and pg.seq_out_degree(t) == 1)
    )
    unwired_gateways = sorted(
        n for n in pg.gateways
        if not (pg.seq_in_degree(n) >= 1 and pg.seq_out_degree(n) >= 1)
    )
    components = pg.components()
    functions_ok = len(pg.tasks) > 0 and not malformed_functions
    gateways_ok = not unwired_gateways
    start_end_ok = len(pg.starts) >= 1 and len(pg.ends) >= 1
    directed_ok = pg.is_directed()
    connected_ok = len(components) <= 1
    return {
        "syn_correct": bool(functions_ok and gateways_ok and start_end_ok
                            and directed_ok and connected_ok),
        "syn_correct_functions": bool(functions_ok),
        "syn_correct_gateways": bool(gateways_ok),
        "syn_correct_start_end": bool(start_end_ok),
        "syn_correct_directed": bool(directed_ok),
        "syn_correct_connected": bool(connected_ok),
        "syn_n_malformed_functions": len(malformed_functions),
        "syn_n_unwired_gateways": len(unwired_gateways),
        # 1 for a coherent model; the count is what says how badly a `no` broke
        # — an orphan node and a model torn in half both read "not connected"
        # otherwise.
        "syn_n_components": len(components),
    }


def evaluate(pg: ProcessGraph) -> Dict[str, Any]:
    n_starts, n_ends = len(pg.starts), len(pg.ends)

    # ── 1/2: existence ──
    has_start = n_starts >= 1
    has_end = n_ends >= 1

    # ── 3/4: exactly one ──
    # Check 3 is a statement about **processes**, not about the file: a model
    # with two pools has two processes and needs one start event in each. It is
    # 1 when every process has exactly one and 0 otherwise (author's
    # instruction, 2026-08-24) — items 23 and 24 of the reference set are
    # multi-pool Camunda models that were failing it for having three and two
    # correct start events. `syntax_rules.processes` supplies the same
    # decomposition the BEF4LLM port scores #3 over, so the two cannot drift
    # apart; a model with no flow objects at all has no process and fails.
    #
    # Check 4 reads the same way since 2026-08-24: one end event in every
    # process. Items 23 and 24 were failing it for having one per pool, which is
    # what a multi-pool model should have. Unlike the BEF4LLM port's #4, which
    # divides by the end events, this notices a process with none.
    procs = syntax_rules.processes(pg)
    single_start = bool(procs) and all(
        sum(1 for s in pg.starts if s in p) == 1 for p in procs
    )
    single_end = bool(procs) and all(
        sum(1 for e in pg.ends if e in p) == 1 for p in procs
    )

    # ── 5: the start event's own degree ──
    # With no start event the rule cannot hold, so it fails rather than passing
    # vacuously. With several, all of them must satisfy it — otherwise a model
    # could pass by having one well-formed start among malformed ones.
    #
    # Counted over **sequence flows only** (`seq_in_degree`/`seq_out_degree`),
    # at the author's instruction on 2026-08-23: a message flow is not control
    # flow, so a start event that also receives a message must not fail for it.
    # This is what BPMN's `bpmn:incoming`/`bpmn:outgoing` mean, what the paper's
    # notation states for in(x)/out(x) (`syntax_rules`, principle 4), and what
    # the correctness verdict below already does. Checks 7/8/9 still count every
    # edge — the rules were not changed with it, so a model that draws message
    # flows can have check 5 and check 7 disagree on their counting.
    start_in0_out1 = has_start and all(
        pg.seq_in_degree(s) == 0 and pg.seq_out_degree(s) == 1 for s in pg.starts
    )

    # ── 6: every task carries a label ──
    # A task's effective label is its `label` attribute, or its node id when it
    # has none (that is what DOT renders). Unlabelled therefore means: renders
    # as nothing. A model with no tasks at all cannot satisfy "each task has a
    # label" in any meaningful sense, so it fails rather than passing over an
    # empty set.
    unlabeled = sorted(t for t in pg.tasks if not pg.label_of(t).strip())
    tasks_labeled = len(pg.tasks) > 0 and not unlabeled

    # Informational, not part of the check: a label that exists but names
    # nothing ("Task_1"). Reported so the placeholder rate stays visible
    # without silently redefining check 6.
    placeholder = sorted(t for t in pg.tasks if pg.has_placeholder_label(t))

    # ── 7: every task sits on exactly one incoming and one outgoing flow ──
    malformed_tasks = sorted(
        t for t in pg.tasks if not (pg.in_degree(t) == 1 and pg.out_degree(t) == 1)
    )
    task_in1_out1 = len(pg.tasks) > 0 and not malformed_tasks

    # ── 8/9: gateway degrees ──
    # A 1→1 pass-through neither splits nor merges: it satisfies neither rule
    # and fails both, and `syn_n_gateways_degenerate` keeps that visible. A
    # gateway that fans in *and* out is in GS and GJ both since 2026-08-24, so
    # it fails each rule on its own account and is counted in `bad_splits` /
    # `bad_joins` rather than here; `syn_n_gateways_mixed` says how many.
    splits, joins = pg.splits(), pg.joins()
    degenerate = pg.degenerate_gateways()
    mixed = pg.mixed_gateways()
    bad_splits = sorted(
        s for s in splits if not (pg.in_degree(s) == 1 and pg.out_degree(s) > 1)
    )
    bad_joins = sorted(
        j for j in joins if not (pg.in_degree(j) > 1 and pg.out_degree(j) == 1)
    )
    # A model with no gateways at all is a valid sequential process: these two
    # rules are then vacuously true, unlike the task/event rules above.
    split_in1_outN = not bad_splits and not degenerate
    join_inN_out1 = not bad_joins and not degenerate

    # ── 10: every split reconverges at a join of its own type ──
    # "Matching" is `graph.matching_joins`: the join must be where the branches
    # this split opened meet again *first*, or the gateway its loop's back edge
    # runs into — being reachable from the split is not enough (author's
    # decision, 2026-08-24; see that function for the item 30 and 46 cases). Type comes from the label/name ("X" vs
    # "+", seg_/spg_, XOR/AND); when either side's type cannot be read, any
    # join counts — an unreadable type is our gap, not the model's mistake.
    #
    # **Event-based gateways are exempt**, at the author's instruction on
    # 2026-08-23. An event-based split does not choose a branch itself: the
    # branch is decided by whichever of the racing events fires first, and BPMN
    # neither requires nor commonly draws a matching event-based join — the
    # branches end in their own end events, or merge at an exclusive gateway.
    # Requiring a partner of the *same* type would therefore fail a correct
    # model for its notation. They are skipped entirely rather than allowed to
    # match any join, so an event-based split can neither fail this check nor
    # satisfy it on behalf of another split; `syn_n_event_based_splits` reports
    # how many were skipped. A model whose splits are all event-based passes
    # vacuously, as a model with no gateways at all already did.
    # ── 11: Sequence-flow connection rules (a ratio, not pass/fail) ──
    seq = sequence_flow_connection_rules(pg)

    event_based_splits = sorted(
        s for s in splits if resolve_gateway_type(pg, s) == "eventbased"
    )
    # Every edge counts here, as it does for checks 7-9 of this module; the
    # BEF4LLM port next door counts sequence flows only.
    succ, pred = fan_maps(pg.unique_edges)
    # A mixed gateway is in neither splits() nor joins(), but it does merge —
    # for this check it counts as a possible partner.
    merging = joins | mixed
    unmatched_splits = []
    for s in sorted(splits):
        if s in event_based_splits:
            continue
        s_type = pg.gateway_types.get(s, "")
        partners = matching_joins(s, succ, pred, splits)
        if not any(
            j in partners
            and (not s_type or not pg.gateway_types.get(j, "") or pg.gateway_types[j] == s_type)
            for j in merging
        ):
            unmatched_splits.append(s)
    split_has_matching_join = not unmatched_splits

    results: Dict[str, Any] = {
        "has_start_event": has_start,
        "has_end_event": has_end,
        "single_start_event": single_start,
        "single_end_event": single_end,
        "start_in0_out1": start_in0_out1,
        "tasks_labeled": tasks_labeled,
        "task_in1_out1": task_in1_out1,
        "split_in1_outN": split_in1_outN,
        "join_inN_out1": join_inN_out1,
        "split_has_matching_join": split_has_matching_join,
        # Ratio-valued, unlike the ten pass/fail checks above. Stored at full
        # precision — see the precision policy in quality/__init__.
        "sequence_flow_connection_rules": seq["score"],
    }
    # Each check contributes its score; a pass/fail check contributes 1.0 or
    # 0.0. `checks_passed` counts only full passes, so the ratio check is
    # counted there only when every sequence flow is valid.
    scores = [float(results[c]) for c in CHECKS]
    passed = sum(1 for s in scores if s == 1.0)

    out: Dict[str, Any] = {f"syn_{k}": v for k, v in results.items()}
    out.update({
        "syn_checks_passed": passed,
        "syn_checks_total": len(CHECKS),
        "syn_score": sum(scores) / len(scores),
        # diagnostics — why a check failed, without reopening the file
        "syn_n_start_events": n_starts,
        "syn_n_end_events": n_ends,
        "syn_n_tasks": len(pg.tasks),
        "syn_n_gateways": len(pg.gateways),
        "syn_n_nodes": len(pg.nodes),
        "syn_n_edges": len(pg.unique_edges),
        "syn_n_unlabeled_tasks": len(unlabeled),
        "syn_n_placeholder_labels": len(placeholder),
        "syn_n_malformed_tasks": len(malformed_tasks),
        "syn_n_splits": len(splits),
        "syn_n_joins": len(joins),
        "syn_n_bad_splits": len(bad_splits),
        "syn_n_bad_joins": len(bad_joins),
        "syn_n_gateways_degenerate": len(degenerate),
        "syn_n_gateways_mixed": len(mixed),
        "syn_n_unmatched_splits": len(unmatched_splits),
        # Splits exempt from check 10 — see the note there.
        "syn_n_event_based_splits": len(event_based_splits),
        "syn_n_intermediate_events": len(pg.intermediates),
        "syn_n_events_unclassified": len(pg.events_unclassified),
        "syn_n_nodes_unclassified": len(pg.unclassified),
        # Sequence-flow connection rules
        "syn_n_sequence_flows": seq["n_sequence_flows"],
        "syn_n_valid_sequence_flows": seq["n_valid_sequence_flows"],
        "syn_n_invalid_sequence_flows": seq["n_invalid_sequence_flows"],
        "syn_n_flows_from_end_event": seq["n_flows_from_end_event"],
        "syn_n_flows_into_start_event": seq["n_flows_into_start_event"],
        "syn_n_duplicate_flows": seq["n_duplicate_flows"],
        "syn_n_flows_unclassified_endpoint": seq["n_flows_unclassified_endpoint"],
    })

    # BEF4LLM size metrics — reported alongside, deliberately NOT folded into
    # `syn_score`. They score *smallness*, not correctness: averaging them in
    # would make a large but perfectly well-formed model look syntactically
    # wrong, and would silently redefine a figure already in use. They carry
    # their own aggregate, `syn_size_score`. Add the four names to CHECKS if
    # that trade-off is ever decided the other way.
    # The yes/no verdict beside the graded scores. Its own columns, no
    # contribution to `syn_score` — see the note above `syntactical_correctness`.
    out.update(syntactical_correctness(pg))
    out.update(size_metrics(pg))
    # The BEF4LLM rule catalogue — the same dimension scored the way that
    # project scores it (error ratios over elements). Separate columns and a
    # separate headline (`syn_bef_score`); nothing here feeds `syn_score`.
    out.update(syntax_rules.evaluate(pg))
    return out


def empty(note: str = "") -> Dict[str, Any]:
    """Column set for a generation that could not be parsed at all.

    Every check is None, not False: "produced no parseable model" is a
    *validity* failure and belongs to that dimension. Scoring it as 0 here
    would blend two different failure modes into one mean. `extract_parse_ok`
    in results.csv already carries the parse-failure rate.
    """
    # The column set is derived from `evaluate()` on an empty graph rather than
    # listed again here — a hand-kept second list silently loses every newly
    # added diagnostic, which would show up as a column that exists for scored
    # rows and vanishes for unparseable ones.
    out: Dict[str, Any] = {k: None for k in evaluate(ProcessGraph())}
    out["syn_checks_total"] = len(CHECKS)
    out["syn_bef_metrics_total"] = len(syntax_rules.METRICS)
    return out
