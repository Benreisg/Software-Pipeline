"""
quality/columns.py — what every column of a quality CSV measures
================================================================
`score_pmo_dataset.py` writes ~180 columns and `run.py` writes the same metric
columns into results.csv. This module is their legend: for each column, which
metric it carries, which dimension and group it belongs to, and where the
definition comes from.

    from quality import columns
    columns.legend_dataframe(df.columns).to_csv("pmo_quality_columns.csv",
                                                index=False)

Every metric family is generated from the same spec objects the scoring reads
(`syntax_rules.METRICS`, `syntactic.CHECKS`, `pragmatic.METRICS`,
`semantic.METRICS`), so a renamed or added metric changes the legend with it —
including the threshold bands, which are read out of `quality/normalize.py`
rather than retyped here. Only the identity and diagnostic columns are
described by hand, in `_STATIC`.

`legend()` never drops a column: one that is neither generated nor listed comes
back marked `undocumented`, so a newly added metric shows up as a visible gap
instead of silently disappearing from the legend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from . import pragmatic, semantic, syntactic, syntax_rules
from .normalize import DIAMETER_THRESHOLDS, Thresholds

# ── the four dimensions, named as the CSV prefixes group them ───────────────
DIM_ID = "identity"
DIM_SYN = "syntactic (dictated checks)"
DIM_BEF = "syntactic (BEF4LLM)"
DIM_PRAG = "pragmatic"
DIM_SEM = "semantic"
DIM_VAL = "validity"

FIELDS = ["column", "dimension", "group", "metric", "kind", "description", "source"]


@dataclass(frozen=True)
class ColumnDoc:
    """One row of the legend."""
    column: str
    dimension: str
    group: str
    metric: str
    kind: str          # score | raw value | count | flag | text | identity
    description: str
    source: str = ""

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def _bands(thresholds: Optional[Thresholds], higher_is_better: bool) -> str:
    """The banding clause appended to every `_score` description, read off the
    thresholds themselves so the legend cannot quote a stale band."""
    if thresholds is None:
        return "reported raw, not scored"
    t = "/".join(f"{v:g}" for v in thresholds)
    fn = "normasc" if higher_is_better else "normdesc"
    direction = "higher is better" if higher_is_better else "lower is better"
    return f"{fn} band (0/0.25/0.5/0.75/1.0) at thresholds {t} — {direction}"


# ── syntactic, the BEF4LLM metric set (syn_bef_*) ───────────────────────────
def _syn_bef_docs() -> List[ColumnDoc]:
    out: List[ColumnDoc] = []
    for spec in syntax_rules.METRICS:
        col = f"syn_bef_{spec.key}"
        source = f"BEF4LLM paper, Table 2 / A.15 #{spec.paper_no}"
        if spec.boolean:
            out.append(ColumnDoc(
                col, DIM_BEF, "Table 2 metric set", spec.label, "score",
                "Boolean metric: 1.0 when the model satisfies the rule, else 0.0.",
                source))
            continue
        out.append(ColumnDoc(
            col, DIM_BEF, "Table 2 metric set", spec.label, "score",
            "Counting metric: conforming elements / covered elements. "
            "1.0 when the rule covers nothing (empty denominator).", source))
        out.append(ColumnDoc(
            f"{col}_conforming", DIM_BEF, "Table 2 metric set",
            f"{spec.label} — conforming elements", "count",
            f"Numerator of {col}: elements that satisfy the rule.", source))
        out.append(ColumnDoc(
            f"{col}_covered", DIM_BEF, "Table 2 metric set",
            f"{spec.label} — covered elements", "count",
            f"Denominator of {col}: elements the rule ranges over.", source))
    return out


_EXTRA_LABELS = {
    "connected_nodes": (
        "Connected nodes",
        "Share of flow objects that lie on a path from a start event to an end "
        "event (start events need out >= 1, end events in >= 1)."),
    "event_gateway_predecessor_successor": (
        "Event gateway: predecessor / successor",
        "Share of event-based split gateways that are followed by events or "
        "preceded by a receive task."),
}


def _syn_extra_docs() -> List[ColumnDoc]:
    source = ("BEF4LLM reference implementation only — in no definition table; "
              "reported, never part of syn_bef_score")
    out: List[ColumnDoc] = []
    for key, (label, desc) in _EXTRA_LABELS.items():
        col = f"syn_extra_{key}"
        out.append(ColumnDoc(col, DIM_BEF, "outside the score", label, "score",
                             desc, source))
        out.append(ColumnDoc(f"{col}_conforming", DIM_BEF, "outside the score",
                             f"{label} — conforming elements", "count",
                             f"Numerator of {col}.", source))
        out.append(ColumnDoc(f"{col}_covered", DIM_BEF, "outside the score",
                             f"{label} — covered elements", "count",
                             f"Denominator of {col}.", source))
    return out


# ── syntactic, the dictated checks (syn_<check>) ────────────────────────────
def _syn_check_docs() -> List[ColumnDoc]:
    out: List[ColumnDoc] = []
    for i, check in enumerate(syntactic.CHECKS, 1):
        label = syntactic.CHECK_LABELS[check]
        source = f"dictated syntactic check {i}/{len(syntactic.CHECKS)}"
        if check == "sequence_flow_connection_rules":
            out.append(ColumnDoc(
                f"syn_{check}", DIM_SYN, "checks", label, "score",
                "Valid sequence flows / all sequence flows — a ratio, not "
                "pass/fail. Invalid: a flow out of an end event or into a "
                "start event.", source))
        else:
            out.append(ColumnDoc(
                f"syn_{check}", DIM_SYN, "checks", label, "flag",
                "True/False; enters syn_score as 1.0/0.0 and counts in "
                "syn_checks_passed only when True.", source))
    return out


def _syn_size_docs() -> List[ColumnDoc]:
    """The four BEF4LLM size metrics under the syn_ prefix. Identical by
    construction to the prag_ ones — both read the same graph primitives."""
    out: List[ColumnDoc] = []
    source = ("BEF4LLM Table A.16 (size); reported beside syn_score, "
              "never folded into it")
    for name in syntactic.SIZE_METRICS:
        # Label and bands come from `syntactic`, not from the pragmatic spec of
        # the same name: the two agree on TNN/TNG/TNSF but not on the diameter,
        # where the syn_ column keeps the literal `max{|p|}` over every flow
        # object while the scored pragmatic one is `diameter_nogw`.
        spec = next((m for m in pragmatic.METRICS if m.key == name), None)
        label = syntactic.SIZE_LABELS[name]
        thresholds = spec.thresholds if spec else DIAMETER_THRESHOLDS
        higher = spec.higher_is_better if spec else False
        formula = (_PRAG_FORMULA[name] + f" Same value as prag_{name}."
                   if spec else _SYN_DIAMETER_FORMULA)
        out.append(ColumnDoc(
            f"syn_{name}", DIM_SYN, "size", label, "raw value", formula, source))
        out.append(ColumnDoc(
            f"syn_{name}_score", DIM_SYN, "size", f"{label} — banded score",
            "score", f"{_bands(thresholds, higher)}. "
                     f"Averaged into syn_size_score.", source))
    return out


# ── pragmatic (prag_*) ──────────────────────────────────────────────────────
# One line per metric: what the raw column actually contains. Kept next to the
# legend rather than inside the specs so the scoring modules stay unchanged.
# `syn_diameter` is the third reading and has no pragmatic column of its own.
_SYN_DIAMETER_FORMULA = (
    "Longest simple start-to-end path, max{|p|} over node sequences with every "
    "flow object counted, gateways included — Table A.16 read literally. Neither "
    "prag_diameter (BEF4LLM's traversal) nor prag_diameter_nogw (the scored one, "
    "gateways skipped)."
)

_PRAG_FORMULA = {
    "tnn": "TNN = |N| — flow objects (events, tasks, gateways); pool/lane anchors "
           "and nodes of unreadable type excluded.",
    "tng": "TNG = |G| — gateways.",
    "tnsf": "TNSF = |F^S| — deduplicated sequence flows (message flows excluded).",
    "diameter_nogw": "Longest simple start-to-end path with activities and events "
                     "counted and gateways skipped — BEF4LLM's counting convention, "
                     "honestly maximised. The scored diameter since 2026-08-29.",
    "density": "Delta = |F^S| / (|N| * (|N| - 1)) — arcs against all arcs that could "
               "exist.",
    "agd": "AGD = sum over gateways of (|in(g)| + |out(g)|) / |G|.",
    "cnc": "CNC = |F^S| / |N| — arcs per node.",
    "gh": "GH = -sum over l in {AND, XOR, OR} of p(l) * log3 p(l) — entropy over the "
          "gateway types the model mixes; 0 when all gateways are of one type.",
    "cfc": "CFC = sum over the split gateways of their branching factor: AND = 1, "
           "XOR = |out(c)|, OR = 2^|out(c)| - 1 (event-based counts as XOR).",
    "cc": "CC = sum over ordered node pairs of the strongest path value / "
          "(|N| * (|N| - 1)); a path's value is the product of its arc weights.",
    "sequentiality": "Xi = arcs between two non-connector nodes / |F^S|.",
    "separability": "Pi = cut vertices / (|N| - 2).",
    "depth": "Lambda = max over nodes of min(in-depth, out-depth) — how deeply the "
             "model nests.",
    "token_split": "TS = sum over AND/OR split gateways of (|out(g)| - 1) — concurrent "
                   "tokens the model can create.",
}


def _prag_docs() -> List[ColumnDoc]:
    out: List[ColumnDoc] = []
    for spec in pragmatic.METRICS:
        group = pragmatic.GROUP_LABELS[spec.group]
        source = "BEF4LLM paper, Table A.16 (pragmatic metrics)"
        out.append(ColumnDoc(f"prag_{spec.key}", DIM_PRAG, group, spec.label,
                             "raw value", _PRAG_FORMULA[spec.key], source))
        out.append(ColumnDoc(
            f"prag_{spec.key}_score", DIM_PRAG, group,
            f"{spec.label} — banded score", "score",
            f"{_bands(spec.thresholds, spec.higher_is_better)}. "
            f"Banded value of prag_{spec.key}; enters prag_score.",
            "BEF4LLM Table A.18 (thresholds), Eq. 1/2"))
    for group in pragmatic.GROUPS:
        members = [m.label for m in pragmatic.METRICS if m.group == group]
        listed = ", ".join(members) if members else "no metric defined in this group"
        out.append(ColumnDoc(
            f"prag_group_{group}_score", DIM_PRAG, pragmatic.GROUP_LABELS[group],
            f"{pragmatic.GROUP_LABELS[group]} — group mean", "score",
            f"Mean of the banded scores in this group ({listed}). Empty when the "
            f"group has no metric.", "BEF4LLM section 4.5"))
    return out


# ── semantic (sem_*) ────────────────────────────────────────────────────────
# semantic.METRICS carries (key, group) only; the paper's names and formulas
# live here. Keys are read from that tuple, so a metric cannot go undescribed.
_SEM_METRICS = {
    "label_sim_syntactic": (
        "Syntactic label similarity",
        "BEF4LLM's own implementation since 2026-08-29. Per pair: "
        "1 - Levenshtein(l1, l2) / max(|l1|, |l2|) on the tokenised, lowercased "
        "labels, two empty labels scoring 0; whole graph: the sum over their "
        "greedy id-keyed matching (one entry per node of *each* model) plus the "
        "participant matching, over non-gateway flow objects + pools + named "
        "lanes of both models."),
    "label_sim_semantic": (
        "Semantic label similarity",
        "BEF4LLM's own implementation since 2026-08-29: "
        "(1.0*same_strings + 0.75*synonyms) / max(|w1|, |w2|) on stemmed, "
        "stopword-free tokens, the overlap counted by their deleting loop; "
        "aggregated like the syntactic one."),
    "label_sim_context": (
        "Context similarity",
        "BEF4LLM's own implementation since 2026-08-30. Per pair: how much of "
        "the semantic-label matching survives one step out from the pair — "
        "(matched predecessors + matched successors) / (max of the two "
        "predecessor counts + max of the two successor counts), gateways "
        "skipped over on the way, 0 when both neighbourhoods are empty; whole "
        "graph: the sum over their greedy id-keyed matching over the "
        "non-gateway flow objects of both models. Pools and lanes are outside "
        "this one, unlike the two label similarities."),
    "graph_edit_distance": (
        "Graph-edit distance",
        "networkx' `optimize_edit_paths` over the two contracted skeletons "
        "(flow objects without gateways, gateways skipped over): the node and "
        "edge insertions, deletions and substitutions that turn one model into "
        "the other, two nodes interchangeable when label and element kind "
        "agree and two edges when the flow type does. Normalised against the "
        "trivial edit script, 1 - ops / (|Vc| + |Vr| + |Ec| + |Er|), so 1.0 "
        "means the same graph. **This port's own reading, not the supplied "
        "code's (snv+sev+sbv)/3.** The exact distance is exponential, so the "
        "search is cut by a wall-clock budget and a cut value is a lower bound "
        "that depends on the machine — read sem_ged_truncated."),
    "common_nodes_edges": (
        "Common nodes and edges",
        "1 - (unmatched nodes + unmatched edges) / (|FOc| + |FOg| + |Fc| + "
        "|Fg|), read through M^opt_Sem: a node is 'in' the other model iff it "
        "is matched, an edge iff its mapped endpoints and flow type exist "
        "there. **This port's own reading — nodes AND edges, as the paper "
        "prints it.** The supplied code calls its common_percentage_similarity "
        "with edges=False over a threshold-0 matching, so their column is 1.0 "
        "for any two models and validates nothing; see pmo_common_edge.py for "
        "the check against the known manipulations that replaces it."),
}

_SEM_GROUP_LABELS = {
    "natural_language": "Natural language",
    "graph_structure": "Graph structure",
}


def _sem_docs() -> List[ColumnDoc]:
    """One entry per semantic metric and per group mean, numbered as Table A.17
    numbers them **after the drops** — the position in `semantic.METRICS`, so
    the legend follows the metric set rather than a hand-kept list."""
    out: List[ColumnDoc] = []
    for number, (key, group) in enumerate(semantic.METRICS, start=1):
        label, formula = _SEM_METRICS[key]
        out.append(ColumnDoc(
            f"sem_{key}", DIM_SEM, _SEM_GROUP_LABELS[group], label, "score",
            formula, f"BEF4LLM paper, Table A.17 #{number}"))
    for group in semantic.GROUPS:
        members = [_SEM_METRICS[k][0] for k, g in semantic.METRICS if g == group]
        out.append(ColumnDoc(
            f"sem_group_{group}_score", DIM_SEM, _SEM_GROUP_LABELS[group],
            f"{_SEM_GROUP_LABELS[group]} — group mean", "score",
            f"Mean of the metrics in this group ({', '.join(members)}). This "
            f"project's roll-up, not the paper's — it aggregates the dimension "
            f"in one step.", "BEF4LLM section 4.5"))
    return out


_STATIC: Dict[str, Tuple[str, str, str, str, str, str]] = {
    "is_few_shot_exemplar": (DIM_ID, "", "Few-shot exemplar", "flag",
                             "True for the items used as prompt exemplars "
                             "(config.FEW_SHOT_IDS). They are scored in real runs "
                             "like every other item — with their own exemplar "
                             "swapped out of the prompt.", "config.FEW_SHOT_IDS"),
    "ground_truth_path": (DIM_ID, "", "Ground-truth file", "text",
                          "Absolute path of the .dot scored. Here it is passed as both "
                          "the generated and the reference model.", ""),
    "quality_note": (DIM_ID, "", "Scoring note", "text",
                     "Empty when scoring succeeded; otherwise why a dimension is "
                     "missing (parse failure).", ""),

    # headline scores
    "syn_score": (DIM_SYN, "headline", "Syntactic score (dictated checks)", "score",
                  "Mean over the dictated checks; the ratio check contributes its "
                  "ratio, the others 1.0/0.0.", "dictated check list"),
    "syn_checks_passed": (DIM_SYN, "headline", "Checks fully passed", "count",
                          "How many of the checks scored exactly 1.0.", ""),
    "syn_checks_total": (DIM_SYN, "headline", "Checks in the set", "count",
                         "Divisor of syn_score.", ""),
    "syn_bef_score": (DIM_BEF, "headline", "Qsyn (BEF4LLM)", "score",
                      "Sum of the metric scores / number of metrics, equal weights. "
                      "The two extras and the size metrics are not included.",
                      "BEF4LLM Eq. 4"),
    "syn_size_score": (DIM_SYN, "size", "Size score", "score",
                       "Mean of the four banded size scores (TNN, TNG, TNSF, "
                       "diameter). Measures smallness, not correctness — deliberately "
                       "not part of syn_score.", "BEF4LLM Table A.16/A.18"),
    "prag_score": (DIM_PRAG, "headline", "Qprag", "score",
                   "Mean over all pragmatic metrics, equal weights; an unmeasurable "
                   "metric enters at 1.0 so the divisor never changes.",
                   "BEF4LLM Eq. 4"),
    "sem_score": (DIM_SEM, "headline", "Qsem", "score",
                  "Mean over all four semantic metrics, equal weights. Against the "
                  "model itself this is NOT 1.0: only the graph-edit distance is "
                  "reflexive, while the three natural-language metrics put a "
                  "self-match at ~0.478 for their shared divisor. A self-comparison "
                  "is a wiring check, never a baseline value.",
                  "BEF4LLM Eq. 4"),

    # BEF4LLM aggregates
    "syn_bef_metrics_total": (DIM_BEF, "aggregate", "Metrics in the set", "count",
                              "Divisor of syn_bef_score.", ""),
    "syn_bef_metrics_perfect": (DIM_BEF, "aggregate", "Metrics scoring 1.0", "count",
                                "How many metrics the model satisfied completely.", ""),
    "syn_bef_metrics_applicable": (DIM_BEF, "aggregate", "Metrics applicable", "count",
                                   "How many metrics had something to measure "
                                   "(non-empty denominator). Not a divisor — it "
                                   "explains a high score on a small model.", ""),
    "syn_bef_n_processes": (DIM_BEF, "aggregate", "Processes", "count",
                            "Processes found (pools/lanes, else the whole model) — the "
                            "denominator of the per-process metrics.", ""),
    "syn_bef_n_pools": (DIM_BEF, "aggregate", "Pools", "count",
                        "Pool subgraphs found in the DOT file.", ""),

    # syntactic diagnostics
    "syn_n_start_events": (DIM_SYN, "diagnostics", "Start events", "count",
                           "Nodes classified as a start event.", ""),
    "syn_n_end_events": (DIM_SYN, "diagnostics", "End events", "count",
                         "Nodes classified as an end event.", ""),
    "syn_n_tasks": (DIM_SYN, "diagnostics", "Tasks", "count",
                    "Nodes classified as a task.", ""),
    "syn_n_gateways": (DIM_SYN, "diagnostics", "Gateways", "count",
                       "Nodes classified as a gateway.", ""),
    "syn_n_nodes": (DIM_SYN, "diagnostics", "Nodes", "count",
                    "All DOT nodes, pool/lane anchors included (unlike syn_tnn).", ""),
    "syn_n_edges": (DIM_SYN, "diagnostics", "Edges", "count",
                    "All deduplicated DOT edges, message flows included.", ""),
    "syn_n_unlabeled_tasks": (DIM_SYN, "diagnostics", "Unlabeled tasks", "count",
                              "Tasks whose effective label is empty — why "
                              "syn_tasks_labeled failed.", ""),
    "syn_n_placeholder_labels": (DIM_SYN, "diagnostics", "Placeholder labels", "count",
                                 "Labels that exist but name nothing ('Task_1', 't3'). "
                                 "Informational, not part of any check.", ""),
    "syn_n_malformed_tasks": (DIM_SYN, "diagnostics", "Malformed tasks", "count",
                              "Tasks violating in = 1, out = 1 (every edge counted).",
                              ""),
    "syn_n_splits": (DIM_SYN, "diagnostics", "Split gateways", "count",
                     "Gateways acting as a split.", ""),
    "syn_n_joins": (DIM_SYN, "diagnostics", "Join gateways", "count",
                    "Gateways acting as a join.", ""),
    "syn_n_bad_splits": (DIM_SYN, "diagnostics", "Bad splits", "count",
                         "Splits violating in = 1, out > 1.", ""),
    "syn_n_bad_joins": (DIM_SYN, "diagnostics", "Bad joins", "count",
                        "Joins violating in > 1, out = 1.", ""),
    "syn_n_gateways_degenerate": (DIM_SYN, "diagnostics", "Degenerate gateways", "count",
                                  "1:1 pass-through gateways — they route nothing "
                                  "and fail both degree checks.", ""),
    "syn_n_gateways_mixed": (DIM_SYN, "diagnostics", "Mixed gateways", "count",
                             "Gateways that fan in and out at once (in > 1 and "
                             "out > 1). Neither split nor join, so outside GS/GJ "
                             "and outside the two degree checks; still eligible as "
                             "the partner of a split.", ""),
    "syn_n_unmatched_splits": (DIM_SYN, "diagnostics", "Unmatched splits", "count",
                               "Splits with no matching join of the same type — why "
                               "syn_split_has_matching_join failed. Event-based "
                               "splits are exempt and never counted here.", ""),
    "syn_n_event_based_splits": (DIM_SYN, "diagnostics", "Event-based splits", "count",
                                 "Splits exempt from syn_split_has_matching_join "
                                 "(author's decision, 2026-08-23).", ""),
    "syn_n_intermediate_events": (DIM_SYN, "diagnostics", "Intermediate events", "count",
                                  "Events the notation declares intermediate (a "
                                  "doublecircle that is not a sink). No metric scores "
                                  "them; they are kept out of the end events.", ""),
    "syn_n_events_unclassified": (DIM_SYN, "diagnostics", "Unclassified events", "count",
                                  "Event-shaped nodes that are neither start nor end "
                                  "nor a declared intermediate event.", ""),
    "syn_n_nodes_unclassified": (DIM_SYN, "diagnostics", "Unclassified nodes", "count",
                                 "Nodes whose BPMN element type could not be read.", ""),
    "syn_n_sequence_flows": (DIM_SYN, "sequence flows", "Sequence flows", "count",
                             "Denominator of syn_sequence_flow_connection_rules.", ""),
    "syn_n_valid_sequence_flows": (DIM_SYN, "sequence flows", "Valid sequence flows",
                                   "count", "Its numerator.", ""),
    "syn_n_invalid_sequence_flows": (DIM_SYN, "sequence flows", "Invalid sequence flows",
                                     "count", "Flows breaking a connection rule.", ""),
    "syn_n_flows_from_end_event": (DIM_SYN, "sequence flows",
                                   "Flows out of an end event", "count",
                                   "One of the two prohibited cases.", ""),
    "syn_n_flows_into_start_event": (DIM_SYN, "sequence flows",
                                     "Flows into a start event", "count",
                                     "The other prohibited case.", ""),
    "syn_n_duplicate_flows": (DIM_SYN, "sequence flows", "Duplicate flows", "count",
                              "Repeated (source, target) pairs in the DOT file.", ""),
    "syn_n_flows_unclassified_endpoint": (DIM_SYN, "sequence flows",
                                          "Flows with an unclassified endpoint", "count",
                                          "Flows whose source or target has no readable "
                                          "element type.", ""),

    # syntactical correctness verdict
    "syn_correct": (DIM_SYN, "correctness verdict", "Syntactically correct", "flag",
                    "True when all five conditions below hold. A verdict, not a score "
                    "— outside syn_score and syn_checks_passed.",
                    "verdict, 5 conditions"),
    "syn_correct_functions": (DIM_SYN, "correctness verdict", "Functions well formed",
                              "flag", "At least one task, and every task has exactly "
                              "one incoming and one outgoing sequence flow.", ""),
    "syn_correct_gateways": (DIM_SYN, "correctness verdict", "Gateways wired", "flag",
                             "Every gateway has in >= 1 and out >= 1. A model without "
                             "gateways passes.", ""),
    "syn_correct_start_end": (DIM_SYN, "correctness verdict", "Start and end present",
                              "flag", "At least one start node and one end node.", ""),
    "syn_correct_directed": (DIM_SYN, "correctness verdict", "Graph is directed", "flag",
                             "digraph, not graph — port of Graphviz agisdirected().",
                             ""),
    "syn_correct_connected": (DIM_SYN, "correctness verdict", "Graph is coherent",
                              "flag", "One connected component — port of Graphviz "
                              "isConnected().", ""),
    "syn_n_malformed_functions": (DIM_SYN, "correctness verdict", "Malformed functions",
                                  "count", "Tasks failing condition 1 (sequence flows "
                                  "only, message flows ignored).", ""),
    "syn_n_unwired_gateways": (DIM_SYN, "correctness verdict", "Unwired gateways",
                               "count", "Gateways failing condition 2.", ""),
    "syn_n_components": (DIM_SYN, "correctness verdict", "Connected components", "count",
                         "1 for a coherent model; how badly a 'no' broke it otherwise.",
                         ""),
    "syn_diameter_nogw": (DIM_SYN, "size", "Diameter (gateways not counted)",
                          "raw value",
                          "Longest start-to-end path with gateways skipped — BEF4LLM's "
                          "own convention. Outside syn_size_score.", ""),
    "syn_diameter_nogw_score": (DIM_SYN, "size",
                                "Diameter (gateways not counted) — banded score",
                                "score", _bands(DIAMETER_THRESHOLDS, False) + ".", ""),
    "syn_diameter_truncated": (DIM_SYN, "size", "Diameter truncated", "flag",
                               "True when the longest-path search hit its expansion "
                               "cap, so the diameter is a lower bound.", ""),

    # pragmatic diagnostics
    "prag_n_metrics_measured": (DIM_PRAG, "aggregate", "Metrics measured", "count",
                                "How many metrics were measurable on this model; the "
                                "rest entered prag_score at 1.0.", ""),
    "prag_n_metrics_total": (DIM_PRAG, "aggregate", "Metrics in the set", "count",
                             "Divisor of prag_score.", ""),
    "prag_n_splits": (DIM_PRAG, "diagnostics", "Split gateways", "count",
                      "Splits on the control-flow view — the set CFC and token split "
                      "sum over.", ""),
    "prag_n_joins": (DIM_PRAG, "diagnostics", "Join gateways", "count",
                     "Joins on the control-flow view.", ""),
    "prag_n_gw_exclusive": (DIM_PRAG, "diagnostics", "XOR gateways", "count",
                            "Gateways resolved to exclusive — a GH share.", ""),
    "prag_n_gw_parallel": (DIM_PRAG, "diagnostics", "AND gateways", "count",
                           "Gateways resolved to parallel — a GH share.", ""),
    "prag_n_gw_inclusive": (DIM_PRAG, "diagnostics", "OR gateways", "count",
                            "Gateways resolved to inclusive — a GH share.", ""),
    "prag_n_gw_eventbased": (DIM_PRAG, "diagnostics", "Event-based gateways", "count",
                             "Counted into the XOR share for GH and CFC, as in "
                             "BEF4LLM.", ""),
    "prag_n_gateways_type_defaulted": (DIM_PRAG, "diagnostics",
                                       "Gateways typed by default", "count",
                                       "Gateways whose type could not be read from "
                                       "name/label and fell back to the default.", ""),
    "prag_n_cut_vertices": (DIM_PRAG, "diagnostics", "Cut vertices", "count",
                            "Articulation points — the numerator of separability.", ""),
    "prag_n_flows_outside_fo": (DIM_PRAG, "diagnostics",
                                "Flows outside the flow objects", "count",
                                "Sequence flows with an endpoint that is not a flow "
                                "object, dropped from the control-flow view. 0 on "
                                "PMo.", ""),
    "prag_n_self_loops": (DIM_PRAG, "diagnostics", "Self-loops", "count",
                          "Flows from a node to itself, dropped from the view.", ""),
    "prag_diameter": (DIM_PRAG, "Size", "Diameter (BEF4LLM's traversal)",
                      "raw value", "Their `countpaths` reproduced with its defects, "
                      "so this column is comparable with their published figures. "
                      "Not the scored one since 2026-08-29 — see prag_diameter_nogw.",
                      "BEF4LLM pragmatic_quality_metrics.py::diameter"),
    "prag_diameter_score": (DIM_PRAG, "Size",
                            "Diameter (BEF4LLM's traversal) — banded score",
                            "score", _bands(DIAMETER_THRESHOLDS, False) +
                            ". Rides along, does not enter prag_score.", ""),
    "prag_diameter_truncated": (DIM_PRAG, "Size", "Diameter truncated", "flag",
                                "True when the longest-path search hit its expansion "
                                "cap.", ""),
    "prag_depth_truncated": (DIM_PRAG, "Partitionability", "Depth truncated", "flag",
                             "True when the depth search hit its expansion cap, so "
                             "depth is a lower bound.", ""),

    # semantic diagnostics
    "sem_n_metrics_measured": (DIM_SEM, "aggregate", "Metrics measured", "count",
                               "How many semantic metrics were measurable; the rest "
                               "entered sem_score at 1.0.", ""),
    "sem_n_metrics_total": (DIM_SEM, "aggregate", "Metrics in the set", "count",
                            "Divisor of sem_score.", ""),
    "sem_n_label_nodes_generated": (DIM_SEM, "diagnostics", "Compared nodes (generated)",
                                    "count", "Non-gateway flow objects of the candidate "
                                    "model — a denominator of the label metrics.", ""),
    "sem_n_label_nodes_reference": (DIM_SEM, "diagnostics", "Compared nodes (reference)",
                                    "count", "The same for the ground truth.", ""),
    "sem_n_nodes_matched": (DIM_SEM, "diagnostics", "Nodes matched", "count",
                            "Size of the optimal semantic matching M^opt_Sem.", ""),
    "sem_n_skeleton_edges_generated": (DIM_SEM, "diagnostics",
                                       "Contracted edges (generated)", "count",
                                       "Edges of the candidate contracted onto the "
                                       "compared nodes.", ""),
    "sem_n_skeleton_edges_reference": (DIM_SEM, "diagnostics",
                                       "Contracted edges (reference)", "count",
                                       "The same for the ground truth.", ""),
    "sem_n_edges_matched": (DIM_SEM, "diagnostics", "Edges matched", "count",
                            "Candidate edges present in the reference through the "
                            "matching.", ""),
    # BEF4LLM's own readings of the four semantic metrics, re-added
    # 2026-08-29 — reproduced from `bef4llm/semantic_quality/`, defects
    # included, and deliberately outside `sem_score`. See the section in
    # `quality/semantic.py` for what each one does differently.
    "sem_label_sim_syntactic_disjoint": (
        DIM_SEM, "natural language",
        "Syntactic label similarity (BEF4LLM's reading, id spaces separated)",
        "raw value", "The same algorithm with the candidate's node ids moved "
        "into their own space, so their id-keyed matching cannot collapse two "
        "entries into one. Compare against a BEF4LLM run over prefix_bpmn_ids.py "
        "output. Not in sem_score.", "BEF4LLM semantic_quality_check"),
    "sem_label_sim_semantic_disjoint": (
        DIM_SEM, "natural language",
        "Semantic label similarity (BEF4LLM's reading, id spaces separated)",
        "raw value", "As above, for their SimSem. Not in sem_score.",
        "BEF4LLM semantic_quality_check"),
    "sem_label_sim_context_disjoint": (
        DIM_SEM, "natural language",
        "Context similarity (BEF4LLM's reading, id spaces separated)",
        "raw value", "As above, for their context similarity — which the "
        "collision hits twice over, since a shared id also makes their "
        "get_graph_for_node resolve both nodes to the same graph. Not in "
        "sem_score.", "BEF4LLM semantic_quality_check"),
    "sem_n_ids_shared": (
        DIM_SEM, "diagnostics", "Node ids shared by both models", "count",
        "How many ids the two models have in common — the collisions their "
        "matching is exposed to. 0 means their reading needs no correction.", ""),
    "sem_n_nodes_matched_greedy": (
        DIM_SEM, "diagnostics", "Nodes matched (BEF4LLM's matching)", "count",
        "Compared nodes of the generated model bound by their greedy matching; "
        "normally all of them.", ""),
    "sem_ged_operations": (
        DIM_SEM, "diagnostics", "Graph-edit distance — raw operations", "count",
        "The edit distance itself, before sem_graph_edit_distance normalises "
        "it against the trivial edit script. 0 = the two models are the same "
        "graph.", "networkx optimize_edit_paths"),
    "sem_ged_truncated": (
        DIM_SEM, "diagnostics", "Graph-edit distance — search truncated", "flag",
        "True = the search ran into its wall-clock budget "
        "(quality/semantic.py::_GED_BUDGET_S), so sem_graph_edit_distance is a "
        "**lower bound** and depends on how fast the machine was. Such a row "
        "is not comparable with one scored to completion. On real generations "
        "this is the common case, not the exception.", ""),
    "sem_common_nodes": (
        DIM_SEM, "diagnostics", "Common nodes and edges — node half", "score",
        "2*matched nodes / (|FOc| + |FOg|). Not a metric of its own: the node "
        "term of sem_common_nodes_edges, reported so a disagreement can be "
        "attributed to nodes or to edges. 1.0 on every model of the "
        "manipulated corpus, since those manipulations never change a node "
        "count.", ""),
    "sem_common_edges": (
        DIM_SEM, "diagnostics", "Common nodes and edges — edge half", "score",
        "2*matched edges / (|Fc| + |Fg|). The edge term of "
        "sem_common_nodes_edges. Note the metric is NOT the mean of the two "
        "halves — it weighs each by how many elements it ranges over.", ""),
    "sem_wordnet": (DIM_SEM, "diagnostics", "WordNet available", "flag",
                    "1 = the synonym term of the semantic label similarity was "
                    "computed; 0 = WordNet absent and that term contributed 0.", ""),

    # validity
    "val_score": (DIM_VAL, "headline", "Qval", "score",
                  "The dimension's headline — mean over its implemented checks, "
                  "currently the DOT check alone.", ""),
    "val_dot_valid": (DIM_VAL, "checks", "Valid DOT", "score",
                      "1.0 / 0.0 when Graphviz accepts / rejects the file; empty when "
                      "the check could not be run.", "Graphviz nop -p"),
    "val_dot_error": (DIM_VAL, "checks", "DOT error", "text",
                      "Graphviz's own message, or why the check could not run.", ""),
    "val_dot_checker": (DIM_VAL, "checks", "DOT checker", "text",
                        "Which binary answered, with its version.", ""),
}


def _generated() -> Dict[str, ColumnDoc]:
    docs: Dict[str, ColumnDoc] = {}
    for doc in (_syn_bef_docs() + _syn_extra_docs() + _syn_check_docs()
                + _syn_size_docs() + _prag_docs() + _sem_docs()):
        docs[doc.column] = doc
    return docs


_GENERATED = _generated()


def describe(column: str) -> ColumnDoc:
    """The legend entry for one column. An unknown column comes back marked
    `undocumented` rather than raising — see the module docstring."""
    if column in _GENERATED:
        return _GENERATED[column]
    if column in _STATIC:
        dimension, group, metric, kind, description, source = _STATIC[column]
        return ColumnDoc(column, dimension, group, metric, kind, description, source)
    return ColumnDoc(column, "", "", "", "undocumented",
                     "No legend entry — add it to quality/columns.py.", "")


def legend(columns: Iterable[str]) -> List[ColumnDoc]:
    """One entry per column, in the order given."""
    return [describe(c) for c in columns]


# ── the published metric sets, without the project's own additions ──────────
# Which columns belong to BEF4LLM's three quality dimensions *as the paper
# defines them*, for a report that has to show that framework and nothing else.
# Built from the spec objects, so it follows the metric sets rather than a
# hand-kept list of names.
#
# In, per dimension:
#   syntactic  Table 2 / A.15 — the metric scores, plus Qsyn
#   pragmatic  Table A.16 — each metric's raw value and its banded score,
#              the seven group means (two of which the paper leaves empty),
#              plus Qprag
#   semantic   Table A.17 — the seven similarities, three group means, Qsem
#
# Out: the dictated syntactic checks (`syn_*`, not BEF4LLM's), the two extras
# the reference code adds outside the definition tables (`syn_extra_*`), the
# size metrics under the `syn_` prefix (the same four values as `prag_tnn`…),
# the validity dimension (this project's, not the framework's), and every
# count/diagnostic column — including the `_conforming` / `_covered` pairs,
# which are a metric's numerator and denominator rather than a metric.
BEF4LLM_HEADLINES = ("syn_bef_score", "prag_score", "sem_score")
BEF4LLM_IDENTITY = ("item_id", "is_few_shot_exemplar")


def bef4llm_metric_columns(columns: Iterable[str]) -> List[str]:
    """The subset of `columns` that is BEF4LLM's own syntactic, pragmatic and
    semantic metric set, in the order given, with the identity columns first."""
    keep = set(BEF4LLM_HEADLINES)
    keep |= {f"syn_bef_{m.key}" for m in syntax_rules.METRICS}
    keep |= {f"prag_{m.key}" for m in pragmatic.METRICS}
    keep |= {f"prag_{m.key}_score" for m in pragmatic.METRICS}
    keep |= {f"prag_group_{g}_score" for g in pragmatic.GROUPS}
    keep |= {f"sem_{k}" for k, _ in semantic.METRICS}
    keep |= {f"sem_group_{g}_score" for g in semantic.GROUPS}

    columns = list(columns)
    identity = [c for c in columns if c in BEF4LLM_IDENTITY]
    return identity + [c for c in columns if c in keep]


def undocumented(columns: Iterable[str]) -> List[str]:
    """The columns `describe` has nothing to say about."""
    return [c for c in columns if c not in _GENERATED and c not in _STATIC]


def legend_dataframe(columns: Iterable[str]):
    """The legend as a DataFrame with the `FIELDS` columns."""
    import pandas as pd
    return pd.DataFrame([d.as_dict() for d in legend(columns)], columns=FIELDS)
