"""
quality/semantic.py — Semantic quality (BEF4LLM's published metric set)
===========================================================================
Does the generated model say the same thing as the reference? The one
dimension that *needs* the ground truth: every metric here compares the
candidate model Mc against the reference model Mg, so without a
`ground_truth_path` the whole column set is None (CSV generation-only mode).

The semantic dimension of BEF4LLM **as the paper defines it** — §4.3, the
formulas in Table A.17 — ported to this project's DOT-based `ProcessGraph`.
Where the paper's definition and the supplied code
(`bef4llm/semantic_quality/`) disagree, this follows the paper, and says so
below. Seven metrics in three groups, the similarity groups of Dijkman
et al. (2011) [55]:

    group               metrics
    ─────────────────   ─────────────────────────────────────────────────
    natural_language    syntactic / semantic label similarity, context
                        similarity

**Two groups are left.** The behaviour group went when both of its metrics were
dropped (2026-08-26), and the graph-structure group when both of its own
followed (2026-08-29) — see `METRICS` for each. The graph-structure group is
back since 2026-09-08 with the graph-edit distance, so the dimension is the two
label similarities, the context similarity (back 2026-08-30) and that distance,
and `sem_score` no longer equals `sem_group_natural_language_score`.

Every metric is a similarity in [0, 1] (the paper: "Each metric is already
scaled to the interval [0,1], so their aggregation yields the overall
semantic score Qsem without additional normalization"), so there is no
raw/banded split as in the pragmatic dimension — one column per metric.

── The moving parts, in the paper's notation ────────────────────────────────
**Compared elements.** Label similarity runs over the flow objects whose
type is not a gateway (Table A.17 writes the denominators as
|{n ∈ FOc | τc(n) ∉ G}| + |{n ∈ FOg | τg(n) ∉ G}|): tasks, start/end events
and intermediate events. Gateways carry routing logic, not statements about
reality. Pools and lanes are **not** included — they are not flow objects
(the supplied code adds them to the label metrics; the paper's formula does
not, and this port follows the paper).

**The matching M^opt.** "Optimal bipartite matchings M^opt_SimX ⊆ FOc × FOg
maximize similarity under criterion X" — computed exactly, per criterion,
with `scipy.optimize.linear_sum_assignment` (Hungarian method). Pairs with
similarity 0 are dropped from the matching afterwards: they add nothing to
any similarity sum, and keeping them would declare two completely unrelated
nodes "the same" for the structural metrics. (The supplied code instead
builds the matching greedily in file order and lets later pairs overwrite
earlier ones one-sidedly; that is one of its divergences, see below.)

**The structural edge set.** For metrics 4–7 the graphs are first
*contracted onto the compared nodes*: every node that is not a non-gateway
flow object (gateways, pool anchors, unclassifiable nodes) is removed and
its predecessors wired to its successors, so `task → gateway → task`
contributes the edge `task → task`. This is how the reference
implementation (and Dijkman's GED) treats ignored node types, and it is
what makes a model identical to the ground truth score 1.0 — without the
contraction, every gateway of both models would sit unmatched in the edit
distance forever, since gateways are excluded from the label matching. An
edge keeps its flow type (message if any contracted step was a message
flow, else sequence), and edge identity for all four structural metrics is
(mapped source, mapped target, flow type). Self-loops are dropped.

**Which Sim the structural metrics use.** Table A.17 writes plain `M` and
`Sim` for metrics 4–7. This port uses the **semantic-label matching**
M^opt_SimSem throughout, as the supplied code does: semantic label
similarity is the framework's notion of node equivalence.

── The seven metrics ────────────────────────────────────────────────────────
1  **Syntactic label similarity** — per pair: 1 − Levenshtein(l1, l2) /
   max(|l1|, |l2|) on the tokenised, lowercased label strings. Whole graph:
   2·Σ SimSyn over M^opt_Syn, divided by the non-gateway node counts of
   both models. **Two ε labels** (the paper: label(t) = ε if missing) make
   the formula 0/0; resolved as 1.0 when both nodes are the same element
   kind (start/end/task/intermediate event) and 0.0 otherwise — see
   *The ε-label rule* below.
2  **Semantic label similarity** — labels become word sets w (tokenised,
   stopwords removed, Porter-stemmed): (2·wi·|w1 ∩ w2| + ws·(s(w1,w2) +
   s(w2,w1))) / (|w1| + |w2|), with s(·,·) counting words that have a
   WordNet synonym on the other side (checked on unstemmed forms, one
   representative per stem). wi = 1.0 and ws = 0.75 — the paper names the
   weights but not their values; these are Dijkman et al.'s, which the
   supplied code also uses. Aggregated like metric 1.
3  **Context similarity** — the paper prints |M_in| / (2·√(|c1_in|·|c2_in|))
   + |M_out| / (2·√(|c1_out|·|c2_out|)), where c_in/c_out are the node's
   predecessors/successors in the contracted graph and M_in/M_out counts
   context nodes whose M^opt_Sem partner lies in the other node's context
   (the mapping-based reading of [56], La Rosa et al.). **What this port
   computes is the supplied code's version of that** — see `_sim_context_bef`:
   the same mapping-based counting over the same neighbourhoods, but divided
   by max(|c1|, |c2|) per direction rather than by the geometric mean, and a
   pair with two empty neighbourhoods scores 0 rather than the paper's 0.5.
   Aggregated over their greedy matching, and over the non-gateway flow
   objects of both models alone.
4  **Graph-edit distance** — 1 − avg(snv, sev, sbv): snv = unmatched nodes
   over all nodes, sev = unmatched edges over all edges, sbv =
   2·Σ (1 − Sim) over matched pairs / (matched nodes) — the mean label
   distance of the matched pairs. Boundary cases the formula leaves 0/0:
   no edges on either side → sev = 0; an empty matching → sbv = 1 (nothing
   aligned: maximal substitution distance, matching the supplied code).
5  **Common nodes and edges** — 1 − (unmatched nodes + unmatched edges) /
   (|FOc| + |FOg| + |Fc| + |Fg|). The paper's set differences read through
   the matching: a candidate node "is in" FOg iff it is matched. (The
   supplied code counts **nodes only** — its `common_percentage_similarity`
   is called with `edges=False` — and so implements only half the printed
   formula.)
6  **Causal-footprint overlap** — removed on 2026-08-26, see `METRICS`.
7  **Dependency-graph overlap** — removed on 2026-08-26, see `METRICS`.

── The ε-label rule ─────────────────────────────────────────────────────────
A node whose label is the empty string is not an accident in this corpus:
`label=""` is how PMo (and two of the prompt templates, which *prescribe*
empty start/end labels) writes BPMN's canonically unlabeled events — every
other DOT node renders its id as its text, so ε means "deliberately no
text". For two ε labels the printed formulas are 0/0 (ed(ε,ε)/max(0,0) and
0/(0+0)). This port resolves them with the one signal the framework still
has, the type map τ it already consults to exclude gateways: **ε ↔ ε
similarity is 1.0 when both nodes are the same element kind (start event,
end event, task, intermediate event), else 0.0.** An ε label against a
non-empty label stays 0. The reference implementation returns 0 for ε ↔ ε,
which silently makes every unlabeled-events model — including ones
following its own dataset's notation — structurally unmatchable at the
start and end of the process. Consequence worth knowing: several
indistinguishable ε nodes of the same kind (e.g. two unlabeled end events)
are matched by an arbitrary-but-deterministic tie-break.

**Metrics 1 and 2 are scored the reference implementation's way** (author's
decision, 2026-08-24): a pair of ε labels contributes 0 — its
`calculate_syntactic_similarity` and `calculate_semantic_similarity` share the
same early return — and `_nl_score` drops the factor 2 for both, so they
reproduce `bef_sem_label_sim_syntactic` and `bef_sem_label_sim_semantic` on 54
of 55 models. A perfect self-match now reads 0.5, less where a model has
unlabelled events. The one that still differs is item 32, whose named lanes
their divisor counts and this port does not — the published formula quantifies
over the flow objects minus the gateways only.

**Scoring and matching are separated for metric 2.** `_sim_sem` keeps the
ε-label rule, because its matching *is* the node equivalence metrics 3-7 read;
only the node equivalence for the structural diagnostics is kept from it.
Applying the rule to the matching itself was tried and reverted the same day: a
zero-similarity pair is dropped from the assignment, so 14 models stopped
matching their own unlabelled events and scored below 1.0 against *themselves*
on context, graph-edit distance and common nodes/edges.

Metrics 6 and 7 — causal-footprint overlap and dependency-graph overlap — are
**no longer computed at all**, dropped at the author's instruction on
2026-08-26, and metrics 4 and 5 followed on 2026-08-29; see `METRICS` for each.
Metric 3 was dropped with them on 2026-08-24 and **came back on 2026-08-30**,
in the same reading as metrics 1 and 2: BEF4LLM's own. What is left is those
three.

**Metric 3 is scored the reference implementation's way too**, and its
aggregation is not quite theirs for metrics 1 and 2: the divisor counts the
non-gateway flow objects of both models and **nothing else** — no pools, no
named lanes — because that is what their own method divides by, three lines
below the two that do count them. Its numerator is again a sum over their
matching's entries, one per node of each model.

── Aggregation ──────────────────────────────────────────────────────────────
`sem_score` = arithmetic mean over all five (§4.5, equal weights — metrics 3
and 6 are not part of the set here, see `METRICS`), with the
same fixed-divisor convention as `prag_score`: a metric that is
unmeasurable on a pair of models (an empty Jaccard universe) enters the
mean at 1.0 — vacuously satisfied — rather than dropping out, so every
model is scored on the same scale. `sem_n_metrics_measured` says how many
of the set were actually measured. Group means likewise.

── Where this port diverges from BEF4LLM's code, following the paper ────────
- **The matching is optimal**, not the code's greedy insertion-order
  matching (which can also hold stale one-sided entries).
- **SimSem divides by |w1| + |w2|** with the exact-overlap term doubled, as
  printed; the code divides by max(|w1|, |w2|) without doubling.
- **Common nodes and edges counts both**, as printed; the code counts nodes
  only (see metric 5 above).
- **Pools and lanes are outside the label metrics**, as printed; the code
  adds them to metrics 1–2.

── WordNet, and why `sem_wordnet` exists ────────────────────────────────────
The synonym term of metric 2 needs WordNet, the one thing that cannot be
embedded here. If NLTK and its wordnet corpus are installed
(`pip install nltk && python -m nltk.downloader wordnet`), the term is
computed; otherwise it contributes 0 and `sem_wordnet` = 0 records that
this row was scored without synonyms. Everything else in the dimension —
tokenisation, stopwords, Porter stemming, Levenshtein — is deterministic
stdlib (`quality/textsim.py`), identical on every machine. English only:
the PMo corpus is English.

Metric 4 is the second and larger caveat on that word *deterministic*: its
search is cut by a wall-clock budget, so a truncated value is a bound that
depends on the machine. `sem_ged_truncated` marks every row it happened to, and
on real generations that is most of them — see `METRICS`.

── Columns ──────────────────────────────────────────────────────────────────
  sem_label_sim_syntactic      metric 1        all four are similarities
  sem_label_sim_semantic       metric 2        in 0.0–1.0, None when
  sem_label_sim_context        metric 3        unmeasurable
  sem_graph_edit_distance      metric 4
  sem_ged_operations           metric 4's raw edit distance, before normalising
  sem_ged_truncated            True = the search ran into its wall-clock budget,
                               so the similarity is a lower bound and is not
                               reproducible on another machine
  sem_group_<group>_score      mean over that group's metrics
  sem_score                    Qsem — mean over the metric set (`METRICS`)
  sem_n_metrics_measured/_total  how many entered the mean as measured
  sem_n_label_nodes_generated/_reference   non-gateway flow objects per side
  sem_n_nodes_matched          |M^opt_Sem|
  sem_n_skeleton_edges_generated/_reference  contracted edges per side
  sem_n_edges_matched          candidate edges present in the reference
  sem_wordnet                  1 = synonym term computed, 0 = WordNet absent

Like every dimension: never a silent zero. No ground truth, or an
unparseable side, yields the all-None column set via `empty()`.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import networkx as nx
except ImportError:  # pragma: no cover — required in requirements.txt
    nx = None  # metric 4 then reports None, never 0

from . import textsim
from .graph import ProcessGraph

# Dijkman et al. (2011): exact word overlap weighs 1.0, a synonym pair 0.75.
# The paper's Table A.17 carries the weights symbolically (wi, ws) and never
# assigns them; these are the source's and the supplied code's values.
WI = 1.0
WS = 0.75

GROUPS = ("natural_language", "graph_structure")

# (column key, group) — order is the paper's Table A.17.
METRICS: Tuple[Tuple[str, str], ...] = (
    ("label_sim_syntactic", "natural_language"),
    ("label_sim_semantic", "natural_language"),
    # Metric 3, context similarity, is **back since 2026-08-30** (author's
    # instruction) — as BEF4LLM's code computes it, the way the two label
    # metrics above have been since 2026-08-29, not as Table A.17 prints it.
    # It was dropped on 2026-08-24 because neither reading was usable *then*:
    # the published formula scores a self-comparison 1.0 where their code
    # scores about 0.48, and reproducing theirs stalled at 42 of 55 models.
    # What has changed since is the ground the comparison stands on. That
    # attempt ran over **self**-comparisons, where every node id collides, so
    # their id-keyed matching collapsed every pair into one entry and
    # `get_graph_for_node` resolved both sides to the candidate graph — two
    # defects on top of the metric. `prefix_bpmn_ids.py` and the `_disjoint`
    # variant take both out, and on that footing the two label metrics reach
    # 55 of 55 and this one **44 of 55** (max |d| 0.167, mean |d| 0.015;
    # `score_pmo_pairs.py`, 2026-08-30).
    #
    # Every one of the 11 that are left has the same cause, and it is not the
    # metric. Their SimSem matrix has ties — two labels made of the same words
    # ("Order was shipped" against "Shipping the order"), and every unlabelled
    # event, which scores 0 against everything — and their greedy matching
    # resolves a tie by *iteration order*, leaving the node bound to whichever
    # partner came first. Theirs is the BPMN document's order; this port's is
    # the sorted DOT id. Metrics 1 and 2 are blind to it, the value being the
    # same either way and only the partner differing; metric 3 reads exactly
    # that partner. Checked node by node on all 11 pairs: the neighbourhoods
    # agree everywhere, and every node bound differently is bound at an *equal*
    # similarity. Nothing in the DOT carries the BPMN's element order, so this
    # residue cannot be closed from this side.
    #
    # `sem_score` is the mean over **three** metrics now, all in the
    # natural-language group, so it still equals that group's mean but no
    # semantic figure is comparable with one produced before this date.
    ("label_sim_context", "natural_language"),
    # Metric 4, graph-edit distance, was **dropped at the author's instruction
    # on 2026-08-29**, right after metric 5. What its reproduction had reached,
    # for the record: their `(snv + sev + sbv)/3` matched on 8 of 55 pairs
    # exactly and 24 within 0.01 once the comparison ran over identical graphs
    # with disjoint ids — the loosest of the three reproductions, because their
    # `sbv` term divides contracted-node similarities by *uncontracted* node
    # counts, and those differ between DOT and BPMN wherever a model carries
    # pool anchors (items 23, 24 and 38). With it the **graph-structure group**
    # went too, having no member left, so `sem_group_graph_structure_score` is
    # no longer emitted. `sem_score` is the mean over **two** metrics now, both
    # in the natural-language group, so it equals that group's mean; no
    # semantic figure is comparable with one produced before this date.
    # Metric 5, common nodes and edges, was **dropped at the author's
    # instruction on 2026-08-29**, the day its reproduction showed what their
    # column is: `common_percentage_similarity` is called with `edges=False`
    # and their matching binds every node at threshold 0.0, so their value is
    # **1.0 for any two models** — reproduced exactly on 55 of 55 pairs of
    # unrelated PMo processes. The published figure therefore carries no
    # information about the models, and this port's own reading of the metric
    # (nodes *and* edges, over the optimal matching) has nothing to be compared
    # against. `sem_score` is the mean over **three** metrics now, and the
    # graph-structure group over one, so neither is comparable with a figure
    # produced before that date. The reproduction column went with it.
    # Metric 6, causal-footprint overlap, was **dropped at the author's
    # instruction on 2026-08-26**. It could not be validated: the paper's
    # Jaccard and the supplied code's cosine are different measures, and a
    # cross-serialisation comparison against their column cannot attribute a
    # difference to the formula rather than to DOT-versus-BPMN (the
    # reproduction that was built for it reached 79 of 155 exact and was
    # removed with the metric). `sem_score` is the mean over **five** metrics
    # now, and the behaviour group over one, so neither is comparable with a
    # figure produced before that date.
    #
    # Metric 7, dependency-graph overlap, followed on **2026-08-26**, also at
    # the author's instruction — and with it the behaviour group, which had no
    # other member left, so `sem_group_behaviour_score` is no longer emitted.
    # The reproduction of the supplied code's reading that had been built for
    # it went too; it had reached 55/55 exact on the PMo self-matches, 45/45 on
    # the basic-vs-full pairs and 35/55 on a generated-vs-truth run once their
    # id-keyed matching collision was taken out of the comparison.
    # `sem_score` is the mean over **four** metrics now.
    #
    # ── Metric 4 is **back since 2026-09-08** (author's instruction) ──
    # Restored on networkx, as `_graph_edit_distance` — a real edit distance on
    # the contracted skeletons, which is the reading this port always computed
    # for the column and *not* the `(snv + sev + sbv)/3` of the supplied code.
    # That distinction is why it can come back while the question that removed
    # it stays open: what could not be validated was the **reproduction** of
    # their formula across the DOT/BPMN boundary (their `sbv` divides
    # contracted-node similarities by uncontracted node counts, so items 23, 24
    # and 38 differ for carrying pool anchors). A real edit distance has no such
    # term, needs no matching of ours, and is reflexive — a model against itself
    # scores exactly 1.0, unlike the two label similarities, whose shared
    # divisor puts a self-match at ~0.478.
    #
    # With it the **graph-structure group is back**, so
    # `sem_group_graph_structure_score` is emitted again and `sem_score` is the
    # mean over **four** metrics. No semantic figure is comparable with one
    # produced between 2026-08-29 and this date.
    #
    # **Read `sem_ged_truncated` before quoting a value.** The exact distance is
    # exponential and does not finish on models this size, so the search is cut
    # by a wall-clock budget and the value is then an upper bound on the
    # distance — a lower bound on the similarity — that depends on how fast the
    # machine was. Measured on `runs/20260824_235554` (55 real generations
    # against their ground truth): **43 of 55 cut at the 5-second default**,
    # mean 4.01 s per pair. Raising the budget does not buy this back — at 60 s
    # half of a 12-pair sample was still cut, for eight times the wall time. The
    # manipulated pairs are the easy case by comparison (3 of 55 cut), because
    # the two sides are nearly the same graph.
    ("graph_edit_distance", "graph_structure"),
    #
    # ── Metric 5 is **back since 2026-09-09** (author's instruction) ──
    # The printed formula over nodes *and* edges — see `_common_nodes_edges` for
    # what that restores and, just as importantly, what it does not: the
    # supplied code's column is a constant 1.0, so nothing here is validated
    # against it and nothing can be. `pmo_common_edge.py` checks it against the
    # known manipulations instead, which confirms the edge half and cannot
    # exercise the node half at all.
    #
    # `sem_score` is the mean over **five** metrics now, and the graph-structure
    # group over two. No semantic figure is comparable with one produced before
    # this date.
    ("common_nodes_edges", "graph_structure"),
)

# Same convention as quality/pragmatic.py: an unmeasurable metric enters the
# mean at 1.0 (vacuously satisfied) so the divisor never changes.
_UNMEASURABLE_SCORE = 1.0

_Edge = Tuple[str, str, str]  # (source, target, "sequence" | "message")


def _contract(pg: ProcessGraph, keep: Set[str],
              self_loops: bool = False) -> Set[_Edge]:
    """Edges of the graph contracted onto `keep`.

    Every node outside `keep` is skipped over: an edge (u, v) exists iff the
    original graph has a path u → … → v whose interior nodes are all outside
    `keep`. The edge is a message flow iff any hop on that path is one.
    Self-loops (a contracted path returning to its origin) are dropped unless
    `self_loops` — as an *edge* one carries no between-node statement to
    compare, but as a *neighbourhood* entry it is real: a task inside a rework
    loop reaches itself over the gateway, and BEF4LLM's `get_predecessors`
    returns it, the origin being the one node their `visited` list starts
    without. Metric 3 therefore asks for them and everything else does not.

    **Pool and lane anchors are walls, not interior nodes.** PMo's DOT draws a
    black-box pool as a node and attaches the message flows of that participant
    to it, so skipping over it the way a gateway is skipped would walk out of
    one participant and into another and state a dependency the model does not
    contain. BPMN attaches a message flow to the participant element, which is
    not a flow node at all, so no path can pass through one.
    """
    adjacency: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)
    walls = pg.markers
    for edge in pg.unique_edges:
        src, dst = edge
        if src != dst and src not in walls and dst not in walls:
            adjacency[src].append((dst, pg.is_message_flow(edge)))

    edges: Set[_Edge] = set()
    for origin in keep:
        stack = list(adjacency.get(origin, ()))
        seen: Set[Tuple[str, bool]] = set()
        while stack:
            node, msg = stack.pop()
            if node in keep:
                if self_loops or node != origin:
                    edges.add((origin, node, "message" if msg else "sequence"))
                continue
            if (node, msg) in seen:
                continue
            seen.add((node, msg))
            for nxt, nxt_msg in adjacency.get(node, ()):
                stack.append((nxt, msg or nxt_msg))
    return edges


class _Side:
    """One model, prepared for comparison: the compared nodes with their
    label representations, and the contracted edge sets."""

    def __init__(self, pg: ProcessGraph):
        # |{n ∈ FO | τ(n) ∉ G}| — flow objects minus gateways. Sorted so the
        # matching (and its tie-breaking) is deterministic.
        self.nodes: List[str] = sorted(
            pg.tasks | pg.starts | pg.ends | pg.intermediates
            | pg.events_unclassified)
        # τ restricted to the compared nodes — the ε-label rule's evidence.
        self.kind: List[str] = [pg.kind_of(n) for n in self.nodes]

        # Metric 1 — the label as a cleaned string.
        self.syn_text: List[str] = [
            " ".join(textsim.tokenize(pg.label_of(n))) for n in self.nodes]

        # Metric 2 — the label as a word set: one representative surface word
        # per Porter stem, stopwords removed. |w| in the formula is the
        # number of distinct stems.
        self.rep: List[Dict[str, str]] = []
        for n in self.nodes:
            by_stem: Dict[str, str] = {}
            for word in textsim.content_words(pg.label_of(n)):
                by_stem.setdefault(textsim.porter_stem(word), word)
            self.rep.append(by_stem)
        self.stems: List[FrozenSet[str]] = [
            frozenset(r.keys()) for r in self.rep]

        # Metrics 3–5 — the graph contracted onto the compared nodes, once
        # with the self-loops metric 3 needs and once without, for the edge
        # sets and counts that must not carry them.
        looped = _contract(pg, set(self.nodes), self_loops=True)
        self.skel: Set[_Edge] = {e for e in looped if e[0] != e[1]}

        # Metric 3 — each compared node's neighbourhood, read off the same
        # contracted graph. `get_predecessors`/`get_successors` walk the raw
        # graph and recurse *through* every gateway (and every node without a
        # `name` attribute) until they reach a named non-gateway node, which
        # is what the contraction already did: a node's contracted
        # predecessors are exactly the non-gateway nodes that reach it over a
        # gateway-only path. Message flows count on both sides — theirs are
        # edges of `process_graph` whenever both ends are flow nodes — and a
        # pool anchor is a wall here because in BPMN it is not in the graph at
        # all. A node in a rework loop is its own neighbour, which is why this
        # reads `looped` and not `skel`. The flow type is dropped: their
        # neighbourhood is a node set.
        index = {node: i for i, node in enumerate(self.nodes)}
        self.pred: List[Set[str]] = [set() for _ in self.nodes]
        self.succ: List[Set[str]] = [set() for _ in self.nodes]
        for src, dst, _kind in looped:
            self.succ[index[src]].add(dst)
            self.pred[index[dst]].add(src)



# ── pairwise label similarities ─────────────────────────────────────────────

def _sim_syn(text1: str, text2: str, same_kind: bool) -> float:
    """1 - ed(l1, l2) / max(|l1|, |l2|), with **two ε labels scoring 0.0** —
    BEF4LLM's reading (author's decision, 2026-08-24).

    Their `calculate_syntactic_similarity` leaves before it measures anything:

        if ("name" not in node1 or "name" not in node2 or
                (node1["name"] == "" and node2["name"] == "")):
            return 0

    so an unlabelled flow object contributes nothing to the sum while still
    counting in the divisor. The ε-label rule this replaces resolved ε ↔ ε by
    element kind and is still what `_sim_sem` does for the node matching — see
    the module docstring.
    """
    longest = max(len(text1), len(text2))
    if longest == 0:
        return 0.0
    return 1.0 - textsim.levenshtein(text1, text2) / longest


def _sim_sem(rep1: Dict[str, str], stems1: FrozenSet[str], text1: str,
             rep2: Dict[str, str], stems2: FrozenSet[str], text2: str,
             same_kind: bool) -> float:
    """(2·wi·|w1 ∩ w2| + ws·(s(w1,w2) + s(w2,w1))) / (|w1| + |w2|)."""
    if not text1 and not text2:  # two ε labels — same resolution as SimSyn had
        return 1.0 if same_kind else 0.0
    denominator = len(stems1) + len(stems2)
    if denominator == 0:  # only stopwords survive — no content to compare
        return 0.0
    overlap = len(stems1 & stems2)
    rest1 = [word for stem, word in rep1.items() if stem not in stems2]
    rest2 = [word for stem, word in rep2.items() if stem not in stems1]
    synonyms = (
        sum(1 for w in rest1 if any(textsim.is_synonym(w, v) for v in rest2))
        + sum(1 for v in rest2 if any(textsim.is_synonym(v, w) for w in rest1))
    )
    return (2.0 * WI * overlap + WS * synonyms) / denominator


def _greedy_matching(matrix: np.ndarray, cand_nodes: List[str],
                     ref_nodes: List[str]) -> Dict[str, Tuple[str, float]]:
    """`create_euquivalence_mapping_for_nodes`, reproduced (author's decision,
    2026-08-24) — the matching all three scored metrics run over.

    Their matching is a dict keyed by node id, filled greedily with
    `threshold=0.0`, so **a pair of similarity 0 enters it too** and the first
    partner seen is only replaced by a *strictly* better one:

        if sim_n1_n2 >= threshold:
            if node1 in matching and matching[node1][1] < sim: matching[node1] = ...
            if node2 in matching and matching[node2][1] < sim: matching[node2] = ...
            if node1 not in matching: matching[node1] = (node2, sim)
            if node2 not in matching: matching[node2] = (node1, sim)

    Both sides share one dict, so a node id present in both models holds a
    single entry — which is why a self-comparison sums to n rather than 2n and
    their scores cap at 0.5. An unlabelled node, similar to nothing, is bound
    to whichever node came first; two nodes with the same label keep their
    first partner because the replacement test is strict.

    **Where the node order still shows.** The *sum* over this dict is the row
    maxima plus the column maxima of the matrix, so it does not depend on the
    order at all — which is why metrics 1 and 2 reproduce exactly from a sorted
    node list. The *partners* do: every candidate for a node whose label
    matches nothing scores 0, and the strict replacement test then leaves it
    bound to whichever node came first. Metric 3 reads those partners rather
    than the values, so a model with unlabelled events can pair them
    differently here than in a run over the BPMN, whose order is the document's.
    """
    matching: Dict[str, Tuple[str, float]] = {}
    for i, node1 in enumerate(cand_nodes):
        for j, node2 in enumerate(ref_nodes):
            similarity = float(matrix[i, j])
            if similarity < 0.0:
                continue
            if node1 in matching and matching[node1][1] < similarity:
                matching[node1] = (node2, similarity)
            if node2 in matching and matching[node2][1] < similarity:
                matching[node2] = (node1, similarity)
            if node1 not in matching:
                matching[node1] = (node2, similarity)
            if node2 not in matching:
                matching[node2] = (node1, similarity)
    return matching


# ── the optimal matching ────────────────────────────────────────────────────

def _optimal_matching(weights: np.ndarray) -> Dict[Tuple[int, int], float]:
    """M^opt for one criterion: the assignment maximising total similarity,
    with zero-similarity pairs dropped — they contribute nothing to any sum,
    and must not count as "the same node" structurally."""
    if weights.size == 0:
        return {}
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return {(int(i), int(j)): float(weights[i, j])
            for i, j in zip(rows, cols) if weights[i, j] > 0.0}


def _pair_matrix(n_rows: int, n_cols: int, sim) -> np.ndarray:
    matrix = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            matrix[i, j] = sim(i, j)
    return matrix


# ── the graph edit distance (metric 4) ──────────────────────────────────────
# Restored 2026-09-08 at the author's instruction, on networkx, which is the
# engine it was given on 2026-08-24 and which `requirements.txt` still names.
#
# **This is the port's own reading of metric 4, not BEF4LLM's.** Theirs is
# `1 - avg(snv, sev, sbv)` — a quota of unmatched nodes and edges plus the mean
# label distance of the matched pairs, which is not an edit distance at all
# despite the name. That reproduction is the one that could not be validated
# across the DOT/BPMN boundary and left with the metric on 2026-08-29: their
# `sbv` divides contracted-node similarities by *uncontracted* node counts, so
# items 23, 24 and 38 differ for carrying pool anchors rather than for anything
# the formula says. A real edit distance has no such term and needs no matching
# of ours — it runs on the contracted skeletons alone, which is why restoring
# it does not restore that open question.
#
# **The exact distance is exponential and does not finish on models this size.**
# `optimize_edit_paths` yields successively cheaper edit paths and its own
# `timeout` cuts the search *inside* the algorithm, so the cheapest path found
# by then is an upper bound on the distance and the similarity a lower bound on
# how alike the two models are.
#
# The timeout has to be networkx's own and not a check between yields, which is
# what an earlier reading did. It does not bind: on item 18 of the manipulated
# pairs the *first* yield alone takes 68 seconds, so a 5-second budget tested
# between yields would have permitted all 68. That matters here more than the
# arithmetic does — scoring runs on a generation worker (`pipeline.run_one`),
# where a metric that does not return takes the whole run with it. Run
# 20260831_182408 lost 21700 of 25740 generations that way to a RecursionError
# in `pragmatic._bef4llm_diameter`, and this is the one metric in the suite able
# to repeat it.
#
# A cut is not cosmetic: the value then depends on how fast the machine was and
# is not reproducible. `sem_ged_truncated` flags every such row, and
# `_GED_BUDGET_S` is the knob — raising it buys tighter bounds on the few hard
# pairs and costs that much wall time on each of them.
_GED_BUDGET_S = 5.0


def _ged_graph(side: "_Side"):
    """One side as networkx sees it: the contracted skeleton, its nodes
    carrying what makes two of them interchangeable.

    A **multi**graph, because the skeleton is keyed on (source, target, flow
    type) and a sequence flow can run beside a message flow between the same
    two nodes. A plain DiGraph would merge the pair into one edge and the
    distance would stop counting one of them.
    """
    graph = nx.MultiDiGraph()
    for node, kind, text in zip(side.nodes, side.kind, side.syn_text):
        graph.add_node(node, label=text, kind=kind)
    for src, dst, flow in side.skel:
        graph.add_edge(src, dst, kind=flow)
    return graph


def _ged_node_match(node1: Dict[str, str], node2: Dict[str, str]) -> bool:
    """Two nodes substitute for free when the label **and** the element kind
    agree — the same evidence the ε-label rule reads, so an unlabelled start
    event and an unlabelled task are not interchangeable."""
    return node1["label"] == node2["label"] and node1["kind"] == node2["kind"]


def _ged_edge_match(edge1: Dict[str, str], edge2: Dict[str, str]) -> bool:
    """Two edges substitute for free when the flow type agrees."""
    return edge1["kind"] == edge2["kind"]


def _graph_edit_distance(cand: "_Side", ref: "_Side",
                         budget_s: float = _GED_BUDGET_S
                         ) -> Tuple[Optional[float], Optional[float], bool]:
    """`(similarity, operations, truncated)` for one pair of contracted models.

    `operations` is the raw distance: the node and edge insertions, deletions
    and substitutions that turn one model into the other. `similarity`
    normalises it against the trivial edit script — delete every node and edge
    of one side, insert every node and edge of the other — so it lands in
    [0, 1] and 1.0 means the two are the same graph:

        1 - operations / (|V_c| + |V_r| + |E_c| + |E_r|)

    Two empty models have no trivial script to normalise against. They are the
    same graph and score 1.0, which is the empty-denominator convention the
    syntactic rules already follow.

    Without networkx the value is None — unmeasurable, never 0.
    """
    if nx is None:
        return None, None, False

    graph_c, graph_r = _ged_graph(cand), _ged_graph(ref)
    trivial = (len(graph_c) + len(graph_r)
               + graph_c.number_of_edges() + graph_r.number_of_edges())
    if trivial == 0:
        return 1.0, 0.0, False

    started = time.perf_counter()
    operations: Optional[float] = None
    for _nodes, _edges, cost in nx.optimize_edit_paths(
            graph_c, graph_r,
            node_match=_ged_node_match, edge_match=_ged_edge_match,
            timeout=budget_s):
        # Paths arrive strictly decreasing, so the last one is the cheapest.
        operations = cost
    elapsed = time.perf_counter() - started

    if operations is None:
        # Cut before it produced even one path. Nothing was measured, and the
        # trivial script is a bound, not a measurement — so, None.
        return None, None, True
    # The generator stops on its own once the search is exhausted; anything
    # that ran into the budget was stopped from outside and holds a bound.
    truncated = elapsed >= budget_s
    return max(0.0, 1.0 - operations / trivial), float(operations), truncated


# ── common nodes and edges (metric 5) ───────────────────────────────────────
# Restored 2026-09-09 at the author's instruction. Like metric 4, what returns
# is **this port's own reading** — the printed formula, over nodes *and* edges —
# and not the supplied code's, whose `common_percentage_similarity` is called
# with `edges=False` and whose matching binds every node at `threshold=0.0`, so
# their published value is 1.0 for any two models whatever. That constant is
# what removed the metric on 2026-08-29: it left this reading with nothing to be
# validated against, the reproduction of a constant being worth nothing.
#
# The validation it has instead is `pmo_common_edge.py`, which scores each
# manipulated PMo model against its original and holds the result against
# `PMo_manipulated/manipulations.json` — the record of what was changed. **Half
# of that check is real and half is vacuous**, and the honest summary is:
#
#   19 rename-only items keep the edge half at exactly 1.0 — a genuine check,
#      and it passes: a label change does not leak into the edge term.
#   18 edge-only items keep the node half at 1.0 — but so does every other
#      item. `sem_common_nodes` is 1.0 on all 55, because the manipulations
#      never change a node count and this metric asks whether a node is matched,
#      never how well, so a rename with any surviving word overlap keeps its
#      node bound.
#
# So on that corpus the metric is carried entirely by its edge half, and it is
# blind to all 46 renames in the rename-only items. That is a property of the
# metric, not a defect of the port — but it is the reason this is a plausibility
# check and not the "reproduced 55/55" standard the label similarities meet.
# Exercising the node half needs a manipulation that adds or deletes a flow
# object, or renames to a label with no word in common.


def _common_nodes_edges(n_c: int, n_r: int, n_matched: int,
                        f_c: int, f_r: int, edges_matched: int
                        ) -> Tuple[float, float, float]:
    """`(metric, node half, edge half)` from counts the dimension already keeps.

        1 - (unmatched nodes + unmatched edges) / (|FOc| + |FOg| + |Fc| + |Fg|)

    read through the matching: a candidate node "is in" FOg iff it is matched,
    and an edge iff its mapped endpoints and flow type exist on the other side.
    Equivalently `(2·matched nodes + 2·matched edges) / that denominator`.

    **The metric is not the mean of its halves.** Each half is the same quotient
    over its own terms, so the combined value weighs each by how many elements
    it ranges over — an edge-rich model moves it more through its edges. Both
    halves are returned because they are what makes a failure legible: they say
    *which* of the two the models disagree on.

    An empty denominator scores 1.0 — nothing to disagree about — the convention
    `quality/syntax_rules.py` already applies to a rule covering no element.
    Arithmetic only: no matching is computed here, so `evaluate` cannot pay for
    one twice and `pmo_common_edge.py` cannot drift from this definition.
    """
    def _ratio(matched: int, denominator: int) -> float:
        return (2 * matched / denominator) if denominator else 1.0

    total = n_c + n_r + f_c + f_r
    unmatched = (n_c + n_r - 2 * n_matched) + (f_c + f_r - 2 * edges_matched)
    return ((1.0 - unmatched / total) if total else 1.0,
            _ratio(n_matched, n_c + n_r),
            _ratio(edges_matched, f_c + f_r))


# ── BEF4LLM's own readings of the same four metrics, reproduced ─────────────
# Re-added 2026-08-29 at the author's instruction, after the reproductions that
# existed for the two behavioural metrics went with them on 2026-08-26.
#
# These columns answer a different question from the four above. Those are the
# paper's Table A.17 definitions; these are what `bef4llm/semantic_quality/`
# actually computes, defects included, so a difference against their published
# column can be attributed: the same algorithm reaching the same number means
# the port is faithful and the divergence lives in the definitions; the same
# algorithm reaching a *different* number means something else is in the way —
# the serialisation, or their id-keyed matching.
#
# None of them enters `sem_score`. They ride along raw, like `prag_diameter`.
#
# What is deliberately *not* reproduced: their matching dict is keyed by node
# id and holds entries for both models at once, so an id occurring in both
# collapses into one entry (12.7 ids per pair on BPMN written to one
# convention, measured 2026-08-26). This project's DOT graphs carry the file's
# own ids, which two models rarely share, so the collision mostly does not
# arise here — the right behaviour, and the reason `dot_to_bpmn.py` has
# `--id-prefix`.


def _bef_tokens(pg: ProcessGraph, node: str) -> Tuple[List[str], List[str]]:
    """Their token pipeline for SimSem: tokenise, **sort**, drop special
    characters and stopwords, then stem — returned as the *word* list and the
    *stem* list, with multiplicity, because their overlap loop deletes from the
    first while walking the second.

    The sort is theirs and is easy to miss —
    `token1 = sorted(tokenize(label_node1.lower()))`, two lines before the loop
    that consumes it. It matters because that loop is order-dependent: its
    shared `factor` guard skips positions once a match is found, so on unsorted
    lists two labels made of the same words can miss each other, and on sorted
    ones they meet like a merge. Dropping stopwords keeps the order, so sorting
    the content words is the same list.

    Without it their *sums* barely move — a pair that loses a word usually
    still is not the row or column maximum — but the **partners** their
    matching binds do, and the context similarity reads the partners. Item 46
    is the case: "Order was shipped" against "Shipping the order" is 1.0 sorted
    and 0.5 unsorted.
    """
    words = sorted(textsim.content_words(pg.label_of(node)))
    return words, [textsim.porter_stem(w) for w in words]


def _sim_sem_bef(words1: List[str], stems1: List[str], text1: str,
                 words2: List[str], stems2: List[str], text2: str) -> float:
    """`natural_language_similarity.calculate_semantic_similarity`, reproduced.

        ((1.0 * same_strings) + (0.75 * synonyms)) / max_length

    Three things separate this from `_sim_sem`, and all three are theirs:

    * the divisor is `max(|w1|, |w2|)` where Table A.17 prints `|w1| + |w2|`,
      and the exact overlap is not doubled;
    * `same_strings` comes from a loop that deletes from `token1`/`token2`
      while walking `stemmed1`/`stemmed2`, guarded by a single shared `factor`,
      so once a match is found the guard `i >= factor and j >= factor` skips
      earlier positions and the deletion indices drift;
    * two ε labels score 0, where the node equivalence this port uses resolves
      them by element kind.

    The synonym count then runs over what the deletions left, nested, so a word
    with two synonyms on the other side counts twice.
    """
    if not text1 and not text2:
        return 0.0
    max_length = max(len(words1), len(words2))
    if max_length == 0:
        return 0.0                       # theirs raises ZeroDivisionError here
    rest1, rest2 = list(words1), list(words2)
    same_strings = 0
    factor = 0
    for i, this_word in enumerate(stems1):
        for j, other_word in enumerate(stems2):
            if this_word == other_word and i >= factor and j >= factor:
                del rest1[i - factor]
                del rest2[j - factor]
                same_strings += 1
                factor += 1
    synonyms = sum(1 for w in rest1 for v in rest2 if textsim.is_synonym(w, v))
    return (1.0 * same_strings + 0.75 * synonyms) / max_length


def _bef_participants(pg: ProcessGraph) -> List[str]:
    """Their participants: every pool, plus every lane that carries a name —
    `create_label_dicts_for_pool_lanes`, and the same set their divisor counts
    as `len(pools) + len([l for l in lanes if "name" in lanes[l]])`.

    **A lane that repeats its pool's name is not a lane here.** This project
    reads pools and lanes from DOT clusters, and a pool cluster with no inner
    lane cluster fills the lane slot with the pool's own name, so counting it
    would count the pool twice — which is exactly what separated this
    reproduction from their column on every pair involving a pooled model
    (12 of 55 on 2026-08-29, all of them, and none of the 43 pool-free pairs).
    Checked against PMo's BPMN participant and named-lane counts: the rule
    agrees on items 21, 22, 23, 24 and 32, and is one too high on item 38,
    where the DOT carries a lane the BPMN leaves unnamed — a difference in the
    two serialisations, not in the arithmetic.
    """
    pools = sorted({p for p in pg.pools.values() if p})
    lanes = sorted({lane for node, lane in pg.lanes.items()
                    if lane and lane != pg.pools.get(node)})
    return pools + lanes


def _greedy_pool_matching(matrix: np.ndarray, labels1: List[str],
                          labels2: List[str]) -> Dict[str, Tuple[str, float]]:
    """`create_euquivalence_mapping_for_pools`, which is *not* the node
    matching: on a replacement it writes **both** directions, and its branches
    are `if/elif/else` rather than the node version's four independent `if`s.
    Reproduced separately for that reason.
    """
    matching: Dict[str, Tuple[str, float]] = {}
    for i, first in enumerate(labels1):
        for j, second in enumerate(labels2):
            similarity = float(matrix[i, j])
            if first in matching:
                if matching[first][1] < similarity:
                    matching[first] = (second, similarity)
                    matching[second] = (first, similarity)
            elif second in matching:
                if matching[second][1] < similarity:
                    matching[first] = (second, similarity)
                    matching[second] = (first, similarity)
            else:
                matching[first] = (second, similarity)
                matching[second] = (first, similarity)
    return matching


def _bef_label_metric(node_matching: Dict[str, Tuple[str, float]],
                      participants1: List[str], participants2: List[str],
                      n_elements1: int, n_elements2: int,
                      prefix: str = "") -> Optional[float]:
    """`semantic_quality_check`'s aggregation for SimSyn and SimSem:

        (Σ over the node matching + Σ over the participant matching)
        / (|elements of model 1| + |elements of model 2|)

    The sum runs over the matching's **entries**, and their dict holds one
    entry per node of *each* model — so a matched pair contributes twice, once
    from each end, while the divisor counts each element once. That is why
    their values sit at roughly twice this port's, which sums each pair once:
    the divisor is the same, the numerator is not.
    """
    total = n_elements1 + n_elements2
    if total == 0:
        return None
    summed = sum(value for _partner, value in node_matching.values())
    if participants1 and participants2:
        matrix = _pair_matrix(
            len(participants1), len(participants2),
            lambda i, j: _sim_syn(participants1[i], participants2[j], True))
        # Their pool matching is keyed by the pool's *id*, which two files do
        # not share; this project's DOT view has only the cluster name, which
        # two halves of one dataset do share. `prefix` moves side 1 into its
        # own space so the same collapse cannot happen here either — the
        # participant half of what the node keys already do above.
        keys1 = [f"{prefix}{p}" for p in participants1] if prefix else participants1
        pool_matching = _greedy_pool_matching(matrix, keys1, participants2)
        summed += sum(value for _partner, value in pool_matching.values())
    return summed / total


def _sim_context_bef(pred1: Set[str], succ1: Set[str],
                     pred2: Set[str], succ2: Set[str],
                     mapping: Dict[str, Tuple[str, float]]) -> float:
    """`natural_language_similarity.calculate_context_similarity`, reproduced.

        dividor = max(|pre1|, |pre2|) + max(|succ1|, |succ2|)
        context = (|map1| + |map2|) / dividor        (0 when dividor == 0)

    where `map1` counts the *reference* node's predecessors that some
    predecessor of the candidate node is mapped to under the **semantic** node
    matching, and `map2` does the same for the successors. So it does not ask
    whether two neighbourhoods look alike — it asks how much of the already
    established SimSem pairing survives one step out from the pair, which is
    the mapping-based reading of La Rosa et al. [56] their code implements.

    Three things about it are theirs and are kept:

    * the *maximum* of the two sides is the divisor, not their sum, so a
      candidate with more predecessors than the reference is charged for them
      but can never be credited past 1.0 per direction;
    * an empty neighbourhood on **both** sides makes the term 0/0 and their
      guard returns 0 for the whole pair — two start events that agree on
      having no predecessor score nothing, and a pair with no neighbours at all
      scores 0 rather than 1;
    * the resolution runs through `mapping` only in one direction (candidate →
      reference), and a neighbour their SimSem matching never bound simply
      drops out of the numerator.

    Their lists may hold a neighbour once at most (the shared `visited` list
    sees to that), so the sets used here count what their `len` counts.
    """
    partners_pred = {mapping[node][0] for node in pred1 if node in mapping}
    partners_succ = {mapping[node][0] for node in succ1 if node in mapping}
    divisor = max(len(pred1), len(pred2)) + max(len(succ1), len(succ2))
    if divisor == 0:
        return 0.0
    return (len(pred2 & partners_pred)
            + len(succ2 & partners_succ)) / divisor


def _bef_context_metric(cand: "_Side", ref: "_Side",
                        sem_matching: Dict[str, Tuple[str, float]],
                        prefix: str = "") -> Optional[float]:
    """`natural_language_similarity()`'s third block: the context similarity
    over its own greedy matching, summed and divided.

        con_mapping = create_euquivalence_mapping_for_nodes(g1, g2, context_sim)
        sum(con_mapping[n][1] for n in con_mapping)
        / (|task/event nodes of g1| + |task/event nodes of g2|)

    Two things separate the aggregation from SimSyn's and SimSem's, and both
    are theirs: the divisor counts **flow objects only** — no pools, no named
    lanes, unlike the two label metrics right above it in their own method —
    and there is no participant matching to add to the numerator, context
    being defined on graph neighbourhoods a pool does not have.

    The numerator is again a sum over the matching's *entries*, one per node of
    each model, so a pair contributes from both ends and a perfect match reads
    1.0 — provided the two id spaces are disjoint. Where they are not, their
    single dict collapses the two entries into one and the value halves, which
    is exactly what `prefix` separates.
    """
    key = (lambda node: f"{prefix}{node}") if prefix else (lambda node: node)
    pred_c = [{key(n) for n in nodes} for nodes in cand.pred]
    succ_c = [{key(n) for n in nodes} for nodes in cand.succ]
    matrix = _pair_matrix(
        len(cand.nodes), len(ref.nodes),
        lambda i, j: _sim_context_bef(pred_c[i], succ_c[i],
                                      ref.pred[j], ref.succ[j], sem_matching))
    total = len(cand.nodes) + len(ref.nodes)
    if total == 0:
        return None
    matching = _greedy_matching(matrix, [key(n) for n in cand.nodes], ref.nodes)
    return sum(value for _partner, value in matching.values()) / total


def _bef_columns(cand: "_Side", ref: "_Side",
                 generated: ProcessGraph, ground_truth: ProcessGraph,
                 syn_matrix: np.ndarray) -> Dict[str, Any]:
    """The five `sem_*_bef` columns for one comparison.

    `syn_matrix` is metric 1's pairwise matrix, which `_sim_syn` already
    computes under their reading (ε ↔ ε scores 0), so it is reused rather than
    built twice; the semantic matrix is theirs and is built here.
    """
    tokens_c = [_bef_tokens(generated, n) for n in cand.nodes]
    tokens_r = [_bef_tokens(ground_truth, n) for n in ref.nodes]

    sem_matrix = _pair_matrix(
        len(cand.nodes), len(ref.nodes),
        lambda i, j: _sim_sem_bef(tokens_c[i][0], tokens_c[i][1], cand.syn_text[i],
                                  tokens_r[j][0], tokens_r[j][1], ref.syn_text[j]))
    parts_c = _bef_participants(generated)
    parts_r = _bef_participants(ground_truth)
    n_c = len(cand.nodes) + len(parts_c)
    n_r = len(ref.nodes) + len(parts_r)

    def values(prefix: str) -> Tuple[Optional[float], Optional[float],
                                     Optional[float]]:
        """All three natural-language metrics over their matchings, with the
        candidate's node keys optionally moved into an id space of their own.

        The semantic matching is built once and used twice, as in their
        `natural_language_similarity`: it scores SimSem and it is the mapping
        the context similarity resolves neighbourhoods through."""
        keys = [f"{prefix}{n}" for n in cand.nodes] if prefix else cand.nodes
        sem_matching = _greedy_matching(sem_matrix, keys, ref.nodes)
        return (_bef_label_metric(_greedy_matching(syn_matrix, keys, ref.nodes),
                                  parts_c, parts_r, n_c, n_r, prefix),
                _bef_label_metric(sem_matching,
                                  parts_c, parts_r, n_c, n_r, prefix),
                _bef_context_metric(cand, ref, sem_matching, prefix))

    syn_bef, sem_bef, con_bef = values("")
    # …and again with the two id spaces separated. Their matching is one dict
    # keyed by node id holding entries for both models, so an id present in
    # both collapses two entries into one and halves that node's contribution.
    # Which ids collide is a property of the *files*, not of the models: PMo's
    # DOT names a task by its label, so a renamed task changes identity, while
    # the BPMN keeps `Task_6` and edits `name`. Measured on the manipulated
    # dataset (2026-08-29): where the two DOT id spaces coincided completely
    # the reproduction hit their column exactly on 34 of 34 pairs, and the
    # deviation on the other 21 tracked the id overlap at r = -0.993. The
    # prefixed variant takes that dependency out, and is what to compare
    # against a BEF4LLM run over `prefix_bpmn_ids.py` output.
    syn_disjoint, sem_disjoint, con_disjoint = values("__GEN__")
    return {
        # The first three are what `evaluate` scores; the `_disjoint` trio
        # rides along outside `sem_score` as the collision-free reading of the
        # same algorithm — the one that matches a BEF4LLM run over prefixed
        # ids.
        "sem_label_sim_syntactic": syn_bef,
        "sem_label_sim_semantic": sem_bef,
        "sem_label_sim_context": con_bef,
        "sem_label_sim_syntactic_disjoint": syn_disjoint,
        "sem_label_sim_semantic_disjoint": sem_disjoint,
        "sem_label_sim_context_disjoint": con_disjoint,
        # How many of the generated model's compared nodes their matching bound
        # — with threshold 0.0 that is normally all of them, and a number below
        # `sem_n_label_nodes_generated` means the other side was empty.
        "sem_n_nodes_matched_greedy": sum(
            1 for n in cand.nodes if n in _greedy_matching(sem_matrix, cand.nodes,
                                                           ref.nodes)),
        # How many node ids the two models share — the collision count their
        # matching is exposed to, and 0 in the prefixed variant by definition.
        "sem_n_ids_shared": len(set(cand.nodes) & set(ref.nodes)),
    }


def evaluate(generated: ProcessGraph,
             ground_truth: Optional[ProcessGraph] = None) -> Dict[str, Any]:
    """Return the `sem_`-prefixed semantic-quality columns for one model.

    Without a ground truth there is nothing to compare against, so the
    full column set comes back None — not 0, which would read as "measured
    and maximally dissimilar" instead of "not comparable".
    """
    if ground_truth is None:
        return empty()

    cand, ref = _Side(generated), _Side(ground_truth)
    n_c, n_r = len(cand.nodes), len(ref.nodes)

    # Metric 1 — syntactic label similarity.
    syn_matrix = _pair_matrix(
        n_c, n_r,
        lambda i, j: _sim_syn(cand.syn_text[i], ref.syn_text[j],
                              cand.kind[i] == ref.kind[j]))
    m_syn = _optimal_matching(syn_matrix)

    # Metric 2 — semantic label similarity. Its matching doubles as the node
    # equivalence for metrics 3–7.
    sem_matrix = _pair_matrix(
        n_c, n_r,
        lambda i, j: _sim_sem(cand.rep[i], cand.stems[i], cand.syn_text[i],
                              ref.rep[j], ref.stems[j], ref.syn_text[j],
                              cand.kind[i] == ref.kind[j]))
    m_sem = _optimal_matching(sem_matrix)
    mapped = {cand.nodes[i]: ref.nodes[j] for (i, j) in m_sem}

    # ── structural bookkeeping for metric 4 and the diagnostics ──
    n_matched = len(m_sem)
    f_c, f_r = len(cand.skel), len(ref.skel)
    # A candidate edge is "in" the reference iff both endpoints are matched
    # and the mapped edge exists there with the same flow type. The node
    # mapping is injective, so this count is symmetric between the sides.
    edges_matched = sum(
        1 for (src, dst, kind) in cand.skel
        if (mapped.get(src), mapped.get(dst), kind) in ref.skel)

    # **All three natural-language metrics are BEF4LLM's own implementation**
    # (author's instruction, the two label ones since 2026-08-29 and the
    # context similarity since 2026-08-30): their greedy id-keyed matching,
    # their SimSem formula, their divisor over non-gateway flow objects plus
    # pools and named lanes — flow objects alone for the context similarity —
    # their ε ↔ ε reading, and their neighbourhood recursion through the
    # gateways. Validated against their code on the manipulated dataset once
    # the id spaces are separated on both sides (`prefix_bpmn_ids.py` there,
    # the `_disjoint` variant here): 55 of 55 pairs exact in both label
    # metrics and 44 of 55 in the context similarity, whose residue is their
    # order-dependent tie-breaking — see `METRICS`.
    #
    # Before that the port computed Table A.17's definitions, which had already
    # been pulled towards this code on three points (2026-08-24: the divisor
    # without the factor 2, and ε ↔ ε scoring 0 in both metrics). The remaining
    # differences were the optimal matching against their greedy one, their
    # `max(|w1|, |w2|)` divisor, and the pools and lanes in their denominators.
    # **No semantic figure is comparable with one produced before this date** —
    # but the published Qsem numbers now are, which is what the change is for.
    bef = _bef_columns(cand, ref, generated, ground_truth, syn_matrix)

    # Metric 4 — the graph edit distance, on the contracted skeletons alone.
    # It reads neither matching above: an edit distance derives its own node
    # correspondence, which is the point of it.
    ged_similarity, ged_operations, ged_truncated = _graph_edit_distance(cand, ref)

    # Metric 5 — common nodes and edges, pure arithmetic over the bookkeeping
    # above. It reads M^opt_Sem, so it costs nothing beyond what metric 2
    # already computed.
    common, common_nodes, common_edges = _common_nodes_edges(
        n_c, n_r, n_matched, f_c, f_r, edges_matched)

    values: Dict[str, Optional[float]] = {
        "label_sim_syntactic": bef["sem_label_sim_syntactic"],
        "label_sim_semantic": bef["sem_label_sim_semantic"],
        "label_sim_context": bef["sem_label_sim_context"],
        "graph_edit_distance": ged_similarity,
        "common_nodes_edges": common,
    }

    out: Dict[str, Any] = {}
    scores: Dict[str, float] = {}
    n_measured = 0
    for key, _group in METRICS:
        value = values[key]
        # Full precision — see the precision policy in quality/__init__.
        out[f"sem_{key}"] = value
        if value is None:
            scores[key] = _UNMEASURABLE_SCORE
        else:
            scores[key] = value
            n_measured += 1

    # §4.5: arithmetic mean over the full metric set, equally weighted — the
    # fixed-divisor convention shared with prag_score. Group means are this
    # project's addition, as in the pragmatic dimension.
    for group in GROUPS:
        in_group = [scores[key] for key, g in METRICS if g == group]
        out[f"sem_group_{group}_score"] = sum(in_group) / len(in_group)
    out["sem_score"] = sum(scores.values()) / len(METRICS)
    out["sem_n_metrics_measured"] = n_measured
    out["sem_n_metrics_total"] = len(METRICS)

    # ── diagnostics ──
    out.update({
        "sem_n_label_nodes_generated": n_c,
        "sem_n_label_nodes_reference": n_r,
        "sem_n_nodes_matched": n_matched,
        "sem_n_skeleton_edges_generated": f_c,
        "sem_n_skeleton_edges_reference": f_r,
        "sem_n_edges_matched": edges_matched,
        "sem_wordnet": 1 if textsim.wordnet_available() else 0,
        # Metric 4's own two. `sem_ged_operations` is the raw edit distance the
        # similarity normalises; `sem_ged_truncated` says the search ran into
        # its budget, so the value is a bound and is not reproducible on
        # another machine. A row with it True is not comparable with one
        # without — see the note in `METRICS`.
        "sem_ged_operations": ged_operations,
        "sem_ged_truncated": ged_truncated,
        # Metric 5's two halves. Not metrics of their own — they are the same
        # quotient over the node and the edge terms separately, and they are
        # what says *which* of the two a disagreement sits in. On the
        # manipulated corpus `sem_common_nodes` is 1.0 on every model, which is
        # exactly the kind of thing these columns exist to make visible.
        "sem_common_nodes": common_nodes,
        "sem_common_edges": common_edges,
    })
    # The collision-free reading and the two matching diagnostics; the metric
    # columns themselves are already in `out`, written from the same dict.
    out.update({k: v for k, v in bef.items() if k not in values
                and f"sem_{k}" not in values})
    return out


def empty() -> Dict[str, Any]:
    """Column set for a generation without a comparable pair — no ground
    truth, or one of the two sides did not parse. Every value is `None`,
    never 0. Derived from `evaluate()` on two empty graphs rather than
    listed again, so a newly added column cannot go missing here."""
    out: Dict[str, Any] = {
        k: None for k in evaluate(ProcessGraph(), ProcessGraph())}
    out["sem_n_metrics_total"] = len(METRICS)
    return out
