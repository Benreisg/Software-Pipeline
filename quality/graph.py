"""
quality/graph.py — DOT loading + BPMN element classification
================================================================
Shared foundation for every quality dimension: turn a `.dot`/`.gv` file into a
`ProcessGraph` whose nodes are classified as tasks, gateways, start events and
end events. Every dimension reads this; none of them parses DOT itself.

Three properties of these files make a naive parser silently wrong — all are
handled here (see README, *DOT quirks any scorer must handle*):

1. **Implicit nodes + default attributes.** PMo declares tasks only through the
   graph-level default `node [shape=box];` and creates them inside edge
   statements. A parser reading explicit `shape=` attributes only sees zero
   tasks and drops every edge touching one. So: defaults are applied per scope,
   and nodes appearing only as edge endpoints are materialised.
2. **The end event is not a `doublecircle`.** PMo draws it as
   `shape=circle, penwidth=4`; the prompt templates variously ask for
   `doublecircle`, or for `circle` with the label "end". Shape alone therefore
   cannot separate start from end — see `_classify_event` for the layered rule.
3. **A node can be declared twice.** Generated models like to declare the shape
   first and colour the node in a styling block at the foot of the file. In DOT
   the second statement *merges* into the node; a parser that lets it replace
   the node loses the shape and reads every start event, end event and gateway
   as a task — see `_collect_declared`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import pydot

# The generation loop scores each reply on the worker that fetched it, so
# this parser now runs on several threads at once. pydot cannot: it binds
# parse actions to module-level pyparsing objects. The lock lives in
# postprocess because that module parses the same library, and the state
# being protected belongs to pydot rather than to either caller - two
# separate locks would protect nothing.
from postprocess import DOT_PARSE_LOCK

# Shape vocabularies. Generous on purpose: the prompt templates in prompts.py
# ask for "box" (Li et al., role_prompt_2) and "rectangle"
# (zero_shot_graph_type_tn_rules) for the same concept, and a model may answer
# with either. Classifying by the modeller's intent beats punishing a synonym.
TASK_SHAPES = {"box", "rect", "rectangle", "square"}
GATEWAY_SHAPES = {"diamond", "mdiamond", "msquare"}
EVENT_SHAPES = {"circle", "doublecircle", "ellipse", "oval"}

# Not flow objects: layout anchors for a pool or lane. PMo writes them as
# `"Pool_1" [shape=point, style=invis];` inside a `cluster_Pool_1` subgraph —
# a positioning device for the message-flow arrows, with no process semantics.
# Kept in the graph (message flows attach to them) but excluded from |FO|.
MARKER_SHAPES = {"point", "none", "plaintext", "plain"}

# BPMN draws a message flow as a dashed line with an open arrowhead; PMo writes
# `[style=dashed, arrowhead=open]`. Everything else is a sequence flow.
_MESSAGE_FLOW_STYLES = {"dashed", "dotted"}

# Node ids that are placeholders rather than activity names ("Task_1", "t3").
_PLACEHOLDER_LABEL = re.compile(r"^(task|activity|act|node|step|t|a|n)[_\-]?\d+$", re.I)

# Gateway naming conventions this corpus actually produces. The
# `zero_shot_graph_type_tn_rules` template prescribes seg_/meg_ (splitting /
# merging exclusive) and spg_/mpg_ (splitting / merging parallel); MaD-style
# models write AND_SPLIT / OR_JOIN / XOR_SPLIT into the node id. Anchored on the
# whole normalised id so an activity called "segment orders" is not caught.
_SEG_SPLIT = re.compile(r"^(seg|spg)\d*$")     # splitting exclusive / parallel
_SEG_JOIN = re.compile(r"^(meg|mpg)\d*$")      # merging  exclusive / parallel


def _clean(s: Optional[str]) -> str:
    """Strip DOT quoting/whitespace from a name or attribute value."""
    if s is None:
        return ""
    s = str(s).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.strip()


# Graphviz wraps a label with \n (centred), \l (left) or \r (right). They are
# *layout*, not text: "Send order\nconfirmation" is one label reading "Send
# order confirmation". Left in place they survive into the token stream — the
# tokeniser drops the backslash and glues the n onto the next word, so
# "\nconfirmation" becomes the word "nconfirmation" and every label similarity
# compares text no reader ever sees. PMo wraps most of its longer labels.
_LINE_BREAK = re.compile(r"\\[nlr]")


def _unwrap(s: str) -> str:
    """A label as it reads, with the line-break escapes resolved to spaces."""
    return " ".join(_LINE_BREAK.sub(" ", s).split())


def _norm_word(s: str) -> str:
    """Lowercase, letters+digits only — so 'StartEvent_1' → 'startevent1'."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _node_defaults(graph: pydot.Graph) -> Dict[str, str]:
    """The scope's `node [...]` defaults. In DOT these apply to every node
    created afterwards, including ones created by an edge statement."""
    defaults: Dict[str, str] = {}
    for d in graph.get_node_defaults() or []:
        defaults.update({k: _clean(v) for k, v in d.items()})
    return defaults


def _collect_declared(graph: pydot.Graph,
                      inherited: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, str]]:
    """Explicitly declared nodes, their own attributes layered over the
    defaults of their scope."""
    defaults = {**(inherited or {}), **_node_defaults(graph)}
    out: Dict[str, Dict[str, str]] = {}
    for node in graph.get_nodes():
        name = _clean(node.get_name())
        if name.upper() in ("GRAPH", "NODE", "EDGE", ""):
            continue
        own = {k: _clean(v) for k, v in node.get_attributes().items()}
        # A node declared twice **merges**, attribute by attribute — what
        # Graphviz does: a second statement adds to the node and overrides only
        # the attributes it names, it does not replace it. Models like to
        # declare the shape first and colour the node in a styling block at the
        # foot of the file (`Start [shape=circle]` … `Start [fillcolor=green]`),
        # and replacing would drop the shape: the graph default
        # `node [shape=box]` then turns every start event, end event and gateway
        # into a task. Checked against `nop`, which prints the merged node.
        out.setdefault(name, dict(defaults)).update(own)
    for sg in graph.get_subgraphs():
        for n, a in _collect_declared(sg, defaults).items():
            out.setdefault(n, a)
    return out


def _collect_implicit(graph: pydot.Graph,
                      inherited: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, str]]:
    """Nodes that only ever appear as an edge endpoint. They inherit the
    defaults of the scope the edge sits in — in PMo, that is every task."""
    defaults = {**(inherited or {}), **_node_defaults(graph)}
    out: Dict[str, Dict[str, str]] = {}
    for e in graph.get_edges():
        for endpoint in (_clean(e.get_source()), _clean(e.get_destination())):
            if endpoint and endpoint.upper() not in ("GRAPH", "NODE", "EDGE"):
                out.setdefault(endpoint, dict(defaults))
    for sg in graph.get_subgraphs():
        for n, a in _collect_implicit(sg, defaults).items():
            out.setdefault(n, a)
    return out


def _collect_clusters(graph: pydot.Graph,
                      stack: Tuple[str, ...] = ()) -> Dict[str, Tuple[str, str]]:
    """Cluster membership per node: `{node: (outermost, innermost)}`.

    DOT's way of drawing a BPMN **pool** is a `subgraph cluster_*`, and a
    **lane** is a cluster nested inside it — PMo writes exactly that
    (`cluster_Pool_1` containing `cluster_Lane_1..3` in item 21). So the
    outermost enclosing cluster is the participant/pool and the innermost is
    the lane; a model with a flat list of `cluster_Pool_*` has pool == lane.

    Nodes outside every cluster are simply absent from the mapping — they
    belong to the model's implicit single pool, which is what a one-participant
    DOT model is. Only `cluster*`-named subgraphs count: a plain `subgraph`
    without that prefix is a layout grouping in DOT, not a drawn box.

    The innermost declaration wins, so a node listed in both the pool and its
    lane gets the lane as its lane.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for sg in graph.get_subgraphs():
        name = _clean(sg.get_name())
        inner = stack + (name,) if name.lower().startswith("cluster") else stack
        # Recurse first: a nested lane must overwrite its enclosing pool.
        out.update(_collect_clusters(sg, inner))
        if not inner:
            continue
        members = [_clean(n.get_name()) for n in sg.get_nodes()]
        for e in sg.get_edges():
            members.extend((_clean(e.get_source()), _clean(e.get_destination())))
        for member in members:
            if member and member.upper() not in ("GRAPH", "NODE", "EDGE"):
                out.setdefault(member, (inner[0], inner[-1]))
    return out


def _collect_edges(graph: pydot.Graph) -> List[Tuple[str, str, Dict[str, str]]]:
    """Edges as (source, target, attributes). Attributes are needed to tell a
    sequence flow from a message flow; no layout data is read beyond that."""
    out: List[Tuple[str, str, Dict[str, str]]] = []
    for e in graph.get_edges():
        src, dst = _clean(e.get_source()), _clean(e.get_destination())
        if src and dst:
            attrs = {k: _clean(v) for k, v in e.get_attributes().items()}
            out.append((src, dst, attrs))
    for sg in graph.get_subgraphs():
        out.extend(_collect_edges(sg))
    return out


@dataclass
class ProcessGraph:
    """A parsed process model, with its elements already classified."""
    nodes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)          # as written, duplicates kept
    unique_edges: Set[Tuple[str, str]] = field(default_factory=set)     # deduplicated
    edge_attrs: Dict[Tuple[str, str], Dict[str, str]] = field(default_factory=dict)

    # `digraph` (True) or `graph` (False) — the keyword the file opens with, and
    # what Graphviz's `agisdirected()` reports for the same file. Defaults to
    # True so a hand-built ProcessGraph in a test is a process model, not an
    # undirected sketch; `load()` sets it from the source.
    directed: bool = True

    tasks: Set[str] = field(default_factory=set)
    gateways: Set[str] = field(default_factory=set)
    starts: Set[str] = field(default_factory=set)
    ends: Set[str] = field(default_factory=set)
    # Intermediate events **as the notation declares them**: a `doublecircle`
    # that is not a sink. PMo's README fixes that notation — "End events are
    # depicted with a bolded outer circle, while intermediate events use a
    # double circle, consistent with BPMN" — and `_classify_event` reads it.
    # An event that merely sits in the middle without saying so stays in
    # `events_unclassified`; both answer "event" to `kind_of`, so nothing that
    # scores on the kind can tell them apart.
    intermediates: Set[str] = field(default_factory=set)
    events_unclassified: Set[str] = field(default_factory=set)
    # Pool/lane anchors and other non-semantic nodes — never flow objects.
    markers: Set[str] = field(default_factory=set)
    # Nodes whose shape matches nothing known (or that carry no shape at all
    # and no default). Reported rather than silently forced into a bucket.
    unclassified: Set[str] = field(default_factory=set)

    # Per gateway: "split" / "join" / "" (neither — degenerate or mixed), and
    # "exclusive" / "parallel" / "inclusive" / "" (type could not be read).
    gateway_roles: Dict[str, str] = field(default_factory=dict)
    # Gateways that fan in *and* out. They keep a split/join role like every
    # other gateway (see `_resolve_mixed_roles`); this records which ones were
    # ambiguous. See `mixed_gateways`.
    mixed: Set[str] = field(default_factory=set)
    gateway_types: Dict[str, str] = field(default_factory=dict)

    # Participant structure: node → enclosing `cluster_*` subgraph. `pools`
    # holds the outermost one (the BPMN pool), `lanes` the innermost. A node
    # outside every cluster is absent from both — it sits in the model's
    # implicit single pool.
    pools: Dict[str, str] = field(default_factory=dict)
    lanes: Dict[str, str] = field(default_factory=dict)

    def in_degree(self, node: str) -> int:
        """Counted on unique edges: a duplicated identical edge is one flow."""
        return sum(1 for s, d in self.unique_edges if d == node)

    def out_degree(self, node: str) -> int:
        return sum(1 for s, d in self.unique_edges if s == node)

    def seq_in_degree(self, node: str) -> int:
        """In-degree counting **sequence flows only** — the control-flow degree.

        This is what BPMN's `bpmn:incoming` means and what BEF4LLM's metrics
        read; `in_degree` above counts every edge, message flows included, which
        is what the "is this node wired at all" checks want. The two differ only
        for a node a message flow attaches to.
        """
        return sum(1 for s, d in self.sequence_flows() if d == node)

    def seq_out_degree(self, node: str) -> int:
        return sum(1 for s, d in self.sequence_flows() if s == node)

    def pool_of(self, node: str) -> str:
        """The node's pool (outermost `cluster_*`), or "" when it is in none."""
        return self.pools.get(node, "")

    def lane_of(self, node: str) -> str:
        return self.lanes.get(node, "")

    def pool_ids(self) -> Set[str]:
        """Every pool the model draws. Empty for a single-participant model."""
        return set(self.pools.values())

    def label_of(self, node: str) -> str:
        """The text a reader sees: the `label` attribute if present, else the
        node id itself (which is how DOT renders an unlabelled node).

        Line-break escapes are resolved either way — see `_unwrap`. The node
        **id** keeps them (it identifies the node and must stay verbatim); only
        the text this returns is unwrapped.
        """
        attrs = self.nodes.get(node, {})
        if "label" in attrs:
            return _unwrap(_clean(attrs["label"]))
        return _unwrap(node)

    def has_placeholder_label(self, node: str) -> bool:
        """True for a label that exists but names nothing — 'Task_1', 't3'."""
        return bool(_PLACEHOLDER_LABEL.match(self.label_of(node).strip()))

    def kind_of(self, node: str) -> str:
        """The SF_BPMN element type of a node:
        "start" | "end" | "task" | "gateway" | "event" | "marker" | "unknown".

        "event" is a circle-shaped node that is neither a start nor an end — a
        declared intermediate event (`intermediates`) or one whose role could
        not be decided (`events_unclassified`). The two are deliberately one
        kind here: no metric distinguishes them, and merging them keeps every
        rule that already reads "event" unchanged. "marker" is a pool/lane
        anchor; "unknown" is a node whose shape matches no vocabulary at all.
        None is folded into a neighbouring category.
        """
        if node in self.starts:
            return "start"
        if node in self.ends:
            return "end"
        if node in self.tasks:
            return "task"
        if node in self.gateways:
            return "gateway"
        if node in self.intermediates or node in self.events_unclassified:
            return "event"
        if node in self.markers:
            return "marker"
        return "unknown"

    def flow_objects(self) -> Set[str]:
        """|FO| — the nodes of the process flow itself: events, tasks and
        gateways. Excludes pool/lane anchors, artefacts, and nodes of unknown
        type, and never counts flows of either kind."""
        return (self.tasks | self.gateways | self.starts | self.ends
                | self.intermediates | self.events_unclassified)

    def is_message_flow(self, edge: Tuple[str, str]) -> bool:
        """True for a dashed/dotted edge — the BPMN message-flow notation."""
        style = self.edge_attrs.get(edge, {}).get("style", "").lower()
        return any(s in style for s in _MESSAGE_FLOW_STYLES)

    def sequence_flows(self) -> Set[Tuple[str, str]]:
        """|F^S| — deduplicated edges that are sequence flows, i.e. every edge
        except the message flows."""
        return {e for e in self.unique_edges if not self.is_message_flow(e)}

    def message_flows(self) -> Set[Tuple[str, str]]:
        return {e for e in self.unique_edges if self.is_message_flow(e)}

    def splits(self) -> Set[str]:
        """GS — every gateway that divides the flow, mixed gateways included.
        See `_gateway_role`; `joins` is built the same way, so a gateway that
        fans in *and* out is in both and both degree rules measure it."""
        return {n for n, role in self.gateway_roles.items()
                if role in ("split", "mixed")}

    def joins(self) -> Set[str]:
        """GJ — every gateway that brings flows together, mixed gateways
        included. See `splits`."""
        return {n for n, role in self.gateway_roles.items()
                if role in ("join", "mixed")}

    def mixed_gateways(self) -> Set[str]:
        """Gateways that fan in **and** out (in > 1 ∧ out > 1) — BPMN's mixed
        gateway.

        Reported as `syn_n_gateways_mixed`, and used for one thing: such a
        gateway can be the **partner of a split** in check 10 / metric #9,
        because it does merge, even though the degree rules see it as a
        gateway with in == out (`degenerate_gateways`) or as whichever of
        split/join its degrees suggest."""
        return set(self.mixed)

    def degenerate_gateways(self) -> Set[str]:
        """Gateways that neither split nor merge the flow: the 1→1
        pass-through, which routes nothing. A gateway that does both at once is
        in GS and GJ both — see `_gateway_role`."""
        return {n for n, role in self.gateway_roles.items() if not role}

    # ── the two graph-level predicates Graphviz answers in C ─────────────
    def is_directed(self) -> bool:
        """What `agisdirected(Agraph_t *g)` reports: was this written as a
        `digraph`?

        A `graph` has no arrow direction at all, so nothing downstream — a
        gateway's split/join role, a start event's in-degree of zero, a path
        from start to end — means what it says. It is read off the source
        keyword, which is precisely what the C function returns: Graphviz sets
        that flag from the keyword when it opens the file, and nothing later
        changes it.
        """
        return bool(self.directed)

    def components(self) -> List[Set[str]]:
        """The connected components, edges followed **in both directions** —
        the search `isConnected(Agraph_t *g)` performs.

        Every node of the file takes part, cluster anchors and unclassified
        shapes included: those are nodes of the root graph to Graphviz
        (`agnnodes`), and a metric that quietly dropped them would not be
        answering the question the C function answers.
        """
        neighbours: Dict[str, Set[str]] = {node: set() for node in self.nodes}
        for src, dst in self.unique_edges:
            neighbours.setdefault(src, set()).add(dst)
            neighbours.setdefault(dst, set()).add(src)

        seen: Set[str] = set()
        found: List[Set[str]] = []
        for node in neighbours:
            if node in seen:
                continue
            stack, group = [node], set()
            while stack:
                current = stack.pop()
                if current in group:
                    continue
                group.add(current)
                stack.extend(neighbours[current] - group)
            seen |= group
            found.append(group)
        return found

    def is_connected(self) -> bool:
        """What `isConnected(Agraph_t *g)` reports: one component covering every
        node. An empty graph is connected there (nothing is unreachable) and is
        connected here, for the same reason."""
        return len(self.components()) <= 1

    def diameter(self, budget: int = 200_000,
                 count_gateways: bool = True) -> Tuple[int, bool]:
        """Longest start-to-end path, counted in flow objects: max{|p|}.

        Returns (length, truncated).

        A path is a non-empty sequence of flow objects connected by **sequence
        flows** (message flows are not process flow) leading from a start event
        to an end event; |p| is the number of flow objects on it.

        **`count_gateways` selects the counting convention**, because the two
        BEF4LLM sources disagree on it and the difference is large:

        * `True` (default) — every flow object on the path counts. This is
          Table A.16's `max{|p| : p ∈ Paths}` read against the paper's own
          notation, where *Paths* are "start-to-end **node** sequences".
        * `False` — gateways do not count, only activities and events. This is
          what BEF4LLM's `pragmatic_quality_metrics.diameter()` computes
          (`if n in conlist: counter = 0`), and therefore the convention its
          published figures were produced under.

        On the 55 PMo reference models the two differ by **7.87 on average**
        (17.89 against 10.93) — the reference models carry 8.4 gateways each —
        which is nearly two threshold bands. Both are reported: `syn_diameter`
        uses the default, `prag_diameter_nogw` / `syn_diameter_nogw` the other.
        **`prag_diameter_nogw` is the one that feeds `prag_score`** since
        2026-08-29 (author's instruction); `prag_diameter` is a third reading
        again — BEF4LLM's own traversal, reproduced in
        `quality/pragmatic.py::_bef4llm_diameter` and no longer scored. See the
        note on the diameter `MetricSpec` there.

        **Cycle handling.** The project had no path or cycle semantics to reuse
        — `reachable_from` is plain reachability, not a path definition — so
        this counts *simple* paths only: no node is visited twice within one
        path. That is the decision the underlying formula leaves open, and it
        is what makes the search terminate on a model with a loop; without it a
        cycle would make the longest path unbounded.

        Longest-simple-path is NP-hard, so the search is additionally capped by
        `budget` expansions. If the cap is hit, the best length found so far is
        returned with truncated=True rather than the run hanging on a
        pathological generated model.
        """
        flows = self.sequence_flows()
        adjacency: Dict[str, List[str]] = {}
        for src, dst in flows:
            adjacency.setdefault(src, []).append(dst)

        fo = self.flow_objects()
        if not count_gateways:
            fo = fo - self.gateways
        best, steps, truncated = 0, 0, False

        for start in sorted(self.starts):
            # (node, nodes already on this path, flow objects counted so far)
            stack: List[Tuple[str, frozenset, int]] = [
                (start, frozenset({start}), 1 if start in fo else 0)
            ]
            while stack:
                node, on_path, length = stack.pop()
                steps += 1
                if steps > budget:
                    truncated = True
                    break
                if node in self.ends:
                    best = max(best, length)
                    # An end event has no outgoing sequence flow in a well
                    # formed model; if it does, the path is not extended past
                    # it — a path ends at an end event by definition.
                    continue
                for nxt in adjacency.get(node, ()):
                    if nxt not in on_path:
                        stack.append((nxt, on_path | {nxt},
                                      length + (1 if nxt in fo else 0)))
            if truncated:
                break

        return best, truncated

    def reachable_from(self, node: str) -> Set[str]:
        """Everything downstream of `node`, following flow direction. Cycle
        safe; the node itself is not included unless a cycle returns to it."""
        seen: Set[str] = set()
        stack = [d for s, d in self.unique_edges if s == node]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(d for s, d in self.unique_edges if s == cur)
        return seen


# ── Match : GS → GJ ─────────────────────────────────────────────────────────
def fan_maps(edges: Iterable[Tuple[str, str]]
             ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Successor and predecessor lists for a set of edges."""
    succ: Dict[str, List[str]] = {}
    pred: Dict[str, List[str]] = {}
    for src, dst in edges:
        succ.setdefault(src, []).append(dst)
        pred.setdefault(dst, []).append(src)
    return succ, pred


def _reconvergence_points(split: str, succ: Mapping[str, Iterable[str]]) -> Set[str]:
    """Where the split's branches meet **first**.

    A level-synchronous walk down every branch at once, labelling each node
    with the branches that reached it. A node reached by two or more branches
    is a reconvergence point and is **not** expanded further: what lies behind
    it is downstream of a merge that already happened, not a merge of this
    split. Without that stop, any join further along the process counts —
    PMo's item 46 has both branches of `"Is in stock?"` running into the task
    `"Check if the whole order is ready for shipment"` and only then, through
    the outer loop, into `ExclusiveGateway_2`, which would otherwise be read as
    the split's partner.

    The walk never passes through the split itself, so a loop returning to it
    does not leak into a sibling branch.
    """
    points: Set[str] = set()
    labels: Dict[str, Set[int]] = {}
    frontier = [(first, i) for i, first in enumerate(succ.get(split, ()))]
    while frontier:
        following = []
        for node, branch in frontier:
            if node == split:
                continue
            seen_by = labels.setdefault(node, set())
            if branch in seen_by:
                continue
            seen_by.add(branch)
            if len(seen_by) >= 2:
                points.add(node)
                continue
            following.extend((nxt, branch) for nxt in succ.get(node, ()))
        frontier = following
    return points


def _loop_merges(split: str, succ: Mapping[str, Iterable[str]],
                 pred: Mapping[str, Iterable[str]], splits: Set[str]) -> Set[str]:
    """The gateways a *loop* merges this split's back edge into.

    A loop's merge sits in front of its split, not behind it, so the
    reconvergence walk above can never find it. It is reached from one of the
    split's branches, has two or more inflows (the back edge and the way in),
    and leads back to the split.

    The walk stops at any **other split gateway**: a merge behind a second
    decision closes *that* decision's branches, not this one's. Item 46 again —
    both ways from `"Is in stock?"` to `ExclusiveGateway_1` and
    `ExclusiveGateway_2` run through another decision first, which is why
    neither is its merge even though both sit on a cycle with it.
    """
    reached: Set[str] = set()
    stack = [first for first in succ.get(split, ())]
    while stack:
        node = stack.pop()
        if node in reached or node == split:
            continue
        reached.add(node)
        if node in splits:          # a further decision — its merges are its own
            continue
        stack.extend(succ.get(node, ()))
    out: Set[str] = set()
    for node in reached:
        if len(list(pred.get(node, ()))) < 2:
            continue
        seen: Set[str] = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(succ.get(cur, ()))
        if split in seen:
            out.add(node)
    return out


def matching_joins(split: str, succ: Mapping[str, Iterable[str]],
                   pred: Mapping[str, Iterable[str]],
                   splits: Set[str]) -> Set[str]:
    """Every gateway that merges what `split` forked — its `Match` candidates.

    The paper declares `Match : GS → GJ` and leaves the partner undefined. Two
    shapes count here, and only these two:

    * **Reconvergence** — the branches meet again, and they meet *there first*
      (`_reconvergence_points`).
    * **Loop merge** — the back edge of a loop runs into a gateway in front of
      the split (`_loop_merges`).

    The caller filters the result by gateway type and by what it considers a
    join; a node in here that is a task is exactly the case this is meant to
    expose — an implicit merge with no gateway at all.

    Reachability alone is **not** enough, which is what this replaces (author's
    decision, 2026-08-24). Two PMo models showed why, and both are named in the
    helpers above: item 30's `"How many people can play on that date?"` reached
    an unrelated join six steps away, item 46's `"Is in stock?"` reached one
    through the outer loop after its branches had already merged at a task.

    `succ`/`pred` decide which edges count; each caller passes the maps for the
    convention it scores by (all edges, or sequence flows only).
    """
    return _reconvergence_points(split, succ) | _loop_merges(split, succ, pred, splits)


def _gateway_role(pg: ProcessGraph, node: str) -> str:
    """Split or join? Declared intent first, topology only as a fallback.

    Deriving the role purely from the degree would make the "split: in=1,
    out>1" and "join: in>1, out=1" checks tautological — every gateway would
    satisfy the rule it was classified by. So a name or label that states the
    role is used first, and topology decides only when nothing is declared.

    The fallback compares the two degrees rather than testing for the exact
    target values: a gateway with in=1, out=3 is *intended* as a split (it fans
    out) and should fail the check, not escape classification.

    The fallback follows the paper's own reading of GS and GJ: **a gateway that
    fans out is a split, one that fans in is a join** (author's instruction,
    2026-08-24). A gateway that does both is `"mixed"`, and `splits()`/`joins()`
    put it in **both** sets — it splits and it merges, so metric #15 asks
    whether exactly one flow enters it and #16 whether exactly one leaves it,
    and it fails both. That is what the two published metrics are for; reading
    it as "neither" let it escape them entirely, which is how PMo's items 02 and
    24 came to score 1.000 on both while every other reading fails them.

    Returns "" only for a gateway that neither fans in nor out — the 1→1
    pass-through, which routes nothing. It is in neither set and is reported as
    `degenerate_gateways`.
    """
    norm = _norm_word(node)
    text = f"{node} {pg.label_of(node)}".upper()
    if _SEG_SPLIT.match(norm) or "SPLIT" in text or "FORK" in text:
        return "split"
    if _SEG_JOIN.match(norm) or "JOIN" in text or "MERGE" in text or "SYNC" in text:
        return "join"

    ind, outd = pg.in_degree(node), pg.out_degree(node)
    if ind > 1 and outd > 1:
        return "mixed"
    if outd > 1:
        return "split"
    if ind > 1:
        return "join"
    return ""


def _gateway_type(pg: ProcessGraph, node: str) -> str:
    """exclusive / parallel / inclusive, or "" when it cannot be read.

    Checked most-specific first: "XOR" contains "OR", and the label "X" must not
    be confused with a "+"/"AND" parallel gateway.
    """
    norm = _norm_word(node)
    label = pg.label_of(node).strip().upper()
    text = f"{node} {pg.label_of(node)}".upper()

    if label == "X" or "XOR" in text or "EXCLUSIVE" in text or norm.startswith(("seg", "meg")):
        return "exclusive"
    if label in ("+", "AND") or "PARALLEL" in text or "AND_" in text or norm.startswith(("spg", "mpg")):
        return "parallel"
    if label in ("O", "OR") or "INCLUSIVE" in text or "OR_" in text:
        return "inclusive"
    return ""


def resolve_gateway_type(pg: ProcessGraph, node: str) -> str:
    """The gateway's BPMN type in the four-way BPMN vocabulary, never "".

    `_gateway_type` above reads only what the notation states and returns "" when
    it states nothing. This resolves the rest, and is what the metric modules
    use:

    1. whatever `_gateway_type` read from the marker or the name;
    2. **event-based** — a `label="E"` diamond, or `EventBasedGateway` in the
       id. PMo's Camunda-derived items write exactly that;
    3. otherwise **exclusive**. Not a guess: BPMN draws the exclusive gateway
       *with or without* the X marker, so an unmarked decision diamond — PMo's
       `"recourse possible?" [shape=diamond]` — *is* an exclusive gateway.

    Deliberately a separate function rather than a change to `gateway_types`:
    that dict feeds the "matching join" check, where a fourth type and a
    defaulted type would change which joins count as matching.
    """
    declared = pg.gateway_types.get(node, "")
    if declared:
        return declared
    label = pg.label_of(node).strip().upper()
    text = f"{node} {pg.label_of(node)}".upper().replace("-", "").replace("_", "")
    if label == "E" or "EVENTBASED" in text:
        return "eventbased"
    return "exclusive"


def _classify_event(pg: ProcessGraph, node: str) -> str:
    """Start, end or intermediate? Layered, because no single signal is
    reliable here.

    1. `doublecircle` **without an outgoing sequence flow** → end. Two prompt
       templates ask for exactly this shape for the end event, and every
       generated end event drawn that way is a sink.

       `doublecircle` **with** an outgoing sequence flow → **intermediate**.
       An end event that something flows out of is a contradiction, and PMo's
       README states the opposite notation outright: "End events are depicted
       with a bolded outer circle, while intermediate events use a double
       circle, consistent with BPMN". Reading every doublecircle as an end
       turned all 17 intermediate events of the reference set — timer, message
       and conditional catch events in items 22/23/24/25/30/38/42/44 — into
       end events, which cost those models #4, #8 and the sequence-flow rules
       and truncated their diameter at the first intermediate event (item 30:
       4 instead of 18). Corrected 2026-08-23; the counts then match the BPMN
       ground truth exactly (58 starts, 61 ends, 17 intermediates).
    2. Name/label keyword ("start…", "end…", "stop…"). PMo's own ground truth
       names them `start` / `end`, and `zero_shot_graph_type_tn_rules` produces
       `start_1` / `end_1`.
    3. Topology — a source (no incoming) is a start, a sink (no outgoing) is an
       end. Catches the Camunda-derived items whose events carry domain names
       like "Order received".

       Counted on **sequence flows only**. Where a process begins is a
       control-flow question, and a message flow is not control flow: PMo's
       `"scoring request received_1"` (item 23) is a *message start event* — no
       incoming sequence flow, one outgoing, and an incoming message flow from
       another pool. Counting every edge made it look like a node in the middle
       of a flow and left it unclassified. Four such events across items 23/24.

    Returns "start", "end", "intermediate", or "" when nothing decides — an
    event that merely sits in the middle without the doublecircle notation is
    left unclassified rather than guessed at.
    """
    attrs = pg.nodes.get(node, {})
    if attrs.get("shape", "").lower() == "doublecircle":
        return "end" if pg.seq_out_degree(node) == 0 else "intermediate"

    for text in (pg.label_of(node), node):
        w = _norm_word(text)
        if w.startswith(("start", "begin")):
            return "start"
        if w.startswith(("end", "stop", "finish", "terminate")):
            return "end"

    ind, outd = pg.seq_in_degree(node), pg.seq_out_degree(node)
    if ind == 0 and outd > 0:
        return "start"
    if outd == 0 and ind > 0:
        return "end"
    return ""


def load(path) -> ProcessGraph:
    """Parse a .dot/.gv file into a classified ProcessGraph.

    Raises ValueError if the file does not parse — callers treat that as a
    validity failure, not a syntactic-quality score of zero.
    """
    with DOT_PARSE_LOCK:
        graphs = pydot.graph_from_dot_file(str(path))
    if not graphs:
        raise ValueError(f"no graph found in {path}")
    root = graphs[0]

    # Implicit first, explicit declarations win: a node written once in an edge
    # and once as `[shape=diamond]` keeps its declared shape. A node written as
    # two declarations keeps the union of both — see `_collect_declared`.
    nodes = {**_collect_implicit(root), **_collect_declared(root)}
    raw_edges = _collect_edges(root)
    # Edges may reference nodes we never saw declared in any scope; keep them.
    for src, dst, _ in raw_edges:
        nodes.setdefault(src, {})
        nodes.setdefault(dst, {})

    edges = [(src, dst) for src, dst, _ in raw_edges]
    edge_attrs: Dict[Tuple[str, str], Dict[str, str]] = {}
    for src, dst, attrs in raw_edges:
        # A repeated edge keeps the attributes of its first occurrence; the
        # dedup elsewhere treats the two as one flow.
        edge_attrs.setdefault((src, dst), attrs)

    clusters = _collect_clusters(root)
    # `digraph` vs `graph`, straight from the source keyword — the same flag
    # Graphviz's `agisdirected()` returns. pydot reports the keyword verbatim.
    directed = str(root.get_type() or "digraph").strip().lower() == "digraph"
    pg = ProcessGraph(nodes=nodes, edges=edges, unique_edges=set(edges),
                      edge_attrs=edge_attrs, directed=directed,
                      pools={n: c[0] for n, c in clusters.items() if n in nodes},
                      lanes={n: c[1] for n, c in clusters.items() if n in nodes})

    events: List[str] = []
    for name, attrs in nodes.items():
        shape = attrs.get("shape", "").lower()
        style = attrs.get("style", "").lower()
        if shape in TASK_SHAPES:
            pg.tasks.add(name)
        elif shape in GATEWAY_SHAPES:
            pg.gateways.add(name)
        elif shape in EVENT_SHAPES:
            events.append(name)
        elif shape in MARKER_SHAPES or "invis" in style:
            pg.markers.add(name)
        else:
            pg.unclassified.add(name)

    for name in events:
        kind = _classify_event(pg, name)
        if kind == "start":
            pg.starts.add(name)
        elif kind == "end":
            pg.ends.add(name)
        elif kind == "intermediate":
            pg.intermediates.add(name)
        else:
            pg.events_unclassified.add(name)

    for name in pg.gateways:
        pg.gateway_roles[name] = _gateway_role(pg, name)
        pg.gateway_types[name] = _gateway_type(pg, name)

    # Recorded for the matching-join check; the degree rules do not use it.
    pg.mixed = {g for g in pg.gateways
                if pg.in_degree(g) > 1 and pg.out_degree(g) > 1}

    return pg
