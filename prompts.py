"""
prompts.py — Li et al. (2025) prompting strategies, reproduced verbatim
=========================================================================
Templates are reproduced from Li et al. (2025), "LLM-based Business Process
Models Generation from Textual Descriptions" (IJCNLP-AACL, 523-533), Section 4
("Methods") and Appendix A.1 ("Steps of Chain_of_Thought").

The paper's boxed Input/Output templates are written for a single linear
completion; the Anthropic Messages API is turn-based, so each "Input: ... /
Output: ..." pair in a few-shot template becomes one user turn + one
assistant turn. The verbatim *text* of every Input and Output block is
preserved; only the framing into API roles differs.

DC2 (faithful prompts): the instruction text, the CoT trigger phrase, and the
Steps-of-CoT scaffold are copied character-for-character from the paper. The
paper's Appendix A.1 box itself uses generic placeholders (<i>, <activity>,
"Sentence_i") rather than an instantiation for one specific example, so the
same generic block is reused for every few-shot-CoT exemplar here rather than
inventing a per-exemplar reasoning trace the paper never published.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dataset

# ── verbatim instruction text (Section 4 boxed templates) ───────────────────
_INSTRUCTION = (
    "Generate the DOT language to present the Business Process Model and "
    "Notation (BPMN) based on the given process textual description: {description}"
)
_COT_TRIGGER = " Please think about it step by step."

# ── verbatim system role (Section 4, "Fine-Tuned LLMs" training-example box) ─
# Used only by `zero_shot_system`, which is **deactivated** (see
# ALL_STRATEGIES below and README "Open decisions" #1): the text is kept so the
# strategy can be switched back on without re-deriving it from the paper, but
# nothing reaches it while the name is out of the strategy lists. It is also
# the pipeline's only system prompt — every other strategy passes `system=None`.
# The four remaining Li-et-al. strategies follow the paper's own boxed
# Input/Output templates, which carry no system role; the user-supplied
# strategies bundle any persona framing into the user turn instead.
FINE_TUNING_SYSTEM_PROMPT = (
    "Your position in a company is to visualize a business process using a "
    "graphical representation of a given piece of text in BPMN. Your "
    "response should use DOT language to do the graphical representation."
)

# ── Role Prompt 1 (Zero-Shot) strategy (user-supplied template, not from Li
# et al.) ── Single zero-shot turn: the persona framing is bundled into the
# same user turn as the task instruction, per the template as given (unlike
# `zero_shot_system`, which follows the paper's own boxed system-role split).
ROLE_PROMPT_1_INSTRUCTION = (
    "You are an expert in BPMN modeling and DOT language. Your task is to "
    "convert detailed textual descriptions of business processes into "
    "accurate BPMN model codes written in DOT language. Label all nodes "
    "with their activity names. Represent all connections between nodes "
    "without labeling the connections. Represent each node and its "
    "connections accurately, ensuring all decision points and flows are "
    "included and connected.\n\n"
    "Now, generate BPMN business process model code in DOT language for the "
    "following textual description of a business process:\n{description}"
)

# ── Role Prompt 2 (Zero-Shot) strategy (user-supplied template, not from Li
# et al.) ── Longer persona variant: adds an explicit BPMN semantics briefing
# (gateway split/merge rules, event cardinality, an error list) and a full
# GraphViz syntax guide with a worked formatting example.
#
# NOTE: this template is stored WITHOUT a `{description}` placeholder and is
# concatenated rather than `.format()`-ed, unlike every other template here.
# Its GraphViz blocks contain literal `{` / `}`, which `str.format` would
# reject; escaping them as `{{` / `}}` would mean the stored constant no
# longer equals the prompt text character-for-character, which matters when
# the prompts are reproduced in the thesis appendix (DC2). The template ends
# on "Process Description:" by design — the description follows it.
ROLE_PROMPT_2_INSTRUCTION = """Task
You are a process modeling expert. Your task is to create a process model based on a textual process description. You need to follow closely the modeling and formatting instructions to output a valid model.

Modeling instructions
A model contains multiple tasks (also called activities), gateways, one start event and one end event. There are multiple tasks and paths between start and end events. A path through the process model represents one of the variants of how the process could be executed.

Tasks
Tasks represent units of work performed within a process. Each task has a label.

Gateways
Gateways are modeling elements that control the flow of work through the paths of the process. Gateways themselves are not tasks. They just split and merge the process flow. There are 2 possible gateway types: the exclusive gateway and the parallel gateway. If, after some particular task, a decision is to be made (i.e., alternative paths occur), a splitting exclusive gateway is to be used. Only one of multiple outgoing paths will be (exclusively) executed. After that, the process flow should be merged again with the help of a merging exclusive gateway. If, after some particular task, multiple tasks should be executed at the same time, a splitting parallel gateway is to be used. After the execution of parallel tasks, there is always a synchronization point (waiting for several separate paths to reach a certain point before the process can continue). This synchronization point is represented by a merging parallel gateway. Every split in a control flow should always be merged. Only gateways of the same type can be used for splitting and merging.

Events
A start event indicates where a process begins. It has solely one outgoing flow and no incoming flow. An end event marks where a process ends. It has one incoming flow and no outgoing flow.

Modeling Errors to Avoid
Create multiple start events or end events.
Forget the merging gateway after a split.
Directly merge splits to a task/event without a merging gateway.

Formatting instructions
This guide provides instructions for generating valid GraphViz syntax.

Basic Structure
GraphViz diagrams should begin with the digraph declaration:
digraph G {
    rankdir=LR;  // Left to Right orientation
    node [shape=box];  // Define default node shape
}

Node Definition
Define nodes with unique identifiers surrounded by quotes, followed by their labels in square brackets. The syntax is as follows:
"NodeId" [label="Node Label", shape=shape type];
Nodes need to be defined only once, and they can be referenced multiple times in the graph using their id. In the case where the node label is unique and not empty, it can be used as the node id. For example:
"Decision?" [shape=diamond];
For tasks, the shape does not need to be specified, as the default is a box. It means that tasks do not need an additional definition after the connection. For example:
"Do something" -> "Decision?";

Node Ids
Node ids should follow these rules:
Must start with the node type followed by an underscore (e.g., ExclusiveGateway_1_, StartEvent_1_).
Must use a unique counter based on the node type (e.g., ExclusiveGateway_1, ExclusiveGateway_2).

Node Types
Possible node types are:
Task (box): "Do task"
StartEvent (circle): "StartEvent_1" [label="start", shape=circle];
EndEvent (circle): "EndEvent_1" [label="end", shape=circle];
ExclusiveGateway (diamond): "ExclusiveGateway_1" [label="X", shape=diamond];
ParallelGateway (diamond): "ParallelGateway_1" [label="+", shape=diamond];
StartEvent and EndEvent label can also be customized or left empty. Exclusive gateways label should be "X" or the name of the decision for the opening gateway. Parallel gateways label should be "+".

Connections
Connect nodes using arrows:
"NodeA" -> "NodeB";

Conditional Connections
For conditional connections (after Exclusive gateway), use the following syntax:
"NodeA" -> "NodeB" [label="Condition"];

Formatting Example
digraph G {
    rankdir=LR;
    node [shape=box];

    "StartEvent_1" -> "Do something";
    "Do something" -> "Decision?";
    "Decision?" -> "Do task" [label="Yes"];
    "Do task" -> "ExclusiveGateway_2";
    "Decision?" -> "ExclusiveGateway_2" [label="No"];
    "ExclusiveGateway_2" -> "EndEvent_1";

    "StartEvent_1" [label="start", shape=circle];
    "Decision?" [shape=diamond];
    "ExclusiveGateway_2" [label="X", shape=diamond];
    "EndEvent_1" [label="end", shape=circle];
}

Process Description:"""

# ── Tree-of-Thought strategy (user-supplied template, not from Li et al.) ───
# Single zero-shot turn: the multi-stage reasoning scaffold is embedded in the
# instruction itself, so no separate CoT trigger or exemplars are needed.
TOT_INSTRUCTION = """Your task is to convert a textual business process description into a BPMN model in DOT format using a Tree-of-Thought (ToT) approach.
Follow this multi-step reasoning:
1) Propose multiple alternatives for each stage:
- Stage 1: Identify tasks (boxes) and start/end events.
- Stage 2: Insert gateways (XOR="X", AND="+") with correct split/join pairing.
- Stage 3: Connect nodes with unlabeled arrows; ensure reachability from start+end.
- Stage 4: Refine into the final DOT graph.
2) At each stage, compare alternatives and keep the best, following criteria:
- Valid BPMN control flow (balanced gateways, one start, one end).
- Completeness (all key tasks/events from the text).
- Minimality (no invented steps, no dangling nodes).
- DOT validity (shapes, rankdir=LR, unlabeled edges).
3) Continue until you reach the final solution.

Final Output: Emit ONLY the DOT code in a single digraph G {{ ... }} block, with:
- rankdir=LR
- shape=box for tasks
- shape=diamond with label="X"/"+" for gateways
- shape=circle for start, shape=doublecircle for end
- all nodes connected, edges unlabeled

Now, generate the BPMN business process model (DOT) for this description:
{description}"""

# ── Zero-Shot + graph type + textual-notation rules (user-supplied template,
# not from Li et al.) ── Single zero-shot turn carrying no persona at all;
# instead it fixes the target notation with an explicit node/edge grammar
# (shapes, empty start/end labels, seg_/meg_/spg_/mpg_ naming convention with
# per-type counters, edge labels for conditions).
#
# ONE DEVIATION FROM THE SOURCE (author's instruction, 2026-08-27): the
# template's own ". " between the description and the briefing is now
# " " + a line break, so the two blocks start on separate lines. The
# source runs them together, which also produced a double full stop for
# every PMo description — they all end in one themselves ("...confirmed..").
# Nothing else is touched, and the trailing space before the break is
# deliberate. The thesis has to report this as a formatting change; the
# strategy is no longer verbatim (README strategy table, DC2).
#
# Unlike ROLE_PROMPT_2 this template contains no literal braces, so the usual
# `.format(description=...)` applies. The description slots in *mid-template*
# (the paper's `<Process Description>` placeholder), not at the end. Kept
# verbatim including its original typos ("incrimented", "Exmaple", "splitted")
# and its unbalanced-looking parenthesis around the rules block.
ZERO_SHOT_GRAPH_TYPE_TN_RULES_INSTRUCTION = """Process description: {description} 
Custom rules for textual notation: (Every graph must have LR (Left to Right) direction. It consists of nodes and edges.
Each node has following structure: "name"[attributes]
There are 5 different types of nodes: start event, end event, task, exclusive gateway and parallel gateway.
Each node has its specific attributes based on the type of the node. Example:
  start node:        "start_1"[shape=circle label=""];
  end node:          "end_1"[shape=doublecircle label=""];
In both start and end nodes labels are always empty. Example:
  task:              "task label"[shape=rectangle];
Task labels are always unique. Example:
  exclusive gateway: "seg_1"[shape=diamond label="X"];
  parallel gateway:  "spg_1"[shape=diamond label="AND"];

Gateways are not tasks. They merely indicate that the control flow of the process is splitted or merged.
Following names "seg_1" and "meg_1" should be used for splitting and merging exclusive gateways.
Following names "spg_1" and "mpg_1" should be used for splitting and merging parallel gateways.

Every time when new start, end or gateways node is used, the counter should be incrimented at 1.

All elements are connected with each other with the help of the edges.
  edge: ->
Examples:
  "start" -> "task 1"
  "task 1" -> "task 2"

If there are conditions or annotations, it is necessary to use text on links (i.e., edge labels).
Exmaple:
  edge label:  "task 1" -> "task 2"[label="condition 1"]). Considering the provided process description and a set of custom rules create a valid Graphviz graph."""

# ── Zero-Shot + short BPMN description (user-supplied template, not from Li
# et al.) ── Structural counterpart to ZERO_SHOT_GRAPH_TYPE_TN_RULES: same
# "Process description: ... (briefing). Considering ..." frame, but the
# briefing covers BPMN *semantics* (event cardinality, task nature, gateway
# split/merge rules) and says nothing about GraphViz syntax — the model is
# still asked for a valid Graphviz graph, with the notation left implicit.
# No literal braces, so `.format(description=...)` applies; the description
# slots in mid-template. Kept verbatim apart from the separator below.
#
# ONE DEVIATION FROM THE SOURCE (author's instruction, 2026-08-27): the
# template's own ". " between the description and the briefing is now
# " " + a line break, so the two blocks start on separate lines. The
# source runs them together, which also produced a double full stop for
# every PMo description — they all end in one themselves ("...confirmed..").
# Nothing else is touched, and the trailing space before the break is
# deliberate. The thesis has to report this as a formatting change; the
# strategy is no longer verbatim (README strategy table, DC2).
ZERO_SHOT_SHORT_BPMN_DESCRIPTION_INSTRUCTION = """Process description: {description} 
Short description of BPMN standard: (Each model has only one start event, one or more end events, activities (also called tasks), and gateways.
A start event shows where a process begins. It has solely one outgoing element and no incoming elements. An end event marks where a process ends. It has one incoming element and no outgoing elements.
There are multiple tasks and paths between start and end events.
A path through the process model represents one of the variants of how the process could be executed.
Tasks represent the work performed within a process. A task will normally take some time to perform, involve one or more resources, require some type of input and produce some sort of output. Each task has a label.
Gateways are modeling elements that control the flow of work through the paths of the process. Gateways themselves are not activities. They just split and merge the process flow. Gateways are unnecessary if the process flow does not require controlling. The two most commonly used gateways are the exclusive and parallel gateways.
If, after some particular task, a decision is to be made (i.e. alternative paths occur), a splitting exclusive gateway is to be used. Only one of multiple outgoing paths will be (exclusively) executed. After that, the process flow should be merged again with the help of a merging exclusive gateway.
If, after some particular task, multiple tasks should be executed at the same time, a splitting parallel gateway is to be used. After the execution of parallel tasks, there is always a synchronization point (waiting for several separate paths to reach a certain point before the process can continue). This synchronization point is represented by a merging parallel gateway.
Every split in a control flow should always be merged. Only gateways of the same type could be used for splitting and merging.). Considering the provided process description and a short description of BPMN standard create a valid Graphviz graph."""

# ── verbatim Steps-of-CoT scaffold (Appendix A.1) ────────────────────────────
STEPS_OF_COT = """Following the steps to answer the question:
STEP1: Claim the task: In order to generate the dot language for the given text, we should generate the subgraph for each sentence.
STEP2: Before start, identify the condition type in each sentence. There are three types of conditions in the subprocess:
a. If the sentence contains the phrases like "the process is split into" and "should be done in parallel", it is the AND condition. For this condition, there are two special nodes in the subgraph: "AND_SPLIT" and "AND_JOIN".
b. If the sentence contains a phrase like "one or more of the following paths", it is OR condition. For this condition, there are two special nodes in the subgraph: "OR_SPLIT" and "OR_JOIN".
c. If the sentence contains phrases like "should be considered" and "should be taken into account", it is XOR condition. For this condition, there are two special nodes in the graph: "XOR_SPLIT" and "XOR_JOIN".
STEP3: Generate the subgraph for each sentence. Based on the given text, there are <i> sentences.
Sentence_1: The process starts with the start node, and then the process goes to <activity>. So the subgraph is:
subgraph sentence_1 {start -> <activity>; }
......
Sentence_i: The process is completed. The previous node is the last node in sentence_i-1. So the subgraph is:
subgraph sentene_i {<activity> -> end; }
STEP4: Join all the subgraphs together. The generated dot language is:"""

# Strategies a run may ask for. `zero_shot_system` is deliberately absent —
# deactivated 2026-08-17, see config.ALL_STRATEGIES. Its builder below stays
# intact but is unreachable while the name is missing here, because
# `strategy_is_known()` gates every entry point.
ALL_STRATEGIES = ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "tree_of_thought", "role_prompt_1", "role_prompt_2", "zero_shot_graph_type_tn_rules", "zero_shot__short_bpmn_description"]


def strategy_is_known(strategy: str) -> bool:
    return strategy in ALL_STRATEGIES


def needs_cot(strategy: str) -> bool:
    return strategy in ("zero_shot_cot", "few_shot_cot")


def needs_exemplars(strategy: str) -> bool:
    return "few_shot" in strategy


# -- prompt version (O6: traceable experimentation) ---------------------------
# Which template constants each strategy's wording is assembled from. The
# fingerprint below hashes exactly those, so it moves when a template is edited
# and stays put when anything else in this module is: two runs recorded under
# the same fingerprint were prompted with the same wording, and a run recorded
# under a different one was not. It is the *version* of the template, not of a
# single prompt -- the prompt actually sent is hashed per generation in
# pipeline.py, which is where the item description enters.
_TEMPLATE_PARTS: Dict[str, Tuple[str, ...]] = {
    "zero_shot": ("_INSTRUCTION",),
    "zero_shot_cot": ("_INSTRUCTION", "_COT_TRIGGER"),
    "zero_shot_system": ("_INSTRUCTION", "FINE_TUNING_SYSTEM_PROMPT"),
    "few_shot": ("_INSTRUCTION",),
    "few_shot_cot": ("_INSTRUCTION", "STEPS_OF_COT"),
    "tree_of_thought": ("TOT_INSTRUCTION",),
    "role_prompt_1": ("ROLE_PROMPT_1_INSTRUCTION",),
    "role_prompt_2": ("ROLE_PROMPT_2_INSTRUCTION",),
    "zero_shot_graph_type_tn_rules": ("ZERO_SHOT_GRAPH_TYPE_TN_RULES_INSTRUCTION",),
    "zero_shot__short_bpmn_description": ("ZERO_SHOT_SHORT_BPMN_DESCRIPTION_INSTRUCTION",),
}


def template_version(strategy: str) -> str:
    """A short fingerprint of the template text `strategy` is built from.

    Empty for a strategy with no entry above, which is the honest answer: an
    unknown wording has no version to record.
    """
    parts = _TEMPLATE_PARTS.get(strategy)
    if not parts:
        return ""
    blob = "".join(f"{name}={globals()[name]}" for name in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ── exemplar loading (independent of the input items being generated for) ──
@dataclass
class Exemplar:
    item_id: str
    description: str
    dot: str


def load_exemplars(paths: List[Path]) -> List[Exemplar]:
    """Load (description, ground-truth DOT) pairs from .gv files to use as
    few-shot in-context examples. Independent of the dataset being scored."""
    exemplars: List[Exemplar] = []
    for p in paths:
        item = dataset.load_item(Path(p))
        dot_text = Path(p).read_text(encoding="utf-8")
        exemplars.append(Exemplar(item_id=item.item_id, description=item.description, dot=dot_text))
    return exemplars


def exemplars_for_item(
    exemplars: List[Exemplar],
    item_id: str,
    reserve: Optional[List[Exemplar]] = None,
    alternates: Optional[List[Exemplar]] = None,
) -> Tuple[List[Exemplar], Optional[Tuple[str, str]]]:
    """The few-shot exemplars to show when generating for `item_id`.

    Every dataset item is evaluated, exemplars included, so a run can reach the
    item whose ground-truth model is sitting in its own prompt. Showing it
    would not measure generation, it would measure copying — so that one
    exemplar is replaced by the first unused stand-in from `reserve` (itself
    never the item being generated for). The other exemplars, and their order,
    are untouched: only the colliding slot changes.

    Returns `(exemplars_to_show, swap)`, where `swap` is `(dropped_id,
    replacement_id)` for the caller to log, or None when nothing collided.
    With no usable reserve left the colliding exemplar is dropped rather than
    shown — one exemplar fewer is a weaker prompt, its own answer is not a
    prompt at all — and `swap` reports `(dropped_id, "")`.

    `alternates` short-circuits all of that. It is a *complete* replacement
    exemplar set, and the caller decides which items get one
    (`cfg.FEW_SHOT_ALT_FOR_IDS`, applied in pipeline.run_experiment). Swapping
    one slot gives each colliding item a different prompt — `01` sees
    `03,02` and `02` sees `01,03` — so the items closest to the exemplar
    set are also the ones whose prompts differ most from every other item's.
    A disjoint alternate set gives all of them the same exemplars instead.
    `swap` then reports the whole set as `("01,02", "40,41")`.
    """
    if alternates:
        # The item is still filtered out of its own prompt: the guarantee this
        # function exists for holds however the set was chosen. A set that is
        # nothing but the item itself is no set at all, so that case falls
        # through to the per-slot swap below rather than returning empty.
        shown = [ex for ex in alternates if ex.item_id != item_id]
        if shown:
            return shown, (",".join(ex.item_id for ex in exemplars),
                           ",".join(ex.item_id for ex in shown))

    hit = next((i for i, ex in enumerate(exemplars) if ex.item_id == item_id), None)
    if hit is None:
        return list(exemplars), None

    used = {ex.item_id for ex in exemplars}
    stand_in = next(
        (ex for ex in (reserve or []) if ex.item_id != item_id and ex.item_id not in used),
        None,
    )
    out = list(exemplars)
    dropped = out[hit].item_id
    if stand_in is None:
        del out[hit]
        return out, (dropped, "")
    out[hit] = stand_in
    return out, (dropped, stand_in.item_id)


# ── message assembly ─────────────────────────────────────────────────────────
def build_messages(
    strategy: str,
    description: str,
    exemplars: Optional[List[Exemplar]] = None,
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Build (system_prompt, messages) for the Anthropic Messages API.

    `messages` alternates user/assistant turns, ending on a user turn (the
    target item) so the model's reply is the generation to score.
    """
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Known: {ALL_STRATEGIES}")

    system: Optional[str] = None
    messages: List[Dict[str, str]] = []

    # Unreachable while `zero_shot_system` is out of ALL_STRATEGIES (the guard
    # above rejects the name first). Kept so re-enabling is a one-line change.
    if strategy == "zero_shot_system":
        system = FINE_TUNING_SYSTEM_PROMPT
        messages.append({"role": "user", "content": _INSTRUCTION.format(description=description)})
        return system, messages

    if strategy == "zero_shot":
        messages.append({"role": "user", "content": _INSTRUCTION.format(description=description)})
        return system, messages

    if strategy == "zero_shot_cot":
        text = _INSTRUCTION.format(description=description) + _COT_TRIGGER
        messages.append({"role": "user", "content": text})
        return system, messages

    if strategy == "tree_of_thought":
        messages.append({"role": "user", "content": TOT_INSTRUCTION.format(description=description)})
        return system, messages

    if strategy == "role_prompt_1":
        messages.append({"role": "user", "content": ROLE_PROMPT_1_INSTRUCTION.format(description=description)})
        return system, messages

    if strategy == "role_prompt_2":
        # concatenated, not .format()-ed — see ROLE_PROMPT_2_INSTRUCTION note
        messages.append({"role": "user", "content": ROLE_PROMPT_2_INSTRUCTION + "\n" + description})
        return system, messages

    if strategy == "zero_shot_graph_type_tn_rules":
        text = ZERO_SHOT_GRAPH_TYPE_TN_RULES_INSTRUCTION.format(description=description)
        messages.append({"role": "user", "content": text})
        return system, messages

    if strategy == "zero_shot__short_bpmn_description":
        text = ZERO_SHOT_SHORT_BPMN_DESCRIPTION_INSTRUCTION.format(description=description)
        messages.append({"role": "user", "content": text})
        return system, messages

    # few_shot / few_shot_cot
    exemplars = exemplars or []
    if not exemplars:
        raise ValueError(f"Strategy {strategy!r} needs at least one exemplar; none given.")

    cot = needs_cot(strategy)
    for ex in exemplars:
        messages.append({"role": "user", "content": _INSTRUCTION.format(description=ex.description)})
        if cot:
            assistant_text = STEPS_OF_COT + "\n" + ex.dot
        else:
            assistant_text = ex.dot
        messages.append({"role": "assistant", "content": assistant_text})

    final_text = _INSTRUCTION.format(description=description)
    messages.append({"role": "user", "content": final_text})
    return system, messages
