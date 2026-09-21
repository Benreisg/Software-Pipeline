"""
debug_bef4llm_context.py — what BEF4LLM's context similarity actually pairs
==========================================================================
Run **in the BEF4LLM environment** (pm4py, networkx, xmlschema). It re-runs
their natural-language similarity pass on one BPMN model against itself and
prints the two things their published columns do not show:

* the **semantic mapping** their context similarity resolves neighbourhoods
  with — which node each node was paired to, and at what similarity;
* the **context mapping** the score sums over, per node.

    python debug_bef4llm_context.py 54 --bpmn-dir "<PMo>\\bpmn" \\
           --bef4llm-src "<bef4llm-main>\\src"

**Why.** `sem_label_sim_context` of a model against itself should be 0.5 under
their divisor — every node paired with itself, every context matched. On 16 of
the 55 PMo models it is less, and the shortfall is *fractional*: 1.0 on items
26 and 32, 1.5 on 49, 51, 52 and 53, 2.62 on item 23. A whole node dropping out
would cost a whole 1.0, so what is being lost are single **directions** of a
node's context — one of `pre`/`succ` matching and the other not.

Reproducing that from the DOT side has been tried and does not close: their
formula, their greedy matching, their recursive neighbourhood, the document
node order and the BPMN input itself each moved the agreement between 35 and 42
of 55 without reaching it. The remaining candidates all live inside the run:
which node the greedy matching binds an unlabelled event to, how far the shared
`visited` list truncates a neighbourhood, and which pair wins a tie. This
prints exactly that.

Read the output like this: any line where `node` and `partner` differ on a
**self**-comparison is a pairing their algorithm got wrong, and every context
value below 1.0 is a node whose neighbourhood did not survive it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dump BEF4LLM's own node and context mappings for one model.")
    ap.add_argument("item", help="PMo item id, e.g. 54")
    ap.add_argument("--bpmn-dir", required=True)
    ap.add_argument("--bef4llm-src", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.bef4llm_src)
    from bef4llm.process_models.importer.bpmn_importer import load_diagram_from_xml
    from bef4llm.process_models.graph_representation.node_types import Event, Task
    from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language
    from bef4llm.semantic_quality.similarity.language_similarity.natural_language_similarity import (
        context_similarity, semantic_similarity)
    from bef4llm.semantic_quality.similarity.similarity_utils import (
        create_euquivalence_mapping_for_nodes)

    path = Path(args.bpmn_dir) / f"{args.item}.bpmn"
    model, reference = load_diagram_from_xml(str(path)), load_diagram_from_xml(str(path))
    g1, g2 = model.process_graph, reference.process_graph

    node_types = list(Task.__members__) + list(Event.__members__)
    compared = [n for n in g1.nodes
                if g1.nodes[n].get("type") in node_types]
    print(f"item {args.item}: {len(compared)} task/event nodes, "
          f"divisor {2 * len(compared)}")

    # 1) the semantic mapping the context similarity resolves neighbours with
    sem_mapping = create_euquivalence_mapping_for_nodes(
        g1, g2, sim=semantic_similarity(lang=Language.ENGLISH), threshold=0.0)
    print("\n-- semantic mapping (node -> partner, similarity) --")
    wrong = 0
    for node, (partner, similarity) in sem_mapping.items():
        flag = ""
        if node != partner:
            flag = "   <- paired with a DIFFERENT node"
            wrong += 1
        name = g1.nodes[node].get("name", "<no name attribute>") if node in g1.nodes else "?"
        print(f"   {str(node)[:34]:36} -> {str(partner)[:34]:36} "
              f"{similarity:.4f}  name={name!r}{flag}")
    print(f"   ({wrong} of {len(sem_mapping)} paired with a different node)")

    # 2) the context mapping the score sums over
    ctx_mapping = create_euquivalence_mapping_for_nodes(
        g1, g2,
        sim=context_similarity(set_of_graphs=[g1, g2], mapping=sem_mapping),
        threshold=0.0)
    total = sum(similarity for _, similarity in ctx_mapping.values())
    print("\n-- context mapping (node -> partner, context value) --")
    for node, (partner, similarity) in sorted(ctx_mapping.items(),
                                              key=lambda kv: kv[1][1]):
        flag = "   <- below 1.0" if similarity < 1.0 else ""
        print(f"   {str(node)[:34]:36} -> {str(partner)[:34]:36} "
              f"{similarity:.4f}{flag}")
    print(f"\nsum {total:.4f} / {2 * len(compared)} = "
          f"{total / (2 * len(compared)):.4f}   (their published value)")


if __name__ == "__main__":
    main()
