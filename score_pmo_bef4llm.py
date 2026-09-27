"""
score_pmo_bef4llm.py — score the PMo models with BEF4LLM's *own* implementation
==============================================================================
The counterpart to `score_pmo_dataset.py`. That one runs this project's port of
the BEF4LLM **paper**; this one runs the authors' **supplied code** over the
same 55 PMo processes, so the two can be compared metric by metric.

    python score_pmo_bef4llm.py --out pmo_quality_bef4llm.csv

**Different input format, same processes.** BEF4LLM reads BPMN 2.0 XML, not
DOT, so this reads the `.bpmn` half of the PMo dataset (`pmo-dataset/bpmn`)
while `score_pmo_dataset.py` reads the `.dot` half. Same 55 processes, two
serialisations — which is itself a source of difference and is flagged in the
report rather than hidden.

**Not importable from the main environment.** BEF4LLM needs pm4py, seaborn,
xmlschema and networkx, none of which this project depends on. Run it in a
separate interpreter with those installed (`.venv-bef4llm` beside the project)
and point `--bef4llm-src` at `bef4llm-main/.../src`; it never imports anything
from `quality/`. `score_pmo_dataset.py` does exactly that on every run, so the
usual way to refresh these numbers is to run *that* script, not this one.

── What the supplied code computes ──────────────────────────────────────────
    validity     validate_bpmn()               XSD schema check
    syntactic    16 `Sytax_Mistakes` metrics   *not* the paper's Table 2 set
    pragmatic    15 metrics in 7 groups        includes TNMF, which this
                                               project deliberately omits
    semantic     7 metrics in 3 groups         model against reference

Every dimension is run on a **freshly loaded model**: the check classes mutate
the graph they are given (`add_subproccess_nodes_to_graph`, `map_edges_to_id`)
and `SyntacticQualityCheckBPMN` never resets some of its error slots between
runs, so feeding one model to two dimensions — or calling a dimension's public
methods twice on one instance — would report a model as more broken than it is.

Within a dimension the measurement runs **once** and all three aggregates are
read off that one result, because the public `*_check()` / `*_check_detailed()`
/ `*_check_metric_results()` methods each recompute everything from scratch and
the semantic pass is expensive (~15 s on the largest model). No aggregation is
reimplemented here: the syntactic and pragmatic classes expose their
aggregators separately (`compute_syntax_score`, `compute_score_per_group`, …),
and for the semantic class — whose aggregators are welded to the computation —
the three `*_similarity()` methods are stubbed out *after* they have run, so
its own `semantic_quality_check()` executes over the state it just produced.
`--verify` re-scores a few items with fresh objects and checks every metric
and aggregate against the single-pass result.

── The semantic columns are a self-comparison ───────────────────────────────
Each model is compared with itself, exactly as in `score_pmo_dataset.py`. The
DOT port currently scores five semantic metrics: its graph edit distance and
common-nodes/edges are 1.0 on self-comparisons, while its three natural-language
metrics are around 0.5. Under the supplied code the three natural-language
metrics also come out at 0.5,
because the published formula divides the summed similarity by
`|FO_c| + |FO_g|` and the paper multiplies that sum by 2 (Table A.17) while the
code does not — a perfect self-match therefore scores n/(n+n). The number is a
property of the implementation, not of the models, and is reported here as
found rather than corrected.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parent
# Both moved into the project on 2026-08-27. Kept relative to this file, so the
# two scripts that read them cannot drift onto different copies of either.
DEFAULT_BPMN_DIR = PROJECT_DIR / "pmo-dataset" / "bpmn"
DEFAULT_BEF_SRC = PROJECT_DIR / "bef4llm-main" / "bef4llm-main" / "src"
DEFAULT_OUT = PROJECT_DIR / "pmo_quality_bef4llm.csv"


def _load_bef4llm(src: str):
    """Import BEF4LLM from `src` and return the handful of names needed."""
    sys.path.insert(0, src)
    from bef4llm.definitions import (Pragmatic_Metrics, Pragmatic_Subgroups,
                                     Similarity_Groups, Similarity_Metrics,
                                     Sytax_Mistakes)
    from bef4llm.pragmatic_quality.pragmatic_quality_check import PragmaticQualityCheckBPMN
    from bef4llm.process_models.importer.bpmn_importer import load_diagram_from_xml
    from bef4llm.semantic_quality.semantic_quality_check import SemanticQualityCheckBPMN
    from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language
    from bef4llm.semantic_quality.similarity.language_similarity.natural_language_similarity import semantic_similarity
    from bef4llm.process_models.graph_representation.node_types import Event, Task
    from bef4llm.synactic_quality.synactic_quality_check import SyntacticQualityCheckBPMN
    from bef4llm.validation.validation import validate_bpmn
    return dict(
        load=load_diagram_from_xml, validate=validate_bpmn,
        Syn=SyntacticQualityCheckBPMN, Prag=PragmaticQualityCheckBPMN,
        Sem=SemanticQualityCheckBPMN, Language=Language,
        SynM=Sytax_Mistakes, PragM=Pragmatic_Metrics, PragG=Pragmatic_Subgroups,
        SemM=Similarity_Metrics, SemG=Similarity_Groups,
        sem_sim=semantic_similarity, Task=Task, Event=Event,
    )


# ── common nodes and edges, computed correctly on BEF4LLM's graphs ──────────
# Their `strucural_similarity` calls `common_percentage_similarity(matching=…)`
# with the default `edges=False`, over a matching built at `threshold=0.0` that
# puts *every* task and event into the dict. `__compute_sn` then only asks
# whether a node id is in that dict, so the value is |N1|+|N2| / |N1|+|N2| = 1.0
# for any two models. The published column is therefore replaced by the printed
# formula (Table A.17), evaluated on the graphs their importer builds and with
# their own `semantic_similarity` as the node similarity:
#
#     1 - (unmatched nodes + unmatched edges) / (|N1| + |N2| + |E1| + |E2|)
#
# Nodes are tasks and events (the domain of their matching); every other node
# is contracted away, its predecessors wired to its successors, as their
# `__compute_nodes_edges` does for gateways. The matching is injective — the
# assignment maximising total similarity, zero-similarity pairs dropped — and
# is held by position, not by node id, so shared ids cannot collide. An edge is
# matched when its mapped endpoints carry an edge of the same flow type.
# The value the supplied code returned is kept as `*_bef4llm_raw`.

def _contracted(graph, B: Dict[str, Any]):
    """Tasks and events of `graph`, every other node bridged over."""
    import networkx as nx

    keep = set(B["Task"].__members__) | set(B["Event"].__members__)
    g = nx.DiGraph()
    g.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        g.add_edge(u, v, type=str(getattr(data.get("type"), "value",
                                          data.get("type"))))
    for node, data in list(g.nodes(data=True)):
        if data.get("type") in keep:
            continue
        preds = [p for p in g.predecessors(node) if p != node]
        succs = [s for s in g.successors(node) if s != node]
        g.remove_node(node)
        for p in preds:
            for s in succs:
                if not g.has_edge(p, s):
                    g.add_edge(p, s, type="sequenceFlow")
    return g


def common_nodes_and_edges(graph1, graph2, B: Dict[str, Any]) -> Dict[str, float]:
    """The metric and its two halves for one pair of BEF4LLM process graphs."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    g1, g2 = _contracted(graph1, B), _contracted(graph2, B)
    n1, n2 = list(g1.nodes), list(g2.nodes)
    sim = B["sem_sim"](lang=B["Language"].ENGLISH)
    weights = np.zeros((len(n1), len(n2)))
    for i, a in enumerate(n1):
        for j, b in enumerate(n2):
            weights[i, j] = sim(g1.nodes[a], g2.nodes[b])

    mapping: Dict[Any, Any] = {}
    if weights.size:
        rows, cols = linear_sum_assignment(weights, maximize=True)
        mapping = {n1[i]: n2[j] for i, j in zip(rows, cols) if weights[i, j] > 0.0}

    edges_matched = sum(
        1 for u, v, data in g1.edges(data=True)
        if u in mapping and v in mapping
        and g2.has_edge(mapping[u], mapping[v])
        and g2[mapping[u]][mapping[v]]["type"] == data["type"])

    nodes_total = len(n1) + len(n2)
    edges_total = g1.number_of_edges() + g2.number_of_edges()
    total = nodes_total + edges_total
    matched = 2 * len(mapping) + 2 * edges_matched
    return {
        "common_percentage": matched / total if total else 1.0,
        "common_nodes": 2 * len(mapping) / nodes_total if nodes_total else 1.0,
        "common_edges": 2 * edges_matched / edges_total if edges_total else 1.0,
    }


def _fix_common_percentage(sem, model_path: Path, reference_path: Path,
                           B: Dict[str, Any], row: Dict[str, Any]) -> None:
    """Replace their constant with the correct value, before the aggregators run.

    Computed on freshly loaded graphs, because the check class mutates the ones
    it holds. The value is written into `sem.similarity_metrics`, so the group
    and dimension scores that `semantic_quality_check()` reads off that state
    aggregate the corrected metric, not the constant.
    """
    key = B["SemM"].common_percentage.value
    group = B["SemG"].graph_structure.value
    fixed = common_nodes_and_edges(B["load"](str(model_path)).process_graph,
                                   B["load"](str(reference_path)).process_graph, B)
    raw = sem.similarity_metrics[group].get(key)
    row[f"bef_sem_{key}_bef4llm_raw"] = None if raw is None else float(raw)
    row["bef_sem_common_nodes"] = fixed["common_nodes"]
    row["bef_sem_common_edges"] = fixed["common_edges"]
    sem.similarity_metrics[group][key] = fixed["common_percentage"]


def score_semantic_pair(generated: Path, reference: Path,
                        B: Dict[str, Any]) -> Dict[str, Any]:
    """The semantic dimension for one **pair** of models, their code, their
    BPMN.

    The counterpart to `score_pmo_pairs.py`: that one scores the same pairs
    over the `.dot` half of the dataset with this project's port, this one over
    the `.bpmn` half with the authors' `SemanticQualityCheckBPMN`. Both key on
    `item_generated` / `item_reference`, so the two CSVs join on those columns
    and every row compares two implementations on the same two processes.

    Only the semantic dimension is run — the syntactic and pragmatic ones are
    properties of a single model and are already in `pmo_quality_bef4llm.csv`.

    The three `*_similarity()` methods are stubbed after they have run, for the
    same reason as in `score_one`: the aggregators would otherwise recompute a
    ~30 s pass over state they already hold.
    """
    load = B["load"]
    row: Dict[str, Any] = {
        "item_generated": generated.stem,
        "item_reference": reference.stem,
        "generated_path": str(generated),
        "reference_path": str(reference),
    }
    sem = B["Sem"](model=load(str(generated)),
                   reference_model=load(str(reference)),
                   lang=B["Language"].ENGLISH)
    sem.semantic_quality_check_metric_results()
    _noop = lambda *a, **k: None                                  # noqa: E731
    sem.natural_language_similarity = _noop
    sem.strucural_similarity = _noop
    sem.behavioural_similarity = _noop
    _fix_common_percentage(sem, generated, reference, B, row)
    sem_m = sem.semantic_quality_check_metric_results()
    row["bef_sem_score"] = sem.semantic_quality_check()
    sem_g = sem.semantic_quality_check_detailed()
    for group in B["SemG"]:
        row[f"bef_sem_group_{group.value}"] = sem_g.get(group.value)
    for metric in B["SemM"]:
        v = sem_m.get(metric.value)
        row[f"bef_sem_{metric.value}"] = None if v is None else float(v)
    row["bef_error"] = ""
    return row


def run_pairs(pairs_csv: Path, bpmn_dir: Path, out: Path,
              B: Dict[str, Any], generated_dir: Path = None) -> None:
    """Score every pair listed in `pairs_csv` and write `out`.

    The file is `score_pmo_pairs.py`'s `*_list.csv`: one row per pair, columns
    `item_generated`, `item_reference`, `direction`.

    `generated_dir` is where the *generated* side is read from; it defaults to
    `bpmn_dir`, which is what the offset pairing needs (both sides are PMo
    models). The basic-variation pairing puts the two sides in different
    directories under the same item id, and needs it set.
    """
    gen_dir = Path(generated_dir) if generated_dir else bpmn_dir
    import csv

    import pandas as pd

    with open(pairs_csv, newline="", encoding="utf-8-sig") as handle:
        pairs = list(csv.DictReader(handle))
    print(f"Scoring {len(pairs)} pairs with BEF4LLM's own semantic check",
          flush=True)

    rows: List[Dict[str, Any]] = []
    failed = 0
    for i, pair in enumerate(pairs, 1):
        generated = gen_dir / f"{pair['item_generated']}.bpmn"
        reference = bpmn_dir / f"{pair['item_reference']}.bpmn"
        try:
            row = score_semantic_pair(generated, reference, B)
            row["direction"] = pair.get("direction", "")
            rows.append(row)
            print(f"  [{i:>3}/{len(pairs)}] {generated.stem} vs {reference.stem}"
                  f"  sem {row['bef_sem_score']:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001 — record, never abort the sweep
            failed += 1
            rows.append({"item_generated": generated.stem,
                         "item_reference": reference.stem,
                         "direction": pair.get("direction", ""),
                         "bef_error": f"{type(exc).__name__}: {exc}"})
            print(f"  ! {generated.stem} vs {reference.stem}: {exc}", flush=True)
            traceback.print_exc(limit=2)

    frame = pd.DataFrame(rows)
    lead = ["item_generated", "item_reference", "direction", "bef_sem_score",
            "bef_error"]
    frame = frame[[c for c in lead if c in frame.columns]
                  + [c for c in frame.columns if c not in lead]]
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} rows x {len(frame.columns)} columns -> {out}"
          + (f"  ({failed} failed)" if failed else ""))


def score_one(path: Path, B: Dict[str, Any]) -> Dict[str, Any]:
    """All four dimensions for one .bpmn file, as the supplied code computes them."""
    row: Dict[str, Any] = {"item_id": path.stem, "bpmn_path": str(path)}
    load = B["load"]

    row["bef_xsd_valid"] = bool(B["validate"](str(path)))

    m = load(str(path))
    row["bef_n_nodes"] = m.process_graph.number_of_nodes()
    row["bef_n_edges"] = m.process_graph.number_of_edges()
    row["bef_n_pools"] = len(m.pools)
    row["bef_n_processes"] = len(m.processes)
    row["bef_n_message_flows"] = len(m.message_flows)
    row["bef_n_lanes"] = len(m.lanes)

    # ── syntactic: the 16 `Sytax_Mistakes` metrics, then the class's own
    #    aggregator over the error counts those checks just produced ──
    syn = B["Syn"](load(str(path)))
    per_metric = syn.syntax_check_metric_results()
    row["bef_syn_score"] = syn.compute_syntax_score()
    for metric in B["SynM"]:
        row[f"bef_syn_{metric.value}"] = per_metric.get(metric.value)

    # ── pragmatic: 15 banded scores, plus the raw measurements behind them,
    #    plus the class's own overall and per-group aggregators ──
    prag = B["Prag"](load(str(path)))
    banded = prag.pragmatic_quality_check_metric_results()
    row["bef_prag_score"] = prag.compute_total_pragmatic_quality_score()
    groups = prag.compute_score_per_group()
    for group in B["PragG"]:
        row[f"bef_prag_group_{group.value}"] = groups.get(group.value)
    # `metric_scores` holds the *raw* measurements; `get_rank` bands them on
    # top, so both come off the same instance.
    raw = {k: v for g in prag.metric_scores.values() for k, v in g.items()}
    for metric in B["PragM"]:
        row[f"bef_prag_{metric.value}"] = raw.get(metric.value)
        row[f"bef_prag_{metric.value}_score"] = banded.get(metric.value)

    # ── semantic: each model against itself ──
    sem = B["Sem"](model=load(str(path)), reference_model=load(str(path)),
                   lang=B["Language"].ENGLISH)
    sem.semantic_quality_check_metric_results()
    # The aggregators recompute the whole similarity pass; stub the three
    # computation steps now that they have run, so the class aggregates the
    # results it already holds instead of spending another ~30 s on them.
    _noop = lambda *a, **k: None                                  # noqa: E731
    sem.natural_language_similarity = _noop
    sem.strucural_similarity = _noop
    sem.behavioural_similarity = _noop
    _fix_common_percentage(sem, path, path, B, row)
    sem_m = sem.semantic_quality_check_metric_results()
    row["bef_sem_score"] = sem.semantic_quality_check()
    sem_g = sem.semantic_quality_check_detailed()
    for group in B["SemG"]:
        row[f"bef_sem_group_{group.value}"] = sem_g.get(group.value)
    for metric in B["SemM"]:
        v = sem_m.get(metric.value)
        row[f"bef_sem_{metric.value}"] = None if v is None else float(v)

    row["bef_error"] = ""
    return row


def verify(path: Path, B: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    """Re-score one item with fresh objects and report any disagreement.

    The naive way is what the module docstring avoids for cost: a fresh
    instance per public method, each recomputing from scratch. Compare every
    individual metric as well as all group and dimension aggregates.
    """
    load, problems = B["load"], []

    def _cmp(label: str, got: Any, want: Any) -> None:
        if want is None and got is None:
            return
        if got is None or want is None or abs(float(got) - float(want)) > 1e-9:
            problems.append(f"{path.stem} {label}: single-pass {got} vs naive {want}")

    _cmp("syn_score", row["bef_syn_score"],
         B["Syn"](load(str(path))).syntax_check())
    naive_syn = B["Syn"](load(str(path))).syntax_check_metric_results()
    for metric in B["SynM"]:
        _cmp(f"syn_{metric.value}", row[f"bef_syn_{metric.value}"],
             naive_syn.get(metric.value))

    _cmp("prag_score", row["bef_prag_score"],
         B["Prag"](load(str(path))).pragmatic_quality_check())
    naive_groups = B["Prag"](load(str(path))).pragmatic_quality_check_detailed()
    for group in B["PragG"]:
        _cmp(f"prag_group_{group.value}", row[f"bef_prag_group_{group.value}"],
             naive_groups.get(group.value))
    naive_prag = B["Prag"](load(str(path)))
    naive_banded = naive_prag.pragmatic_quality_check_metric_results()
    naive_raw = {k: v for group in naive_prag.metric_scores.values()
                 for k, v in group.items()}
    for metric in B["PragM"]:
        _cmp(f"prag_{metric.value}", row[f"bef_prag_{metric.value}"],
             naive_raw.get(metric.value))
        _cmp(f"prag_{metric.value}_score",
             row[f"bef_prag_{metric.value}_score"],
             naive_banded.get(metric.value))

    # The naive path returns their unfixed common_percentage, so it is held
    # against the `_bef4llm_raw` column, and the two aggregates it feeds (the
    # dimension score and the graph-structure group) are not compared.
    fixed_key = B["SemM"].common_percentage.value
    fixed_group = B["SemG"].graph_structure.value
    naive_sem_g = B["Sem"](model=load(str(path)), reference_model=load(str(path)),
                           lang=B["Language"].ENGLISH).semantic_quality_check_detailed()
    for group in B["SemG"]:
        if group.value == fixed_group:
            continue
        _cmp(f"sem_group_{group.value}", row[f"bef_sem_group_{group.value}"],
             naive_sem_g.get(group.value))
    naive_sem = B["Sem"](model=load(str(path)), reference_model=load(str(path)),
                         lang=B["Language"].ENGLISH).semantic_quality_check_metric_results()
    for metric in B["SemM"]:
        column = f"bef_sem_{metric.value}"
        if metric.value == fixed_key:
            column += "_bef4llm_raw"
        _cmp(f"sem_{metric.value}", row[column], naive_sem.get(metric.value))
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score the PMo BPMN models with BEF4LLM's own implementation.")
    ap.add_argument("--bpmn-dir", default=str(DEFAULT_BPMN_DIR))
    ap.add_argument("--bef4llm-src", default=str(DEFAULT_BEF_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--pairs", metavar="LIST.csv",
                    help="score the semantic dimension over the pairs in this "
                         "file (score_pmo_pairs.py's *_list.csv) instead of "
                         "scoring every model against itself")
    ap.add_argument("--generated-dir",
                    help="with --pairs: read the generated side from this "
                         "directory instead of --bpmn-dir (the basic-variation "
                         "pairing has both sides under the same item id)")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="re-score the first N items with fresh objects and "
                         "assert all metrics and aggregates match")
    args = ap.parse_args()

    import pandas as pd

    B = _load_bef4llm(args.bef4llm_src)

    if args.pairs:
        out = Path(args.out)
        if out == DEFAULT_OUT:            # the self-match default would mislead
            out = out.with_name("pmo_semantic_pairs_bef4llm.csv")
        run_pairs(Path(args.pairs), Path(args.bpmn_dir), out, B,
                  args.generated_dir)
        return

    paths = sorted(Path(args.bpmn_dir).glob("*.bpmn"), key=lambda p: p.stem)
    print(f"Scoring {len(paths)} BPMN models with BEF4LLM's own code", flush=True)

    rows: List[Dict[str, Any]] = []
    failed, problems = 0, []
    for i, p in enumerate(paths, 1):
        try:
            row = score_one(p, B)
            rows.append(row)
            print(f"  [{i:>2}/{len(paths)}] {p.name}  "
                  f"syn {row['bef_syn_score']:.4f}  prag {row['bef_prag_score']:.4f}  "
                  f"sem {row['bef_sem_score']:.4f}", flush=True)
            if i <= args.verify:
                problems.extend(verify(p, B, row))
        except Exception as exc:  # noqa: BLE001 — record, never abort the sweep
            failed += 1
            rows.append({"item_id": p.stem, "bpmn_path": str(p),
                         "bef_error": f"{type(exc).__name__}: {exc}"})
            print(f"  ! {p.name}: {exc}", flush=True)
            traceback.print_exc(limit=2)

    if args.verify:
        print(f"\n[verify] {args.verify} item(s) re-scored the naive way: "
              + ("all metrics and aggregates match" if not problems else "MISMATCH"), flush=True)
        for problem in problems:
            print(f"  ! {problem}", flush=True)
        if problems:
            raise RuntimeError("BEF4LLM verification disagreed: " + "; ".join(problems))

    df = pd.DataFrame(rows)
    lead = ["item_id", "bef_xsd_valid", "bef_syn_score", "bef_prag_score",
            "bef_sem_score", "bef_error"]
    df = df[[c for c in lead if c in df.columns]
            + [c for c in df.columns if c not in lead]]
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} columns → {args.out}"
          + (f"  ({failed} failed)" if failed else ""))


if __name__ == "__main__":
    main()
