"""
quality — model quality metrics, one module per dimension
============================================================

    validity.py    Validity (DOT check)      — EMPTY, to be defined
    syntactic.py   Syntactic quality         — 11 checks + 4 size metrics
    semantic.py    Semantic quality          — 5 metrics (BEF4LLM suite),
                                               needs the ground truth; metric 4 is
                                               networkx' graph edit distance
    pragmatic.py   Pragmatic quality         — 14 metrics (BEF4LLM suite)

    syntax_rules.py  BEF4LLM's own syntactic rule catalogue (15 rules scored as
                     error ratios), merged into the syntactic dimension as
                     `syn_bef_*`
    graph.py       shared DOT parsing + BPMN element classification
    normalize.py   shared BEF4LLM banding + threshold constants
    textsim.py     label-text primitives for the semantic dimension
                   (tokeniser, stopwords, Porter stemmer, Levenshtein,
                   optional WordNet synonyms)
    score.py       runs the dimensions over one model or a whole run

Each dimension exposes `evaluate(...) -> Dict[str, Any]`, a flat dict of
prefixed columns (`syn_`, `sem_`, `prag_`, `val_`). An empty dict means the
dimension contributes no columns — which is exactly what the unimplemented
validity dimension returns today, so nothing about it is implied or faked in
results.csv.

Scoring runs as a separate pass over a finished run directory, so metrics can
be changed and re-applied without re-generating anything:

    python score_run.py runs/<run_id>

── Precision policy ─────────────────────────────────────────────────────────
Metric values are stored at **full float precision**; rounding is a
presentation concern and belongs to whatever prints or plots them
(`score_pmo_dataset.py`'s report, the HTML pages, a thesis table), never to the
stored column.

This was not always so: the dimensions used to round ratios to 6 and scores to
4 decimals on the way into the dict. That silently made every stored value
differ from its own definition in the 5th decimal, and comparing the columns
against another implementation then reported the *rounding* as disagreement —
AGD, density, CNC and the task-degree metric all looked as though they diverged
from BEF4LLM's implementation when they in fact agree exactly on all 55 PMo
models. Keep values exact; round when you display them.
"""
from .score import score_generation, score_model, score_run, scored_frame

__all__ = ["score_generation", "score_model", "score_run", "scored_frame"]
