#!/usr/bin/env python3
"""Build a self-contained per-entry HTML comparison of PMo quality CSVs."""
from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "pmo_quality_comparison_20260816"


MAPPINGS = [
    # section, label, original BEF4LLM column, local pipeline column, comparability
    ("Headline scores", "Syntactic quality", "bef_syn_score", "syn_bef_score", "Approximate: supplied-code set vs paper-defined port"),
    ("Headline scores", "Pragmatic quality", "bef_prag_score", "prag_score", "Comparable dimension; implementation details differ"),
    ("Headline scores", "Semantic quality", "bef_sem_score", "sem_score", "Self-comparison; original code is not reflexive"),
    ("Syntactic metrics", "Existence of start event", "bef_syn_existence_start_event", "syn_bef_existence_start_event", "Mapped"),
    ("Syntactic metrics", "Existence of end event", "bef_syn_existence_end_event", "syn_bef_existence_end_event", "Mapped"),
    ("Syntactic metrics", "One start event per process", "bef_syn_one_start_event", "syn_bef_one_start_event_per_process", "Approximate denominator semantics"),
    ("Syntactic metrics", "One end event per process", "bef_syn_one_end_event", "syn_bef_one_end_event_per_process", "Approximate denominator semantics"),
    ("Syntactic metrics", "Start-event degree", "bef_syn_start_event_in_out_degree", "syn_bef_start_event_in_out_degree", "Approximate: Boolean vs ratio"),
    ("Syntactic metrics", "End-event degree", "bef_syn_end_event_in_out_degree", "syn_bef_end_event_in_out_degree", "Approximate: Boolean vs ratio"),
    ("Syntactic metrics", "Labeled tasks", "bef_syn_labeled_tasks", "syn_bef_labeled_tasks", "Mapped"),
    ("Syntactic metrics", "Task in/out degree", "bef_syn_tasks_in_outdegree", "syn_bef_task_in_out_degree", "Mapped"),
    ("Syntactic metrics", "Intermediate-event degree", "bef_syn_intermediate_event_in_out_degree", "syn_bef_intermediate_event_in_out_degree", "Approximate event classification"),
    ("Syntactic metrics", "Gateway in/out degree", "bef_syn_gateway_in_out_degree", "syn_bef_gateway_in_out_degree", "Same predicate and denominator"),
    ("Syntactic metrics", "Connected nodes", "bef_syn_connected_nodes", "syn_extra_connected_nodes", "Mapped; outside paper score in port"),
    ("Syntactic metrics", "Event/gateway predecessor-successor", "bef_syn_event_gateway_predecessor_successor_wrong", "syn_extra_event_gateway_predecessor_successor", "Inverse/approximate definitions"),
    ("Syntactic metrics", "One process per pool", "bef_syn_one_process_in_pool", "syn_bef_one_process_per_pool", "Mapped"),
    ("Syntactic metrics", "Wrong sequence flow", "bef_syn_wrong_sequence_flow", None, "Unavailable in DOT port"),
    ("Syntactic metrics", "Wrong message flow", "bef_syn_wrong_message_flow", None, "Unavailable in DOT port"),
    ("Pragmatic groups", "Size", "bef_prag_group_size", "prag_group_size_score", "Mapped"),
    ("Pragmatic groups", "Density", "bef_prag_group_density", "prag_group_density_score", "Mapped"),
    ("Pragmatic groups", "Connector interplay", "bef_prag_group_connector_interplay", "prag_group_connector_interplay_score", "Mapped"),
    ("Pragmatic groups", "Partitionability", "bef_prag_group_partionability", "prag_group_partitionability_score", "Mapped"),
    ("Pragmatic groups", "Cyclicity", "bef_prag_group_cyclicity", "prag_group_cyclicity_score", "Mapped"),
    ("Pragmatic groups", "Concurrency", "bef_prag_group_concurrency", "prag_group_concurrency_score", "Mapped"),
    ("Pragmatic groups", "Other", "bef_prag_group_other_metrics", "prag_group_other_score", "Mapped; metric membership differs"),
    ("Pragmatic raw metrics", "Total gateways", "bef_prag_total_numbers_of_gateways", "prag_tng", "Mapped"),
    ("Pragmatic raw metrics", "Total nodes", "bef_prag_total_numbers_of_nodes", "prag_tnn", "Mapped"),
    ("Pragmatic raw metrics", "Total sequence flows", "bef_prag_total_numbers_of_sequence_flows", "prag_tnsf", "Mapped"),
    ("Pragmatic raw metrics", "Total message flows", "bef_prag_total_numbers_of_message_flows", None, "Unavailable in DOT"),
    ("Pragmatic raw metrics", "Diameter", "bef_prag_diameter", "prag_diameter", "Mapped"),
    ("Pragmatic raw metrics", "Density", "bef_prag_density", "prag_density", "Mapped"),
    ("Pragmatic raw metrics", "Average gateway degree", "bef_prag_average_gateway_degree", "prag_agd", "Mapped"),
    ("Pragmatic raw metrics", "Connectivity coefficient", "bef_prag_connectivity_coefficient", "prag_cnc", "Mapped"),
    ("Pragmatic raw metrics", "Gateway heterogeneity", "bef_prag_gateway_heterogeneity", "prag_gh", "Mapped"),
    ("Pragmatic raw metrics", "Control-flow complexity", "bef_prag_control_flow_complexity", "prag_cfc", "Mapped"),
    ("Pragmatic raw metrics", "Sequentiality", "bef_prag_sequentiality", "prag_sequentiality", "Mapped"),
    ("Pragmatic raw metrics", "Separability", "bef_prag_seperatibility", "prag_separability", "Mapped"),
    ("Pragmatic raw metrics", "Depth", "bef_prag_depth", "prag_depth", "Mapped"),
    ("Pragmatic raw metrics", "Token split", "bef_prag_token_split", "prag_token_split", "Mapped"),
    ("Pragmatic raw metrics", "Cross connectivity", "bef_prag_cross_connectivity", None, "Omitted from paper-defined DOT port"),
    ("Semantic groups", "Natural language", "bef_sem_group_natural_language", "sem_group_natural_language_score", "Mapped"),
    ("Semantic groups", "Graph structure", "bef_sem_group_graph_structure", "sem_group_graph_structure_score", "Mapped"),
    ("Semantic metrics", "Label similarity: syntactic", "bef_sem_label_sim_syntactic", "sem_label_sim_syntactic", "Mapped; formula corrections in port"),
    ("Semantic metrics", "Label similarity: semantic", "bef_sem_label_sim_semantic", "sem_label_sim_semantic", "Mapped; formula corrections in port"),
    ("Semantic metrics", "Graph edit distance", "bef_sem_graph_edit_distance", "sem_graph_edit_distance", "Mapped"),
    ("Semantic metrics", "Common nodes/edges", "bef_sem_common_percentage", "sem_common_nodes_edges", "Mapped"),
]


def esc(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return html.escape(str(value))


def number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    n = number(value)
    if n is None:
        return esc(value)
    if n.is_integer():
        return str(int(n))
    return f"{n:.6f}".rstrip("0").rstrip(".")


def mean(df: pd.DataFrame, column: str) -> str:
    return fmt(pd.to_numeric(df[column], errors="coerce").mean()) if column in df else "—"


def delta(original: Any, current: Any) -> tuple[str, str]:
    left, right = number(original), number(current)
    if left is None or right is None:
        return "—", ""
    value = right - left
    cls = "positive" if value > 1e-12 else "negative" if value < -1e-12 else "neutral"
    return f"{value:+.6f}".rstrip("0").rstrip("."), cls


def field_group(column: str) -> str:
    if "_syn_" in column or column.startswith("syn_"):
        return "Syntactic"
    if "_prag_" in column or column.startswith("prag_"):
        return "Pragmatic"
    if "_sem_" in column or column.startswith("sem_"):
        return "Semantic"
    if "valid" in column or column.startswith("val_"):
        return "Validity"
    if column.startswith(("bef_n_", "syn_n_")):
        return "Counts and structure"
    return "Metadata and other fields"


def raw_tables(row: pd.Series, columns: Iterable[str], source: str) -> str:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        grouped.setdefault(field_group(column), []).append(column)
    chunks = []
    for group, names in grouped.items():
        rows = "".join(
            f"<tr><th>{html.escape(name)}</th><td>{fmt(row.get(name))}</td></tr>"
            for name in names
        )
        chunks.append(
            f'<details class="raw-group"><summary>{html.escape(group)} ({len(names)})</summary>'
            f'<div class="table-wrap"><table class="raw"><tbody>{rows}</tbody></table></div></details>'
        )
    return f'<section class="raw-source"><h4>{html.escape(source)}</h4>{"".join(chunks)}</section>'


def mapped_table(original: pd.Series, current: pd.Series) -> str:
    sections = []
    section_names = list(dict.fromkeys(mapping[0] for mapping in MAPPINGS))
    for section in section_names:
        rows = []
        for _, label, original_col, current_col, note in (m for m in MAPPINGS if m[0] == section):
            original_value = original.get(original_col) if original_col else None
            current_value = current.get(current_col) if current_col else None
            diff, cls = delta(original_value, current_value)
            rows.append(
                f"<tr><th>{html.escape(label)}</th><td>{fmt(original_value)}</td>"
                f"<td>{fmt(current_value)}</td><td class=\"{cls}\">{diff}</td>"
                f"<td class=\"note\">{html.escape(note)}</td></tr>"
            )
        sections.append(
            f'<tr class="section"><th colspan="5">{html.escape(section)}</th></tr>' + "".join(rows)
        )
    return (
        '<div class="table-wrap"><table class="mapped"><thead><tr><th>Metric</th>'
        '<th>Original BEF4LLM</th><th>My pipeline</th><th>Δ (mine − original)</th>'
        '<th>Comparability</th></tr></thead><tbody>' + "".join(sections) + "</tbody></table></div>"
    )


def build_html(original: pd.DataFrame, current: pd.DataFrame, original_path: Path,
               current_path: Path) -> str:
    original = original.set_index("item_id", drop=False)
    current = current.set_index("item_id", drop=False)
    ids = sorted(current.index)
    original_fields = [c for c in original.columns if c != "item_id"]
    current_fields = [c for c in current.columns if c != "item_id"]

    overview_rows = []
    for label, original_col, current_col in [
        ("Syntactic", "bef_syn_score", "syn_bef_score"),
        ("Pragmatic", "bef_prag_score", "prag_score"),
        ("Semantic", "bef_sem_score", "sem_score"),
    ]:
        left, right = mean(original, original_col), mean(current, current_col)
        diff, cls = delta(float(left), float(right))
        overview_rows.append(
            f"<tr><th>{label}</th><td>{left}</td><td>{right}</td><td class=\"{cls}\">{diff}</td></tr>"
        )

    cards = []
    for item_id in ids:
        old, new = original.loc[item_id], current.loc[item_id]
        headline = []
        for label, old_col, new_col in [
            ("Syntactic", "bef_syn_score", "syn_bef_score"),
            ("Pragmatic", "bef_prag_score", "prag_score"),
            ("Semantic", "bef_sem_score", "sem_score"),
        ]:
            d, cls = delta(old.get(old_col), new.get(new_col))
            headline.append(
                f'<div class="score"><span>{label}</span><b>{fmt(old.get(old_col))} → '
                f'{fmt(new.get(new_col))}</b><small class="{cls}">Δ {d}</small></div>'
            )
        cards.append(
            f'<article class="entry" id="item-{esc(item_id)}" data-id="{esc(item_id)}">'
            f'<details><summary><strong>PMo {esc(item_id)}</strong><div class="scores">{"".join(headline)}</div></summary>'
            '<div class="entry-body"><h3>Mapped comparison</h3>' + mapped_table(old, new) +
            '<details class="all-fields"><summary>All calculated fields from both pipelines</summary>'
            '<p class="note-block">These tables preserve the source column names and values without forcing non-equivalent metrics into a shared definition.</p>'
            '<div class="raw-grid">' + raw_tables(old, original_fields, "Original BEF4LLM pipeline") +
            raw_tables(new, current_fields, "My DOT pipeline") + '</div></details></div></details></article>'
        )

    options = "".join(f'<option value="{esc(i)}">PMo {esc(i)}</option>' for i in ids)
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PMo Quality Comparison</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--ink:#17212b;--muted:#66727e;--line:#d9e0e6;--blue:#175cd3;--green:#087443;--red:#b42318;--soft:#eef4ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:auto;padding:32px 24px 64px}} h1{{font-size:30px;margin:0 0 6px}} h2{{margin-top:34px}} h3{{margin:0 0 12px}}
.subtitle,.note-block{{color:var(--muted)}} .panel,.entry{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 1px 2px #0000000a}}
.panel{{padding:20px;margin:18px 0}} .meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}} code{{word-break:break-all}}
.toolbar{{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;background:#f4f6f8ee;backdrop-filter:blur(8px);padding:12px 0}}
select,button{{border:1px solid #aeb8c2;border-radius:8px;background:white;padding:9px 12px;color:var(--ink)}} button{{cursor:pointer}} .count{{margin-left:auto;color:var(--muted)}}
.entry{{margin:12px 0;overflow:hidden}} .entry>details>summary{{display:flex;align-items:center;gap:22px;padding:16px 18px;cursor:pointer;list-style:none}} .entry>details>summary::-webkit-details-marker{{display:none}}
.entry>details>summary>strong{{font-size:18px;min-width:78px;color:var(--blue)}} .entry-body{{border-top:1px solid var(--line);padding:20px}}
.scores{{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:10px;flex:1}} .score{{background:var(--soft);border-radius:8px;padding:8px 10px;display:grid;grid-template-columns:80px 1fr auto;gap:8px}}
.score span,.score small{{color:var(--muted)}} .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:8px}} table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}} thead th{{position:sticky;top:0;background:#edf2f7;z-index:1;white-space:nowrap}} tbody tr:last-child th,tbody tr:last-child td{{border-bottom:0}}
.mapped .section th{{background:#f7f9fb;color:var(--blue);font-size:13px;text-transform:uppercase;letter-spacing:.04em}} .mapped td:nth-child(2),.mapped td:nth-child(3),.mapped td:nth-child(4){{font-variant-numeric:tabular-nums;white-space:nowrap}} .note{{color:var(--muted);min-width:260px}}
.positive{{color:var(--green)!important}} .negative{{color:var(--red)!important}} .neutral{{color:var(--muted)!important}} .all-fields{{margin-top:18px}} .all-fields>summary,.raw-group>summary{{cursor:pointer;font-weight:650;padding:10px 0}}
.raw-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .raw-source h4{{font-size:16px}} .raw th{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;word-break:break-word;width:67%}} .raw td{{font-variant-numeric:tabular-nums}}
.hidden{{display:none}} @media(max-width:900px){{main{{padding:18px 10px}}.scores,.raw-grid{{grid-template-columns:1fr}}.entry>details>summary{{align-items:flex-start;flex-direction:column}}.score{{grid-template-columns:70px 1fr auto}}}}
@media print{{.toolbar{{display:none}}.entry{{break-inside:avoid}}body{{background:white}}main{{max-width:none;padding:0}}}}
</style></head><body><main>
<h1>PMo Quality Comparison</h1><p class="subtitle">Original BEF4LLM BPMN pipeline vs this repository's Graphviz/DOT pipeline · generated {esc(generated)}</p>
<section class="panel"><h2>Scope and interpretation</h2><div class="meta">
<div><b>Entries</b><br>{len(ids)} matched PMo models</div><div><b>Original fields</b><br>{len(original.columns)} columns</div><div><b>My pipeline fields</b><br>{len(current.columns)} columns</div>
<div><b>Original input</b><br>BPMN 2.0 XML</div><div><b>My input</b><br>Graphviz/DOT</div><div><b>Validity</b><br>Original XSD result only; DOT validity is not implemented</div></div>
<p>The report compares the same process IDs in two serializations. A delta is descriptive, not automatically an accuracy improvement: the original supplied code and the paper-defined port differ in metric definitions and aggregation. Semantic values are self-comparisons; the port is reflexive (1.0), while the original implementation is not.</p>
<p><b>Original CSV:</b> <code>{esc(original_path)}</code><br><b>My pipeline CSV:</b> <code>{esc(current_path)}</code></p></section>
<section class="panel"><h2>Mean headline scores</h2><div class="table-wrap"><table><thead><tr><th>Dimension</th><th>Original BEF4LLM</th><th>My pipeline</th><th>Δ</th></tr></thead><tbody>{''.join(overview_rows)}</tbody></table></div></section>
<div class="toolbar"><label for="entry-filter"><b>Show entry</b></label><select id="entry-filter"><option value="">All PMo entries</option>{options}</select><button id="expand">Expand visible</button><button id="collapse">Collapse all</button><span class="count" id="count">{len(ids)} entries</span></div>
<section id="entries">{''.join(cards)}</section>
</main><script>
const entries=[...document.querySelectorAll('.entry')], filter=document.getElementById('entry-filter'), count=document.getElementById('count');
function applyFilter(){{let n=0;entries.forEach(e=>{{const show=!filter.value||e.dataset.id===filter.value;e.classList.toggle('hidden',!show);if(show)n++;}});count.textContent=n+' '+(n===1?'entry':'entries');if(filter.value){{const d=document.querySelector('#item-'+CSS.escape(filter.value)+' > details');d.open=true;}}}}
filter.addEventListener('change',applyFilter);document.getElementById('expand').onclick=()=>entries.filter(e=>!e.classList.contains('hidden')).forEach(e=>e.querySelector(':scope > details').open=true);document.getElementById('collapse').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=false);
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original", type=Path,
        default=DEFAULT_OUTPUT_DIR / "original_pipeline_quality_metrics.csv",
        help="CSV produced by the original BEF4LLM pipeline",
    )
    parser.add_argument(
        "--current", type=Path,
        default=DEFAULT_OUTPUT_DIR / "my_pipeline_quality_metrics.csv",
        help="CSV produced by this repository's DOT pipeline",
    )
    parser.add_argument(
        "--html", type=Path,
        default=DEFAULT_OUTPUT_DIR / "comparison.html",
        help="self-contained HTML report to write",
    )
    parser.add_argument(
        "--merged", type=Path,
        default=DEFAULT_OUTPUT_DIR / "all_metrics_merged.csv",
        help="merged machine-readable CSV to write",
    )
    args = parser.parse_args()

    original = pd.read_csv(args.original, dtype={"item_id": str})
    current = pd.read_csv(args.current, dtype={"item_id": str})
    original["item_id"] = original["item_id"].str.zfill(2)
    current["item_id"] = current["item_id"].str.zfill(2)
    if original["item_id"].duplicated().any() or current["item_id"].duplicated().any():
        raise ValueError("Duplicate item_id in an input CSV")
    if set(original["item_id"]) != set(current["item_id"]):
        raise ValueError("The input CSVs do not contain the same PMo IDs")

    if args.merged:
        merged = current.merge(original, on="item_id", how="inner", validate="one_to_one")
        args.merged.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.merged, index=False)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(build_html(original, current, args.original.resolve(), args.current.resolve()), encoding="utf-8")
    print(f"Wrote {len(current)} entries to {args.html}")


if __name__ == "__main__":
    main()
