"""The results page and the two aggregations behind it.

Builds from the synthetic run directory in `conftest.py` (`fake_run`/`scored`),
so the suite never depends on an API call having happened.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

import aggregate as agg
import results_report


# ── aggregation ───────────────────────────────────────────────────────────────
def test_by_model_pools_the_strategies(scored):
    frame = agg.by_model(scored)
    assert list(frame["model_key"]) == ["alpha", "beta"]
    assert frame.set_index("model_key").loc["alpha", "n_calls"] == 2


def test_by_strategy_pools_the_models(scored):
    frame = agg.by_strategy(scored)
    assert list(frame["strategy"]) == ["few_shot", "zero_shot"]
    assert frame.set_index("strategy").loc["zero_shot", "n_calls"] == 2


def test_every_aggregate_row_carries_its_sample_size(scored):
    """A mean without n and the parse rate is not readable — the rule the
    report is built on."""
    for frame in (agg.by_model(scored), agg.by_strategy(scored)):
        for column in ("n_calls", "n_valid", "n_scored", "parse_rate"):
            assert column in frame.columns
            assert frame[column].notna().all()


def test_a_failed_call_lowers_the_parse_rate_of_its_row(scored):
    beta = agg.by_model(scored).set_index("model_key").loc["beta"]
    assert beta["n_calls"] == 2
    assert beta["parse_rate"] == pytest.approx(0.5)


def test_tokens_get_a_median_next_to_the_mean(scored):
    """Reasoning models skew the distribution; the mean alone would mislead."""
    frame = agg.by_model(scored)
    assert "total_tokens_mean" in frame.columns
    assert "total_tokens_median" in frame.columns
    assert "cost_usd_sum" in frame.columns


def test_an_empty_error_string_is_not_counted_as_a_failure(scored):
    """In JSONL "no error" is "", not NaN — notna() would call every
    successful call a failure."""
    assert agg.totals(scored)["n_errors"] == 1        # only the one that really failed
    by_model = agg.by_model(scored).set_index("model_key")
    assert by_model.loc["alpha", "n_errors"] == 0


def test_pragmatic_quality_is_declared_directionless():
    directions = {column: direction for column, _label, direction in agg.QUALITY_METRICS}
    assert directions["prag_score"] == "none"
    assert directions["syn_bef_score"] == "up"


def test_only_the_befllm_syntactic_score_is_a_headline():
    """Decided 2026-08-18. The dictated checks stay in the data as diagnosis,
    but two syntactic headlines would mean two rankings of the strategies."""
    columns = [column for column, _label, _direction in agg.QUALITY_METRICS]
    assert "syn_bef_score" in columns
    assert "syn_score" not in columns


def test_the_dictated_checks_are_still_recorded(scored):
    """Removed from the report, not from the run: they name which rule broke."""
    from quality import syntactic as syn_module
    assert "syn_score" in scored.columns
    for check in syn_module.CHECKS:
        assert f"syn_{check}" in scored.columns


# ── the page ──────────────────────────────────────────────────────────────────
def test_page_has_both_views_and_all_three_metric_blocks(fake_run, scored):
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    for view in ("model", "strategy"):
        for block in ("quality", "cost", "latency"):
            assert f'data-view="{view}" data-tab="{block}"' in html
    assert html.count('<section class="panel"') == 6


def test_the_metric_selector_is_one_flat_row(fake_run, scored):
    """Every block is its own button and all of them are on screen at all
    times: no second level, and pressing one reveals nothing that was not
    there."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    buttons = re.findall(r'<button type="button" class="tabbtn"[^>]*?data-tab="(\w+)"', html)
    assert buttons == ["quality", "cost", "latency"]
    # Nothing of the old two-level selector survives — no nesting, no hiding.
    for dead in ("data-top=", 'class="tabbtn sub"', 'class="subtabs"', "effgroup"):
        assert dead not in html, dead
    # The landing tab is pressed with JS disabled too.
    assert 'class="tabbtn" data-tab="quality" aria-pressed="true"' in html


def test_the_quality_panel_is_reachable_and_carries_its_metrics(fake_run, scored):
    """The Quality button, its panel, and the spec the fold-out under it reads —
    all three, because any one of them missing makes the block unreadable."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    assert '>Quality</button>' in html
    assert 'data-block="quality"' in html
    assert 'id="paperspec"' in html
    assert 'block === "quality" ? SPEC' in html


def test_an_unfolded_call_shows_only_its_own_tables_subject(fake_run, scored):
    """Opened from the cost or latency table, a call shows that table's
    metrics — not the quality metrics, which belong to a different question."""
    eff = results_report.efficiency_metric_spec()
    assert set(eff) == {"cost", "latency"}
    quality_keys = {m["key"] for part in results_report.paper_metric_spec()
                    for m in part["metrics"]}
    for block, parts in eff.items():
        keys = {m["key"] for part in parts for m in part["metrics"]}
        assert keys, f"{block} shows nothing"
        assert not keys & quality_keys, f"{block} still carries quality metrics"


def test_every_efficiency_metric_is_a_field_the_provider_records():
    """A key nobody records would silently render as 'not computed'. Checked
    against the provider result rather than a run, because that is the contract
    the rows are written from — renaming a field there must break this."""
    import dataclasses

    import providers
    # `call_wall_s` is measured around the provider call by the pipeline itself,
    # so it is a row field without being a provider field (pipeline.py).
    recorded = {f.name for f in dataclasses.fields(providers.GenerationResult)}
    recorded |= {"call_wall_s"}
    for parts in results_report.efficiency_metric_spec().values():
        for part in parts:
            assert part["headline"] in recorded, part["headline"]
            for metric in part["metrics"]:
                assert metric["key"] in recorded, metric["key"]


def test_the_foldout_picks_its_metrics_off_the_table(fake_run, scored):
    """The block comes from the table the row sits in, not from the selector —
    so a fold-out built once stays correct when another tab is opened later."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    assert 'data-block="cost"' in html and 'data-block="latency"' in html
    assert 'var host = row.closest("table")' in html
    assert "host.dataset.block" in html
    assert 'id="effspec"' in html


def test_the_cost_breakdown_adds_up_to_the_recorded_total():
    """Each token class is priced at the rates its own row was billed at, and
    the parts must reproduce the total the vendor charged — otherwise the card
    shows a decomposition of a different number than the one above it."""
    import config as cfg
    model = cfg.model_by_key("google_standard")
    record = {"model_key": "google_standard", "cost_basis": "intro",
              "input_tokens": 1_000_000, "cached_tokens": 500_000,
              "cache_write_tokens": 0, "cache_write_1h_tokens": 0,
              "output_tokens": 200_000}
    record["cost_usd"] = cfg.cost_usd(
        model,
        cfg.TokenUsage(input_tokens=1_000_000, cache_read_tokens=500_000,
                       output_tokens=200_000),
        on_date="2026-08-18")[0]

    results_report.add_cost_breakdown([record])
    parts = sum(record[results_report.COST_FIELD_PREFIX + f] or 0
                for f in results_report.PRICED_TOKEN_FIELDS)
    assert parts == pytest.approx(record["cost_usd"])
    assert record["cost_breakdown_note"] == ""
    # Priced at the intro rates the row recorded, not at today's list rates.
    assert record[results_report.COST_FIELD_PREFIX + "input_tokens"] == pytest.approx(0.75)


def test_the_breakdown_follows_the_basis_the_row_recorded():
    """An off-peak DeepSeek row must be split at half rates — recomputing from
    the model alone would price it at whatever the clock says now."""
    peak = {"model_key": "deepseek_efficient", "cost_basis": "list+peak",
            "input_tokens": 1_000_000, "output_tokens": 0, "cost_usd": 0.44}
    off = dict(peak, cost_basis="list+offpeak", cost_usd=0.22)
    results_report.add_cost_breakdown([peak, off])
    key = results_report.COST_FIELD_PREFIX + "input_tokens"
    assert peak[key] == pytest.approx(0.44)
    assert off[key] == pytest.approx(0.22)


def test_a_model_the_catalogue_no_longer_knows_keeps_its_total():
    """Vendors retire model ids — Gemini 2.5 and deepseek-chat both went. An old
    run must still render, with the total intact and the split declined."""
    record = {"model_key": "gone_forever", "cost_basis": "list",
              "input_tokens": 100, "output_tokens": 100, "cost_usd": 0.5}
    results_report.add_cost_breakdown([record])
    assert record["cost_usd"] == 0.5
    assert record[results_report.COST_FIELD_PREFIX + "input_tokens"] is None
    assert "no rates on file" in record["cost_breakdown_note"]


def test_long_text_values_are_marked_as_prose():
    """`reasoning_note` and `untracked_usage` are sentences. In the narrow value
    column a long string wraps one character per line and renders vertically —
    the prose flag is what moves them to the full width of the card."""
    cost_block = results_report.efficiency_metric_spec()["cost"]
    by_key = {m["key"]: m for part in cost_block for m in part["metrics"]}
    for key in ("reasoning_note", "untracked_usage"):
        assert by_key[key].get("prose") is True, key
    # Numbers must not be: prose styling would left-align them out of the column.
    assert not by_key["cost_usd"].get("prose")


def test_only_classes_with_their_own_rate_carry_a_cost():
    """A subtotal or a subset priced again would double-count the call."""
    tokens = results_report.efficiency_metric_spec()["cost"][0]
    priced = {m["key"] for m in tokens["metrics"] if m.get("cost")}
    assert priced == set(results_report.PRICED_TOKEN_FIELDS)
    unpriced = {m["key"] for m in tokens["metrics"] if not m.get("cost")}
    # reported_input_tokens is the vendor's raw figure, not a class of its own —
    # pricing it would charge the whole prompt a second time.
    assert unpriced == {"reported_input_tokens", "reported_total_tokens",
                        "reported_cache_creation_tokens", "reported_output_tokens",
                        "tool_use_prompt_tokens",
                        "billable_input_tokens", "thinking_tokens"}
    assert tokens["total"] == {"label": "Total", "key": "total_tokens",
                               "cost": "cost_usd"}


def test_the_embedded_dataset_is_valid_json_for_a_browser(fake_run, scored):
    """`json.dumps` writes NaN and Infinity as bare literals. Python reads those
    back happily; no browser will. One of them makes `JSON.parse` throw at page
    load, which aborts the whole script — every fold-out then does nothing at
    all, silently, on a page that looks perfect. Python's own `json.loads` is
    too lenient to catch it, so the constants are rejected explicitly here."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")

    def strict(marker):
        raw = html.split(f'id="{marker}" type="application/json">')[1].split("</script>")[0]
        assert not re.search(r'(?<![\"\w])(NaN|-?Infinity)(?![\"\w])', raw),             f"{marker} contains a non-JSON constant"
        return json.loads(raw, parse_constant=lambda c: pytest.fail(
            f"{marker} contains the non-JSON constant {c!r}"))

    calls = strict("calls")
    strict("paperspec")
    strict("effspec")
    # A fixture row deliberately has unscored metrics — the NaN source.
    assert any(v is None for row in calls for v in row.values())


def test_a_metric_a_run_never_produced_travels_as_null():
    """The replacement is None, not 0.0 — a missing measurement must not become
    a number the reader can average."""
    records = [{"a": float("nan"), "b": float("inf"), "c": 1.5, "d": "text", "e": None}]
    results_report.json_safe(records)
    assert records[0] == {"a": None, "b": None, "c": 1.5, "d": "text", "e": None}


def test_page_is_named_results_html(fake_run, scored):
    assert results_report.build(fake_run, scored=scored).name == "results.html"


def test_missing_metrics_read_as_not_computed(fake_run, scored):
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    assert "not computed" in html


def test_page_is_self_contained(fake_run, scored):
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    assert not re.findall(r'(?:src|href)="https?://', html)
    assert not re.search(r"cdn|googleapis|unpkg|jsdelivr", html, re.IGNORECASE)


def test_per_call_data_travels_with_the_page(fake_run, scored):
    """The aggregates have to be re-derivable without the run directory."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    blob = re.search(r'<script id="calls" type="application/json">(.*?)</script>',
                     html, re.S).group(1)
    assert len(json.loads(blob)) == len(scored)


def test_no_api_key_reaches_the_generated_page(fake_run, scored, monkeypatch):
    """Neither local-file nor environment keys appear in the report."""
    import config as cfg

    local_key = "LOCAL_API_KEY_TEST_SENTINEL_1234567890"
    env_key = "ENV_API_KEY_TEST_SENTINEL_1234567890"
    monkeypatch.setattr(cfg, "_CONFIGURED_KEYS", {"anthropic": local_key})
    monkeypatch.setenv("OPENAI_API_KEY", env_key)
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")

    for secret in (local_key, env_key):
        assert secret not in html
        assert secret[:24] not in html
    assert not re.search(r"sk-(ant-)?(api|proj)[\w-]{16,}", html)


# ── expandable rows ───────────────────────────────────────────────────────────
def panels_of(html):
    """The six panels of the page, keyed by (view, block)."""
    out = {}
    for chunk in html.split('<section class="panel" ')[1:]:
        head = chunk[:chunk.index(">")]
        view = re.search(r'data-view="(\w+)"', head).group(1)
        tab = re.search(r'data-tab="(\w+)"', head).group(1)
        out[(view, tab)] = chunk[:chunk.index("</section>")]
    return out


def test_every_aggregate_row_can_be_expanded(fake_run, scored):
    """Each row head is a button, and each row is followed by its fold-out —
    in the pooled tables and in the per-vendor token tables alike."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    rows = re.findall(r'<tr class="aggrow" data-key="([^"]+)">(.*?)</tr>\s*'
                      r'<tr class="detailrow" hidden>', html, re.S)
    assert rows, "no expandable rows rendered"
    for _key, cells in rows:
        assert 'class="expander" aria-expanded="false"' in cells
    panels = panels_of(html)
    # Quality and latency: one pooled table per view, 2 models / 2 strategies.
    for block in ("quality", "latency"):
        assert panels[("model", block)].count('class="aggrow"') == 2
        assert panels[("strategy", block)].count('class="aggrow"') == 2
    # Tokens: no pooled table in either view — one row per model in its
    # vendor's table, and by strategy the same rows once per strategy.
    assert panels[("model", "cost")].count('class="aggrow"') == 2
    assert panels[("strategy", "cost")].count('class="aggrow"') == 2 * 2


def test_every_row_head_says_it_is_a_mean_over_n_calls(fake_run, scored):
    """The row head carries the sample it averages, in words — in the pooled
    tables and in the per-vendor token tables alike. Without it a folded row
    reads like one measurement of one model."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    heads = re.findall(r'<tr class="aggrow" data-key="[^"]+">\s*<th class="rowhead"[^>]*>'
                       r'<button[^>]*>.*?</button>', html, re.S)
    assert heads, "no row heads rendered"
    for head in heads:
        assert re.search(r'<span class="rowmeta">\((Average from|From) (all \d+ API '
                         r'Calls|1 API Call)\)</span>', head), head
    # The number is this row's own sample, not the run's: it must agree with
    # the `n` cell the row already carries, in every table.
    pairs = re.findall(r'(?:Average from|From) (?:all )?(\d+) API Calls?\)</span>'
                       r'</button></th>\s*<td class="num sample[^"]*"[^>]*>(\d+)</td>', html)
    assert len(pairs) == len(heads), (len(pairs), len(heads))
    for said, shown in pairs:
        assert said == shown, (said, shown)


def test_a_latency_row_is_not_called_an_average(fake_run, scored):
    """Its columns are a mean, a median and a max side by side — "Average from"
    would name the first and misdescribe the other two. Quality and tokens are
    one mean per column and keep the word."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    panels = panels_of(html)
    for view in ("model", "strategy"):
        latency = panels[(view, "latency")]
        assert "Average from" not in latency
        assert latency.count('class="rowmeta">(From all ') == 2
        for block in ("quality", "cost"):
            assert "Average from" in panels[(view, block)]


def test_row_keys_exist_in_the_embedded_dataset(fake_run, scored):
    """The fold-out filters the dataset by these keys — a mismatch would open
    an empty panel."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    calls = json.loads(re.search(r'<script id="calls" type="application/json">(.*?)</script>',
                                 html, re.S).group(1))
    for group, key in re.findall(r'<table class="agg" data-group="(\w+)">(?:(?!</table>).)*?'
                                 r'data-key="([^"]+)"', html, re.S):
        assert any(str(c[group]) == key for c in calls), f"{group}={key} matches no call"


def test_foldout_spans_the_whole_table_width(fake_run, scored):
    """A colspan that disagrees with the real column count breaks the layout.

    Checked per table rather than per block: the tokens block is one table per
    vendor now, and those tables do not share a width — each vendor has as many
    columns as it has token quantities.
    """
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    tables = re.findall(r'<table class="agg[^"]*"[^>]*>(.*?)</table>', html, re.S)
    assert tables
    for table in tables:
        # The second header row carries exactly one cell per column, including
        # the row head — which is what the fold-out has to span.
        width = re.search(r'<tr class="substats">(.*?)</tr>', table, re.S).group(1).count("<th")
        spans = {int(s) for s in re.findall(r'<td colspan="(\d+)">', table)}
        assert spans == {width}, f"fold-out spans {spans}, table is {width} wide"


def test_the_tokens_block_is_per_vendor_tables_in_both_views(fake_run, scored):
    """Neither view carries a pooled tokens table any more: the vendors do not
    report the same quantities, so every row lives in its vendor's table."""
    panels = panels_of(results_report.build(fake_run, scored=scored)
                       .read_text(encoding="utf-8"))
    for view in ("model", "strategy"):
        panel = panels[(view, "cost")]
        assert '<table class="agg" data-group="strategy"' not in panel
        assert '<table class="agg" data-group="model_key"' not in panel
        assert 'class="agg vendortable"' in panel
    # Quality and latency still are one pooled table per view.
    assert '<table class="agg" data-group="strategy"' in panels[("strategy", "quality")]


# ── the fold-out shows the paper's metrics, not the code's ────────────────────
def test_foldout_covers_exactly_the_ported_metric_sets():
    """Derived from the metric modules, so the two cannot drift apart."""
    from quality import pragmatic, semantic, syntax_rules

    spec = {block["title"]: [m["key"] for m in block["metrics"]]
            for block in results_report.paper_metric_spec()}
    syntactic = next(v for k, v in spec.items() if k.startswith("Syntactic"))
    sem = next(v for k, v in spec.items() if k.startswith("Semantic"))
    prag = next(v for k, v in spec.items() if k.startswith("Pragmatic"))

    assert syntactic == [f"syn_bef_{m.key}" for m in syntax_rules.METRICS]
    assert sem == [f"sem_{key}" for key, _g in semantic.METRICS]
    assert prag == [f"prag_{m.key}" for m in pragmatic.METRICS]


def test_group_scores_are_not_shown_because_they_are_not_the_papers():
    """Both metric modules state the group means are this project's addition;
    the paper aggregates each dimension in one step."""
    keys = [m["key"] for block in results_report.paper_metric_spec()
            for m in block["metrics"]]
    assert not [k for k in keys if "_group_" in k]
    # the grouping itself is the paper's and survives as a label
    labels = [m["label"] for block in results_report.paper_metric_spec()
              for m in block["metrics"]]
    assert any("[Size]" in l for l in labels)
    assert any("[natural language]" in l for l in labels)
    # the dimension headlines stay — those are the paper's §4.5 aggregates
    assert {b["headline"] for b in results_report.paper_metric_spec()} == {
        "syn_bef_score", "sem_score", "prag_score"}


def test_foldout_excludes_everything_that_is_not_from_the_paper():
    from quality import syntactic as syn_module

    keys = [m["key"] for block in results_report.paper_metric_spec()
            for m in block["metrics"]]

    # the author's eleven dictated checks are not BEF4LLM's
    for check in syn_module.CHECKS:
        assert f"syn_{check}" not in keys
    # nor is the Graphviz DOT gate, nor code-only extras
    assert not [k for k in keys if k.startswith(("val_", "syn_extra_"))]
    # nor this project's own instrumentation. `prag_diameter_nogw` is the one
    # `_nogw` column that is not instrumentation: since 2026-08-29 it *is* the
    # scored diameter (author's instruction), so it stands in the fold-out as
    # the paper's metric #5 — the column name kept, the reading swapped.
    assert not [k for k in keys
                if "_n_" in k or k.endswith("_truncated")
                or (k.endswith("_nogw") and k != "prag_diameter_nogw")]
    assert "prag_diameter_nogw" in keys and "prag_diameter" not in keys
    # nor the duplicated size block under syn_
    assert not [k for k in keys if k.startswith(("syn_tnn", "syn_tng", "syn_tnsf",
                                                 "syn_diameter", "syn_size"))]
    # nor cost, tokens or timing — named exactly, since Token Split *is* one of
    # the paper's concurrency metrics and a substring match would exclude it
    for generation_field in ("input_tokens", "output_tokens", "total_tokens",
                             "cached_tokens", "billable_input_tokens", "cost_usd",
                             "api_latency_s", "latency_s"):
        assert generation_field not in keys
    assert "prag_token_split" in keys        # …and it must still be there


def test_gateway_metric_is_drawn_by_default_and_can_be_hidden(fake_run, scored, monkeypatch):
    """All scored BEF4LLM metrics appear unless explicitly hidden."""
    hidden = {"syn_bef_gateway_in_out_degree": "Gateway in/out degree"}
    assert results_report.HIDDEN_COLUMNS == set()

    # The published set and the page agree by default.
    published = [m["key"] for block in results_report.paper_metric_spec()
                 for m in block["metrics"]]
    spec = results_report.quality_metric_spec()
    drawn_keys = [m["key"] for block in spec for m in block["metrics"]]
    assert "syn_bef_split_has_matching_join" not in published
    for key in hidden:
        assert key in published
        assert key in drawn_keys
    assert [b["headline"] for b in spec] == [
        b["headline"] for b in
        [results_report.CORRECTNESS_BLOCK] + results_report.paper_metric_spec()]

    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    for label in hidden.values():
        assert label in html

    monkeypatch.setattr(results_report, "HIDDEN_COLUMNS", set(hidden))
    hidden_html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    drawn = json.loads(re.search(
        r'<script id="paperspec" type="application/json">(.*?)</script>',
        hidden_html, re.S).group(1))
    calls = json.loads(re.search(
        r'<script id="calls" type="application/json">(.*?)</script>',
        hidden_html, re.S).group(1))
    for key, label in hidden.items():
        assert label not in hidden_html
        assert key not in [m["key"] for block in drawn for m in block["metrics"]]
        assert key in calls[0]                    # still in the data
    assert calls[0].get("syn_bef_score") is not None


def test_the_spec_travels_with_the_page(fake_run, scored):
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    blob = re.search(r'<script id="paperspec" type="application/json">(.*?)</script>',
                     html, re.S).group(1)
    spec = json.loads(blob)
    assert [b["title"] for b in spec] == [b["title"]
                                          for b in results_report.quality_metric_spec()]
    # The verdict's block goes in front of the paper's set, kept apart on the
    # page because the two are not the same kind of statement; the paper's set
    # follows it unchanged. (While a verdict is in `HIDDEN_COLUMNS` its block is
    # dropped and the fold-out is the paper's set alone.)
    assert spec[0]["title"] == results_report.CORRECTNESS_BLOCK["title"]
    assert [b["title"] for b in spec[1:]] == [b["title"]
                                              for b in results_report.paper_metric_spec()]


def test_every_spec_key_can_actually_be_filled(scored):
    """A key the scorer never produces would render as a permanent
    'not computed' row — a silent hole in the card."""
    produced = set(scored.columns)
    for block in results_report.quality_metric_spec():
        assert block["headline"] in produced
        for metric in block["metrics"]:
            assert metric["key"] in produced, metric["key"]
            for extra in (metric.get("ratio") or []):
                assert extra in produced, extra
            if metric.get("score"):
                assert metric["score"] in produced, metric["score"]


# ── request ids ───────────────────────────────────────────────────────────────
def _reqid_footer(html: str) -> str:
    match = re.search(r'<section class="reqids">.*?</section>', html, re.S)
    return match.group(0) if match else ""


def test_only_vendors_actually_used_are_listed(tmp_path, fake_run, scored):
    """The fixture run uses one vendor; the other four must not appear."""
    footer = _reqid_footer(results_report.build(fake_run, scored=scored)
                           .read_text(encoding="utf-8"))
    assert "Request-IDs" in footer
    for absent in ("Anthropic", "Google", "Mistral", "OpenAI", "DeepSeek"):
        assert absent not in footer, f"{absent} was not used in this run"


def test_trackable_vendors_link_to_their_section(fake_run, scored):
    scored = scored.copy()
    scored["vendor"] = ["anthropic", "openai", "anthropic", "openai"]
    footer = _reqid_footer(results_report.build(fake_run, scored=scored)
                           .read_text(encoding="utf-8"))
    assert '<a href="request_ids.html#anthropic">Anthropic</a>' in footer
    assert '<a href="request_ids.html#openai">OpenAI</a>' in footer

    page = (fake_run / "request_ids.html").read_text(encoding="utf-8")
    assert '<section id="anthropic">' in page and '<section id="openai">' in page


def test_deepseek_links_now_that_it_returns_ids(fake_run, scored):
    """It was listed as untrackable while it was mocked. Since it went live it
    returns a completion id per call, so labelling it 'not trackable' would hide
    data the run holds."""
    scored = scored.copy()
    scored["vendor"] = ["deepseek"] * len(scored)
    footer = _reqid_footer(results_report.build(fake_run, scored=scored)
                           .read_text(encoding="utf-8"))
    assert "DeepSeek" in footer
    assert "not trackable" not in footer
    assert "href=" in footer


def test_a_vendor_without_ids_is_still_named_rather_than_dropped(fake_run, scored):
    """The untrackable path stays exercised: a vendor the report cannot link
    must be listed and marked, never silently omitted."""
    scored = scored.copy()
    scored["vendor"] = ["someunknownvendor"] * len(scored)
    footer = _reqid_footer(results_report.build(fake_run, scored=scored)
                           .read_text(encoding="utf-8"))
    assert "someunknownvendor (not trackable)" in footer


def test_a_call_without_a_recorded_id_says_so(fake_run, scored):
    """An older run, a failed request or the mock provider — never a blank."""
    scored = scored.copy()
    scored["vendor"] = "openai"
    results_report.build(fake_run, scored=scored)
    page = (fake_run / "request_ids.html").read_text(encoding="utf-8")
    assert "not recorded" in page


def test_recorded_ids_reach_the_page(fake_run, scored):
    scored = scored.copy()
    scored["vendor"] = "anthropic"
    scored["request_id"] = [f"req_{i}" for i in range(len(scored))]
    scored["response_id"] = [f"msg_{i}" for i in range(len(scored))]
    results_report.build(fake_run, scored=scored)
    page = (fake_run / "request_ids.html").read_text(encoding="utf-8")
    assert "req_0" in page and "msg_0" in page


def test_provider_captures_the_ids_the_sdk_sets(settings):
    """`_request_id` is attached to the response object by the SDK, not
    declared on the model — so it is read off a live-shaped response."""
    import types
    from anthropic.types import Usage
    from providers import AnthropicProvider, _request_ids

    reply = types.SimpleNamespace(
        id="msg_01ABC", stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text="digraph g { a -> b }")],
        usage=Usage(input_tokens=10, output_tokens=20))
    reply._request_id = "req_01XYZ"
    assert _request_ids(reply) == {"request_id": "req_01XYZ", "response_id": "msg_01ABC"}

    provider = AnthropicProvider(api_key="sk-test-not-used")
    provider._client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: reply))
    import config as cfg
    gen = provider.generate(cfg.model_by_key("anthropic_standard"), None,
                            [{"role": "user", "content": "hi"}], settings)
    assert gen.request_id == "req_01XYZ"
    assert gen.response_id == "msg_01ABC"


def test_a_response_without_ids_yields_empty_strings_not_errors():
    """The mock provider and older SDK paths have neither field."""
    import types
    from providers import _request_ids
    assert _request_ids(types.SimpleNamespace()) == {"request_id": "", "response_id": ""}


def test_builder_can_score_a_run_on_its_own(fake_run):
    """`python results_report.py <run>` must work without a scored frame."""
    out = results_report.build(fake_run)
    assert out.exists() and out.stat().st_size > 0


# ── token usage as the vendor reported it ─────────────────────────────────────
# Two vendor-shaped payloads, because the point of the section is that the two
# do not line up: Anthropic tiers its cache writes and states no total, Gemini
# counts thoughts and breaks the prompt down by modality.
ANTHROPIC_USAGE = {
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 4},
    "cache_creation_input_tokens": 4,
    "cache_read_input_tokens": 0,
    "input_tokens": 154,
    "output_tokens": 754,
    "output_tokens_details": None,
    "server_tool_use": None,
    "service_tier": "standard",
}
GEMINI_USAGE = {
    "candidates_token_count": 581,
    "cached_content_token_count": None,
    "prompt_token_count": 141,
    "prompt_tokens_details": [{"modality": "TEXT", "token_count": 141}],
    "thoughts_token_count": None,
    "total_token_count": 722,
    "traffic_type": None,
}
MISTRAL_USAGE = {
    "prompt_tokens": 10_000,
    "completion_tokens": 2_000,
    "total_tokens": 12_000,
    # The cached share, under all four names the schema gives it.
    "prompt_tokens_details": {"cached_tokens": 6_000},
    "prompt_token_details": {"cached_tokens": 6_000},
    "num_cached_tokens": 6_000,
    "cached_tokens": 6_000,
    # A breakdown of the completion count.
    "completion_tokens_details": {"reasoning_tokens": 800},
    # Billed by duration rather than by token, and null on every text call —
    # kept anyway, because "this vendor states no figure" is itself the fact
    # this section reports.
    "prompt_audio_seconds": None,
    "request_count": 1,
    # Not a measurement: it has its own column on the page.
    "service_tier": "standard",
}


@pytest.fixture
def usage_run(fake_run):
    """The same run, with the usage objects a live run archives beside each
    reply. `beta`'s second call failed, so it has none — the coverage a card
    has to state rather than average over."""
    raw = fake_run / "raw"
    for strategy in ("zero_shot", "few_shot"):
        (raw / f"01__alpha__{strategy}.usage.json").write_text(
            json.dumps(ANTHROPIC_USAGE), encoding="utf-8")
    (raw / "01__beta__zero_shot.usage.json").write_text(
        json.dumps(GEMINI_USAGE), encoding="utf-8")
    return fake_run


def test_a_nested_usage_object_flattens_to_dotted_names():
    flat = results_report.flatten_usage(ANTHROPIC_USAGE)
    assert flat["cache_creation.ephemeral_5m_input_tokens"] == 4
    assert flat["input_tokens"] == 154


def test_a_modality_list_is_keyed_by_its_modality_not_its_position():
    """Gemini sends the prompt breakdown as a list. Keyed by index, the name
    would depend on the order the vendor happened to send in and could not be
    aggregated across calls."""
    flat = results_report.flatten_usage(GEMINI_USAGE)
    assert flat["prompt_tokens_details.TEXT.token_count"] == 141
    assert not any("[0]" in name for name in flat)


def test_a_field_the_vendor_left_empty_is_never_read_as_zero():
    """`thoughts_token_count: null` means "not reported", not "no thinking" —
    averaging it as 0 would invent a measurement the API never made."""
    usage = results_report.usage_metrics(GEMINI_USAGE)
    assert usage["thoughts_token_count"] is None
    assert usage["candidates_token_count"] == 581


def test_only_token_fields_survive_the_archive():
    """The usage object also carries the tier and the routing. Those are not
    token metrics, and the ones worth reporting have their own column already."""
    anthropic = results_report.usage_metrics(ANTHROPIC_USAGE)
    assert "service_tier" not in anthropic          # text, not a count
    assert "server_tool_use" not in anthropic       # empty, and counts no tokens
    gemini = results_report.usage_metrics(GEMINI_USAGE)
    assert "traffic_type" not in gemini
    # Empty *and* named as a token count: kept, because a vendor offering the
    # field and never filling it is the very thing this section reports.
    assert gemini["cached_content_token_count"] is None


def test_every_documented_mistral_token_metric_reaches_the_results_report():
    """Every quantity the Chat usage schema documents reaches the page under
    its own name — the four aliases for the cached prompt share included, which
    pricing counts once but the vendor's own table has to show all of. A field
    the vendor left empty stays too, reported as "not reported": that Mistral
    offers `prompt_audio_seconds` and fills it on no text call is a fact about
    its reporting, and this section is where such facts live."""
    usage = results_report.usage_metrics(MISTRAL_USAGE)
    expected = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details.cached_tokens",
        "prompt_token_details.cached_tokens",
        "num_cached_tokens",
        "cached_tokens",
        "completion_tokens_details.reasoning_tokens",
        "prompt_audio_seconds",
        "request_count",
    }
    assert set(usage) == expected
    assert usage["prompt_audio_seconds"] is None
    # `service_tier` is a name, not a quantity; it has a column of its own.
    assert "service_tier" not in usage

    records = [{
        "vendor": "mistral",
        "model_key": "mistral_standard",
        "model_id": "mistral-large-latest",
        results_report.VENDOR_USAGE_FIELD: usage,
    }]
    # A name the vendor stated but left empty stays in the dataset above; it
    # does not become a column, because a column whose every row reads "not
    # reported" is noise in a table meant for comparing figures. The two halves
    # are deliberately different, so both are pinned here.
    stated_with_a_figure = expected - {"prompt_audio_seconds"}
    columns = results_report.vendor_token_columns(records)
    assert set(columns["mistral"]["native"]) == stated_with_a_figure
    html = results_report._vendor_tables_by_model(records)
    for name in stated_with_a_figure:
        assert name in html


def test_an_unreadable_archive_costs_its_detail_not_the_page(tmp_path):
    broken = tmp_path / "broken.usage.json"
    broken.write_text("{not json", encoding="utf-8")
    assert results_report.read_usage_file(broken) == {}
    assert results_report.read_usage_file(tmp_path / "absent.usage.json") == {}


def test_every_call_finds_the_usage_object_it_archived(usage_run, scored):
    records = scored.to_dict(orient="records")
    found = results_report.attach_vendor_usage(usage_run, records)
    assert found == 3                                # the failed call archived none
    by_call = {(r["model_key"], r["strategy"]): r["vendor_usage"] for r in records}
    assert by_call[("alpha", "zero_shot")]["input_tokens"] == 154
    assert by_call[("beta", "zero_shot")]["candidates_token_count"] == 581
    assert by_call[("beta", "few_shot")] == {}


def test_a_run_without_an_archive_still_renders(fake_run, scored):
    """Runs made before the archive existed, and every mock run, have no usage
    object at all. The vendor's own columns are then impossible, so the classes
    this project records stand in — under a header that says why."""
    records = scored.to_dict(orient="records")
    assert results_report.attach_vendor_usage(fake_run, records) == 0
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    assert "token classes recorded (no usage object archived)" in html
    assert 'class="aggrow"' in html
def test_a_row_states_how_many_of_its_calls_reported(usage_run, scored):
    """`beta` has two calls and one usage object. A mean over what it did
    report is fine; letting it look like it covers both calls is not — which is
    what the `usage` column beside `n` says."""
    records = scored.to_dict(orient="records")
    results_report.attach_vendor_usage(usage_run, records)
    columns = results_report.vendor_token_columns(records)["fixture"]
    rows = {row["model_key"]: row["values"]
            for row in results_report.vendor_token_rows(records, "fixture", columns)}
    assert (rows["beta"]["n_calls"], rows["beta"]["n_usage"]) == (2, 1)
    assert (rows["alpha"]["n_calls"], rows["alpha"]["n_usage"]) == (2, 2)
    # Named by the vendor and never filled: no column at all, rather than a
    # column of zeros.
    assert "thoughts_token_count" not in columns["native"]
def test_the_statistics_pool_the_calls_of_the_model(usage_run, scored):
    records = scored.to_dict(orient="records")
    results_report.attach_vendor_usage(usage_run, records)
    columns = results_report.vendor_token_columns(records)["fixture"]
    rows = {row["model_key"]: row["values"]
            for row in results_report.vendor_token_rows(records, "fixture", columns)}
    key = results_report.NATIVE_PREFIX + "input_tokens"
    assert rows["alpha"]["n_calls"] == 2                      # both its calls
    assert rows["alpha"][key] == pytest.approx(154)           # the mean of the two
def test_no_per_model_usage_cards_are_left_on_the_page(usage_run, scored):
    """The cards under the by-LLM tokens table are gone: what a vendor reported
    is the columns of that vendor's table, and nothing repeats it below."""
    html = results_report.build(usage_run, scored=scored).read_text(encoding="utf-8")
    for dead in ('class="usagecard"', 'class="usagecards"', "Token usage as each vendor",
                 '<table class="native">'):
        assert dead not in html, dead
    # The fold-out still shows a call's own usage object, empty fields and all.
    assert '"not reported"' in html
def test_the_tables_show_every_field_the_vendor_filled(usage_run, scored):
    """Every field a vendor actually reported is a column of its table. A field
    it names but never fills is not: it would be blank in every row."""
    html = results_report.build(usage_run, scored=scored).read_text(encoding="utf-8")
    reported = {**results_report.usage_metrics(ANTHROPIC_USAGE),
                **results_report.usage_metrics(GEMINI_USAGE)}
    for name, value in reported.items():
        header = re.compile(r'<th class="num substat fieldhead[^"]*">'
                            + re.escape(name) + "</th>")
        # A never-filled field keeps its place in the per-call usage object, so
        # the name is still in the embedded data — it just has no column.
        assert bool(header.search(html)) is (value is not None), name
def test_the_unfolded_call_carries_its_own_usage_object(usage_run, scored):
    """Per call the page shows the vendor's object verbatim — built client-side
    from the embedded dataset, so it travels with the page."""
    html = results_report.build(usage_run, scored=scored).read_text(encoding="utf-8")
    calls = json.loads(re.search(r'<script id="calls" type="application/json">(.*?)</script>',
                                 html, re.S).group(1))
    assert all("vendor_usage" in call for call in calls)
    assert any(call["vendor_usage"].get("input_tokens") == 154 for call in calls)
    # Shown under tokens & cost, where the reader is asking what a call cost,
    # and never as a zero for a field the vendor left empty.
    assert 'block === "cost" && call.vendor_usage' in html
    assert '"not reported"' in html


def test_the_comparable_classes_stay_what_cost_is_computed_from():
    """The normalised classes are what makes two vendors comparable at all, and
    stay the basis of every cost figure and of each call's breakdown."""
    columns = [column for column, _label, _direction in agg.COST_METRICS]
    for column in ("input_tokens", "output_tokens", "total_tokens", "cost_usd"):
        assert column in columns
def test_a_token_count_of_zero_prints_as_zero():
    """`num` keeps two decimals below 10, which is right for a quality score and
    wrong for a cache-read count: 0.00 tokens reads as a measurement error."""
    assert results_report.count(0) == "0"
    assert results_report.count(154) == "154"
    assert results_report.count(154.5) == "154.5"
    assert results_report.count(1103) == "1" + results_report.GROUP_SEP + "103"


# ── one token table per vendor ────────────────────────────────────────────────
@pytest.fixture
def vendor_run(tmp_path):
    """Two vendors, two models each, two strategies — the shape the split
    exists for. The two vendors report different quantities: only Anthropic
    fills a 1-hour cache-write class, only Google a reasoning count, and their
    usage objects share not one field name.
    """
    run_dir = tmp_path / "20260102_000000_vendors"
    (run_dir / "generated").mkdir(parents=True)
    (run_dir / "raw").mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "created_at": "2026-01-02T00:00:00", "mode": "live", "n_items": 1,
        "repetitions": 1, "strategies": ["zero_shot", "few_shot"],
        "input_source": "fixture", "settings": {"max_output_tokens": 4096},
        "n_generations_planned": 8, "n_generations_completed": 8,
        "stopped_early": False, "pricing": {"as_of": "fixture"},
    }), encoding="utf-8")

    rows = []
    vendors = (("anthropic", ("claude_a", "claude_b"), ANTHROPIC_USAGE),
               ("google", ("gemini_a", "gemini_b"), GEMINI_USAGE))
    # Two strategies with different prompt lengths, so a mean over the two is
    # not the same number as either of them and the per-strategy tables cannot
    # accidentally agree with the pooled one.
    prompts = {"zero_shot": 100, "few_shot": 200}
    for vendor, models, usage in vendors:
        for model in models:
            for strategy, prompt_tokens in prompts.items():
                tag = f"01__{model}__{strategy}"
                (run_dir / "generated" / f"{tag}.gv").write_text(
                    'digraph G { s [shape=circle label=""]; s -> "do it"; '
                    '"do it" -> e; e [shape=doublecircle label=""] }', encoding="utf-8")
                (run_dir / "raw" / f"{tag}.usage.json").write_text(
                    json.dumps(usage), encoding="utf-8")
                row = {
                    "item_id": "01", "model_key": model, "model_id": f"{model}-1",
                    "vendor": vendor, "strategy": strategy, "repetition": 1,
                    "is_mock": False, "generation_error": None,
                    "ground_truth_path": "", "generated_gv": f"generated/{tag}.gv",
                    "extract_parse_ok": True, "tokens_estimated": False,
                    "input_tokens": prompt_tokens, "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    # The second model of each vendor answers at length, so
                    # a column has a spread and the shading has something to
                    # point at.
                    "output_tokens": 300 if model.endswith("_a") else 500,
                    "billable_input_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens + (300 if model.endswith("_a") else 500),
                    "cost_usd": (0.001 if strategy == "zero_shot" else 0.002)
                                * (1 if model.endswith("_a") else 2),
                    "latency_s": 1.0, "api_latency_s": 1.0, "api_attempts": 1,
                }
                # The two vendor-specific classes, each recorded by one vendor
                # only — a column the other one must not be given.
                if vendor == "anthropic":
                    row["cache_write_1h_tokens"] = 0
                else:
                    row["thinking_tokens"] = 40
                rows.append(row)
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return run_dir


@pytest.fixture
def vendor_scored(vendor_run):
    import quality
    return quality.score_run(vendor_run, verbose=False)


@pytest.fixture
def vendor_records(vendor_run, vendor_scored):
    """The per-call records the page is built from, usage objects attached."""
    records = vendor_scored.where(pd.notna(vendor_scored), None).to_dict(orient="records")
    results_report.attach_vendor_usage(vendor_run, records)
    return records


def test_a_column_appears_only_where_the_vendor_reports_it(vendor_records):
    """The reason the table is split at all: a class one vendor states and the
    other does not must not become a column of blanks in the other's table."""
    columns = results_report.vendor_token_columns(vendor_records)
    anthropic = [key for key, _label, _direction in columns["anthropic"]["project"]]
    google = [key for key, _label, _direction in columns["google"]["project"]]
    assert "cache_write_1h_tokens" in anthropic and "cache_write_1h_tokens" not in google
    assert "thinking_tokens" in google and "thinking_tokens" not in anthropic
    # And each vendor's own field names, which share nothing at all.
    assert "cache_read_input_tokens" in columns["anthropic"]["native"]
    assert "candidates_token_count" in columns["google"]["native"]
    assert not set(columns["anthropic"]["native"]) & set(columns["google"]["native"])


def test_every_model_has_one_row_in_its_vendors_table(vendor_records):
    columns = results_report.vendor_token_columns(vendor_records)
    assert set(columns) == {"anthropic", "google"}
    for vendor, expected in (("anthropic", ["claude_a", "claude_b"]),
                             ("google", ["gemini_a", "gemini_b"])):
        rows = results_report.vendor_token_rows(vendor_records, vendor, columns[vendor])
        assert [row["model_key"] for row in rows] == expected


def test_an_entry_is_the_mean_over_all_of_that_models_calls(vendor_records):
    """Two calls of 100 and 200 prompt tokens: the entry is 150, and `n` says
    it is a mean of two."""
    columns = results_report.vendor_token_columns(vendor_records)
    rows = results_report.vendor_token_rows(vendor_records, "anthropic", columns["anthropic"])
    values = rows[0]["values"]
    assert values["n_calls"] == 2
    assert values["input_tokens"] == pytest.approx(150)
    assert values["total_tokens"] == pytest.approx(450)
    assert values["cost_mean"] == pytest.approx(0.0015)
    assert values["cost_sum"] == pytest.approx(0.003)
    # The vendor's own fields are averaged the same way.
    assert values[results_report.NATIVE_PREFIX + "cache_creation_input_tokens"]         == pytest.approx(4)


def test_a_vendors_own_field_never_overwrites_the_class_of_the_same_name(vendor_records):
    """A vendor names fields as it likes, and some of those names are this
    project's names for something else: OpenAI's `input_tokens` includes its
    cache reads, this project's excludes them. One key for both would print one
    figure in the other's column — and the gap between the two is exactly what
    the table is there to show. The fixture makes them differ on purpose."""
    columns = results_report.vendor_token_columns(vendor_records)
    rows = results_report.vendor_token_rows(vendor_records, "anthropic", columns["anthropic"])
    values = rows[0]["values"]
    assert values["input_tokens"] == pytest.approx(150)                      # the class
    assert values[results_report.NATIVE_PREFIX + "input_tokens"] == pytest.approx(154)


def test_a_per_strategy_entry_is_the_mean_within_that_strategy(vendor_records):
    """The per-strategy tables must not quietly show the pooled figure."""
    columns = results_report.vendor_token_columns(vendor_records)
    for strategy, expected in (("zero_shot", 100), ("few_shot", 200)):
        rows = results_report.vendor_token_rows(
            vendor_records, "anthropic", columns["anthropic"], "strategy", strategy)
        assert rows[0]["values"]["n_calls"] == 1
        assert rows[0]["values"]["input_tokens"] == pytest.approx(expected)


def test_a_model_with_no_call_in_scope_is_dropped_not_blanked(vendor_records):
    columns = results_report.vendor_token_columns(vendor_records)
    rows = results_report.vendor_token_rows(
        vendor_records, "anthropic", columns["anthropic"], "strategy", "never_ran")
    assert rows == []


def test_cost_is_the_only_shaded_column(vendor_run, vendor_scored):
    """Shading points at the best value in a column, relative to the other rows
    of the same table. Cheaper is better and can be said; whether more of a
    vendor-specific token quantity is better is not something this project can
    declare, so those columns carry no direction and stay flat."""
    panel = panels_of(results_report.build(vendor_run, scored=vendor_scored)
                      .read_text(encoding="utf-8"))[("model", "cost")]
    table = re.search(r'<div class="vendorsection">.*?</table>', panel, re.S).group(0)
    rows = dict(re.findall(r'<tr class="aggrow" data-key="([^"]+)">(.*?)</tr>', table, re.S))

    def weight(cells, text):
        found = re.search(rf'<td class="num cell-scale[^"]*" style="--w:([\d.]+)">{text}</td>',
                          cells)
        return float(found.group(1)) if found else None

    assert weight(rows["claude_a"], r"\$0.001500") == pytest.approx(1.0)   # the cheaper model
    assert weight(rows["claude_b"], r"\$0.003000") == pytest.approx(0.0)
    # Every token column is a vendor field, and none of them is shaded: the two
    # shaded cells in the row are the two cost cells and nothing else.
    shaded = re.findall(r'<td class="num cell-scale[^>]*>([^<]*)</td>', rows["claude_a"])
    assert shaded == ["$0.001500", "$0.003000"]


def test_the_vendor_tables_list_only_the_vendors_own_fields(vendor_run, vendor_scored):
    """No second group of this project's classes beside them: for four of the
    five vendors those are the same numbers again under different names."""
    panel = panels_of(results_report.build(vendor_run, scored=vendor_scored)
                      .read_text(encoding="utf-8"))[("model", "cost")]
    assert "priced token classes" not in panel
    table = re.search(r'<div class="vendorsection">.*?</table>', panel, re.S).group(0)
    heads = re.findall(r'<th class="num group" colspan="\d+">([^<]+)</th>', table)
    assert heads == ["calls", "as Anthropic reported it", "cost"]


def test_a_run_without_usage_objects_falls_back_to_the_recorded_classes(fake_run, scored):
    """A mock run, or one older than the archive, has no vendor fields at all.
    The classes this project records are then the only token figures that
    exist — a table of nothing but call counts would be worse."""
    panel = panels_of(results_report.build(fake_run, scored=scored)
                      .read_text(encoding="utf-8"))[("model", "cost")]
    assert "no usage object archived" in panel
    assert "input (uncached)" in panel and "total" in panel


def test_the_by_llm_tokens_panel_is_one_table_per_vendor(vendor_run, vendor_scored):
    html = results_report.build(vendor_run, scored=vendor_scored).read_text(encoding="utf-8")
    panel = panels_of(html)[("model", "cost")]
    assert panel.count('class="agg vendortable"') == 2
    # The pooled table is gone: every model of the run has a row in a vendor's
    # table, so nothing it showed was lost.
    assert '<table class="agg" data-group="model_key"' not in panel
    for model in ("claude_a", "claude_b", "gemini_a", "gemini_b"):
        assert f'data-key="{model}"' in panel
    # Each vendor's own field names are column heads, under its own name.
    assert "as Anthropic reported it" in panel and "as Google reported it" in panel
    assert "cache_creation.ephemeral_5m_input_tokens" in panel
    assert "candidates_token_count" in panel


def test_the_strategy_panel_adds_a_table_per_strategy_and_vendor(vendor_run, vendor_scored):
    html = results_report.build(vendor_run, scored=vendor_scored).read_text(encoding="utf-8")
    panel = panels_of(html)[("strategy", "cost")]
    # 2 strategies × 2 vendors, and nothing pooled above them.
    assert '<table class="agg" data-group="strategy"' not in panel
    assert panel.count('class="agg vendortable"') == 4
    for strategy in ("zero_shot", "few_shot"):
        assert f'data-filter-key="{strategy}"' in panel
    assert panel.count('data-filter-group="strategy"') == 4


def test_each_strategy_heading_carries_its_total_cost(vendor_run, vendor_scored):
    """The tables below a heading are means per model, one vendor at a time —
    nothing in them adds the strategy up, so the heading does."""
    html = results_report.build(vendor_run, scored=vendor_scored).read_text(encoding="utf-8")
    panel = panels_of(html)[("strategy", "cost")]
    records = vendor_scored.to_dict(orient="records")
    for strategy in ("zero_shot", "few_shot"):
        total = sum(float(r["cost_usd"]) for r in records
                    if r["strategy"] == strategy and not results_report.missing(r["cost_usd"]))
        assert total > 0
        head = re.search(rf'<h5 class="strategyhead">{strategy}(.*?)</h5>', panel, re.S)
        assert head, strategy
        assert f'(Total Cost: {results_report.money(total)})' in head.group(1)
    # Every strategy of the run gets one, and the figures are its own calls, so
    # they add up to the run's total cost.
    assert panel.count('class="strategycost"') == 2


def test_an_unpriced_call_is_not_counted_as_zero_in_a_strategy_total():
    """A model whose rate is unverified has no cost. Adding it as 0 would make
    the strategy look cheaper than it was; the heading says so instead."""
    records = [{"strategy": "zero_shot", "cost_usd": 0.5},
               {"strategy": "zero_shot", "cost_usd": None},
               {"strategy": "few_shot", "cost_usd": None}]
    priced = results_report._strategy_total_cost(records, "zero_shot")
    assert "$0.5000" in priced
    assert "1 of 2 calls carry no price" in priced
    none_at_all = results_report._strategy_total_cost(records, "few_shot")
    assert "not computed" in none_at_all
    assert "$" not in none_at_all


def test_a_per_strategy_row_unfolds_only_that_strategys_calls(vendor_run, vendor_scored):
    """A row in a strategy's table means this model *within* that strategy, so
    the fold-out has to filter on both keys or it would show the whole run."""
    html = results_report.build(vendor_run, scored=vendor_scored).read_text(encoding="utf-8")
    assert "host.dataset.filterGroup" in html
    assert "String(c[filterGroup]) === filterKey" in html


def test_the_by_llm_tables_carry_no_second_filter(vendor_run, vendor_scored):
    """There a row is the model's whole run, pooled over every strategy."""
    panel = panels_of(results_report.build(vendor_run, scored=vendor_scored)
                      .read_text(encoding="utf-8"))[("model", "cost")]
    assert "data-filter-group" not in panel


def test_a_vendor_field_is_never_shaded_as_if_it_had_a_direction(vendor_records):
    """`_share` returns nothing for a directionless column: this project has no
    basis to declare that more of a vendor's own field is better or worse."""
    assert results_report._share([1, 2, 3], 2, "none") is None
    assert results_report._share([100, 200], 100, "down") == pytest.approx(1.0)
    assert results_report._share([100, 200], 200, "down") == pytest.approx(0.0)
    assert results_report._share([100, 100], 100, "down") is None      # no spread


def test_latency_drops_the_columns_that_answer_a_different_question(fake_run, scored):
    """A latency is measured on the API call, which happens whether or not the
    reply parsed — so `valid` and `parse rate` describe something the row is not
    about. `n` stays: it is the sample the row's own figures come from."""
    panels = panels_of(results_report.build(fake_run, scored=scored)
                       .read_text(encoding="utf-8"))
    for view in ("model", "strategy"):
        latency = panels[(view, "latency")]
        assert '<th class="num sample">n</th>' in latency
        assert "valid" not in latency
        assert "parse" not in latency
        # Quality keeps all three: there a mean is only readable against them.
        quality = panels[(view, "quality")]
        for column in ("n", "valid", "parse&nbsp;rate"):
            assert f'<th class="num sample">{column}</th>' in quality


# ── Syntactical Correctness ──────────────────────────────────────────────
def _correctness(dot, tmp_path):
    from quality import graph as g, syntactic
    path = tmp_path / "m.gv"
    path.write_text(dot, encoding="utf-8")
    return syntactic.syntactical_correctness(g.load(path))


_WELL_FORMED = """digraph {
  s [shape=circle, label="start"];
  a [shape=box, label="Do A"];
  b [shape=box, label="Do B"];
  e [shape=doublecircle, label="end"];
  s -> a; a -> b; b -> e;
}"""


def test_a_well_formed_model_is_syntactically_correct(tmp_path):
    """All three conditions hold: one arc in and out of every function, no
    gateway to wire, and a start and an end node."""
    r = _correctness(_WELL_FORMED, tmp_path)
    assert r["syn_correct"] is True
    assert (r["syn_correct_functions"], r["syn_correct_gateways"],
            r["syn_correct_start_end"]) == (True, True, True)


def test_a_function_with_two_incoming_arcs_fails(tmp_path):
    """`exactly one` is the rule for functions — a loop rejoining a task
    without a gateway is the common way real models break it."""
    dot = _WELL_FORMED.replace("b -> e;", "b -> e; b -> a;")
    r = _correctness(dot, tmp_path)
    assert r["syn_correct"] is False
    assert r["syn_correct_functions"] is False
    assert r["syn_n_malformed_functions"] == 2      # a gains an arc, b splits
    assert r["syn_correct_start_end"] is True       # the other two still hold


def test_a_gateway_needs_only_one_arc_each_way(tmp_path):
    """Functions are held to `exactly one`, gateways to `at least one`: a split
    with two outgoing arcs is correct, a dangling one is not."""
    ok = """digraph {
      s [shape=circle]; g [shape=diamond, label="X"];
      a [shape=box, label="A"]; b [shape=box, label="B"];
      e [shape=doublecircle]; f [shape=doublecircle];
      s -> g; g -> a; g -> b; a -> e; b -> f;
    }"""
    assert _correctness(ok, tmp_path)["syn_correct"] is True

    dangling = ok.replace("s -> g; ", "")           # gateway with no incoming arc
    r = _correctness(dangling, tmp_path)
    assert r["syn_correct"] is False
    assert r["syn_correct_gateways"] is False
    assert r["syn_n_unwired_gateways"] == 1


def test_a_model_without_a_start_or_an_end_fails(tmp_path):
    r = _correctness(_WELL_FORMED.replace('s [shape=circle, label="start"];', ""), tmp_path)
    assert r["syn_correct_start_end"] is False and r["syn_correct"] is False


def test_the_verdict_stays_out_of_the_graded_scores(tmp_path):
    """It has no partial credit, so folding it into `syn_score` would change a
    figure already in use — and it repeats checks the score already has."""
    from quality import syntactic
    assert "syn_correct" not in syntactic.CHECKS
    for key in ("correct", "correct_functions"):
        assert key not in syntactic.CHECKS


def test_a_row_counts_the_calls_that_passed(fake_run, scored, monkeypatch):
    """Per call the metric is yes or no; a row shows how many of its calls said
    yes, out of all of them. A call that produced no model did not pass, so it
    stays in the denominator.

    `HIDDEN_COLUMNS` is emptied here so the test states what it needs rather
    than inheriting it: the page draws the column today, and this test is about
    the arithmetic in the cell, not about whether it is drawn."""
    monkeypatch.setattr(results_report, "HIDDEN_COLUMNS", set())
    import pandas as pd

    frame = pd.DataFrame([
        {"model_key": "all_good", "n_calls": 2, "n_valid": 2, "parse_rate": 1.0,
         "syn_correct_mean": 1.0, "syn_correct_count": 2},
        {"model_key": "one_bad", "n_calls": 2, "n_valid": 2, "parse_rate": 1.0,
         "syn_correct_mean": 0.5, "syn_correct_count": 2},
        {"model_key": "one_unparsed", "n_calls": 2, "n_valid": 1, "parse_rate": 0.5,
         "syn_correct_mean": 1.0, "syn_correct_count": 1},
        {"model_key": "nothing", "n_calls": 2, "n_valid": 0, "parse_rate": 0.0,
         "syn_correct_mean": None, "syn_correct_count": 0},
    ])
    # Every other quality column has to exist for the table to render at all;
    # none of them is what this test is about.
    for column, _label, _direction in agg.QUALITY_METRICS:
        for stat in ("mean", "count"):
            if f"{column}_{stat}" not in frame:
                frame[f"{column}_{stat}"] = None
    html = results_report._metric_table(frame, "model_key", "quality")
    rows = dict(re.findall(r'data-key="(\w+)">(.*?)</tr>', html, re.S))
    cell = {key: re.search(r'<td class="num[^"]*stat-verdict[^"]*"[^>]*>(.*?)</td>',
                           cells).group(1)
            for key, cells in rows.items()}
    assert cell["all_good"] == "2/2"
    assert cell["one_bad"] == "1/2"
    # Every call that produced a model passed, but one produced none: 1/2, and
    # the tooltip says why the other call is not in the numerator.
    assert cell["one_unparsed"] == "1/2"
    assert "1 of 2 call(s) passed" in rows["one_unparsed"]
    assert "no verdict" in rows["one_unparsed"]
    # No verdict anywhere in the row: "not computed", never a 0/2 this project
    # cannot stand behind.
    assert "not computed" in cell["nothing"]
    # Shaded against the other rows, like every other column: the row that got
    # it right most often is the darkest.
    weights = {key: float(re.search(r'stat-verdict[^"]*" style="--w:([\d.]+)"',
                                    cells).group(1))
               for key, cells in rows.items()}
    assert weights["all_good"] == 1.0 and weights["one_bad"] == 0.0


def test_the_quality_table_shows_the_verdict(fake_run, scored):
    """Drawn again since 2026-08-26 (author's instruction): a column in the
    Quality table of both views, a block in a call's fold-out, and the verdict
    and its five conditions still in the embedded dataset for every call."""
    html = results_report.build(fake_run, scored=scored).read_text(encoding="utf-8")
    assert "syn_correct" not in results_report.HIDDEN_COLUMNS
    for view in ("model", "strategy"):
        panel = panels_of(html)[(view, "quality")]
        assert "Syntactical Correctness" in panel
        assert "stat-verdict" in panel
    # The fold-out block is the one that names the five conditions.
    assert "all five must hold" in html
    assert "syn_correct_connected" in html
    calls = json.loads(re.search(r'<script id="calls" type="application/json">(.*?)</script>',
                                 html, re.S).group(1))
    assert "syn_correct" in calls[0] and "syn_correct_connected" in calls[0]



def test_an_undirected_graph_is_never_syntactically_correct(tmp_path):
    """`graph` instead of `digraph`: Graphviz's own agisdirected() says no, and
    without arrow direction nothing the other conditions check means what it
    says — a start event with in = 0 cannot even be defined."""
    directed = _correctness(_WELL_FORMED, tmp_path)
    assert directed["syn_correct_directed"] is True and directed["syn_correct"] is True

    undirected = _WELL_FORMED.replace("digraph {", "graph {").replace("->", "--")
    r = _correctness(undirected, tmp_path)
    assert r["syn_correct_directed"] is False
    assert r["syn_correct"] is False


def test_a_model_in_two_pieces_is_not_coherent(tmp_path):
    """isConnected() searches with edges in both directions, over every node of
    the file. One orphan node is enough: a model in two pieces is two
    processes, not one."""
    dot = _WELL_FORMED.replace("}", 'orphan [shape=box, label="Do C"];\n}')
    r = _correctness(dot, tmp_path)
    assert r["syn_correct_connected"] is False
    assert r["syn_n_components"] == 2
    assert r["syn_correct"] is False
    # The whole model still passes once the piece is wired in.
    joined = dot.replace("b -> e;", "b -> orphan; orphan -> e;")
    ok = _correctness(joined, tmp_path)
    assert ok["syn_correct_connected"] is True and ok["syn_n_components"] == 1
    assert ok["syn_correct"] is True


def test_connectivity_matches_graphviz_ccomps(tmp_path):
    """The component count is the one Graphviz reports for the same file. Run
    against `ccomps` where it is installed — the check that keeps this a port of
    isConnected() rather than a lookalike."""
    import re
    import shutil
    import subprocess

    from quality import graph as g

    ccomps = shutil.which("ccomps") or r"C:/Program Files/Graphviz/bin/ccomps.exe"
    if not shutil.which(ccomps) and not Path(ccomps).exists():
        pytest.skip("Graphviz ccomps not installed")

    cases = {
        "one.gv": _WELL_FORMED,
        "two.gv": _WELL_FORMED.replace("}", 'orphan [shape=box];\n}'),
        "three.gv": _WELL_FORMED.replace("}", 'x [shape=box]; y [shape=box];\n}'),
    }
    for name, dot in cases.items():
        path = tmp_path / name
        path.write_text(dot, encoding="utf-8")
        out = subprocess.run([ccomps, "-sv", str(path)], capture_output=True, text=True)
        expected = int(re.search(r"(\d+) components", out.stdout + out.stderr).group(1))
        assert len(g.load(path).components()) == expected, name


def test_a_node_declared_twice_keeps_its_shape(tmp_path):
    """A second statement for the same node **adds** attributes; it does not
    replace the node.

    Models routinely declare the shape first and colour the node in a styling
    block at the foot of the file. Replacing dropped the shape, and the graph
    default `node [shape=box]` then read every start event, end event and
    gateway as a task: one generated model came out as 15 tasks, no events and
    no gateways, with `syn_bef_score` computed over a process that was not the
    one in the file.
    """
    from quality import graph as g

    dot = """digraph {
      node [shape=box, style=rounded];
      Start [shape=circle, label="Need"];
      Decide [shape=diamond, label="Ok?"];
      Work [shape=box, label="Do the work"];
      End [shape=doublecircle, label="Done"];
      Start -> Decide; Decide -> Work; Decide -> End; Work -> End;

      // styling block: same nodes again, colour only
      Start [fillcolor=lightgreen, style="filled,rounded"];
      Decide [fillcolor=lightyellow, style=filled];
      End [fillcolor=lightcoral, style="filled,rounded"];
    }"""
    path = tmp_path / "styled.gv"
    path.write_text(dot, encoding="utf-8")

    pg = g.load(path)
    assert pg.starts == {"Start"} and pg.ends == {"End"}
    assert pg.gateways == {"Decide"} and pg.tasks == {"Work"}
    # The colour from the second statement is there too — merged, not ignored.
    assert pg.nodes["Start"]["fillcolor"] == "lightgreen"
    assert pg.nodes["Start"]["style"] == "filled,rounded"


def test_merged_node_attributes_match_graphviz(tmp_path):
    """The merge is Graphviz's, not an invention: `nop` prints each node once,
    with the attributes of every statement folded in. Compared against it where
    Graphviz is installed — the check that keeps `_collect_declared` a port."""
    import re
    import shutil
    import subprocess

    from quality import graph as g

    nop = shutil.which("nop") or r"C:/Program Files/Graphviz/bin/nop.exe"
    if not shutil.which(nop) and not Path(nop).exists():
        pytest.skip("Graphviz nop not installed")

    dot = """digraph {
      node [shape=box];
      Start [shape=circle]; Start [fillcolor=lightgreen, style=filled];
      Gate [shape=diamond]; Gate [label="Ok?"];
      Start -> Gate;
    }"""
    path = tmp_path / "merge.gv"
    path.write_text(dot, encoding="utf-8")

    printed = subprocess.run([nop, str(path)], capture_output=True, text=True).stdout
    ours = g.load(path).nodes
    for name in ("Start", "Gate"):
        block = re.search(rf"{name}\s*\[(.*?)\];", printed, re.S).group(1)
        for pair in block.split(","):
            key, _, value = pair.partition("=")
            assert ours[name][key.strip()] == value.strip().strip('"'), (name, key)


def test_the_bef4llm_readings_ride_along_outside_the_score(tmp_path):
    """The `sem_*_bef` columns reproduce BEF4LLM's own four semantic metrics
    (re-added 2026-08-29). They are diagnostics: present on every comparison,
    absent from `METRICS`, and therefore never part of `sem_score`.

    The set shrank twice on 2026-08-29, each time with the metric it mirrored:
    *common nodes and edges*, whose reading of theirs returned 1.0 for any two
    models, and the *graph-edit distance*. Both metrics have since returned
    (2026-09-08 and 2026-09-09) **without** their mirrors: the restored readings
    are this port's own, so there is nothing of theirs to reproduce beside them.
    What is left mirrors the two label similarities and — since 2026-08-30 —
    the context similarity.
    """
    from quality import graph as g, semantic

    one = tmp_path / "one.gv"
    one.write_text("""digraph {
      s [shape=circle]; a [shape=box, label="Collect the order"];
      e [shape=doublecircle]; s -> a; a -> e;
    }""", encoding="utf-8")
    other = tmp_path / "other.gv"
    other.write_text("""digraph {
      s2 [shape=circle]; b [shape=box, label="Fly the aeroplane"];
      c [shape=box, label="Land again"]; e2 [shape=doublecircle];
      s2 -> b; b -> c; c -> e2;
    }""", encoding="utf-8")

    out = semantic.evaluate(g.load(one), g.load(other))
    bef = [k for k in out if k.endswith("_disjoint")]
    assert set(bef) == {"sem_label_sim_syntactic_disjoint",
                        "sem_label_sim_semantic_disjoint",
                        "sem_label_sim_context_disjoint"}

    # Not one of them is a scored metric …
    scored = {f"sem_{key}" for key, _group in semantic.METRICS}
    assert not scored & set(bef)
    # … and the score is the mean of the ones that are.
    assert out["sem_score"] == pytest.approx(
        sum(out[k] for k in scored) / len(semantic.METRICS))

    # Common nodes and edges came back on 2026-09-09 as a scored metric — but
    # only on *this* side. Their reading of it stays gone, and that asymmetry is
    # the whole point: `common_percentage_similarity` returns 1.0 for any two
    # models, so a reproduction column would carry a constant and validate
    # nothing. There is therefore a `sem_common_nodes_edges` and no
    # `sem_common_nodes_edges_disjoint`.
    assert "sem_common_nodes_edges" in scored
    assert not [k for k in out if "common" in k and k.endswith("_disjoint")]
    # Its two halves ride along as diagnostics, outside the score.
    assert {"sem_common_nodes", "sem_common_edges"} <= set(out)
    assert not scored & {"sem_common_nodes", "sem_common_edges"}
    # The scored metrics are their algorithm as it runs on real files, so the
    # collision-free reading is the one that can differ — never the other way,
    # since separating the id spaces can only *add* matching entries.
    assert out["sem_label_sim_syntactic_disjoint"] >= out["sem_label_sim_syntactic"]
    assert out["sem_n_ids_shared"] >= 0

    # A comparison without a partner has the columns too, all None.
    assert set(bef) <= set(semantic.empty())
    assert all(semantic.empty()[k] is None for k in bef)


def test_a_message_flow_is_not_an_arc_of_the_control_flow(tmp_path):
    """Only sequence flows count for the degree rules: an activity that also
    sends a message is still an activity with one arc in and one out. BPMN draws
    a message flow dashed, which is how `graph.py` tells the two apart."""
    from quality import graph as g, syntactic

    dot = """digraph {
      s [shape=circle]; e [shape=doublecircle];
      a [shape=box, label="Do A"]; p [shape=point, style=invis];
      s -> a; a -> e;
      a -> p [style=dashed, arrowhead=open];
    }"""
    path = tmp_path / "m.gv"
    path.write_text(dot, encoding="utf-8")
    pg = g.load(path)
    assert (pg.in_degree("a"), pg.out_degree("a")) == (1, 2)          # every edge
    assert (pg.seq_in_degree("a"), pg.seq_out_degree("a")) == (1, 1)  # control flow

    r = syntactic.syntactical_correctness(pg)
    assert r["syn_correct_functions"] is True and r["syn_correct"] is True
    # Check 7 keeps counting every edge — the two are allowed to disagree here,
    # and this is the model where they do.
    assert syntactic.evaluate(pg)["syn_task_in1_out1"] is False


def test_coherence_still_counts_every_edge(tmp_path):
    """Ignoring message flows is about the degree rules. Coherence is the port
    of isConnected(), which counts every edge of the file — and message flows
    are exactly what joins two pools."""
    from quality import graph as g, syntactic

    dot = """digraph {
      subgraph cluster_a { s [shape=circle]; t [shape=box]; e [shape=doublecircle];
                           s -> t; t -> e; }
      subgraph cluster_b { u [shape=box]; }
      t -> u [style=dashed, arrowhead=open];
    }"""
    path = tmp_path / "pools.gv"
    path.write_text(dot, encoding="utf-8")
    r = syntactic.syntactical_correctness(g.load(path))
    assert r["syn_correct_connected"] is True and r["syn_n_components"] == 1


# ── what a request id unfolds into (O6) ───────────────────────────────────────
def _archive_prompt(run_dir, stem="01__alpha__zero_shot"):
    """The record `pipeline._write_provenance` writes beside a reply."""
    (run_dir / "prompts").mkdir(exist_ok=True)
    (run_dir / "prompts" / f"{stem}.json").write_text(json.dumps({
        "input": {"source": "descriptions/01.txt", "chars": 24,
                  "sha256": "aa11bb22", "text": "A does X, then B does Y."},
        "prompt": {"source": "Li et al. (2025), Sec. 4 - verbatim",
                   "template_version": "e0fd20c1362f", "sha256": "cc33dd44",
                   "n_turns": 1, "chars": 99, "system": None,
                   "messages": [{"role": "user",
                                 "content": "Generate the DOT language ..."}],
                   "exemplar_item_ids": []},
        "parameters": {"temperature": None, "temperature_supported": False,
                       "max_output_tokens": 4096, "thinking": False},
        "pricing": {"sheet_as_of": "2026-01-01", "basis": "list",
                    "price_verified": True,
                    "rates_per_mtok": {"input_tokens": 1.0, "output_tokens": 5.0}},
        "artefacts": {"raw_output": f"raw/{stem}.txt"},
    }), encoding="utf-8")


def test_every_call_unfolds_into_the_conditions_that_produced_it(fake_run, scored):
    """The id is the handle; clicking it has to answer what produced the row,
    not only what the vendor called it."""
    scored = scored.copy()
    scored["vendor"] = "anthropic"
    results_report.build(fake_run, scored=scored)
    page = (fake_run / "request_ids.html").read_text(encoding="utf-8")

    assert page.count('<button type="button" class="rid"') == len(scored)
    assert page.count('<tr class="det"') == len(scored)
    assert 'aria-controls="c1"' in page and 'id="c1" hidden' in page
    for heading in ("Input", "Prompt", "Model and parameters", "Timestamps",
                    "Raw output", "Price list", "Derived measurements"):
        assert f"<h4>{heading}</h4>" in page, heading


def test_the_archived_prompt_is_shown_under_its_request_id(fake_run, scored):
    _archive_prompt(fake_run)
    scored = scored.copy()
    scored["vendor"] = "anthropic"
    results_report.build(fake_run, scored=scored)
    page = (fake_run / "request_ids.html").read_text(encoding="utf-8")

    assert "A does X, then B does Y." in page          # the input, verbatim
    assert "Generate the DOT language ..." in page     # the prompt, verbatim
    assert "e0fd20c1362f" in page                      # which template version


def test_a_run_that_archived_no_prompt_says_so(fake_run, scored):
    """An older run has its reply but not the prompt behind it — the fold-out
    has to say that rather than show empty fields."""
    scored = scored.copy()
    scored["vendor"] = "anthropic"
    results_report.build(fake_run, scored=scored)
    page = (fake_run / "request_ids.html").read_text(encoding="utf-8")
    assert "Prompt and input not archived" in page


def test_a_failed_call_unfolds_too(fake_run, scored):
    """It has no request id and no reply, but it has conditions — and the error
    they produced."""
    scored = scored.copy()
    scored["vendor"] = "anthropic"
    results_report.build(fake_run, scored=scored)
    page = (fake_run / "request_ids.html").read_text(encoding="utf-8")
    assert "Error code: 500 - fixture failure" in page
    assert page.count('<button type="button" class="rid"') == len(scored)


def test_the_price_list_shows_the_rates_the_call_was_charged(fake_run, scored):
    _archive_prompt(fake_run)
    scored = scored.copy()
    scored["vendor"] = "anthropic"
    results_report.build(fake_run, scored=scored)
    page = (fake_run / "request_ids.html").read_text(encoding="utf-8")
    assert "2026-01-01" in page                        # the sheet's date
    assert "$1.0000" in page and "$5.0000" in page     # input / output per Mtok
