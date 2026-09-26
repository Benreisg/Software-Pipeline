"""
score_pmo_dataset.py — score every PMo ground-truth model with both pipelines
============================================================================
The reference baseline. Runs all four quality dimensions of **this project's
port** over the 55 PMo ground-truth models, runs **BEF4LLM's own supplied
code** over the same 55 processes, and puts the two side by side.

    python score_pmo_dataset.py                  # → pmo_quality.csv
                                                 #   + pmo_quality_columns.csv
                                                 #   + pmo_quality_bef4llm.csv
                                                 #   + pmo_quality.xlsx
    python score_pmo_dataset.py --out foo.csv --dataset-dir <path>

**Both sides are recomputed on every run.** Nothing is read from a cached CSV:
the authors' code is executed over the `.bpmn` half of the dataset each time,
even though the procedure is deterministic and the numbers come out the same.
A stale `pmo_quality_bef4llm.csv` from an older checkout, an older dataset copy
or an older metric set can therefore no longer be interleaved silently — the
file this run writes is the file this run read. `--no-bef4llm` skips their side
entirely (the workbook is then written without their columns).

It costs about six seconds per model, so roughly five to seven minutes for the
55; their semantic pass is the expensive part. The progress line names the item
being scored.

── The two halves of the dataset ────────────────────────────────────────────
    <dataset>/graphviz/<id>.dot     this project's port reads DOT
    <dataset>/bpmn/<id>.bpmn        BEF4LLM's code reads BPMN 2.0 XML
Same 55 processes, two serialisations — itself a source of difference, and
flagged in the legend rather than hidden. `--bpmn-dir` moves the BPMN half on
its own; by default it is the `bpmn/` folder beside the `graphviz/` one.

── A separate interpreter for their code ────────────────────────────────────
BEF4LLM needs pm4py, seaborn, xmlschema and networkx. If this interpreter can
import them, their code runs in-process; otherwise the scoring is handed to
`.venv-bef4llm` (or whatever `--bef4llm-python` names) as a subprocess running
`score_pmo_bef4llm.py`, and its CSV is read back. Either way the work is the
same and `pmo_quality_bef4llm.csv` is written beside `--out`.

Beside the data CSV goes **`<out>_columns.csv`**, one row per column saying
which metric that column carries, which dimension and group it belongs to,
and which table of the paper defines it (`quality/columns.py`, generated from
the same metric specs the scoring reads). `--no-legend` skips it.

**`<out>.xlsx`** is the same two tables as sheets `quality` and `columns`, with
the header frozen, filterable, and each header cell carrying its legend entry
as a hover note — the metric is readable where the column is. It carries
**BEF4LLM's syntactic, pragmatic and semantic metric sets only** (the three
definition tables as this port implements them, their group means, the three
Q-scores); the dictated checks, the reference code's extras, validity and the
diagnostics stay in the CSVs. `--xlsx-all-columns` puts everything in,
`--no-xlsx` skips the workbook. Needs `openpyxl`.

**Why this exists.** Pragmatic quality rewards smallness, the syntactic checks
are strict, and several BEF4LLM threshold bands are calibrated for smaller
models than PMo contains — so 1.0 is *not* the ceiling a generated model should
be compared against. The reference models' own scores are, and this produces
them. Every number a generated model is reported against belongs next to the
matching column here.

**What is scored, and how.** Exactly the same code path a real run uses:
`quality.score.score_model(generated, ground_truth)`, with the ground-truth
`.dot` passed as *both* arguments. Consequences worth knowing before reading
the output:

* **Syntactic** (`syn_*`, `syn_bef_*`) and **pragmatic** (`prag_*`) are
  model-internal — these are genuine measurements of the reference set, and the
  interesting part of this table.
* **Semantic** (`sem_*`) compares a model with a reference. Here they are the
  same file, so these are self-comparisons — never baseline values. One of the
  four metrics is reflexive and lands on exactly 1.0
  (`sem_graph_edit_distance`, restored 2026-09-08); a value below 1.0 there
  would mean the semantic layer is miswired. **The label similarities and the
  context similarity are not**, and are not meant
  to be: the port divides the summed similarity by |FO_c| + |FO_g| like the
  reference implementation does, where the paper's Table A.17 multiplies that
  sum by 2, so a perfect self-match scores n/(n+n) ≈ 0.5. Both sides of the
  workbook show it, and they agree there almost exactly — which is the point:
  the ~0.478 is a property of the shared formula, not of the models.
* **Validity** (`val_*`) asks Graphviz (`nop -p`) whether the file is valid DOT
  language. On the ground truth this is a wiring check: the reference models are
  Graphviz files, so anything below 1.0 would mean the checker is
  misconfigured, not that PMo is malformed.

Items `01`/`02` are the few-shot exemplars (`config.FEW_SHOT_IDS`). A run
covers them like every other item (their own model is swapped out of the prompt
for those two prompts), so this baseline and a run cover the same 55 items. They
are flagged in `is_few_shot_exemplar` anyway, so the two items that ran on a
swapped exemplar set can still be separated out.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config

# The per-item report is drawn with box characters, which a legacy console code
# page (cp1252) cannot encode — printing one would kill the script *after* the
# scoring is done. Same guard run.py already carries.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
from dataset import resolve_dirs
from quality import columns, pragmatic, syntactic, syntax_rules
from quality.score import score_model

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_DIR / "pmo_quality.csv"

# This script's own dataset and reference-code locations. Both moved into the
# project on 2026-08-27; `config.PMO_DATASET_DIR` still points wherever a *run*
# reads its descriptions from, which is a different question, so this baseline
# carries its own default rather than changing that one underneath run.py.
DEFAULT_DATASET_DIR = PROJECT_DIR / "pmo-dataset"
DEFAULT_BEF4LLM_ROOT = PROJECT_DIR / "bef4llm-main"

# Interpreters tried, in order, when this one cannot import BEF4LLM's deps.
_VENV_PYTHONS = (
    PROJECT_DIR / ".venv-bef4llm" / "Scripts" / "python.exe",
    PROJECT_DIR / ".venv-bef4llm" / "bin" / "python",
)


def legend_path(out: Path) -> Path:
    """Where the column legend goes: `<out>_columns.csv` beside the data."""
    return out.with_name(f"{out.stem}_columns.csv")


def bef4llm_path(out: Path) -> Path:
    """Where their side is written: `<out>_bef4llm.csv` beside the data."""
    return out.with_name(f"{out.stem}_bef4llm.csv")


def write_legend(df: pd.DataFrame, out: Path) -> Path:
    """One row per column of `df` — which metric it carries, from which
    dimension and definition. Generated from the same spec objects the
    scoring reads, so it cannot drift from the columns it describes.

    A column `quality/columns.py` has no entry for is written out marked
    `undocumented` and named on the console, rather than quietly missing.
    """
    path = legend_path(out)
    columns.legend_dataframe(df.columns).to_csv(path, index=False,
                                                encoding="utf-8-sig")
    missing = columns.undocumented(df.columns)
    if missing:
        print(f"  [legend] {len(missing)} column(s) without an entry in "
              f"quality/columns.py: {', '.join(missing)}")
    return path


# ── BEF4LLM's own implementation, recomputed every run ──────────────────────
def resolve_bef4llm_src(root: Path) -> Path:
    """The importable `src` directory under a bef4llm checkout.

    Accepts the checkout root, the doubled `bef4llm-main/bef4llm-main` layout a
    GitHub zip unpacks to, or the `src` directory itself — whichever the folder
    on this machine happens to be.
    """
    root = Path(root)
    seen: List[Path] = [root, root / "src"]
    seen += [p / "src" for p in sorted(root.glob("*")) if p.is_dir()]
    seen += [p / "src" for p in sorted(root.glob("*/*")) if p.is_dir()]
    for candidate in seen:
        if (candidate / "bef4llm" / "__init__.py").exists():
            return candidate.resolve()
    raise SystemExit(
        f"No BEF4LLM source under {root} — looked for a 'src' directory "
        f"containing 'bef4llm/__init__.py'. Point --bef4llm-src at it.")


def _bef4llm_interpreter(explicit: Optional[str]) -> Path:
    """The interpreter that can run their code, when this one cannot."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"--bef4llm-python does not exist: {path}")
        return path
    for candidate in _VENV_PYTHONS:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "BEF4LLM needs pm4py/seaborn/xmlschema/networkx, which this "
        "interpreter does not have, and no .venv-bef4llm was found beside "
        "the project. Create one, or name an interpreter with "
        "--bef4llm-python, or skip their side with --no-bef4llm.")


def score_bef4llm(bpmn_dir: Path, src: Path, out_csv: Path,
                  python_exe: Optional[str] = None,
                  verify: int = 0) -> Tuple[pd.DataFrame, str]:
    """Run the authors' code over `bpmn_dir` and return `(frame, provenance)`.

    In-process when this interpreter can import their dependencies, as a
    subprocess running `score_pmo_bef4llm.py` otherwise. `out_csv` is written
    either way, so `build_pmo_quality_comparison.py` and anything else reading
    that file gets this run's numbers rather than an older run's.
    """
    paths = sorted(Path(bpmn_dir).glob("*.bpmn"), key=lambda p: p.stem)
    if not paths:
        raise SystemExit(f"No .bpmn files in {bpmn_dir}")

    import score_pmo_bef4llm as bef

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    provenance = f"bef4llm-main at {src}, BPMN from {bpmn_dir}, computed {stamp}"

    try:
        B = bef._load_bef4llm(str(src))
    except ImportError as exc:
        interpreter = _bef4llm_interpreter(python_exe)
        print(f"  [bef4llm] {exc.name} missing here — scoring {len(paths)} "
              f"models with {interpreter}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        cmd = [str(interpreter), str(PROJECT_DIR / "score_pmo_bef4llm.py"),
               "--bpmn-dir", str(bpmn_dir), "--bef4llm-src", str(src),
               "--out", str(out_csv)]
        if verify:
            cmd += ["--verify", str(verify)]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_DIR), env=env)
        return pd.read_csv(out_csv, dtype={"item_id": str}), provenance

    print(f"  [bef4llm] scoring {len(paths)} models in this interpreter")
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for i, path in enumerate(paths, 1):
        print(f"  [{i:>2}/{len(paths)}] {path.name}", end="\r", flush=True)
        try:
            row = bef.score_one(path, B)
            if i <= verify:
                problems.extend(bef.verify(path, B, row))
        except Exception as exc:  # noqa: BLE001 — record, never abort the sweep
            row = {"item_id": path.stem, "bpmn_path": str(path),
                   "bef_error": f"{type(exc).__name__}: {exc}"}
            print(f"  ! {path.name}: {exc}", flush=True)
        rows.append(row)
    print(" " * 60, end="\r")
    if verify:
        print(f"  [bef4llm] {verify} item(s) re-scored the naive way: "
              + ("all aggregates match" if not problems else "MISMATCH"))
        for problem in problems:
            print(f"    ! {problem}")

    frame = pd.DataFrame(rows)
    lead = ["item_id", "bef_xsd_valid", "bef_syn_score", "bef_prag_score",
            "bef_sem_score", "bef_error"]
    frame = frame[[c for c in lead if c in frame.columns]
                  + [c for c in frame.columns if c not in lead]]
    frame.to_csv(out_csv, index=False)
    return frame, provenance


def _bef4llm_pairs() -> Dict[str, Tuple[str, str]]:
    """Our column → (the column BEF4LLM's own code writes, how comparable).

    Read off `build_pmo_quality_comparison.MAPPINGS`, the one place that says
    which metric of theirs answers which metric of ours, so the workbook cannot
    drift from the comparison report. The comparability note travels with the
    pair: several of these are approximations, and the workbook is where that
    has to be readable. Every pragmatic pair is extended by its banded score
    (`*_score` on both sides): the metric and the score it contributes are two
    different questions, and both are asked here.
    """
    from build_pmo_quality_comparison import MAPPINGS

    pairs: Dict[str, Tuple[str, str]] = {}
    for _section, _label, theirs, ours, note in MAPPINGS:
        if not ours or not theirs:
            continue
        pairs.setdefault(ours, (theirs, note))
        if ours.startswith("prag_") and not ours.endswith("_score"):
            pairs.setdefault(f"{ours}_score", (f"{theirs}_score", note))
    # `prag_cc` reproduces their cross-connectivity since 2026-08-24 — the
    # mapping still carries the old "omitted" note.
    reproduced = "Reproduced in the port since 2026-08-24"
    pairs.setdefault("prag_cc", ("bef_prag_cross_connectivity", reproduced))
    pairs.setdefault("prag_cc_score",
                     ("bef_prag_cross_connectivity_score", reproduced))
    return pairs


def _interleave_bef4llm(df: pd.DataFrame, theirs: Optional[pd.DataFrame],
                        provenance: str = ""):
    """Put BEF4LLM's own value for a metric immediately right of ours.

    Returns `(df, docs)` — the widened frame and one legend entry per column,
    because the inserted columns are not in `quality/columns.py`: they do not
    come from this pipeline at all, they are this run's output from the
    authors' code, and each carries where that came from and how far the two
    definitions actually agree.

    A metric of ours with no counterpart over there is left alone; so is every
    column when their side was not computed.
    """
    docs = columns.legend(df.columns)
    if theirs is None or theirs.empty:
        return df, docs

    keyed = theirs.copy()
    keyed["item_id"] = keyed["item_id"].astype(str)
    keyed = keyed.set_index("item_id")
    index = df["item_id"].astype(str)
    pairs = _bef4llm_pairs()

    out_frame: Dict[str, Any] = {}
    out_docs: List[columns.ColumnDoc] = []
    for name, doc in zip(df.columns, docs):
        out_frame[name] = df[name].to_numpy()
        out_docs.append(doc)
        pair = pairs.get(name)
        if pair is None or pair[0] not in keyed.columns:
            continue
        source, note = pair
        column = f"{source}  (BEF4LLM)"
        out_frame[column] = keyed[source].reindex(index).to_numpy()
        out_docs.append(columns.ColumnDoc(
            column=column,
            dimension="BEF4LLM's own implementation",
            group=doc.group,
            metric=f"{doc.metric} — as their code computes it",
            kind=doc.kind,
            description=(f"`{source}`, computed by the authors' code on the "
                         f"same process. The column to its left is this "
                         f"project's `{name}`. Comparability: {note}."),
            source=provenance or "bef4llm-main, via score_pmo_bef4llm.py",
        ))
    return pd.DataFrame(out_frame), out_docs


def write_workbook(df: pd.DataFrame, out: Path,
                   theirs: Optional[pd.DataFrame] = None,
                   provenance: str = "",
                   bef4llm_only: bool = True) -> Optional[Path]:
    """`<out>.xlsx` — the same numbers, readable without a CSV viewer.

    Two sheets: `quality` carries the data with the header frozen and
    filterable, and `columns` carries the matching legend. The part that only
    Excel can do is on the data sheet: **every header cell holds its legend
    entry as a hover note**, so what a column measures is readable where the
    column is, not two files away.

    By default the workbook carries **BEF4LLM's own syntactic, pragmatic and
    semantic metric sets only** (`columns.bef4llm_metric_columns` — the three
    definition tables as this port implements them, their group means and the
    three Q-scores). The dictated syntactic checks, the reference code's two
    extras, the validity dimension and every diagnostic count stay out of it;
    `bef4llm_only=False` keeps all of them. Both CSVs always carry the complete
    table either way.

    **Beside each metric stands what the authors' own code computes for it**
    (`_interleave_bef4llm`), from the `theirs` frame this run produced — so the
    two numbers are one column apart instead of two files apart, and each
    inserted header says how comparable the pair actually is. A metric with no
    counterpart over there keeps its single column, and without their side the
    workbook is written exactly as before.

    Needs `openpyxl`; without it the workbook is skipped with a note and the
    CSVs are unaffected.
    """
    try:
        from openpyxl.comments import Comment
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("  [xlsx] openpyxl is not installed — skipped (pip install openpyxl)")
        return None

    path = out.with_suffix(".xlsx")
    if bef4llm_only:
        df = df[columns.bef4llm_metric_columns(df.columns)]
    df, docs = _interleave_bef4llm(df, theirs, provenance)
    legend = pd.DataFrame([d.as_dict() for d in docs], columns=columns.FIELDS)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="quality", index=False)
        legend.to_excel(writer, sheet_name="columns", index=False)

        data, cols = writer.sheets["quality"], writer.sheets["columns"]
        header = Font(bold=True)

        # Data sheet: header frozen next to item_id, one note per column.
        data.freeze_panes = "B2"
        data.auto_filter.ref = data.dimensions
        for i, doc in enumerate(docs, 1):
            cell = data.cell(row=1, column=i)
            cell.font = header
            cell.alignment = Alignment(vertical="top", wrap_text=False)
            note = "\n".join(filter(None, [
                f"{doc.metric}  [{doc.kind}]",
                f"{doc.dimension}" + (f" — {doc.group}" if doc.group else ""),
                doc.description,
                f"Source: {doc.source}" if doc.source else "",
            ]))
            cell.comment = Comment(note, "quality/columns.py", height=150, width=360)
            data.column_dimensions[get_column_letter(i)].width = max(
                12, min(len(cell.value) + 2, 34))

        # Legend sheet: the descriptions are prose, so they get the room.
        cols.freeze_panes = "A2"
        cols.auto_filter.ref = cols.dimensions
        for i in range(1, len(columns.FIELDS) + 1):
            cols.cell(row=1, column=i).font = header
        for letter, width in zip("ABCDEFG", (38, 28, 22, 44, 12, 90, 40)):
            cols.column_dimensions[letter].width = width
        for row in cols.iter_rows(min_row=2, min_col=6, max_col=7):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
    return path


# Column order in the CSV: identity, then the headline scores, then each
# dimension's own block. Anything not named here keeps its natural order at the
# end, so a newly added metric column can never be dropped by this ordering.
_IDENTITY = ["item_id", "is_few_shot_exemplar", "ground_truth_path", "quality_note"]
_HEADLINE = ["syn_score", "syn_checks_passed", "syn_checks_total",
             "syn_bef_score", "syn_size_score", "prag_score", "sem_score"]
_BLOCKS = ["syn_bef_", "syn_extra_", "syn_", "prag_", "sem_", "val_"]


def _order_columns(df: pd.DataFrame) -> List[str]:
    ordered = [c for c in _IDENTITY + _HEADLINE if c in df.columns]
    seen = set(ordered)
    for prefix in _BLOCKS:
        for c in df.columns:
            if c.startswith(prefix) and c not in seen:
                ordered.append(c)
                seen.add(c)
    ordered += [c for c in df.columns if c not in seen]
    return ordered


def score_dataset(dataset_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Score every `graphviz/<id>.dot` in the dataset. One row per item."""
    _, gv_dir = resolve_dirs(Path(dataset_dir))
    paths = sorted(gv_dir.glob("*.dot"), key=lambda p: p.stem)
    if not paths:
        raise FileNotFoundError(f"No .dot files in {gv_dir}")

    rows: List[Dict[str, Any]] = []
    for i, path in enumerate(paths, 1):
        if verbose:
            print(f"  [{i:>2}/{len(paths)}] {path.name}", end="\r", flush=True)
        row: Dict[str, Any] = {
            "item_id": path.stem,
            "is_few_shot_exemplar": path.stem in config.FEW_SHOT_IDS,
            "ground_truth_path": str(path),
        }
        # The reference model is scored *as* the generation, against itself —
        # see the module docstring on what that means per dimension.
        row.update(score_model(path, path))
        rows.append(row)
    if verbose:
        print(" " * 60, end="\r")

    df = pd.DataFrame(rows)
    return df[_order_columns(df)]


# ── console report ──────────────────────────────────────────────────────────
def _n_checks() -> str:
    """The "/11" in the header and in each row, read off the check list
    itself so retiring or adding a check leaves no stale denominator here."""
    return f"/{len(syntactic.CHECKS)}"


def _fmt(v: Any, width: int = 6, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—".rjust(width)
    if isinstance(v, bool):
        return ("yes" if v else "no").rjust(width)
    if isinstance(v, float):
        return f"{v:.{nd}f}".rjust(width)
    return str(v).rjust(width)


def print_report(df: pd.DataFrame) -> None:
    """One line per PMo entry, then the aggregates a baseline is quoted from."""
    print("\n" + "=" * 130)
    print("PER-ENTRY QUALITY METRICS — PMo ground-truth models")
    print("=" * 130)
    print(f"{'item':>4} │ {'TNN':>4}{'TNG':>5}{'TNSF':>6}{'diam':>6} │ "
          f"{'syn':>6}{('ok' + _n_checks()):>7} │ {'synBEF':>7}{'perf':>6} │ "
          f"{'prag':>6}{'size':>6}{'dens':>6}{'conn':>6}{'part':>6}{'cycl':>6}"
          f"{'conc':>6}{'othr':>6} │ {'sem':>6}")
    print("─" * 130)
    for _, r in df.iterrows():
        marker = "*" if r["is_few_shot_exemplar"] else " "
        print(f"{r['item_id']:>3}{marker} │ "
              f"{_fmt(r['prag_tnn'], 4)}{_fmt(r['prag_tng'], 5)}"
              f"{_fmt(r['prag_tnsf'], 6)}{_fmt(r['prag_diameter'], 6)} │ "
              f"{_fmt(r['syn_score'])}"
              f"{(str(r['syn_checks_passed']) + _n_checks()):>7} │ "
              f"{_fmt(r['syn_bef_score'], 7)}"
              f"{_fmt(r['syn_bef_metrics_perfect'], 6)} │ "
              f"{_fmt(r['prag_score'])}"
              + "".join(_fmt(r.get(f"prag_group_{g}_score"))
                        for g in pragmatic.GROUPS)
              + f" │ {_fmt(r.get('sem_score'))}")
    print("─" * 130)
    n = len(df)

    def _mean(column: str) -> Any:
        return df[column].mean() if column in df else None

    print(f"{'mean':>4} │ "
          f"{_fmt(_mean('prag_tnn'), 4, 1)}{_fmt(_mean('prag_tng'), 5, 1)}"
          f"{_fmt(_mean('prag_tnsf'), 6, 1)}{_fmt(_mean('prag_diameter'), 6, 1)} │ "
          f"{_fmt(_mean('syn_score'))}"
          f"{_fmt(_mean('syn_checks_passed'), 7, 1)} │ "
          f"{_fmt(_mean('syn_bef_score'), 7)}"
          f"{_fmt(_mean('syn_bef_metrics_perfect'), 6, 1)} │ "
          f"{_fmt(_mean('prag_score'))}"
          + "".join(_fmt(_mean(f"prag_group_{g}_score")) for g in pragmatic.GROUPS)
          + f" │ {_fmt(_mean('sem_score'))}")
    print("=" * 130)
    print("* = few-shot exemplar (config.FEW_SHOT_IDS); scored in real runs too, "
          "with its own model swapped out of the prompt")
    print("sem = self-comparison (model vs itself). GED and common-nodes/edges are "
          "reflexive → 1.0; the two label")
    print("      similarities are not, and are not meant to be — the shared "
          "|FO_c|+|FO_g| divisor puts a perfect")
    print("      self-match at ≈0.5. None of these are baseline values.")

    # ── validity: a wiring check on this input, a real metric on generated
    # models. Printed first because everything below it presumes the file
    # parsed at all.
    if "val_dot_valid" in df:
        values = df["val_dot_valid"]
        checker = df["val_dot_checker"].dropna().iloc[0] if df["val_dot_checker"].notna().any() else "?"
        print(f"\nVALIDITY — Graphviz accepts the file as DOT [{checker}]")
        print("─" * 130)
        if values.notna().any():
            print(f"  valid DOT{'':<35} mean {values.mean():.4f}  "
                  f"{int(values.sum())}/{n} valid")
            for _, r in df[values.fillna(1.0) < 1.0].iterrows():
                print(f"    ✗ {r['item_id']}: {r['val_dot_error']}")
        else:
            print(f"  not checked — {checker}")

    # ── per-metric detail: where the reference set itself is imperfect ──
    print(f"\nSYNTACTIC — the {len(syntactic.CHECKS)} dictated checks, pass rate over {n} models")
    print("─" * 130)
    for check in syntactic.CHECKS:
        col = f"syn_{check}"
        values = df[col].astype(float)
        n_perfect = int((values == 1.0).sum())
        bar = "█" * round(20 * values.mean())
        print(f"  {syntactic.CHECK_LABELS[check]:<42} mean {values.mean():.4f}  "
              f"{n_perfect:>2}/{n} perfect  {bar}")

    print(f"\nSYNTACTIC — BEF4LLM metric set (paper Table 2/A.15), {len(syntax_rules.METRICS)} metrics")
    print("─" * 130)
    for spec in syntax_rules.METRICS:
        col = f"syn_bef_{spec.key}"
        values = df[col].astype(float)
        n_perfect = int((values == 1.0).sum())
        pooled = ""
        if not spec.boolean:
            c = df[f"{col}_conforming"].sum()
            t = df[f"{col}_covered"].sum()
            pooled = f"  pooled {int(c)}/{int(t)}" if t else "  (never applicable)"
        print(f"  #{spec.paper_no:>2} {spec.label:<48} mean {values.mean():.4f}  "
              f"{n_perfect:>2}/{n} perfect{pooled}")

    print(f"\nPRAGMATIC — {len(pragmatic.METRICS)} metrics, raw value range and mean banded score")
    print("─" * 130)
    for spec in pragmatic.METRICS:
        raw = pd.to_numeric(df[f"prag_{spec.key}"], errors="coerce")
        score = pd.to_numeric(df[f"prag_{spec.key}_score"], errors="coerce")
        direction = "higher better" if spec.higher_is_better else "lower better"
        n_zero = int((score == 0.0).sum())
        print(f"  {spec.label:<44} raw {raw.min():>8.3f} – {raw.max():<8.3f} "
              f"mean {raw.mean():>8.3f} │ score {score.mean():.3f}  "
              f"({n_zero:>2}/{n} at 0.0, {direction})")


def print_bef4llm_report(df: pd.DataFrame, theirs: pd.DataFrame) -> None:
    """Both pipelines' means, metric by metric, for the pairs that have one.

    Two numbers under one label are only worth reading with the caveat that
    binds them, so the comparability note from `MAPPINGS` is printed with the
    pair rather than left in a file.
    """
    keyed = theirs.copy()
    keyed["item_id"] = keyed["item_id"].astype(str)
    keyed = keyed.set_index("item_id")
    index = df["item_id"].astype(str)
    pairs = _bef4llm_pairs()

    failed = int((keyed.get("bef_error", pd.Series(dtype=str))
                  .fillna("").astype(str) != "").sum())
    print(f"\nBOTH PIPELINES — this port against BEF4LLM's own code, "
          f"{len(keyed)} models" + (f"  ({failed} failed)" if failed else ""))
    print("─" * 130)
    print(f"  {'metric':<44}{'port':>9}{'BEF4LLM':>10}{'Δ':>9}   comparability")
    for name in df.columns:
        pair = pairs.get(name)
        if pair is None or pair[0] not in keyed.columns:
            continue
        source, note = pair
        ours = pd.to_numeric(df[name], errors="coerce")
        them = pd.to_numeric(keyed[source].reindex(index), errors="coerce")
        if ours.isna().all() or them.isna().all():
            continue
        a, b = ours.mean(), them.mean()
        print(f"  {name:<44}{a:>9.4f}{b:>10.4f}{b - a:>+9.4f}   {note[:38]}")

    # Gateway degree now follows the original predicate and aggregation on both
    # sides. Compare each PMo item as well as the means above.
    per_model_checks = (("syn_bef_gateway_in_out_degree", "Gateway-degree"),)
    ids = df["item_id"].astype(str).reset_index(drop=True)
    for metric, label in per_model_checks:
        pair = pairs.get(metric)
        if not pair or pair[0] not in keyed.columns:
            continue
        source, _ = pair
        ours = pd.to_numeric(df[metric], errors="coerce").reset_index(drop=True)
        original_rows = keyed.reindex(index).reset_index(drop=True)
        them = pd.to_numeric(original_rows[source], errors="coerce")
        valid = ours.notna() & them.notna()
        same = (ours.sub(them).abs() <= 1e-12) & valid
        n_valid, n_same = int(valid.sum()), int(same.sum())
        print(f"\n  {label} per-model agreement: {n_same}/{n_valid} "
              f"within 1e-12 (DOT port vs original BEF4LLM)")
        if n_same == n_valid:
            continue

        mismatches = [i for i in range(len(df))
                      if valid.iloc[i] and not same.iloc[i]]
        for i in mismatches[:10]:
            print(f"    {ids.iloc[i]}: port={ours.iloc[i]:.12g}, "
                  f"BEF4LLM={them.iloc[i]:.12g}, "
                  f"delta={them.iloc[i] - ours.iloc[i]:+.3g}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score every PMo ground-truth model with both pipelines.")
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR),
                    help=f"PMo dataset root, the graphviz/ half is scored by "
                         f"this project (default: {DEFAULT_DATASET_DIR})")
    ap.add_argument("--bpmn-dir", default=None,
                    help="BPMN half, scored by BEF4LLM's own code "
                         "(default: <dataset-dir>/bpmn)")
    ap.add_argument("--bef4llm-root", default=str(DEFAULT_BEF4LLM_ROOT),
                    help=f"bef4llm checkout (default: {DEFAULT_BEF4LLM_ROOT})")
    ap.add_argument("--bef4llm-src", default=None,
                    help="their importable src/ directly, skipping the search "
                         "under --bef4llm-root")
    ap.add_argument("--bef4llm-python", default=None,
                    help="interpreter with pm4py/seaborn/xmlschema/networkx "
                         "(default: .venv-bef4llm beside the project)")
    ap.add_argument("--bef4llm-verify", type=int, default=0, metavar="N",
                    help="re-score the first N models the naive way and assert "
                         "their single-pass aggregates match")
    ap.add_argument("--no-bef4llm", action="store_true",
                    help="skip their side; the workbook is written without it")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"CSV to write (default: {DEFAULT_OUT.name})")
    ap.add_argument("--quiet", action="store_true", help="write the CSV, print no report")
    ap.add_argument("--no-legend", action="store_true",
                    help="skip the <out>_columns.csv column legend")
    ap.add_argument("--no-xlsx", action="store_true",
                    help="skip the <out>.xlsx workbook (data + legend + header notes)")
    ap.add_argument("--xlsx-all-columns", action="store_true",
                    help="put every column in the workbook, not only BEF4LLM's "
                         "syntactic/pragmatic/semantic metric sets")
    args = ap.parse_args()

    out = Path(args.out)
    dataset_dir = Path(args.dataset_dir)

    print(f"Scoring PMo ground-truth models from {dataset_dir}")
    df = score_dataset(dataset_dir)

    theirs: Optional[pd.DataFrame] = None
    provenance = ""
    their_csv: Optional[Path] = None
    if not args.no_bef4llm:
        src = (Path(args.bef4llm_src) if args.bef4llm_src
               else resolve_bef4llm_src(Path(args.bef4llm_root)))
        bpmn_dir = Path(args.bpmn_dir) if args.bpmn_dir else dataset_dir / "bpmn"
        if not bpmn_dir.is_dir():
            raise SystemExit(f"No BPMN directory at {bpmn_dir} — name one with "
                             f"--bpmn-dir, or skip with --no-bef4llm.")
        print(f"Scoring the same models with BEF4LLM's own code from {src}")
        their_csv = bef4llm_path(out)
        theirs, provenance = score_bef4llm(
            bpmn_dir, src, their_csv,
            python_exe=args.bef4llm_python, verify=args.bef4llm_verify)

    df.to_csv(out, index=False)
    legend = None if args.no_legend else write_legend(df, out)
    workbook = None if args.no_xlsx else write_workbook(
        df, out, theirs=theirs, provenance=provenance,
        bef4llm_only=not args.xlsx_all_columns)

    if not args.quiet:
        print_report(df)
        if theirs is not None:
            print_bef4llm_report(df, theirs)
    print(f"\nWrote {len(df)} rows × {len(df.columns)} columns → {out}")
    if their_csv is not None and theirs is not None:
        print(f"Wrote BEF4LLM's own scores ({len(theirs)} rows × "
              f"{len(theirs.columns)} columns) → {their_csv}")
    if legend is not None:
        print(f"Wrote the column legend ({len(df.columns)} rows) → {legend}")
    if workbook is not None:
        scope = ("every column" if args.xlsx_all_columns
                 else "BEF4LLM's syntactic/pragmatic/semantic metrics only")
        print(f"Wrote the workbook ({scope}, notes on every header) → {workbook}")


if __name__ == "__main__":
    main()
