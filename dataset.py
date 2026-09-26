"""
dataset.py — PMo dataset access (Graphviz/DOT branch)
=====================================================

The PMo dataset (55 pairs) keeps the two halves of an item in two sibling
folders, matched by file number:

    <pmo-dataset>/descriptions/01.txt   ← natural-language process description
                                          (this is the INPUT given to the LLM)
    <pmo-dataset>/graphviz/01.dot       ← ground-truth process model in DOT
                                          (this is what generations are scored
                                          against)

So `01.txt` and `01.dot` are one item; the stem (`01`) is its `item_id` and
appears in every output filename and results row.

This differs from the MaD layout this branch previously used, where a single
`.gv` file carried the description in its graph-level `label="..."` attribute
and the model structure in its body. Nothing outside this module needs to know:
`ProcessItem` is unchanged, so generation and aggregation are untouched by the
switch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# No DOT parsing happens here: descriptions come from plain .txt files, and the
# ground-truth .dot is only ever passed on as a path — whatever scores it later
# opens it itself. pydot is still used by postprocess.py to validate replies.

DESCRIPTIONS_SUBDIR = "descriptions"
GRAPHVIZ_SUBDIR = "graphviz"
_GT_SUFFIXES = (".dot", ".gv")     # PMo ships .dot; .gv accepted for tolerance


@dataclass
class ProcessItem:
    item_id: str            # logical id (the shared file stem, or a CSV id/row number)
    description: str        # the textual process description (LLM input)
    ground_truth_path: Optional[Path] = None
    #   path to the reference model for this item. Carried through to
    #   results.csv (`ground_truth_path`) for the quality scoring that is
    #   currently being restructured; the run itself does not open it.
    #   None  → generation-only: tokens/cost/latency are still recorded, but
    #           there is nothing to compare the output against. This is what
    #           the CSV-input mode produces when a row carries only a
    #           description.


def resolve_dirs(dataset_dir: Path) -> Tuple[Path, Path]:
    """Return (descriptions_dir, graphviz_dir) for a PMo dataset location.

    Accepts the dataset root *or* either of its two subfolders, so
    `--dataset-dir .../pmo-dataset`, `.../pmo-dataset/graphviz` and
    `.../pmo-dataset/descriptions` all work — the sibling folder is derived.
    """
    p = Path(dataset_dir)
    desc, gv = p / DESCRIPTIONS_SUBDIR, p / GRAPHVIZ_SUBDIR
    if desc.is_dir() and gv.is_dir():
        return desc, gv
    if p.name in (DESCRIPTIONS_SUBDIR, GRAPHVIZ_SUBDIR):
        desc, gv = p.parent / DESCRIPTIONS_SUBDIR, p.parent / GRAPHVIZ_SUBDIR
        if desc.is_dir() and gv.is_dir():
            return desc, gv
    raise ValueError(
        f"{p} is not a PMo dataset directory — expected it to contain both a "
        f"'{DESCRIPTIONS_SUBDIR}/' and a '{GRAPHVIZ_SUBDIR}/' folder (or to be "
        f"one of those two folders)."
    )


def ground_truth_path_for(item_id: str, graphviz_dir: Path) -> Optional[Path]:
    for suffix in _GT_SUFFIXES:
        p = Path(graphviz_dir) / f"{item_id}{suffix}"
        if p.exists():
            return p
    return None


def description_path_for(gt_path: Path) -> Path:
    """The `descriptions/<id>.txt` belonging to a `graphviz/<id>.dot`."""
    gt_path = Path(gt_path)
    desc_dir, _ = resolve_dirs(gt_path.parent)
    return desc_dir / f"{gt_path.stem}.txt"


def read_description(txt_path: Path) -> str:
    """Read one description file.

    The paragraph structure of the source file is preserved (one sentence per
    line in PMo) — only line endings are normalised and surrounding blank space
    trimmed, so the text handed to the model is what the dataset ships.
    """
    text = Path(txt_path).read_text(encoding="utf-8-sig")
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def load_item(gt_path: Path) -> ProcessItem:
    """Load one item from the path of its ground-truth model file."""
    gt_path = Path(gt_path)
    desc_path = description_path_for(gt_path)
    if not desc_path.exists():
        raise ValueError(
            f"No description found for {gt_path.name}: expected {desc_path}. "
            "PMo items are a <id>.dot / <id>.txt pair sharing one file number."
        )
    description = read_description(desc_path)
    if not description:
        raise ValueError(f"Description file is empty: {desc_path}")
    return ProcessItem(
        item_id=gt_path.stem,
        description=description,
        ground_truth_path=gt_path,
    )


def exemplar_paths(dataset_dir: Path, item_ids: List[str]) -> List[Path]:
    """Ground-truth paths for the given item ids, for use as few-shot examples.

    Raises on an id that has no model file, rather than silently running a
    few-shot strategy with fewer examples than the run was configured for.
    """
    _, gv_dir = resolve_dirs(dataset_dir)
    paths: List[Path] = []
    for item_id in item_ids:
        p = ground_truth_path_for(item_id, gv_dir)
        if p is None:
            raise FileNotFoundError(
                f"No ground-truth model for exemplar id {item_id!r} in {gv_dir} "
                f"(looked for {item_id}{' / '.join(_GT_SUFFIXES)})."
            )
        paths.append(p)
    return paths


def load_dataset(
    dataset_dir: Path,
    limit: Optional[int] = None,
    sort: bool = True,
    exclude_ids: Optional[set] = None,
) -> List[ProcessItem]:
    """Load all `descriptions/<id>.txt` + `graphviz/<id>.dot` pairs.

    Parameters
    ----------
    limit : cap the number of items (for cost-bounded sample runs).
    exclude_ids : item ids to skip (e.g. the few-shot exemplars, to prevent
                  evaluating on examples the model was shown).
    """
    desc_dir, gv_dir = resolve_dirs(dataset_dir)

    paths = [p for suffix in _GT_SUFFIXES for p in gv_dir.glob(f"*{suffix}")]
    if sort:
        # PMo ids are zero-padded ("01".."55"), so lexicographic order is
        # numeric order; sorting on the stem keeps mixed suffixes together.
        paths.sort(key=lambda p: p.stem)
    exclude_ids = exclude_ids or set()

    # A description with no model (or vice versa) is a dataset problem, not a
    # run problem — report it once instead of letting the item vanish silently.
    gv_ids = {p.stem for p in paths}
    desc_ids = {p.stem for p in desc_dir.glob("*.txt")}
    for orphan in sorted(desc_ids - gv_ids):
        print(f"  [dataset] {orphan}.txt has no model in {gv_dir.name}/ — not loadable")

    items: List[ProcessItem] = []
    for p in paths:
        if p.stem in exclude_ids:
            continue
        try:
            items.append(load_item(p))
        except Exception as exc:  # keep going; report unusable pairs
            print(f"  [dataset] skipping {p.name}: {exc}")
        if limit is not None and len(items) >= limit:
            break
    return items


# ──────────────────────────────────────────────────────────────────────────
#  CSV input
# ──────────────────────────────────────────────────────────────────────────
# A CSV is the convenient way to hand the pipeline a batch of process
# descriptions to generate for. One row = one process item. The cross product
# Prompt × Model × (CSV rows) is then run exactly as for the .gv dataset.
#
# Expected columns (matched case-insensitively; first match wins). All but the
# description column are optional:
#   description     (REQUIRED)  the textual process description, the LLM input
#                   aliases: process_description, text, process, input,
#                            prozessbeschreibung, beschreibung
#   id              (optional)  a stable id used in output filenames / tables
#                   aliases: item_id, process_id, name, case_id
#                   → if absent, ids are generated as row_0001, row_0002, …
#   ground_truth    (optional)  path to a reference .gv for scoring this row
#                   aliases: ground_truth_path, gv_path, gv, reference,
#                            reference_path
#                   relative paths are resolved against the CSV's own folder;
#                   if absent/empty the row runs in generation-only mode.
#
# German Excel exports often use ';' as the separator — the delimiter is sniffed
# automatically, or can be forced via `delimiter`.

_DESC_ALIASES = [
    "description", "process_description", "text", "process", "input",
    "nl_description", "prozessbeschreibung", "beschreibung",
]
_ID_ALIASES = ["id", "item_id", "process_id", "name", "case_id"]
_GT_ALIASES = [
    "ground_truth", "ground_truth_path", "groundtruth", "gv_path", "gv",
    "reference", "reference_path", "ground_truth_gv",
]


def _resolve_col(columns, aliases, override=None):
    """Return the actual column name matching one of `aliases` (case-insensitive),
    or `override` if given and present, else None."""
    lower = {str(c).strip().lower(): c for c in columns}
    if override:
        if override in columns:
            return override
        if override.strip().lower() in lower:
            return lower[override.strip().lower()]
        raise KeyError(f"Column {override!r} not found. Available: {list(columns)}")
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def load_dataset_from_csv(
    csv_path: Path,
    limit: Optional[int] = None,
    exclude_ids: Optional[set] = None,
    *,
    description_col: Optional[str] = None,
    id_col: Optional[str] = None,
    ground_truth_col: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> List[ProcessItem]:
    """Load process items from a CSV file. See the module note above for the
    accepted column names. Returns one `ProcessItem` per usable row."""
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Robust read: sniff the delimiter (handles ';' exports), keep everything as
    # strings, and do NOT turn empty cells into NaN floats.
    read_kwargs = dict(dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if delimiter:
        df = pd.read_csv(csv_path, sep=delimiter, **read_kwargs)
    else:
        df = pd.read_csv(csv_path, sep=None, engine="python", **read_kwargs)

    if df.shape[1] == 0:
        raise ValueError(f"CSV {csv_path.name} has no columns.")

    desc_c = _resolve_col(df.columns, _DESC_ALIASES, description_col)
    if desc_c is None:
        if df.shape[1] == 1:
            desc_c = df.columns[0]          # single-column file → that's the text
        else:
            raise ValueError(
                f"Could not find a description column in {csv_path.name}. "
                f"Columns present: {list(df.columns)}. Expected one of "
                f"{_DESC_ALIASES}, or pass --csv-description-col."
            )
    id_c = _resolve_col(df.columns, _ID_ALIASES, id_col)
    gt_c = _resolve_col(df.columns, _GT_ALIASES, ground_truth_col)

    exclude_ids = exclude_ids or set()
    csv_dir = csv_path.parent

    items: List[ProcessItem] = []
    seen_ids: dict = {}
    n_gt = 0
    for i, (_, row) in enumerate(df.iterrows()):
        description = str(row[desc_c]).strip()
        if not description:
            print(f"  [csv] row {i + 1}: empty description — skipped")
            continue

        # id (auto-generate + de-duplicate so output files never collide)
        raw_id = str(row[id_c]).strip() if id_c else ""
        item_id = raw_id or f"row_{i + 1:04d}"
        if item_id in seen_ids:
            seen_ids[item_id] += 1
            item_id = f"{item_id}_{seen_ids[item_id]}"
            print(f"  [csv] row {i + 1}: duplicate id → renamed to {item_id}")
        else:
            seen_ids[item_id] = 1

        if item_id in exclude_ids:
            continue

        # optional ground truth
        gt_path: Optional[Path] = None
        if gt_c:
            raw_gt = str(row[gt_c]).strip()
            if raw_gt:
                p = Path(raw_gt)
                if not p.is_absolute():
                    p = (csv_dir / p).resolve()
                if p.exists():
                    gt_path = p
                    n_gt += 1
                else:
                    print(f"  [csv] row {i + 1} ({item_id}): ground-truth file "
                          f"not found: {raw_gt} — running generation-only")

        items.append(ProcessItem(item_id=item_id, description=description,
                                 ground_truth_path=gt_path))
        if limit is not None and len(items) >= limit:
            break

    mode = (f"{n_gt}/{len(items)} rows have ground truth (others generation-only)"
            if n_gt else "generation-only (no ground-truth column populated)")
    print(f"  [csv] loaded {len(items)} item(s) from {csv_path.name} — {mode}")
    return items
