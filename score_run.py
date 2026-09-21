#!/usr/bin/env python3
"""
score_run.py — re-score a finished run and rebuild its results page
=======================================================================
Scoring is decoupled from generation: this reads a run directory, scores every
model in `generated/` against the metrics as they stand today, and rewrites
`results.html`. No API calls, nothing regenerated, nothing spent — so it can be
re-run as often as the metrics change.

    python score_run.py runs/<run_id>

A run started with `python run.py` scores itself and writes the page at the end
already (unless `--no-score`); this is for re-scoring an older run, or after
editing a metric.

The run's `csv/` export is rewritten from the same scored frame, so the tables
and the page always state the same numbers; `--no-csv` skips it.

Nothing else is written. The run's data files — `manifest.json`,
`results.jsonl`, `raw/`, `generated/` — are inputs and are never modified, so a
re-score cannot corrupt what a paid run produced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import csv_export
import results_report
from quality import score_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, e.g. runs/20260815_004512")
    parser.add_argument("--no-report", action="store_true",
                        help="Score only; don't rewrite results.html")
    parser.add_argument("--no-csv", action="store_true",
                        help="Don't rewrite the run's csv/ export")
    args = parser.parse_args(argv)

    if not args.run_dir.is_dir():
        sys.exit(f"Not a directory: {args.run_dir}")
    if not (args.run_dir / "results.jsonl").exists():
        sys.exit(f"No results.jsonl in {args.run_dir} — is that a run directory?")

    scored = score_run(args.run_dir)
    if not args.no_report:
        out = results_report.build(args.run_dir, scored=scored)
        print(f"  [report] wrote {out}")
    if not args.no_csv:
        out = csv_export.build(args.run_dir, scored=scored)
        print(f"  [csv] wrote {out}")


if __name__ == "__main__":
    main()
