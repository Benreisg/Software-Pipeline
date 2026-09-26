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
re-score cannot corrupt what a paid run produced. One consequence worth knowing:
a run that scored itself during generation keeps **those** quality columns in
`results.jsonl` afterwards, so after a metric change that file and the rebuilt
`results.html` / `csv/` state different numbers. The rebuilt ones are current;
`quality.scored_frame` reads the file and would report the stale ones, which is
why `--workers` exists here rather than a rewrite of the log.

`--workers N` spreads the scoring over N processes. Since the semantic dimension
grew the graph edit distance (bounded per pair by a wall-clock budget), a large
run is no longer a minutes-long job: 25217 models take about 33 hours in one
process.
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
    parser.add_argument("--cache", type=Path, default=None, metavar="FILE",
                        help="Append each scored row to FILE and resume from it "
                             "on a later call. For a run of this size the "
                             "scoring is measured in hours, so an interruption "
                             "without one costs all of it.")
    parser.add_argument("--workers", type=int, default=1, metavar="N",
                        help="Score over N processes (default 1). Scoring one "
                             "generation touches nothing shared, and the "
                             "semantic graph edit distance is bounded by a "
                             "wall-clock budget per pair, so a large run is "
                             "worth spreading out: 25217 models take ~33 h in "
                             "one process and ~2.5 h over 14.")
    args = parser.parse_args(argv)

    if not args.run_dir.is_dir():
        sys.exit(f"Not a directory: {args.run_dir}")
    if not (args.run_dir / "results.jsonl").exists():
        sys.exit(f"No results.jsonl in {args.run_dir} — is that a run directory?")

    scored = score_run(args.run_dir, workers=args.workers,
                       cache_path=args.cache)
    if not args.no_report:
        out = results_report.build(args.run_dir, scored=scored)
        print(f"  [report] wrote {out}")
    if not args.no_csv:
        out = csv_export.build(args.run_dir, scored=scored)
        print(f"  [csv] wrote {out}")


if __name__ == "__main__":
    main()
