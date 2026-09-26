"""
quality/validity.py — Validity (DOT format check)
=====================================================
Is the output a well-formed DOT artefact at all? The gate every other dimension
sits behind: a model that does not parse cannot be scored for syntax, semantics
or pragmatics.

Definition used here
--------------------
    **DOT validity = Graphviz accepts the artefact as valid DOT language.**

The judge is Graphviz itself, not a re-implementation of its grammar: the
`nop` tool with `-p`, which the Graphviz documentation describes as producing
"no output, just checks the input for valid DOT language". Exit code 0 = valid,
exit code 1 + a message on stderr = invalid:

    echo 'digraph {}' | nop -p        → exit 0, silent
    echo 'digraph {'  | nop -p        → exit 1, "Error: <stdin>: syntax error in line 2"

Two things this deliberately does NOT claim:

- It says nothing about whether the graph is a meaningful BPMN process model.
  That is what the syntactic, semantic and pragmatic dimensions are for. A
  `digraph { a -> a }` is perfectly valid DOT and a nonsense process.
- It judges the **extracted** artefact (`generated/<...>.gv`), not the raw
  reply. Markdown fences and prose around the graph are removed during
  generation by `postprocess.extract_and_validate()`, whose own
  `extract_parse_ok` / `extract_error` columns cover *that* step (pydot-level).
  Measured against the raw reply almost every model would fail, because almost
  every model wraps its answer in ``` fences — which is a presentation
  convention, not a format error. The two signals stack: `extract_parse_ok`
  says something DOT-shaped could be pulled out of the reply, `val_dot_valid`
  says Graphviz accepts what was pulled out.

One correction on top of the bare exit code, verified against Graphviz 15.0.0:
`nop -p` accepts an **empty** document (empty input, whitespace, comments only)
as valid DOT and exits 0. A model that returned nothing would therefore score
1.0. So a text that declares no graph at all is reported invalid here, with
`no graph declared in the artefact` as the reason, before Graphviz is asked.

Columns
-------
    val_dot_valid    1.0 / 0.0 / None — None = the check could not be run
                     (Graphviz missing, timeout), never a silent 0
    val_dot_error    Graphviz's own message, or why the check could not run
    val_dot_checker  which binary answered, with its version
    val_score        the dimension's headline. The mean over its implemented
                     checks — today the DOT check is the only member, so the
                     two coincide; the name is what aggregate.py and the run
                     report read, and stays stable if a second check is added.

Requires Graphviz on PATH (`nop`, shipped with every Graphviz install). Set
$GRAPHVIZ_NOP to point at the binary directly if it is installed somewhere
unusual. Without it the dimension reports None and says so, rather than
failing the run or scoring zeroes.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from .graph import ProcessGraph

# Env override for a Graphviz that is installed but not on PATH.
NOP_ENV_VAR = "GRAPHVIZ_NOP"

# Where Graphviz's installer puts things on Windows, where PATH is often not
# updated. Checked only after PATH; harmless (and skipped) on other platforms.
_FALLBACK_PATHS = (
    r"C:\Program Files\Graphviz\bin\nop.exe",
    r"C:\Program Files (x86)\Graphviz\bin\nop.exe",
)

# Does the artefact declare a graph at all? Accepts every form nop does:
# `graph`, `digraph`, `strict graph`, `strict digraph`. See the empty-document
# note in the module docstring for why this guard exists.
_GRAPH_DECL = re.compile(r"\b(?:strict\s+)?(?:di)?graph\b", re.IGNORECASE)

_TIMEOUT_S = 30


@dataclass
class DotCheck:
    """One verdict from the Graphviz checker."""
    valid: Optional[bool]   # None = not checkable, NOT invalid
    error: str = ""         # Graphviz's message, or why no verdict was possible
    checker: str = ""       # binary + version that answered


@lru_cache(maxsize=1)
def nop_executable() -> Optional[str]:
    """Path to Graphviz's `nop`, or None. Resolved once per process."""
    override = os.environ.get(NOP_ENV_VAR)
    if override and Path(override).exists():
        return override
    found = shutil.which("nop")
    if found:
        return found
    for candidate in _FALLBACK_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def _graphviz_version() -> str:
    """' (graphviz X.Y.Z)' if it can be determined, else ''.

    `nop` itself has no version flag — its usage lists only `-p` and `-?`, and
    `-V` makes it print usage and exit 1 — so the version is read from `dot` in
    the same installation directory (falling back to whatever `dot` is on
    PATH). Cosmetic: it only labels the verdict for the record.
    """
    candidates = []
    exe = nop_executable()
    if exe:
        candidates.append(Path(exe).with_name("dot" + Path(exe).suffix))
    on_path = shutil.which("dot")
    if on_path:
        candidates.append(Path(on_path))

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            proc = subprocess.run([str(candidate), "-V"], capture_output=True,
                                  stdin=subprocess.DEVNULL, timeout=_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            continue
        raw = ((proc.stderr or b"") + (proc.stdout or b"")).decode("utf-8", "replace")
        match = re.search(r"version\s+(\S+)", raw)
        if match:
            return f" (graphviz {match.group(1)})"
    return ""


@lru_cache(maxsize=1)
def checker_label() -> str:
    """'nop -p (graphviz X.Y.Z)', or why there is no checker. Resolved once."""
    exe = nop_executable()
    if exe is None:
        return (f"unavailable — Graphviz 'nop' not found on PATH; install Graphviz "
                f"or set ${NOP_ENV_VAR}")
    return f"nop -p{_graphviz_version()}"


def check_dot(text: str) -> DotCheck:
    """Ask Graphviz whether `text` is valid DOT.

    The text is handed to `nop -p` on stdin as UTF-8 bytes — encoded here
    rather than through `text=True`, which would use the console's code page
    and can neither carry a label like "Prüfung – Angebot" nor be trusted to
    leave the bytes alone (a UTF-8 BOM in front of `digraph` is itself a
    syntax error to Graphviz).
    """
    exe = nop_executable()
    label = checker_label()
    if exe is None:
        return DotCheck(valid=None, error=label, checker=label)

    if not _GRAPH_DECL.search(text or ""):
        # Graphviz would exit 0 on this: an empty document is vacuously valid
        # DOT. As a metric over model output it is a failure, not a pass.
        return DotCheck(valid=False, error="no graph declared in the artefact",
                        checker=label)

    try:
        proc = subprocess.run(
            [exe, "-p"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return DotCheck(valid=None, error=f"{exe} timed out after {_TIMEOUT_S}s",
                        checker=label)
    except OSError as exc:
        return DotCheck(valid=None, error=f"could not run {exe}: {exc}", checker=label)

    if proc.returncode == 0:
        return DotCheck(valid=True, error="", checker=label)

    message = (proc.stderr or b"").decode("utf-8", "replace").strip()
    message = " | ".join(line.strip() for line in message.splitlines() if line.strip())
    return DotCheck(valid=False, error=message or f"nop -p exited {proc.returncode}",
                    checker=label)


def _columns(check: DotCheck) -> Dict[str, Any]:
    score = None if check.valid is None else float(check.valid)
    return {
        "val_score": score,
        "val_dot_valid": score,
        "val_dot_error": check.error,
        "val_dot_checker": check.checker,
    }


def empty() -> Dict[str, Any]:
    """The full column set, unmeasured. For rows where no verdict is possible —
    a generation that never returned, or a file that is gone."""
    return _columns(DotCheck(valid=None, error="", checker=checker_label()))


def invalid(reason: str) -> Dict[str, Any]:
    """A verdict of 0.0 without asking Graphviz, for output that never got as
    far as an artefact — a reply with no DOT block in it is a format failure,
    not a missing measurement."""
    return _columns(DotCheck(valid=False, error=reason, checker=checker_label()))


def evaluate(generated: Optional[ProcessGraph],
             raw_dot: Optional[str] = None,
             parse_error: str = "",
             source_path=None) -> Dict[str, Any]:
    """Return the `val_`-prefixed columns for one generated artefact.

    Give it the DOT text (`raw_dot`) or the file to read it from
    (`source_path`); with neither, the columns come back unmeasured. Graphviz
    judges the bytes, so `generated` (this project's own parse, None when it
    failed) and `parse_error` are not consulted — the point of this dimension
    is a verdict independent of `quality/graph.py`.
    """
    text = raw_dot
    if text is None and source_path is not None:
        try:
            text = Path(source_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _columns(DotCheck(valid=None, error=f"could not read {source_path}: {exc}",
                                     checker=checker_label()))
    if text is None:
        return empty()
    return _columns(check_dot(text))
