"""
postprocess.py — extract & validate DOT from a raw model reply
==================================================================
Generation-only failures (couldn't extract / couldn't parse) are reported,
never silently dropped (DC6 Robustness): the caller still gets a row with
evaluable=False rather than the item disappearing from the results.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional

import pydot

# pydot is NOT thread-safe. Its parser (pydot.dot_parser) binds parse actions to
# module-level pyparsing grammar objects and pushes results through module-level
# state, so two threads parsing at the same time corrupt each other's parse.
# The failure is not a wrong graph but a confusing TypeError from inside pydot -
# "push_ID() missing 1 required positional argument: 'toks'" - which this module
# would then report as an ordinary `parse error: ...` on a perfectly valid DOT
# document. Measured on a parallel mock run of 104 replies: 102 of 104 valid
# graphs came back unparseable that way.
#
# Parsing is local, fast and a tiny share of a run that spends its time waiting
# on HTTP, so serialising it costs nothing worth measuring.
#
# Any other pydot caller that may run beside this one has to take the SAME lock -
# the state being protected belongs to pydot, not to this module.
# `quality/graph.py` also parses, but only in the scoring pass, which runs after
# generation on one thread; parallelise that and it needs this lock too.
DOT_PARSE_LOCK = threading.Lock()

_FENCED_BLOCK = re.compile(r"```(?:dot|graphviz)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DIGRAPH_START = re.compile(r"\b(strict\s+)?digraph\b", re.IGNORECASE)


@dataclass
class ExtractResult:
    dot_text: Optional[str]
    parse_ok: bool
    error: str = ""


def _extract_braced_block(text: str, start: int) -> Optional[str]:
    """From `start` (index of 'digraph'/'strict'), return the substring up to
    the matching closing brace of the first '{', balancing nested braces."""
    open_idx = text.find("{", start)
    if open_idx == -1:
        return None
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # unbalanced — never closed


def extract_dot(raw_reply: str) -> str:
    """Best-effort extraction of a DOT graph from a raw model reply. Does not
    validate parseability — see `extract_and_validate` for that."""
    fenced = _FENCED_BLOCK.search(raw_reply)
    if fenced:
        candidate = fenced.group(1).strip()
        m = _DIGRAPH_START.search(candidate)
        if m:
            block = _extract_braced_block(candidate, m.start())
            if block:
                return block
        if candidate:
            return candidate

    m = _DIGRAPH_START.search(raw_reply)
    if m:
        block = _extract_braced_block(raw_reply, m.start())
        if block:
            return block

    return raw_reply.strip()


def extract_and_validate(raw_reply: str) -> ExtractResult:
    if not raw_reply or not raw_reply.strip():
        return ExtractResult(dot_text=None, parse_ok=False, error="empty reply")

    dot_text = extract_dot(raw_reply)
    if not dot_text or not _DIGRAPH_START.search(dot_text):
        return ExtractResult(dot_text=dot_text or None, parse_ok=False, error="no 'digraph' found in reply")

    try:
        with DOT_PARSE_LOCK:
            graphs = pydot.graph_from_dot_data(dot_text)
        if not graphs:
            return ExtractResult(dot_text=dot_text, parse_ok=False, error="pydot returned no graphs")
    except Exception as exc:
        return ExtractResult(dot_text=dot_text, parse_ok=False, error=f"parse error: {exc}")

    return ExtractResult(dot_text=dot_text, parse_ok=True, error="")
