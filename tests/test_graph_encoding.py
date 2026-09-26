"""UTF-8 artifacts must not depend on the operating system's text encoding."""
from __future__ import annotations

import pytest
import pydot

from quality import graph, score


@pytest.mark.parametrize("label", ["Pr\u00fcfung", "Approve \u2713", "Send \u201cOK\u201d"])
def test_load_preserves_utf8_labels_with_a_windows_default(tmp_path, monkeypatch, label):
    path = tmp_path / "unicode.gv"
    path.write_text(
        'digraph G { s [shape=circle]; t [shape=box, label="' + label
        + '"]; e [shape=doublecircle]; s -> t -> e; }', encoding="utf-8")
    real_read = pydot.graph_from_dot_file

    def windows_read(path, encoding=None):
        return real_read(path, encoding=encoding or "cp1252")

    monkeypatch.setattr(pydot, "graph_from_dot_file", windows_read)
    pg = graph.load(path)
    assert pg.nodes["t"]["label"] == label
    assert pg.tasks == {"t"}
    assert pg.unique_edges == {("s", "t"), ("t", "e")}


def test_locale_dependent_checkpoint_is_not_reused(tmp_path):
    import json
    header = score._cache_header(1)
    assert header["dot_encoding"] == "utf-8"
    del header["dot_encoding"]
    cache = tmp_path / "old_scoring.jsonl"
    cache.write_text(json.dumps(header) + '\n{"syn_score": null}\n', encoding="utf-8")
    assert score._load_cache(cache, 1, verbose=False) == []
