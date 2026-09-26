"""
quality/textsim.py — label-text primitives for the semantic dimension
=========================================================================
Tokenisation, stopword removal, stemming, string edit distance and synonym
lookup — the text layer under `quality/semantic.py`'s label-similarity
metrics (BEF4LLM Table A.17, metrics 1–3, via Dijkman et al. 2011).

Everything except the synonym lookup is **pure stdlib and deterministic**:
the same label produces the same tokens, stems and distances on every
machine, with no corpus download and no model file. That matters because
these values feed published scores — a metric that silently changes with
the local NLTK version is not reproducible.

The one thing the stdlib cannot provide is WordNet. The synonym term of
semantic label similarity (`ws · (s(w1,w2) + s(w2,w1))`) therefore uses
NLTK's WordNet **if it is installed and its corpus is present**, and
otherwise contributes 0 — with `sem_wordnet` recording per row which of
the two modes computed the score, so the degradation is never silent.
Install with:

    pip install nltk
    python -m nltk.downloader wordnet

Differences from BEF4LLM's `language_similarity/lanuage_utils.py`, all
deliberate:

- **Tokens are pure alphanumeric words.** The reference splits on
  whitespace and then substitutes special characters with spaces *inside*
  a token, so "re-check" survives as the single word "re check". Here
  "re-check" is two words. Affects word counts, not semantics.
- **The stemmer is the original Porter (1980) algorithm**, implemented
  here, not NLTK's `PorterStemmer` (which applies its own extensions by
  default). Same paper-citable algorithm on every machine.
- **The stopword list is embedded** (NLTK's canonical English list),
  not downloaded at runtime. English only — the PMo corpus is English;
  the reference's German branch is not ported.
"""
from __future__ import annotations

import threading

import re
from functools import lru_cache
from typing import FrozenSet, List

# ── tokenisation ────────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(label: str) -> List[str]:
    """Lowercased alphanumeric words, in order. '' → []."""
    return _WORD.findall(label.lower())


# NLTK's English stopword list, embedded verbatim. Contractions are listed in
# their written form and normalised through `tokenize` below, so both halves
# ("you're" → "you", "re") land in the final set.
_STOPWORDS_RAW = """
i me my myself we our ours ourselves you you're you've you'll you'd your
yours yourself yourselves he him his himself she she's her hers herself it
it's its itself they them their theirs themselves what which who whom this
that that'll these those am is are was were be been being have has had
having do does did doing a an the and but if or because as until while of
at by for with about against between into through during before after
above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other
some such no nor not only own same so than too very s t can will just don
don't should should've now d ll m o re ve y ain aren aren't couldn
couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven haven't
isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't
shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
"""

STOPWORDS: FrozenSet[str] = frozenset(
    w for token in _STOPWORDS_RAW.split() for w in tokenize(token)
)


def content_words(label: str) -> List[str]:
    """Tokens with stopwords removed — the word list the formulas call w."""
    return [w for w in tokenize(label) if w not in STOPWORDS]


# ── Levenshtein distance ────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Plain edit distance (insert/delete/substitute, all cost 1)."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1,          # delete from a
                           cur[j - 1] + 1,       # insert into a
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ── Porter stemmer (Porter 1980, "An algorithm for suffix stripping") ───────
# The original published algorithm, without the later revisions that NLTK
# applies by default. Words of length ≤ 2 are returned unchanged, as in every
# standard implementation.

_VOWELS = "aeiou"


def _cons(w: str, i: int) -> bool:
    c = w[i]
    if c in _VOWELS:
        return False
    if c == "y":
        return True if i == 0 else not _cons(w, i - 1)
    return True


def _measure(w: str) -> int:
    """m in Porter's [C](VC)^m[V] decomposition of a stem."""
    m, i, n = 0, 0, len(w)
    while i < n and _cons(w, i):
        i += 1
    while i < n:
        while i < n and not _cons(w, i):
            i += 1
        if i >= n:
            break
        m += 1
        while i < n and _cons(w, i):
            i += 1
    return m


def _has_vowel(w: str) -> bool:
    return any(not _cons(w, i) for i in range(len(w)))


def _ends_double_cons(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    """*o: stem ends consonant-vowel-consonant, final not w, x or y."""
    if len(w) < 3:
        return False
    i = len(w) - 3
    return (_cons(w, i) and not _cons(w, i + 1) and _cons(w, i + 2)
            and w[-1] not in "wxy")


def _longest_rule(w: str, rules):
    """Porter's within-step semantics: only the rule with the longest
    matching suffix is considered; if its condition fails, no other rule in
    the step fires."""
    best = None
    for suffix, replacement, min_m in rules:
        if w.endswith(suffix) and (best is None or len(suffix) > len(best[0])):
            best = (suffix, replacement, min_m)
    if best is None:
        return w
    suffix, replacement, min_m = best
    stem = w[: len(w) - len(suffix)]
    if _measure(stem) > min_m:
        return stem + replacement
    return w


_STEP2 = [("ational", "ate", 0), ("tional", "tion", 0), ("enci", "ence", 0),
          ("anci", "ance", 0), ("izer", "ize", 0), ("abli", "able", 0),
          ("alli", "al", 0), ("entli", "ent", 0), ("eli", "e", 0),
          ("ousli", "ous", 0), ("ization", "ize", 0), ("ation", "ate", 0),
          ("ator", "ate", 0), ("alism", "al", 0), ("iveness", "ive", 0),
          ("fulness", "ful", 0), ("ousness", "ous", 0), ("aliti", "al", 0),
          ("iviti", "ive", 0), ("biliti", "ble", 0)]

_STEP3 = [("icate", "ic", 0), ("ative", "", 0), ("alize", "al", 0),
          ("iciti", "ic", 0), ("ical", "ic", 0), ("ful", "", 0),
          ("ness", "", 0)]

_STEP4 = [("al", "", 1), ("ance", "", 1), ("ence", "", 1), ("er", "", 1),
          ("ic", "", 1), ("able", "", 1), ("ible", "", 1), ("ant", "", 1),
          ("ement", "", 1), ("ment", "", 1), ("ent", "", 1), ("ou", "", 1),
          ("ism", "", 1), ("ate", "", 1), ("iti", "", 1), ("ous", "", 1),
          ("ive", "", 1), ("ize", "", 1)]


@lru_cache(maxsize=65536)
def porter_stem(word: str) -> str:
    w = word
    if len(w) <= 2:
        return w

    # step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif not w.endswith("ss") and w.endswith("s"):
        w = w[:-1]

    # step 1b
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        stripped = None
        if w.endswith("ed") and _has_vowel(w[:-2]):
            stripped = w[:-2]
        elif w.endswith("ing") and _has_vowel(w[:-3]):
            stripped = w[:-3]
        if stripped is not None:
            w = stripped
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif _ends_double_cons(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif _measure(w) == 1 and _cvc(w):
                w += "e"

    # step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    w = _longest_rule(w, _STEP2)
    w = _longest_rule(w, _STEP3)

    # step 4 — ION carries the extra "stem ends s or t" condition. No other
    # step-4 suffix ends in "n", so for a word ending in "ion" it is always
    # the longest (indeed only) match and can be handled apart.
    if w.endswith("ion"):
        stem = w[:-3]
        if stem.endswith(("s", "t")) and _measure(stem) > 1:
            w = stem
    else:
        w = _longest_rule(w, _STEP4)

    # step 5a
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem

    # step 5b
    if _measure(w) > 1 and _ends_double_cons(w) and w.endswith("l"):
        w = w[:-1]

    return w


# ── WordNet synonym lookup (optional, flagged) ──────────────────────────────

_WORDNET = None  # None = not tried yet, False = unavailable, else the module
# nltk's WordNet reader is NOT thread-safe, and not only while it is loading:
# it serves every lookup out of one shared zip handle, so two concurrent reads
# trip `assert self.fp is None` inside nltk.data. Scoring now happens on the
# generation workers, so this is reached from a dozen threads at once - measured
# on 13 concurrent scorings of one model: 12 of 13 raised.
#
# The lock therefore covers every access, not just the one-time load. That
# serialises the synonym lookups, which is affordable: they are memoised per
# word by `_lemma_names`, so a run pays for each distinct label once and the
# threads spend their time waiting on HTTP regardless.
_WORDNET_LOCK = threading.RLock()


def _wordnet():
    global _WORDNET
    if _WORDNET is None:
        with _WORDNET_LOCK:
            if _WORDNET is None:      # another thread may have won the race
                try:
                    from nltk.corpus import wordnet
                    wordnet.synsets("test")  # force the corpus open, fail here or never
                    _WORDNET = wordnet
                except Exception:  # noqa: BLE001 — no nltk, no corpus, broken zip: all mean "off"
                    _WORDNET = False
    return _WORDNET


def wordnet_available() -> bool:
    """True when the synonym term is actually being computed."""
    return bool(_wordnet())


@lru_cache(maxsize=65536)
def _lemma_names(word: str) -> FrozenSet[str]:
    wn = _wordnet()
    if not wn:
        return frozenset()
    # Held across the whole traversal: `synsets()` and the `lemmas()` walk both
    # read from the same corpus handle, so releasing between them would leave
    # exactly the gap the lock exists to close. Re-entrant because _wordnet()
    # takes it too on the very first call.
    with _WORDNET_LOCK:
        return frozenset(
            lemma.name().lower() for synset in wn.synsets(word)
            for lemma in synset.lemmas()
        )


def is_synonym(w1: str, w2: str) -> bool:
    """True when the two words share a WordNet synset (checked on the
    unstemmed forms — WordNet indexes real words, not Porter stems). Always
    False when WordNet is unavailable; `wordnet_available()` says which."""
    if w1 == w2:
        return False  # identity is exact overlap, counted by wi, not ws
    return w2 in _lemma_names(w1) or w1 in _lemma_names(w2)
