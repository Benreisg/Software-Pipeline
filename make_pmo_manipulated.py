#!/usr/bin/env python3
"""Erzeugt PMo_manipulated: manipulierte Varianten aller 55 PMo-Prozessmodelle.

Fuer jeden Eintrag werden Task-/Gateway-Namen umbenannt und/oder Kanten
hinzugefuegt bzw. entfernt. Die Manipulation wird auf dem BPMN-Modell geplant
(laut PMo-README die Ground Truth) und identisch auf die Graphviz-Variante
angewendet, damit BPMN/xx.bpmn und Graphviz/xx.dot weiterhin dasselbe
Prozessmodell beschreiben.

Das Mapping BPMN-Element -> Graphviz-Knoten laeuft ueber die Variante
`variations/graphviz_with_ids`, die dieselbe Kantenreihenfolge wie `graphviz`
verwendet und die Element-IDs als Knotennamen fuehrt.

Aufruf:
    python make_pmo_manipulated.py            # schreibt PMo_manipulated/
    python make_pmo_manipulated.py --verify   # nur Mapping/Konsistenz pruefen
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATASET = BASE / "pmo-dataset"
OUT = BASE / "PMo_manipulated"

BPMN_NS = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"

TASK_TAGS = {"task", "userTask", "serviceTask", "manualTask", "sendTask",
             "receiveTask", "scriptTask", "businessRuleTask", "subProcess",
             "callActivity"}
EVENT_TAGS = {"startEvent", "endEvent", "intermediateCatchEvent",
              "intermediateThrowEvent", "boundaryEvent"}
GATEWAY_TAGS = {"exclusiveGateway", "parallelGateway", "inclusiveGateway",
                "eventBasedGateway", "complexGateway"}
FLOW_NODE_TAGS = TASK_TAGS | EVENT_TAGS | GATEWAY_TAGS

DOT_WRAP_WIDTH = 20

# ---------------------------------------------------------------------------
# Umbenennungen
# ---------------------------------------------------------------------------

# Verb-Synonyme (erstes Wort eines Task-Namens)
VERB_SYNONYMS = {
    "send": "dispatch", "receive": "obtain", "check": "verify", "verify": "validate",
    "create": "generate", "generate": "produce", "make": "create", "prepare": "set up",
    "update": "modify", "record": "log", "collect": "gather", "provide": "supply",
    "place": "submit", "address": "handle", "guide": "assist", "confirm": "acknowledge",
    "approve": "authorize", "review": "examine", "notify": "inform", "inform": "notify",
    "store": "save", "process": "handle", "assign": "allocate", "calculate": "compute",
    "deliver": "ship", "complete": "finalize", "start": "begin", "select": "choose",
    "print": "output", "sign": "endorse", "analyze": "assess", "evaluate": "assess",
    "reject": "decline", "cancel": "abort", "register": "log", "search": "look for",
    "close": "finalize", "hand": "pass", "hand over": "pass on",
    "hand out": "give out", "book": "reserve", "pay": "settle",
    "order": "request", "fill": "complete", "add": "insert", "remove": "delete",
    "open": "launch", "return": "send back", "forward": "route", "schedule": "plan",
    "contact": "reach out to", "request": "ask for", "issue": "release",
    "perform": "carry out", "execute": "run", "handle": "process", "get": "fetch",
    "put": "place", "give": "hand out", "show": "display",
    "read": "scan", "write": "note down", "test": "inspect", "inspect": "examine",
    "clean": "wash", "wash": "clean", "pack": "wrap", "load": "fill",
    "unload": "empty", "measure": "gauge", "count": "tally", "compare": "match",
    "define": "specify", "specify": "define", "plan": "schedule", "arrange": "organize",
    "organize": "arrange", "submit": "hand in", "publish": "release", "archive": "file",
    "print out": "output", "enter": "input", "input": "enter", "choose": "select",
    "decide": "determine", "determine": "decide", "inform about": "report on",
    "book in": "register", "set": "configure", "install": "set up", "repair": "fix",
    "fix": "repair", "replace": "swap", "start up": "boot", "shut": "close",
    "hold": "keep", "keep": "retain", "move": "transfer", "transfer": "move",
    "transition": "move",
    "ship": "deliver", "buy": "purchase", "purchase": "buy", "sell": "market",
    "hire": "recruit", "recruit": "hire", "train": "instruct", "instruct": "train",
    "sort": "classify", "classify": "categorize", "filter": "screen", "screen": "filter",
    "scan": "read", "copy": "duplicate", "merge": "combine", "combine": "merge",
    "split": "divide", "divide": "split", "extract": "retrieve", "retrieve": "extract",
    "import": "load", "export": "extract", "upload": "transmit", "download": "retrieve",
    "reserve": "book", "release": "issue", "block": "lock", "lock": "block",
    "grant": "award", "deny": "refuse", "refuse": "deny", "accept": "approve",
    "escalate": "forward", "resolve": "settle", "settle": "clear", "clear": "settle",
    "monitor": "track", "track": "monitor", "report": "document", "document": "record",
}

# Nomen-Synonyme (irgendwo im Namen)
NOUN_SYNONYMS = {
    "customer": "client", "client": "customer", "order": "purchase order",
    "invoice": "bill", "bill": "invoice", "ticket": "booking", "product": "item",
    "information": "data", "data": "information", "request": "application",
    "document": "file", "file": "document", "payment": "transaction",
    "report": "summary", "email": "e-mail message", "form": "sheet",
    "goods": "merchandise", "delivery": "shipment", "shipment": "delivery",
    "supplier": "vendor", "vendor": "supplier", "employee": "staff member",
    "manager": "supervisor", "department": "unit", "system": "platform",
    "account": "profile", "contract": "agreement", "agreement": "contract",
    "offer": "proposal", "proposal": "offer", "quote": "quotation",
    "complaint": "grievance", "claim": "case", "case": "claim",
    "application": "request", "appointment": "meeting", "meeting": "appointment",
    "room": "space", "material": "supplies", "stock": "inventory",
    "inventory": "stock", "price": "cost", "cost": "price", "budget": "funds",
    "result": "outcome", "outcome": "result", "decision": "verdict",
    "feedback": "response", "response": "reply", "reply": "response",
    "confirmation": "acknowledgement", "notification": "alert", "alert": "notification",
    "package": "parcel", "parcel": "package", "list": "register",
    "database": "data store", "record": "entry", "entry": "record",
    "student": "learner", "patient": "case", "doctor": "physician",
    "driver": "courier", "warehouse": "storage facility", "office": "branch",
    "team": "group", "group": "team", "project": "initiative", "task": "activity",
    "process": "procedure", "procedure": "process", "step": "stage",
    "quality": "standard", "test": "check", "check": "inspection",
    "machine": "device", "device": "machine", "tool": "instrument",
    "user": "operator", "operator": "user", "credit": "loan", "loan": "credit",
    "score": "rating", "rating": "score", "risk": "exposure",
    "method": "technique", "technique": "method", "configuration": "setup",
    "setup": "configuration", "requirement": "specification",
    "specification": "requirement", "dismissal": "termination",
    "notice": "notification", "signature": "sign-off", "approval": "sign-off",
    "training": "instruction", "solution": "remedy", "issue": "problem",
    "problem": "issue", "error": "fault", "fault": "error", "defect": "flaw",
    "prototype": "mock-up", "design": "layout", "layout": "design",
    "visuals": "graphics", "campaign": "promotion", "audit": "inspection",
    "director": "head", "submission": "hand-in", "grades": "marks",
    "appeal": "objection", "installment": "instalment", "seat": "place",
    "route": "itinerary", "journey": "trip", "trip": "journey",
    "reminder": "prompt", "agency": "bureau", "booking": "reservation",
    "insurance": "coverage", "label": "tag", "offers": "bids", "offer": "bid",
    "logistics": "distribution", "dish": "meal", "meal": "dish",
    "restaurant": "canteen", "computer": "device", "repair": "restoration",
    "calculation": "estimate", "estimate": "calculation", "cost": "price",
    "identification": "detection", "detection": "identification",
    "ultrasound": "sonography", "vein": "vessel", "blood": "plasma",
    "wire": "guidewire", "gaps": "shortfalls", "compliance": "conformity",
    "mitigation": "countermeasure", "responsibilities": "duties",
    "role": "position", "transition": "handover", "complaint": "grievance",
    "inquiry": "enquiry", "enquiry": "inquiry", "prospect": "lead",
    "site": "location", "visit": "call", "part": "component",
    "parts": "components", "machine": "unit", "shift": "work period",
    "invoicing": "billing", "packaging": "wrapping", "storage": "warehousing",
    "type": "variant", "subscription": "membership", "injury": "harm",
    "company": "firm", "set": "kit", "games": "titles", "income": "earnings",
}

# Umformulierungen fuer Gateway-Bedingungen (Adjektive, Vergleiche, Zustaende)
GATEWAY_QUESTION_SYNONYMS = {
    "ok": "acceptable", "available": "on hand", "possible": "feasible",
    "received": "arrived", "correct": "accurate", "valid": "legitimate",
    "complete": "finished", "approved": "authorized", "needed": "required",
    "required": "needed", "successful": "effective", "enough": "sufficient",
    "sufficient": "enough", "empty": "cleared", "full": "at capacity",
    "new": "unseen", "known": "familiar", "paid": "settled", "open": "pending",
    "ready": "prepared", "yes": "affirmative", "no": "negative",
    "in stock": "on hand", "special": "exceptional", "urgent": "time-critical",
    "large": "big", "small": "minor", "high": "elevated", "low": "reduced",
    "done": "finished", "below": "under", "above": "over", "more than": "over",
    "less than": "under", "greater than": "over", "worth": "viable for",
    "days": "calendar days", "minutes": "min", "hour": "hr", "hours": "hrs",
    "service": "maintenance", "position": "placement", "good": "proper",
    "bad": "poor", "desired": "requested", "already": "previously",
    "under": "below", "over": "above", "still": "as yet",
    "shipping": "delivery", "billing": "invoicing", "adress": "location",
    "address": "location", "selected": "chosen", "found": "located",
    "same": "identical", "different": "diverging", "further": "additional",
    "additional": "further", "first": "initial", "last": "final",
    "internal": "in-house", "external": "outside", "positive": "favourable",
    "negative": "unfavourable", "critical": "severe", "minor": "slight",
}

QUALIFIERS = ["manually", "in the system", "for review", "immediately",
              "in advance", "on site", "by phone", "in writing"]
QUESTION_QUALIFIERS = ["right now", "at this point", "in this case",
                       "according to the record", "for this run"]
NOMINAL_PREFIXES = ["Perform", "Carry out", "Execute", "Handle"]

VOWEL_SOUND = tuple("aeiou")


def _is_acronym(word: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", word)
    return len(letters) > 1 and letters.isupper()


def _match_case(original: str, replacement: str) -> str:
    """Uebertraegt die Grossschreibung des Originalworts auf den Ersatz."""
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _inflect(base: str, suffix: str) -> str:
    """Setzt die Flexion des Originalverbs auf das Synonym (nur erstes Wort)."""
    head, _, tail = base.partition(" ")
    if suffix == "s":
        if head.endswith(("s", "x", "z", "ch", "sh")):
            head += "es"
        elif head.endswith("y") and head[-2:-1] not in "aeiou":
            head = head[:-1] + "ies"
        else:
            head += "s"
    elif suffix == "ing":
        head = (head[:-1] if head.endswith("e") else head) + "ing"
    elif suffix == "ed":
        head = head + ("d" if head.endswith("e") else "ed")
    return head + ((" " + tail) if tail else "")


def _verb_lookup(word: str, inflected_only: bool = False) -> str | None:
    """Sucht ein Verb-Synonym und uebernimmt die Flexion des Originals.

    Mit `inflected_only` werden nur eindeutig flektierte Formen (-s/-ing/-ed)
    akzeptiert - so wird ein Nomen wie "request" mitten im Namen nicht
    faelschlich als Verb ersetzt.
    """
    key = re.sub(r"[^A-Za-z]", "", word).lower()
    if key in VERB_SYNONYMS and not inflected_only:
        return VERB_SYNONYMS[key]
    for suffix, stems in (("s", (key[:-1],)),
                          ("ing", (key[:-3], key[:-3] + "e")),
                          ("ed", (key[:-2], key[:-2] + "e", key[:-1]))):
        if not key.endswith(suffix) or len(key) <= len(suffix) + 1:
            continue
        for stem in stems:
            if stem in VERB_SYNONYMS:
                return _inflect(VERB_SYNONYMS[stem], suffix)
    return None


# Woerter, deren Anlaut nicht zum ersten Buchstaben passt
CONSONANT_START = ("uni", "use", "usu", "uti", "eu", "one", "once")
VOWEL_START = ("hour", "honest", "honou", "honor", "heir")


def _needs_an(word: str) -> bool:
    lowered = word.lower()
    if lowered.startswith(VOWEL_START):
        return True
    if lowered.startswith(CONSONANT_START):
        return False
    if _is_acronym(word):
        return word[0].upper() in "AEFHILMNORSX"
    return lowered[:1] in VOWEL_SOUND


def _looks_finite_verb(words: list[str], i: int) -> bool:
    """Schaetzt, ob words[i] ein finites Verb ist (Praesens-s mit Objekt).

    "IP checks the request of the INQ" -> "checks" ist ein Verb,
    "More issues?" -> "issues" ist ein Nomen (es folgt kein Objekt).
    """
    if i == 0:
        return False
    key = re.sub(r"[^A-Za-z]", "", words[i]).lower()
    if not key.endswith("s") or key.endswith("ss") or key[:-1] not in VERB_SYNONYMS:
        return False
    return len(words) - i - 1 >= 2


def _noun_lookup(word: str) -> str | None:
    """Sucht ein Nomen-Synonym; erkennt auch Pluralformen."""
    key = re.sub(r"[^A-Za-z-]", "", word).lower()
    if not key:
        return None
    if key in NOUN_SYNONYMS:
        return NOUN_SYNONYMS[key]
    for suffix, stems in (("ies", (key[:-3] + "y",)),
                          ("es", (key[:-2],)),
                          ("s", (key[:-1],))):
        if not key.endswith(suffix) or len(key) <= len(suffix) + 1:
            continue
        for stem in stems:
            if stem in NOUN_SYNONYMS:
                return _inflect(NOUN_SYNONYMS[stem], "s")
    return None


def fix_articles(text: str) -> str:
    """Korrigiert a/an, falls eine Ersetzung den Anlaut geaendert hat."""
    def repl(m: re.Match) -> str:
        art, nxt = m.group(1), m.group(2)
        fixed = "an" if _needs_an(nxt) else "a"
        if art[0].isupper():
            fixed = fixed.capitalize()
        return f"{fixed} {nxt}"
    return re.sub(r"\b([Aa]n?)\s+([\w-]+)", repl, text)


def rename_label(name: str, rng: random.Random,
                 is_gateway: bool) -> tuple[str, str] | None:
    """Liefert (neuer Name, angewandte Regel) oder None."""
    stripped = name.strip()
    if not stripped:
        return None

    trailing = ""
    core = stripped
    while core and core[-1] in "?!.:":
        trailing = core[-1] + trailing
        core = core[:-1]
    core = core.rstrip()
    if not core:
        return None

    words = core.split()
    is_question = trailing.startswith("?")

    def finish(text: str, rule: str) -> tuple[str, str]:
        return fix_articles(text) + trailing, rule

    # 1) Bedingungen in Gateway-Fragen: Schluesselwort austauschen
    if is_gateway or is_question:
        lowered = core.lower()
        for src, dst in sorted(GATEWAY_QUESTION_SYNONYMS.items(),
                               key=lambda kv: -len(kv[0])):
            idx = lowered.find(src)
            if idx < 0 or (idx > 0 and lowered[idx - 1].isalnum()):
                continue
            end = idx + len(src)
            if end != len(lowered) and lowered[end].isalnum():
                continue
            repl = _match_case(core[idx:idx + 1], dst)
            return finish(core[:idx] + repl + core[end:], "condition")

    # 2) Verb am Wortanfang (auch zweiwortig wie "hand over", inkl. Flexion)
    for span in (2, 1):
        if len(words) < span:
            continue
        key = " ".join(words[:span]).lower()
        if key in VERB_SYNONYMS:
            return finish(" ".join([_match_case(words[0], VERB_SYNONYMS[key])]
                                   + words[span:]), "verb")
    syn = _verb_lookup(words[0])
    if syn:
        return finish(" ".join([_match_case(words[0], syn)] + words[1:]), "verb")

    # 3) Nomen irgendwo im Namen - vor der Verbsuche, weil eine Pluralform
    #    ("issues", "gaps") sonst faelschlich als Verb gelesen wuerde.
    #    Ein Wort, das auch ein Verb sein kann ("checks", "sets"), gilt nur
    #    dann als Nomen, wenn ihm kein Objekt folgt.
    for i, word in enumerate(words):
        if _is_acronym(word) or _looks_finite_verb(words, i):
            continue
        syn = _noun_lookup(word)
        if syn:
            new_words = list(words)
            new_words[i] = _match_case(word, syn)
            return finish(" ".join(new_words), "noun")

    # 4) Flektiertes Verb an spaeterer Stelle ("MPOO reviews the dismissal")
    for i in range(1, len(words)):
        if not _looks_finite_verb(words, i):
            continue
        if i + 1 < len(words):
            pair = f"{words[i]} {words[i + 1]}".lower()
            stem = re.sub(r"s\b", "", pair, count=1)
            if stem in VERB_SYNONYMS and pair != stem:
                syn = _inflect(VERB_SYNONYMS[stem], "s")
                new_words = words[:i] + [_match_case(words[i], syn)] + words[i + 2:]
                return finish(" ".join(new_words), "verb")
        syn = _verb_lookup(words[i], inflected_only=True)
        if syn:
            new_words = list(words)
            new_words[i] = _match_case(words[i], syn)
            return finish(" ".join(new_words), "verb")

    # 5) Fallback: praezisierender Zusatz bzw. Verb vor die Nominalphrase
    if is_question:
        return finish(core + " " + rng.choice(QUESTION_QUALIFIERS), "qualifier")
    if _verb_lookup(words[0]) or words[0].lower() in VERB_SYNONYMS:
        return finish(core + " " + rng.choice(QUALIFIERS), "qualifier")
    head = core if _is_acronym(words[0]) else core[0].lower() + core[1:]
    return finish(rng.choice(NOMINAL_PREFIXES) + " " + head, "prefix")


# ---------------------------------------------------------------------------
# Graphviz-Label-Umbruch (Breite 20, greedy - wie im Original-Dataset)
# ---------------------------------------------------------------------------

def wrap_label(text: str, width: int = DOT_WRAP_WIDTH) -> str:
    """Zeilenumbruch wie im PMo-Dataset: greedy nach Woertern, ueberlange
    Woerter anschliessend hart auf `width` geschnitten."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    out: list[str] = []
    for line in lines:
        while len(line) > width:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return "\\n".join(out)


# ---------------------------------------------------------------------------
# BPMN-Parsing (zeilenbasiert, damit die Datei formattreu bleibt)
# ---------------------------------------------------------------------------

class BpmnDoc:
    def __init__(self, path: Path):
        self.path = path
        self.lines = path.read_text(encoding="utf-8").splitlines()
        self.nodes: dict[str, dict] = {}       # id -> {tag, name, line, proc}
        self.flows: dict[str, dict] = {}       # id -> {src, tgt, name, line, proc}
        self.proc_ranges: list[tuple[int, int]] = []
        self._parse()

    def _parse(self) -> None:
        root = ET.parse(self.path).getroot()
        # Zeilenbereiche der process-Bloecke
        proc_start = None
        for i, line in enumerate(self.lines):
            if re.match(r"\s*<process[\s>]", line):
                proc_start = i
            elif re.match(r"\s*</process>", line) and proc_start is not None:
                self.proc_ranges.append((proc_start, i))
                proc_start = None

        def proc_of(line_idx: int) -> int:
            for p, (a, b) in enumerate(self.proc_ranges):
                if a <= line_idx <= b:
                    return p
            return -1

        # Element-IDs den Zeilen zuordnen
        id_line: dict[str, int] = {}
        for i, line in enumerate(self.lines):
            m = re.search(r'<([a-zA-Z]+)\s+id="([^"]+)"', line)
            if m and m.group(2) not in id_line:
                id_line[m.group(2)] = i

        for proc in root.iter(BPMN_NS + "process"):
            for el in proc:
                tag = el.tag.replace(BPMN_NS, "")
                eid = el.get("id")
                if tag in FLOW_NODE_TAGS:
                    line = id_line.get(eid, -1)
                    self.nodes[eid] = {"tag": tag, "name": el.get("name"),
                                       "line": line, "proc": proc_of(line)}
                elif tag == "sequenceFlow":
                    line = id_line.get(eid, -1)
                    self.flows[eid] = {"src": el.get("sourceRef"),
                                       "tgt": el.get("targetRef"),
                                       "name": el.get("name"),
                                       "line": line, "proc": proc_of(line)}

    # -- Manipulationen -----------------------------------------------------

    def rename_node(self, node_id: str, new_name: str) -> None:
        info = self.nodes[node_id]
        i = info["line"]
        line = self.lines[i]
        escaped = (new_name.replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;").replace('"', "&quot;"))
        new_line, count = re.subn(r'(\sname=")[^"]*(")',
                                  lambda m: m.group(1) + escaped + m.group(2),
                                  line, count=1)
        if count != 1:
            raise RuntimeError(f"{self.path.name}: name= nicht ersetzbar in {node_id!r}")
        self.lines[i] = new_line
        info["name"] = new_name

    def remove_flow(self, flow_id: str) -> None:
        keep: list[str] = []
        skip_until_edge_end = False
        removed_flow = False
        for line in self.lines:
            if skip_until_edge_end:
                if "</bpmndi:BPMNEdge>" in line:
                    skip_until_edge_end = False
                continue
            if re.search(rf'<sequenceFlow id="{re.escape(flow_id)}"', line):
                removed_flow = True
                continue
            if re.fullmatch(rf'\s*<(incoming|outgoing)>{re.escape(flow_id)}</\1>\s*', line):
                continue
            if re.search(rf'<bpmndi:BPMNEdge[^>]*bpmnElement="{re.escape(flow_id)}"', line):
                if "/>" in line and "</bpmndi:BPMNEdge>" not in line:
                    continue
                skip_until_edge_end = True
                continue
            keep.append(line)
        if not removed_flow:
            raise RuntimeError(f"{self.path.name}: sequenceFlow {flow_id} nicht gefunden")
        self.lines = keep
        del self.flows[flow_id]
        self._reindex()

    def add_flow(self, flow_id: str, src: str, tgt: str) -> None:
        # 1) outgoing / incoming in die Knotenelemente einhaengen
        self._insert_ref(src, "outgoing", flow_id)
        self._insert_ref(tgt, "incoming", flow_id)
        # 2) sequenceFlow-Element hinter den letzten Flow desselben process
        proc = self.nodes[src]["proc"]
        last = -1
        for fid, f in self.flows.items():
            if f["proc"] == proc and f["line"] > last:
                last = f["line"]
        if last < 0:
            last = self.proc_ranges[proc][1] - 1
        indent = re.match(r"\s*", self.lines[last]).group(0)
        self.lines.insert(last + 1,
                          f'{indent}<sequenceFlow id="{flow_id}" '
                          f'sourceRef="{src}" targetRef="{tgt}"/>')
        self.flows[flow_id] = {"src": src, "tgt": tgt, "name": None,
                               "line": last + 1, "proc": proc}
        self._reindex()
        # 3) Diagramm-Kante ergaenzen (falls die Datei ein Diagramm enthaelt)
        self._add_di_edge(flow_id, src, tgt)

    def _insert_ref(self, node_id: str, kind: str, flow_id: str) -> None:
        info = self.nodes[node_id]
        i = info["line"]
        line = self.lines[i]
        indent = re.match(r"\s*", line).group(0)
        if re.search(r"/>\s*$", line):
            # selbstschliessendes Element aufbrechen
            tag = re.search(r"<([a-zA-Z]+)", line).group(1)
            head = re.sub(r"\s*/>\s*$", ">", line)
            self.lines[i:i + 1] = [head,
                                   f"{indent}  <{kind}>{flow_id}</{kind}>",
                                   f"{indent}</{tag}>"]
        else:
            tag = re.search(r"<([a-zA-Z]+)", line).group(1)
            end = i
            while end < len(self.lines) and f"</{tag}>" not in self.lines[end]:
                end += 1
            self.lines.insert(end, f"{indent}  <{kind}>{flow_id}</{kind}>")
        self._reindex()

    def _bounds(self, node_id: str) -> tuple[float, float, float, float] | None:
        for i, line in enumerate(self.lines):
            if re.search(rf'<bpmndi:BPMNShape[^>]*bpmnElement="{re.escape(node_id)}"', line):
                for j in range(i, min(i + 4, len(self.lines))):
                    m = re.search(r'<omgdc:Bounds x="([-\d.]+)" y="([-\d.]+)" '
                                  r'width="([-\d.]+)" height="([-\d.]+)"', self.lines[j])
                    if m:
                        return tuple(float(g) for g in m.groups())
        return None

    def _add_di_edge(self, flow_id: str, src: str, tgt: str) -> None:
        plane_end = None
        for i, line in enumerate(self.lines):
            if "</bpmndi:BPMNPlane>" in line:
                plane_end = i
                break
        if plane_end is None:
            return
        sb, tb = self._bounds(src), self._bounds(tgt)
        if sb is None or tb is None:
            return
        x1, y1 = sb[0] + sb[2], sb[1] + sb[3] / 2
        x2, y2 = tb[0], tb[1] + tb[3] / 2
        indent = re.match(r"\s*", self.lines[plane_end]).group(0) + "  "
        block = [
            f'{indent}<bpmndi:BPMNEdge bpmnElement="{flow_id}" id="{flow_id}_di">',
            f'{indent}  <omgdi:waypoint x="{x1:g}" y="{y1:g}"/>',
            f'{indent}  <omgdi:waypoint x="{x2:g}" y="{y2:g}"/>',
            f"{indent}</bpmndi:BPMNEdge>",
        ]
        self.lines[plane_end:plane_end] = block
        self._reindex()

    def _reindex(self) -> None:
        id_line: dict[str, int] = {}
        for i, line in enumerate(self.lines):
            m = re.search(r'<([a-zA-Z]+)\s+id="([^"]+)"', line)
            if m and m.group(2) not in id_line:
                id_line[m.group(2)] = i
        for eid, info in self.nodes.items():
            if eid in id_line:
                info["line"] = id_line[eid]
        for fid, info in self.flows.items():
            if fid in id_line:
                info["line"] = id_line[fid]
        self.proc_ranges = []
        start = None
        for i, line in enumerate(self.lines):
            if re.match(r"\s*<process[\s>]", line):
                start = i
            elif re.match(r"\s*</process>", line) and start is not None:
                self.proc_ranges.append((start, i))
                start = None

    def text(self) -> str:
        # Die Originaldateien enden ohne Zeilenumbruch - das bleibt so.
        return "\n".join(self.lines)


# ---------------------------------------------------------------------------
# Graphviz-Parsing
# ---------------------------------------------------------------------------

EDGE_RE = re.compile(r'^(\s*)"([^"]*)"\s*->\s*"([^"]*)"\s*(\[[^\]]*\])?\s*;\s*$')
DECL_RE = re.compile(r'^(\s*)"([^"]*)"\s*(\[[^\]]*\])\s*;\s*$')
MEMBER_RE = re.compile(r'^(\s*)"([^"]*)"\s*;\s*$')


class DotDoc:
    def __init__(self, path: Path):
        self.path = path
        self.lines = path.read_text(encoding="utf-8").splitlines()
        self.edges: list[tuple[int, str, str, str]] = []   # (line, src, tgt, attrs)
        self.decls: dict[str, int] = {}
        for i, line in enumerate(self.lines):
            m = EDGE_RE.match(line)
            if m:
                self.edges.append((i, m.group(2), m.group(3), (m.group(4) or "").strip()))
                continue
            m = DECL_RE.match(line)
            if m:
                self.decls.setdefault(m.group(2), i)

    def rename_node(self, key: str, new_label: str) -> str:
        """Benennt einen Knoten um und liefert seinen neuen Schluessel.

        Im Original-Dataset ist der Knotenschluessel der Elementname; nur bei
        mehrfach vergebenen Namen traegt er ein Suffix und den Namen im
        label-Attribut. Da der neue Name eindeutig ist, wird hier immer der
        Schluessel ersetzt und ein jetzt ueberfluessiges label entfernt.
        """
        wrapped = wrap_label(new_label)
        if wrapped in self.decls or any(wrapped in (s, t) for _, s, t, _ in self.edges):
            raise RuntimeError(f"{self.path.name}: neuer Knotenname {wrapped!r} "
                               f"kollidiert mit einem vorhandenen Knoten")
        old, new = f'"{key}"', f'"{wrapped}"'
        for i, line in enumerate(self.lines):
            if line is not None and old in line:
                self.lines[i] = line.replace(old, new)
        decl_line = self.decls.pop(key, None)
        if decl_line is not None:
            self.decls[wrapped] = decl_line
            line = self.lines[decl_line]
            if 'label="' in line:
                stripped = re.sub(r'label="[^"]*"(,\s*)?', "", line, count=1)
                if re.match(r'^\s*"[^"]*"\s*\[\s*\]\s*;\s*$', stripped):
                    self.lines[decl_line] = None   # Deklaration ohne Inhalt
                    del self.decls[wrapped]
                else:
                    self.lines[decl_line] = stripped.replace("[, ", "[")
        self.edges = [(i, wrapped if s == key else s, wrapped if t == key else t, a)
                      for i, s, t, a in self.edges]
        return wrapped

    def remove_edge(self, line_idx: int) -> None:
        self.lines[line_idx] = None  # spaeter herausgefiltert

    def add_edge(self, src_key: str, tgt_key: str) -> None:
        last = max(i for i, _, _, _ in self.edges if self.lines[i] is not None)
        indent = re.match(r"\s*", self.lines[last]).group(0)
        self.lines.insert(last + 1, f'{indent}"{src_key}" -> "{tgt_key}";')
        self.edges = [(i + 1 if i > last else i, s, t, a) for i, s, t, a in self.edges]
        self.edges.append((last + 1, src_key, tgt_key, ""))
        self.decls = {k: (v + 1 if v > last else v) for k, v in self.decls.items()}

    def text(self) -> str:
        return "\n".join(l for l in self.lines if l is not None)


# ---------------------------------------------------------------------------
# Mapping BPMN-Element -> Graphviz-Knoten
# ---------------------------------------------------------------------------

def category(tag: str) -> str:
    if tag in GATEWAY_TAGS:
        return "gateway"
    if tag in EVENT_TAGS:
        return "event"
    return "task"


def wid_category(wid_id: str) -> str:
    if "Gateway" in wid_id:
        return "gateway"
    if "Event" in wid_id:
        return "event"
    return "task"


GATEWAY_SYMBOL = {"exclusiveGateway": "X", "parallelGateway": "+",
                  "inclusiveGateway": "O", "eventBasedGateway": "E",
                  "complexGateway": "*"}


def parse_wid(path: Path):
    edges, labels = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = EDGE_RE.match(line)
        if m:
            edges.append((m.group(2), m.group(3), (m.group(4) or "").strip()))
            continue
        m = DECL_RE.match(line)
        if m:
            lm = re.search(r'label="([^"]*)"', m.group(3))
            labels[m.group(2)] = lm.group(1) if lm else None
    return edges, labels


def build_mapping(item: str, doc: BpmnDoc, dot: DotDoc):
    """Liefert (bpmn_id -> gv_key, flow_id -> gv_edge_index) oder wirft."""
    wid_edges, wid_labels = parse_wid(DATASET / "variations" / "graphviz_with_ids" / f"{item}.dot")
    gv_edges = [(s, t, a) for _, s, t, a in dot.edges]
    if len(gv_edges) != len(wid_edges):
        raise RuntimeError(f"{item}: Kantenzahl graphviz={len(gv_edges)} "
                           f"with_ids={len(wid_edges)}")

    # graphviz-Knoten <-> with_ids-Knoten ueber die Kantenreihenfolge
    gv_of_wid: dict[str, str] = {}
    for (gs, gt, ga), (ws, wt, wa) in zip(gv_edges, wid_edges):
        if ga != wa:
            raise RuntimeError(f"{item}: Kantenattribute weichen ab: {ga!r} / {wa!r}")
        for g, w in ((gs, ws), (gt, wt)):
            if gv_of_wid.setdefault(w, g) != g:
                raise RuntimeError(f"{item}: Knotenmapping widerspruechlich bei {w!r}")

    # with_ids-Knoten <-> BPMN-Element
    wid_ids = list(gv_of_wid)
    domains: dict[str, list[str]] = {}
    for bid, info in doc.nodes.items():
        cat = category(info["tag"])
        name = (info["name"] or "").strip()
        cands = []
        for wid in wid_ids:
            if wid_category(wid) != cat:
                continue
            label = wid_labels.get(wid)
            if cat == "gateway":
                if name:
                    if label != wrap_label(name):
                        continue
                elif label != GATEWAY_SYMBOL.get(info["tag"]):
                    continue
            else:
                if (label or "") != wrap_label(name):
                    continue
            cands.append(wid)
        # exakte ID-Gleichheit hat Vorrang
        if bid in cands:
            cands = [bid] + [c for c in cands if c != bid]
        if not cands:
            raise RuntimeError(f"{item}: kein Graphviz-Knoten fuer {bid} "
                               f"({info['tag']}, name={info['name']!r})")
        domains[bid] = cands

    target = sorted((s, t, a) for s, t, a in wid_edges)
    order = sorted(domains, key=lambda b: len(domains[b]))
    assign: dict[str, str] = {}
    used: set[str] = set()

    def backtrack(k: int) -> bool:
        if k == len(order):
            mapped = sorted((assign[f["src"]], assign[f["tgt"]],
                             f'[label="{wrap_label(f["name"])}"]' if f["name"] else "")
                            for f in doc.flows.values())
            # Message Flows sind im DOT enthalten, im flows-Dict aber nicht
            return all(e in target for e in mapped)
        bid = order[k]
        for cand in domains[bid]:
            if cand in used:
                continue
            assign[bid] = cand
            used.add(cand)
            if backtrack(k + 1):
                return True
            used.discard(cand)
            del assign[bid]
        return False

    if not backtrack(0):
        raise RuntimeError(f"{item}: kein konsistentes Element-Mapping gefunden")

    node_map = {bid: gv_of_wid[wid] for bid, wid in assign.items()}

    # sequenceFlow -> Index der Graphviz-Kante
    by_key: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, (s, t, a) in enumerate(gv_edges):
        by_key[(s, t, a)].append(idx)
    flow_map: dict[str, int] = {}
    for fid, f in doc.flows.items():
        key = (node_map[f["src"]], node_map[f["tgt"]],
               f'[label="{wrap_label(f["name"])}"]' if f["name"] else "")
        if not by_key.get(key):
            raise RuntimeError(f"{item}: Kante fuer {fid} nicht im DOT gefunden ({key})")
        flow_map[fid] = by_key[key].pop(0)
    return node_map, flow_map


# ---------------------------------------------------------------------------
# Manipulationsplanung
# ---------------------------------------------------------------------------

def plan(item: str, doc: BpmnDoc, rng: random.Random) -> list[dict]:
    ops: list[dict] = []
    idx = int(item)
    profile = "rename" if idx % 3 == 1 else ("edge" if idx % 3 == 2 else "both")

    renamable = [bid for bid, i in doc.nodes.items()
                 if i["tag"] in TASK_TAGS and (i["name"] or "").strip()]
    gateways = [bid for bid, i in doc.nodes.items()
                if i["tag"] in GATEWAY_TAGS and (i["name"] or "").strip()]
    taken = {(i["name"] or "").strip() for i in doc.nodes.values()}

    def do_renames(count: int) -> int:
        done = 0
        pool = renamable + gateways
        rng.shuffle(pool)
        # Gateways bevorzugt mitnehmen, wenn vorhanden
        pool.sort(key=lambda b: 0 if b in gateways and done == 0 else 1)
        for bid in pool:
            if done >= count:
                break
            info = doc.nodes[bid]
            old = (info["name"] or "").strip()
            result = rename_label(old, rng, bid in gateways)
            if not result:
                continue
            new, rule = result
            if new == old or new in taken:
                continue
            ops.append({"op": "rename", "element": bid, "type": info["tag"],
                        "old": old, "new": new, "rule": rule})
            taken.add(new)
            done += 1
        return done

    def edge_candidates_remove() -> list[str]:
        out_deg, in_deg = defaultdict(int), defaultdict(int)
        for f in doc.flows.values():
            out_deg[f["src"]] += 1
            in_deg[f["tgt"]] += 1
        branch, plain = [], []
        for fid, f in doc.flows.items():
            if out_deg[f["src"]] > 1 or in_deg[f["tgt"]] > 1:
                branch.append(fid)
            else:
                plain.append(fid)
        return branch or plain

    def edge_candidate_add() -> tuple[str, str] | None:
        succ = defaultdict(list)
        existing = set()
        for f in doc.flows.values():
            succ[f["src"]].append(f["tgt"])
            existing.add((f["src"], f["tgt"]))
        nodes = list(doc.nodes)
        shortcuts, generic = [], []
        for u in nodes:
            if doc.nodes[u]["tag"] == "endEvent":
                continue
            two = {w for v in succ[u] for w in succ[v]}
            three = {x for w in two for x in succ[w]}
            for v in (two | three):
                if v == u or (u, v) in existing:
                    continue
                if doc.nodes[v]["tag"] == "startEvent":
                    continue
                if doc.nodes[u]["proc"] != doc.nodes[v]["proc"]:
                    continue
                shortcuts.append((u, v))
        if not shortcuts:
            for u in nodes:
                for v in nodes:
                    if u == v or (u, v) in existing:
                        continue
                    if doc.nodes[u]["tag"] == "endEvent" or doc.nodes[v]["tag"] == "startEvent":
                        continue
                    if doc.nodes[u]["proc"] != doc.nodes[v]["proc"]:
                        continue
                    generic.append((u, v))
        pool = shortcuts or generic
        return rng.choice(sorted(pool)) if pool else None

    def next_flow_id() -> str:
        nums = [int(m.group(1)) for fid in doc.flows
                for m in [re.search(r"(\d+)$", fid)] if m]
        return f"SequenceFlow_{max(nums, default=0) + 1 + len([o for o in ops if o['op'] == 'add_edge'])}"

    def do_remove() -> bool:
        cands = edge_candidates_remove()
        if not cands:
            return False
        fid = rng.choice(sorted(cands))
        f = doc.flows[fid]
        ops.append({"op": "remove_edge", "flow": fid, "source": f["src"],
                    "target": f["tgt"],
                    "label": f"{f['src']} -> {f['tgt']}"})
        return True

    def do_add() -> bool:
        pair = edge_candidate_add()
        if not pair:
            return False
        u, v = pair
        ops.append({"op": "add_edge", "flow": next_flow_id(), "source": u,
                    "target": v, "label": f"{u} -> {v}"})
        return True

    n_tasks = len(renamable) + len(gateways)
    if profile == "rename":
        want = min(max(2, round(0.15 * n_tasks)), 4)
        if do_renames(want) == 0:
            do_remove() or do_add()
    elif profile == "edge":
        ok = do_remove()
        ok = do_add() or ok
        if not ok:
            do_renames(2)
    else:
        do_renames(2)
        if idx % 2 == 0:
            do_remove() or do_add()
        else:
            do_add() or do_remove()
    if not ops:
        raise RuntimeError(f"{item}: keine Manipulation moeglich")
    return ops


# ---------------------------------------------------------------------------
# Anwendung
# ---------------------------------------------------------------------------

def apply_ops(item: str, doc: BpmnDoc, dot: DotDoc, node_map, flow_map,
              ops: list[dict]) -> dict[str, str]:
    """Wendet jede geplante Operation auf BPMN und Graphviz gleichermassen an."""
    for op in ops:
        if op["op"] == "rename":
            doc.rename_node(op["element"], op["new"])
            old_key = node_map[op["element"]]
            new_key = dot.rename_node(old_key, op["new"])
            if new_key != old_key:
                node_map = {b: (new_key if g == old_key else g)
                            for b, g in node_map.items()}
        elif op["op"] == "remove_edge":
            dot.remove_edge(dot.edges[flow_map[op["flow"]]][0])
            doc.remove_flow(op["flow"])
        elif op["op"] == "add_edge":
            doc.add_flow(op["flow"], op["source"], op["target"])
            dot.add_edge(node_map[op["source"]], node_map[op["target"]])
    return node_map


# ---------------------------------------------------------------------------
# Verifikation
# ---------------------------------------------------------------------------

def verify_pair(item: str, bpmn_text: str, dot_text: str, node_map) -> None:
    root = ET.fromstring(bpmn_text)
    nodes, flows = {}, {}
    for proc in root.iter(BPMN_NS + "process"):
        for el in proc:
            tag = el.tag.replace(BPMN_NS, "")
            if tag in FLOW_NODE_TAGS:
                nodes[el.get("id")] = tag
            elif tag == "sequenceFlow":
                flows[el.get("id")] = (el.get("sourceRef"), el.get("targetRef"),
                                       el.get("name"))
    for fid, (s, t, _) in flows.items():
        if s not in nodes or t not in nodes:
            raise RuntimeError(f"{item}: Flow {fid} zeigt auf unbekanntes Element")
    # incoming/outgoing muessen zu den Flows passen
    for proc in root.iter(BPMN_NS + "process"):
        for el in proc:
            eid = el.get("id")
            if el.tag.replace(BPMN_NS, "") not in FLOW_NODE_TAGS:
                continue
            for ref in el.findall(BPMN_NS + "incoming"):
                fid = (ref.text or "").strip()
                if fid in flows and flows[fid][1] != eid:
                    raise RuntimeError(f"{item}: incoming {fid} bei {eid} inkonsistent")
            for ref in el.findall(BPMN_NS + "outgoing"):
                fid = (ref.text or "").strip()
                if fid in flows and flows[fid][0] != eid:
                    raise RuntimeError(f"{item}: outgoing {fid} bei {eid} inkonsistent")
        for fid, (s, t, _) in flows.items():
            pass
    # jede Flow-ID muss genau einmal als incoming und einmal als outgoing stehen
    inc, out = defaultdict(int), defaultdict(int)
    for el in root.iter():
        tag = el.tag.replace(BPMN_NS, "")
        if tag == "incoming":
            inc[(el.text or "").strip()] += 1
        elif tag == "outgoing":
            out[(el.text or "").strip()] += 1
    for fid in flows:
        if inc[fid] != 1 or out[fid] != 1:
            raise RuntimeError(f"{item}: Flow {fid} hat incoming={inc[fid]} "
                               f"outgoing={out[fid]}")
    for fid in list(inc) + list(out):
        if fid not in flows:
            raise RuntimeError(f"{item}: Referenz auf entfernten Flow {fid}")

    # Graphviz muss dieselben Kanten tragen
    dot_edges = []
    for line in dot_text.splitlines():
        m = EDGE_RE.match(line)
        if m:
            dot_edges.append((m.group(2), m.group(3)))
    mapped = sorted((node_map[s], node_map[t]) for s, t, _ in flows.values())
    msg = sorted(e for e in dot_edges if e not in mapped)
    remaining = sorted(dot_edges)
    for e in mapped:
        if e in remaining:
            remaining.remove(e)
        else:
            raise RuntimeError(f"{item}: Kante {e} fehlt im DOT")
    # remaining = Message Flows, die nicht im sequenceFlow-Dict stehen


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="nur Mapping und Konsistenz pruefen, nichts schreiben")
    args = ap.parse_args()

    if not args.verify:
        # Verzeichnisse nicht loeschen (unter Windows haeufig gesperrt),
        # sondern anlegen und alte Ausgaben einzeln ersetzen.
        for sub, pattern in (("BPMN", "*.bpmn"), ("Graphviz", "*.dot")):
            (OUT / sub).mkdir(parents=True, exist_ok=True)
            for stale in (OUT / sub).glob(pattern):
                stale.unlink()

    log: dict[str, list[dict]] = {}
    stats = defaultdict(int)
    for i in range(1, 56):
        item = f"{i:02d}"
        doc = BpmnDoc(DATASET / "bpmn" / f"{item}.bpmn")
        dot = DotDoc(DATASET / "graphviz" / f"{item}.dot")
        node_map, flow_map = build_mapping(item, doc, dot)
        rng = random.Random(20260829 + i)
        ops = plan(item, doc, rng)
        node_map = apply_ops(item, doc, dot, node_map, flow_map, ops)
        bpmn_text, dot_text = doc.text(), dot.text()
        verify_pair(item, bpmn_text, dot_text, node_map)
        for op in ops:
            stats[op["op"]] += 1
        log[item] = ops
        if not args.verify:
            (OUT / "BPMN" / f"{item}.bpmn").write_text(bpmn_text, encoding="utf-8")
            (OUT / "Graphviz" / f"{item}.dot").write_text(dot_text, encoding="utf-8")
        print(f"{item}: " + ", ".join(
            f"{o['op']}({o.get('old', o.get('label'))}"
            + (f" -> {o['new']}" if o["op"] == "rename" else "") + ")"
            for o in ops))

    print("\nSumme:", dict(stats))
    if not args.verify:
        (OUT / "manipulations.json").write_text(
            json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"geschrieben nach {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
