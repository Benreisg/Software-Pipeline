# PMo_manipulated

Manipulierte Varianten aller 55 PMo-Prozessmodelle, in zwei Repräsentationen:

- `BPMN/01.bpmn` … `BPMN/55.bpmn` (Quelle: `pmo-dataset/bpmn/`)
- `Graphviz/01.dot` … `Graphviz/55.dot` (Quelle: `pmo-dataset/graphviz/`)

**Beide Ordner tragen für jeden Eintrag exakt dieselbe Manipulation.** Die
Änderungen werden auf dem BPMN-Modell geplant (laut PMo-README die Ground Truth)
und über ein verifiziertes Element-Mapping identisch auf die Graphviz-Datei
angewendet. `BPMN/xx.bpmn` und `Graphviz/xx.dot` beschreiben also weiterhin
dasselbe Prozessmodell — nur eben das manipulierte.

## Art der Manipulationen

| Operation | Anzahl | Beschreibung |
|---|---|---|
| `rename` | 82 | Task- bzw. Gateway-Name durch eine semantisch verschobene Variante ersetzt (Synonym, umformulierte Bedingung) |
| `remove_edge` | 27 | Eine Sequenzkante entfernt, bevorzugt ein Gateway-Zweig |
| `add_edge` | 27 | Eine Sequenzkante ergänzt, bevorzugt als Abkürzung über 2–3 Schritte |

Verteilung über die 55 Einträge: 19 nur Umbenennungen, 18 nur Kantenänderungen,
18 mit beidem. Pro Eintrag 2 bis 4 Operationen.

Umbenannt werden ausschließlich **Task-** und **benannte Gateway-Namen**;
Ereignisnamen und Elementtypen bleiben unangetastet. Element-IDs bleiben stabil,
damit sich manipulierte und originale Modelle direkt gegenüberstellen lassen.

## Protokoll

`manipulations.json` listet je Eintrag alle Operationen mit Element-ID, altem und
neuem Namen bzw. Quelle und Ziel der betroffenen Kante.

## Konsistenz der Dateien

- Alle 55 BPMN-Dateien sind wohlgeformtes XML; `incoming`/`outgoing` und die
  Diagrammkanten (`BPMNEdge`) wurden zu jeder Kantenänderung mitgeführt.
- Alle 55 DOT-Dateien rendern fehlerfrei mit `dot`.
- Ein Isomorphie-Test über alle Paare zeigt dieselben Abweichungen wie im
  Original-Dataset — die Manipulation führt keine neuen ein.

## Neu erzeugen

    python make_pmo_manipulated.py            # schreibt diesen Ordner neu
    python make_pmo_manipulated.py --verify   # prüft nur, schreibt nichts

Das Skript arbeitet deterministisch (fester Seed pro Eintrag), ein erneuter Lauf
liefert dieselben Dateien.
