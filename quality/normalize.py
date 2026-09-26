"""
quality/normalize.py — BEF4LLM score normalisation
======================================================
Shared normalisation so no dimension re-implements the banding logic.

The paper defines two normalisation functions (§4.2, Eq. 1 and Eq. 2), each
banding a raw value against four empirically validated thresholds into five
groups. **Both take the thresholds in ascending order**, exactly as Table A.18
prints them; the two differ only in which end is the good end:

* `normdesc` — lower is better. The size metrics, density, gateway
  heterogeneity, control-flow complexity, depth and token split use it.

      score = 1.0    if        x <  t1
              0.75   if t1 <=  x <  t2
              0.5    if t2 <=  x <  t3
              0.25   if t3 <=  x <  t4
              0.0    if t4 <=  x

* `normasc` — higher is better. Cross-connectivity, sequentiality and
  separability use it. **It follows BEF4LLM's `get_rank`** since 2026-08-24,
  not Eq. 2 as printed:

      score = 1.0    if       x <= t1     ← their default rank, see below
              0.25   if t1 <  x <= t2
              0.5    if t2 <  x <= t3
              0.75   if t3 <  x <= t4
              1.0    if t4 <  x

`normdesc` uses half-open bands `[t_i, t_(i+1))`, so a value landing exactly on
a threshold belongs to the band that threshold opens, as the paper spells out.
`normasc` is half-open at the other end, because their comparison is `score >
ts`.

The paper's own worked example (§4.2): a model with 45 nodes, TNN thresholds
29.9 / 43.7 / 58.1 / 81.1, lower is better, `43.7 <= 45 < 58.1` → **0.5**,
group 3. `normdesc(45, TNN_THRESHOLDS)` returns exactly that.

> ⚠ **The "higher is better" banding scores its worst input as its best.**
> `get_rank` handles that direction with `rank = 0` as the default and lowers
> it only on the first `score > ts` hit, so a value below *every* threshold
> never breaks out of the loop and comes out at **1.0**. `normasc` reproduces
> that (author's decision, 2026-08-24) because these columns are meant to be
> comparable with theirs — 35 of the 55 reference models land there on
> sequentiality. Do not read `prag_sequentiality_score`,
> `prag_separability_score` or `prag_cc_score` as quality signals; the raw
> columns beside them are the measurements.

Thresholds are the published values from **Table A.18** and live here as named
constants so a metric never carries loose numbers.
"""
from __future__ import annotations

from typing import Optional, Tuple

Thresholds = Tuple[float, float, float, float]

# ── Table A.18, verbatim and in the paper's ascending order ─────────────────
# Size
TNN_THRESHOLDS: Thresholds = (29.9, 43.7, 58.1, 81.1)              # [48]
TNG_THRESHOLDS: Thresholds = (1.42, 3.36, 5.3, 6.49)               # [48]
TNSF_THRESHOLDS: Thresholds = (19.4, 34.8, 50.2, 74.8)             # [48]
DIAMETER_THRESHOLDS: Thresholds = (7.92, 12.2, 16.5, 23.4)         # [48]
# Table A.18 also publishes TNMF (message flows) = (1.09, 7.15, 13.2, 22.8).
# No metric uses it: TNMF is deliberately not implemented in either dimension.

# Density
DENSITY_THRESHOLDS: Thresholds = (0.1361169, 0.357143, 0.741667, 2.33333)   # [51]
AGD_THRESHOLDS: Thresholds = (3.67, 3.88, 4.06, 4.18)              # [49]
# ⚠ t4: Table A.18 prints 4.18 — the same value as AGD's t4 in the row directly
# above, and out of step with this row's own 0.53 spacing. BEF4LLM's code has
# 2.28 here. The published value is used, because that is what a thesis cites;
# on PMo nothing turns on it (CNC never exceeds 1.43 on the reference models,
# so no model lands between the two candidates).
CNC_THRESHOLDS: Thresholds = (0.37, 0.9, 1.43, 4.18)               # [50]

# Connector interplay
GH_THRESHOLDS: Thresholds = (0.62, 0.79, 0.92, 0.94)               # [49]
CFC_THRESHOLDS: Thresholds = (13, 22, 37, 51)                      # [49]
CROSS_CONNECTIVITY_THRESHOLDS: Thresholds = (0.007996, 0.030407, 0.061814, 0.112903)  # [51]

# Partitionability
SEQUENTIALITY_THRESHOLDS: Thresholds = (0.25, 0.48, 0.7, 1.07)     # [48]
SEPARABILITY_THRESHOLDS: Thresholds = (0.03, 0.37, 0.71, 1.24)     # [50]
DEPTH_THRESHOLDS: Thresholds = (0.42, 1.72, 3.02, 5.09)            # [49]

# Concurrency
TOKEN_SPLIT_THRESHOLDS: Thresholds = (0.12, 0.21, 0.6, 1.36)       # [50]


def normdesc(x: Optional[float], thresholds: Thresholds) -> Optional[float]:
    """Eq. 1 — band `x` against `(t1, t2, t3, t4)`, **lower is better**.

    Bands are half-open on the left (`t1 <= x < t2`), exactly as printed, so a
    value landing precisely on a threshold falls into the band that threshold
    opens. `None` in, `None` out — an unmeasurable value must not become a 0.0
    that reads like a measured worst case.
    """
    if x is None:
        return None
    t1, t2, t3, t4 = thresholds
    if x < t1:
        return 1.0
    if x < t2:
        return 0.75
    if x < t3:
        return 0.5
    if x < t4:
        return 0.25
    return 0.0


def normasc(x: Optional[float], thresholds: Thresholds) -> Optional[float]:
    """Eq. 2 — band `x` against `(t1, t2, t3, t4)`, **higher is better**, as
    BEF4LLM's `get_rank` bands it (author's decision, 2026-08-24).

    Their thresholds for these metrics are stored in *descending* order, which
    sends `get_rank` down its second branch:

        rank = 0                                   # the best rank, as default
        for ts in list_thresholds:
            if score > ts:
                rank = list_thresholds.index(ts)
                break
        rank_score = (4 - rank) * 0.25

    Two consequences, both reproduced here:

    * the bands are half-open on the **upper** side — `t1 < x ≤ t2` is the
      0.25 band, so a value sitting exactly on a threshold falls into the band
      *below* it, the opposite of `normdesc`;
    * **a value at or below `t1` scores 1.0**, the best. Nothing breaks the
      loop, so `rank` keeps its default of 0. For a metric where more is
      better, that is the worst input scoring like the best one — 35 of the 55
      reference models land there on sequentiality, item 07 with 0.0238 among
      them.

    That second point is a defect, not a reading: it is kept because these
    columns are meant to be comparable with theirs, and it is why
    `prag_sequentiality_score` and `prag_separability_score` must not be quoted
    as quality signals. `normdesc` is unaffected — for a "lower is better"
    metric their ascending list and this project's bands already agree.

    `None` in, `None` out.
    """
    if x is None:
        return None
    t1, t2, t3, t4 = thresholds
    if x > t4:
        return 1.0
    if x > t3:
        return 0.75
    if x > t2:
        return 0.5
    if x > t1:
        return 0.25
    return 1.0
