# PMo / Graphviz-DOT — Cost & Token Comparison Pipeline (multi-vendor)

Branch 1 of the thesis software:
**"Large Language Models in Process Model Generation: A Cost and Token Usage
Comparison Across Models and Prompting Strategies."**

This is the **multi-vendor** build of the pipeline: it generates **BPMN process
models as Graphviz/DOT** from the natural-language descriptions in the **PMo**
dataset, using models from **Anthropic, OpenAI,
Mistral, and Google**, and measures **token usage, cost, and latency**
across **models**, **vendors**, and **prompting strategies** — matching the
thesis decision log's "one Efficient / Standard / Advanced model per vendor"
requirement. (See `Software/` one level up for the Anthropic-only build this
was extended from.)

> ⚠ **Before any paid run:**
> - **Anthropic** model IDs/prices in `config.py` were checked against current
>   Claude API documentation and are believed correct.
> - **OpenAI, Mistral, and Google** model IDs are best-effort placeholders and
>   their prices are deliberately left unset (`None`) so the pipeline never
>   fabricates a dollar figure for them — confirm both against each vendor's
>   current docs and fill in real prices in `config.py` before trusting
>   `cost_usd` for those vendors, or before spending the ~€200 budget.
> - The OpenAI/Mistral/Google **provider integrations themselves**
>   (`providers.py`) are best-effort — this assistant had no live, verified
>   SDK documentation for those three vendors when writing them. Smoke-test
>   each with `--limit 1` before a full run.

---

## What it does

For every `(model × strategy × process item)` combination — repeated
`--repetitions` times, 1 by default (see *Repetitions*) — it:

1. Builds the prompt for that strategy (see *Prompting strategies* below).
2. Calls that model's vendor, capturing **every billable token class**
   (uncached input, output, cache read, cache writes per TTL — see *Cost*),
   **USD cost** at the rate actually billed, the **round trip of the
   prompt** (`api_latency_s` — request sent → full reply received, plus two
   wider spans and both wall-clock timestamps; see *Timing* below), and an
   explicit note on how — or whether — that vendor reports reasoning tokens.
3. Extracts the DOT from the reply and validates it parses.
4. Persists the raw reply and the extracted `.gv`, then appends a row to
   `results.csv` / `results.jsonl`.
5. Scores every generated model (see *Quality metrics*) and merges the columns
   into `results.csv`.
6. Aggregates everything into summary and pivot tables for the analysis chapter.

## Dataset — PMo

An item is a **pair of files sharing one number**, in two sibling folders:

```
<PMO_DATASET_DIR>/descriptions/03.txt   → the process description (LLM input)
<PMO_DATASET_DIR>/graphviz/03.dot       → the ground-truth model (scored against)
```

55 pairs, ids `01`–`55`. The stem (`03`) is the `item_id` in every output
filename and results row.

The dataset lives **outside** the project — it is data, not code. The default
location is set in `config.py`:

```python
PMO_DATASET_DIR = Path(os.environ.get("PMO_DATASET_DIR")
                       or r"C:\Users\breis\Desktop\Data_Sets\pmo-dataset")
```

Override it with the `PMO_DATASET_DIR` environment variable (per machine) or
`--dataset-dir` (per run). `--dataset-dir` accepts the dataset root **or**
either of its two subfolders — the sibling is derived.

**Few-shot exemplars** are PMo items too: `cfg.FEW_SHOT_IDS` (default `01`, `02`)
are shown as in-context examples. They are **not** held out — a run covers all
**55** items, so every strategy is scored on the whole dataset. Leakage is
prevented per item instead of per run: when the item being generated for *is*
one of the exemplars, that one exemplar is swapped for a stand-in from
`cfg.FEW_SHOT_FALLBACK_IDS` (default `03`, `04`), so no model ever sees the
answer to the item it is asked to model.

The three items nearest the exemplar set skip that per-slot swap entirely:
`cfg.FEW_SHOT_ALT_FOR_IDS` (default `01`, `02`, `03`) run on the **complete**
alternate set `cfg.FEW_SHOT_ALT_IDS` (default `40`, `41`). `01` and `02` are the
exemplars themselves; `03` does not leak — it is not an exemplar — but it is
the stand-in the other two are shown, so covering it too means all three see
*the same* exemplar pair instead of three differently-swapped prompts
(`03`+`02`, `01`+`03`, `01`+`02`). The alternate ids must be disjoint from both
`FEW_SHOT_IDS` and `FEW_SHOT_FALLBACK_IDS`; emptying either list falls back to
the per-slot swap, which still keeps every item out of its own prompt.

Both mechanisms are recorded in the run's `manifest.json`: which items got which
set under `few_shot_substitutions` (`"01": "40,41"` for a complete replacement,
`"01": "03"` for a single slot), the files under `exemplar_files` /
`fallback_exemplar_files` / `alt_exemplar_files`. Change the exemplars with
`--few-shot-ids 07,19`, the stand-ins with `--few-shot-fallback-ids`, the
alternate set with `--few-shot-alt-ids` / `--few-shot-alt-for-ids`, or point
outside the dataset with `--few-shot-files`. `--hold-out-few-shot-items`
restores the old behaviour of dropping the exemplars from the run (55 → 53).

### ⚠ DOT quirks any scorer must handle

Learned the hard way while wiring up the (now removed) metric suite — carried
here so the next one does not repeat it:

1. **Tasks are implicit.** PMo declares tasks *only* via the graph-level default
   `node [shape=box];` and creates them inside edge statements, never as their
   own node statement. A parser that reads explicit `shape=` attributes only
   (as the supplied suite did) sees **zero tasks and drops every edge touching
   one** — on `01.dot` that was 5 of 18 edges surviving. A scorer must apply DOT
   default node attributes and materialise nodes that appear only as edge
   endpoints. This applies to *generated* models too: several prompt templates
   explicitly tell the model that tasks may rely on the `shape=box` default.
2. **The end event is not a `doublecircle`.** PMo draws it as
   `shape=circle, penwidth=4`, so pure shape matching classifies it as a
   *start*. A topology rule works across conventions: a circle-shaped sink (no
   outgoing edge, ≥1 incoming) is an end event.
3. **`doublecircle` means something different on each side.** The prompt
   templates tell the model to use it for the **end event**, so that is what
   generated models mean by it. The Camunda-derived PMo items use it for
   **intermediate catching events** that legitimately have outgoing flows
   (item 24: `"Guest appears" -> "Hand over meal"`), exactly as PMo's own
   README states: "End events are depicted with a bolded outer circle, while
   intermediate events use a double circle, consistent with BPMN".

   Resolved by letting the shape decide only for a **sink** (2026-08-23): a
   `doublecircle` with no outgoing sequence flow is an end event — what every
   generated model drawing one means — and a `doublecircle` with an outgoing
   flow is an intermediate event, which is what an end event with something
   after it can only be. Before that, all **17** intermediate events of the
   reference set (items 22, 23, 24, 25, 30, 38, 42, 44) were counted as end
   events, which cost those models metrics #4 and #8, produced 17 phantom
   sequence-flow violations, and truncated their diameter at the first
   intermediate event — item 30 measured a diameter of 4 instead of 18. The
   scorer now counts 58 starts, 61 ends and 17 intermediates, which is exactly
   what the BPMN ground truth contains.

4. **A node can be declared twice, and the second statement merges.** This one
   is a *generated*-model quirk, found 2026-08-28. Models like to declare the
   shape first and colour the node in a styling block at the foot of the file:

   ```dot
   node [shape=box, style=rounded];      // graph-level default
   Start [shape=circle, label="Need for New Hire"];
   Gateway1 [shape=diamond, label="Select candidates?"];
   …
   Start    [fillcolor=lightgreen, style="filled,rounded"];   // styling block
   Gateway1 [fillcolor=lightyellow, style=filled];
   ```

   In DOT the second statement **adds to** the node — Graphviz's own `nop`
   prints `Start` once, with `shape=circle` intact. A parser that lets the later
   statement *replace* the earlier one loses the shape, the graph default
   `node [shape=box]` takes over, and every start event, end event and gateway
   in that file is read as a **task**. That is what `_collect_declared` did
   until 2026-08-28: one generated model (`02 · anthropic_efficient ·
   zero_shot_cot · r2`) scored as 15 tasks, 0 gateways, no start and no end
   event, against a file that plainly draws 8 tasks, 5 diamonds and two circles
   — and its `syn_bef_score` of 0.673 was computed over a process that is not
   the one in the file, with three gateway metrics counted as "not applicable"
   (and so as 1.0) because their elements had vanished. Fixed by merging
   attribute by attribute, as Graphviz does; two tests hold it there —
   `test_a_node_declared_twice_keeps_its_shape` for the classification, and
   `test_merged_node_attributes_match_graphviz`, which compares our merged
   attributes against what `nop` prints.

   **15 of 2851** parsed models in `runs/` were affected (1–12 nodes each,
   across 10 runs); the PMo reference models are not — they never re-declare a
   node. The 10 runs were re-scored on 2026-08-28. The corrected scores mostly
   go **down**, not up: gateways that were invisible are now measured by the
   degree rules and by metric 9, where before they were silently absent. The
   one exception is the call above, whose start and end events came back:
   `syn_bef_score` 0.673 → 0.923.

Also worth knowing: items `23`/`24` genuinely have several start and end events
(they are multi-pool Camunda models) — that is the dataset, not a parser bug.

## Vendors and models

One Efficient / Standard / Advanced model per vendor, per the supervisor's
decision. Model keys are `{vendor}_{tier}`:

| vendor    | efficient              | standard             | advanced                                 | live? | prices verified? |
|-----------|------------------------|----------------------|------------------------------------------|-------|-------------------|
| anthropic | claude-haiku-4-5       | claude-sonnet-5      | claude-opus-5 (adaptive, effort medium)  | ✅ | ✅ 2026-08-30 |
| openai    | gpt-5.6-luna           | gpt-5.6-terra        | gpt-5.6-sol (effort medium)              | ✅ | ✅ 2026-08-30 |
| mistral   | mistral-small-latest   | mistral-large-latest | mistral-medium-latest (effort high)      | ✅ | ✅ 2026-08-30 |
| google    | gemini-3.5-flash-lite  | gemini-3.6-flash     | gemini-3.1-pro-preview (level MEDIUM)    | ✅ | ✅ 2026-08-30 |
| deepseek  | deepseek-v4-flash (reasoning **off**) | *(none)* | deepseek-v4-pro (reasoning ON, effort high) | ✅ | ✅ 2026-08-30 |

> ⚠ **Reasoning depth changed on 2026-08-31.** Three Advanced tiers were moved
> from `high` (Google: unstated) to **`medium`**, and Mistral's Advanced tier
> uses **`high`**. Rows recorded before that date were measured at the old
> depths and are not comparable with rows recorded after it. `effort` is written
> per row into the manifest, which is what tells the two apart.
> The re-enabled Pro tier requests reasoning at `high`; DeepSeek's scale is
> `low | high | max`, so it has no medium to match the other Advanced tiers with.
>
> DeepSeek's **Flash tier was switched to reasoning on and back off again on the
> same day** and is reasoning-**off** now — see the DeepSeek notes below for the
> measurement that decided it.

That is **14 models, not 15**: DeepSeek sells two models and has no Standard
tier. `deepseek-v4-pro` is enabled as its Advanced tier, so this vendor
contributes two rows where the others contribute three.
Nothing in the pipeline assumes three tiers per vendor.

> **Renamed 2026-08-20.** The third tier was called `reasoning` and is now
> `advanced` — on Google and DeepSeek the slot is filled by a larger model
> class rather than by a reasoning switch, so the old name described only three
> of the five vendors. The keys changed with it (`*_reasoning` → `*_advanced`).
> The five retired specs stay in `ALL_MODELS` with `enabled=False`: they are
> never selectable and never run, but `model_by_key` still finds them, so runs
> recorded before the rename keep their per-token-class cost breakdown. Three
> of them carry live measurements.

**All five vendors are wired up** (`cfg.LIVE_VENDORS`) and call their real APIs;
the mock provider remains available through `--provider mock`. The OpenAI
provider uses the **Responses API** — see *Cost* for why that choice is not
cosmetic.

DeepSeek needs four notes of its own. It serves **only two models**, so it
fills Efficient and Advanced and leaves **Standard empty**. Both models reason
by default at effort `high`; this configuration requests that setting explicitly
on the Advanced tier and switches it off explicitly on the Efficient one.

> ⚠ **Changed twice on 2026-08-31: the Flash tier reasons again — no.** It was
> switched on in the morning (`thinking={"type": "enabled"}` +
> `reasoning_effort="high"`) and **back off the same day**; it sends
> `{"type": "disabled"}` again, as it did until then, so the cheap tier does not
> bill thinking nobody asked for. `tier="efficient"` describes the setting again.
>
> The reversal is measured, on the 245 Flash calls in `runs/`:
>
> | Flash | n | median latency | median output tokens | of which reasoning | median cost |
> |---|---:|---:|---:|---:|---:|
> | reasoning **on** | 62 | 81.0 s | 10082 | 9505 (94 %) | $0.00673 |
> | reasoning **off** | 183 | 5.6 s | 640 | 0 | $0.00047 |
>
> **14x the latency and 14x the cost** for the same task. Per-token throughput
> was never the issue — DeepSeek streams at ~108 tok/s, mid-field across the five
> vendors and ahead of OpenAI's ~81 — the token count was. Two comparability
> faults came with it as well: this tier ran at `high` while the other four
> vendors' Advanced tiers were set to `medium` the same day (DeepSeek has no
> `medium`), and every other vendor's **Efficient** tier does not reason at all.
> Rows from that window stay identifiable: `thinking_tokens > 0`, a
> `reasoning_note` naming `thinking={"type": "enabled"}`, and `params` in each
> `prompts/<tag>.json`. They are not comparable with rows on either side of it.

> ⚠ **Corrected 2026-08-20.** That switch was `reasoning_effort="none"` and did
> nothing. DeepSeek's `reasoning_effort` accepts `low | high | max`; thinking is
> turned on and off through a separate `thinking` object
> (`{"type": "enabled"}` / `{"type": "disabled"}`), which the OpenAI SDK can only pass as
> `extra_body` while `reasoning_effort` stays a top-level argument — the split
> DeepSeek's own SDK example shows. DeepSeek **ignores parameter values it does
> not recognise rather than erroring**, so the Efficient tier kept reasoning and
> kept billing it while the row said reasoning was off. Any DeepSeek measurement
> taken before this date has to be read with that in mind. Verified against the
> API reference and the Thinking Mode guide, 2026-08-20.

Until 2026-08-20 the two models were stretched over three tiers by running one
of them twice, with and without the reasoning switch, and the reasoning tier ran
on **Flash, not Pro**. That is gone: each model now appears once, in the tier its
class belongs to.

> ⚠ **Re-enabled 2026-08-31.** `deepseek-v4-pro` is configured as the Advanced
> tier with reasoning enabled at effort `high`. It had been removed on
> 2026-08-25 because the reasoning-on configuration took a median **269.6 s**
> per call (n=3, range 137–402 s), versus 43.4 s for the next-slowest Advanced
> tier. The reinstated configuration uses that same reasoning depth, so this
> runtime warning applies directly; time a `--limit 1` call before a paid sweep.

The numbers an earlier Pro run with the same reasoning depth produced
(2026-08-18, 32768-token budget, two items, at peak rates):

| | output tokens | of which reasoning | cost per call |
|---|---|---|---|
| `deepseek-v4-pro` (advanced, reasoning on) | 10642 / 9014 | 9893 / 8246 | $0.042 / $0.036 |
| `deepseek-v4-flash` (efficient, reasoning on) | 6622 / 5866 | 6105 / 5261 | $0.0088 / $0.0078 |

A later measurement (2026-08-25, two items, off-peak) also compared the two
reasoning efforts DeepSeek accepts, which is what settled the removal:
`effort="high"` took 335.6 s per call at 11398 output tokens, `effort="low"`
65.7 s at 4891 — 5.1× faster for 43 % of the cost, with valid DOT in 4 of 4.
Lowering the effort was rejected as a fix because it changes tokens, cost and
quality at once, not just the runtime.

> ⚠ **The Advanced tiers need a large output budget — this is now the
> default (32768), and lowering it breaks them.** Reasoning is billed inside the
> output allowance, so a cap that fits the answer does not fit the thinking that
> precedes it. Measured on the PMo prompts, 2026-08-18: at 4096 both
> `mistral-medium-latest` and the reasoning DeepSeek model spent the **entire**
> budget reasoning and returned nothing — `stop_reason=length`, 0/4 valid,
> billed in full. A run now warns when a call is truncated with nothing usable,
> naming the models and the money spent on nothing.
>
> ⚠ **DeepSeek in particular needs it:** It spends
> 6k–11k output tokens per call, nearly all of it on reasoning. At the default
> 4096 the entire allowance goes to reasoning, `stop_reason` is `length`, no
> answer is returned — **and the call is billed in full**. This was measured on
> both models: four calls out of four produced nothing at 4096 and at 8192.

Its **cache hit costs about a thirtieth of a miss**,
not the usual tenth, so the cached rate is stated per model and never derived
from `CACHE_READ_MULTIPLIER`. And its **rates double during peak hours**
(01:00–04:00 and 06:00–10:00 UTC, **Monday to Friday**): the table below lists
the peak rates, `cost_usd` halves them off-peak, and every row records which
applied in `cost_basis` (`list+peak` / `list+offpeak`). A DeepSeek cost figure
is therefore only reproducible together with the date *and* time of day the
call ran.

> ⚠ **Corrected 2026-08-30.** `is_peak_hour` read only the hours and ignored
> the "Monday through Friday" half of the same published sentence, so every
> weekend call was priced at the peak rate — exactly double. Weekends are when
> a long run is most likely to be left going unattended, so this was the
> costly direction of the two.

Mistral needs two notes of its own. Its **naming no longer tracks its price
ladder**: Large 3 is the cheap fast model at $0.50/$1.50 and Medium 3.5 the
premium one at $1.50/$7.50, so the tiers follow price, not the word in the name.
Small 4 is reasoning-capable — it absorbed Magistral's reasoning into the
general line — and Mistral documents `reasoning_effort="none"` ("the model
thinks minimally and the thinking chunk is omitted") but **not** what an omitted
parameter does. Since 2026-08-20 the Efficient tier therefore sends `"none"`
explicitly. Before that it sent nothing, so its reasoning state was undocumented
rather than known; the ten live `mistral_efficient` rows already in `runs/` were
measured under that unknown and are not strictly comparable with rows recorded
after the change. Large 3 is not documented as taking the parameter at all, so
the Standard tier keeps sending nothing.

And it has **no reasoning-specialised model** left for the Advanced slot —
`magistral-medium-latest` was retired, and Mistral Small 4 absorbed Magistral's
reasoning into the general line. `magistral-small-latest` still resolves but
carries no published price, and an unpriced model cannot fill a tier when cost
is a result, so the Advanced tier is Medium 3.5 with reasoning explicitly
enabled via `reasoning_effort="high"`. Because the naming runs backwards, the Advanced tier is
the *Medium* model and the Standard tier the *Large* one.

Two things about the Google row are worth carrying into the write-up. The
Gemini **2.5** family this project started with (flash / pro / flash-lite) is
closed to new API keys and answers 404, so the tiers come from the 3.x line.
Google sells exactly **one Pro-class text model** on the Gemini API and only
under a preview id: `gemini-3.1-pro-preview` (`gemini-3-pro-image` generates
images; every other 3.x model is a Flash or a Flash-Lite). It fills the Advanced
tier as of 2026-08-20, replacing the older `gemini-3.5-flash` that stood in
while the slot had no Pro-class model to put in it.

> ⚠ **A preview model is a reproducibility risk and the thesis has to say so.**
> Google can change or withdraw a preview id without notice; there is no stable
> alias to pin instead (checked 2026-08-20). A withdrawal fails loudly in
> `check_connection` before a run spends anything, rather than quietly measuring
> a different model — but a re-run months later may not be able to reproduce
> this tier at all.

3.1 Pro is also the only model here with **prompt-size-tiered pricing**: $2/$12
per Mtok up to a 200k-token prompt, $4/$18 above it (cache read $0.20 / $0.40).
`config.py` carries the ≤200k rates, which are the only ones PMo prompts — a few
hundred tokens each — can reach. `cost_usd` has no notion of prompt-size tiers,
so a future long-context run on this model would be priced too low.

Every Gemini 3.x model also reasons *by default*, so even the non-thinking tiers
produce billable thinking tokens — the provider records them on all three.

Edit `config.py`'s `ALL_MODELS` to correct IDs/prices or add/disable models.

## Cost

Cost is a headline result of this thesis, so the pipeline prices **every
billable quantity an API call produces**, not just input and output.

### What is billed

Each vendor has its own price sheet with its own date (`cfg.PRICING_SHEETS`,
recorded per run in `manifest.pricing.sheets` — a cost figure without the date
of the price list behind it is not reproducible):

| model | input $/Mtok | cached input $/Mtok | output $/Mtok |
|---|---|---|---|
| **Anthropic** — sheet of 2026-08-30 | | | |
| `claude-sonnet-5` (standard) | 2.00 | 0.20 | 10.00 |
| `claude-opus-5` (advanced) | 5.00 | 0.50 | 25.00 |
| `claude-haiku-4-5` (efficient) | 1.00 | 0.10 | 5.00 |
| **OpenAI** — sheet of 2026-08-30 | | | |
| `gpt-5.6-terra` (standard) | 2.00 | 0.20 | 12.00 |
| `gpt-5.6-sol` (advanced) | 5.00 — **4.00 promotional through 2026-11-21** | 0.50 — **0.40 promotional** | 30.00 — **20.00 promotional** |
| `gpt-5.6-luna` (efficient) | 0.20 | 0.02 | 1.20 |
| **Google** — sheet of 2026-08-30 | | | |
| `gemini-3.6-flash` (standard) | 1.50 — **0.75 promotional through 2026-12-31** | 0.15 — **0.075 promotional** | 7.50 — **3.75 promotional** |
| `gemini-3.1-pro-preview` (advanced) | 2.00 — *$4.00 above a 200k-token prompt* | 0.20 | 12.00 — *$18.00 above 200k* |
| `gemini-3.5-flash-lite` (efficient) | 0.30 | 0.03 | 2.50 |
| **Mistral** — sheet of 2026-08-30 | | | |
| `mistral-large-latest` — Large 3 (standard) | 0.50 | 0.05 | 1.50 |
| `mistral-medium-latest` — Medium 3.5 (advanced) | 1.50 | 0.15 | 7.50 |
| `mistral-small-latest` — Small 4 (efficient) | 0.15 | 0.015 | 0.60 |
| **DeepSeek** — sheet of 2026-08-30, **peak rates; halved off-peak** | | | |
| `deepseek-v4-pro` (advanced, reasoning on) | 1.32 | 0.044 | 3.96 |
| `deepseek-v4-flash` (efficient) | 0.44 | 0.014 | 1.32 |

> ⚠ **DeepSeek's sheet changed shortly before this was written.** The
> peak/off-peak rates above are what the official pricing page states verbatim
> on 2026-08-20; the changeover date, 2026-08-16 16:00 UTC, comes from secondary
> sources rather than from the page itself, so treat the date as the softer of
> the two claims. Until then DeepSeek billed one flat rate around the clock —
> $0.14 in / $0.0028 cached / $0.28 out on Flash, $0.435 / $0.003625 / $0.87 on
> Pro — and since then the peak/off-peak rates above. A DeepSeek call measured
> before that moment therefore **cannot** be re-priced with the numbers in
> `config.py`, and the hit/miss ratio differs too (1/50 and 1/120 under the old
> sheet, ~1/30 under this one). No run in `runs/` is affected: every DeepSeek
> row on file so far is a mock row that cost nothing.

Anthropic's cache-read rate is a flat 0.10× the input rate across the line;
OpenAI publishes one **per model** (0.10× for the gpt-5.6 family, but 0.5× on
o1 and 0.25× on o3), so `ModelSpec.price_cached_in_per_mtok` carries the rate
explicitly and the multiplier is only a fallback.

On top of the two per-token rates, four multipliers apply to the **input** rate,
plus one per-request tool price:

| billable class | rate | column |
|---|---|---|
| uncached prompt | 1× input | `input_tokens` |
| completion (thinking included) | 1× output | `output_tokens` |
| cache **read** | 0.10× input | `cached_tokens` |
| cache **write**, standard rate | 1.25× input | `cache_write_tokens` |
| cache **write**, 1-hour TTL (Anthropic only) | 2× input | `cache_write_1h_tokens` |
| Batches API tier | 0.5× everything | `service_tier` = `batch` |
| server-side web search | $10 per 1,000 requests | `web_search_requests` |

**`input_tokens` is the uncached prompt only.** Anthropic reports it that way;
OpenAI, Mistral, DeepSeek and Gemini report the whole prompt instead, and the
providers normalise theirs down so the column means one thing in every row. The
vendor's own figure survives verbatim in **`reported_input_tokens`** — the
provenance for that normalisation, so a row can be checked against the vendor's
dashboard. It is never priced and, for the same reason the normalisation exists,
never comparable across vendors. The whole prompt as billed is
`billable_input_tokens` = input + cache read + cache writes, and `total_tokens`
is that plus output. Pricing `input_tokens` alone therefore *undercounts* the
moment caching is switched on.

**`total_tokens` is this project's own sum**, so the vendor's grand total is
kept beside it in **`reported_total_tokens`** — OpenAI's, Mistral's and
DeepSeek's `total_tokens`, Gemini's `total_token_count` — and the two are
compared on every call (see *Nothing billed may go untracked*). A `0` there
means the vendor states no total: Anthropic's Messages usage carries none, which
is a fact about that API rather than a gap. **`reported_output_tokens`** is the
same story on the answer side, and it is Gemini that makes it necessary: its
`candidates_token_count` counts the answer alone, while `output_tokens` is the
billable figure and adds the thoughts Gemini charges at the output rate. On the
other four vendors the two are the same number. **`tool_use_prompt_tokens`** is
Gemini's `tool_use_prompt_token_count`. Google's REST contract presents it as a
tool-use prompt breakdown and defines the total without adding it again. Some
SDK schemas describe it as a separate term, so the provider reconciles it
against `total_token_count` and counts it exactly once in either response shape.
The column stays informational and is never priced directly. Anthropic
contributes the last of these columns: **`reported_cache_creation_tokens`** is its
`cache_creation_input_tokens`, the flat total of all cache writes, which it
states *alongside* the 5m/1h split rather than instead of it — it is the only
vendor here that reports both, and the only one where the raw figure is not
already a column. Between the three `reported_*` columns, every token figure the
five APIs return is now in the row verbatim, next to the normalised columns
derived from it. The modality breakdowns stay out of the table and go to the
run's raw archive instead (see *Output*), where they bloat nothing.

`cost_usd` is the full sum; `cost_basis` records which rate produced it
(`list`, `intro`, `unverified`, `+batch`, `mock`). Unverified prices give an
empty `cost_usd` — never `0.0`, which would read as "this call was free".

### Promotional rates

Billing applies a vendor promotion automatically, so pricing at the list rate
inside the window overstates the run's own cost. `cfg.prices_for(model)`
resolves which rate applies **on the day the run happens** and every row records
the basis. As of the 2026-08-30 sheet check the window that matters is
`gpt-5.6-sol`'s: $4/$20 through 2026-11-21, then $5/$30 with no code change.

> **Sonnet 5's promotion became the price.** Until 2026-08-30 this project
> encoded Sonnet 5 as $3/$15 list with a $2/$10 promotion expiring 2026-08-31.
> Anthropic cancelled the increase and made $2/$10 standard, so the promotional
> fields are gone from `config.py`. Had they stayed, every Sonnet 5 call from
> 2026-09-01 on would have been priced 50 % above what it cost — and nothing in
> the pipeline would have flagged it, because a stale *list* rate is exactly
> what `cost_basis: "list"` claims to be.

> ⚠ The live run of 2026-08-16 (`runs/20260816_152038`) predates this and
> recorded **$0.073572** at the list rate. It actually billed **$0.049048** —
> the recorded figure is 50 % too high. Cost is captured at generation time and
> is not recomputed by re-scoring, so that run's number has to be corrected by
> hand or the run repeated.

### Why OpenAI runs on the Responses API

`OpenAIProvider` calls `client.responses.create`, not `chat.completions`. Two
reasons, both load-bearing:

1. **Chat Completions cannot see what it is billed.** OpenAI charges cache
   writes at 1.25× the uncached input rate, and only the Responses usage object
   reports them (`input_tokens_details.cache_write_tokens`). Chat Completions
   returns `prompt_tokens_details.cached_tokens` alone — a run there would pay
   for writes it cannot report, which is precisely the untracked cost this
   project forbids. The few-shot prompts are well past the caching threshold,
   so this is not hypothetical.
2. **OpenAI documents Responses as the recommended endpoint for reasoning
   models** and states Chat Completions can yield lower model performance.
   Measuring gpt-5.6 through the weaker endpoint would understate it against
   Claude — a validity problem for the comparison, not just a cost one.

Mapping: the transcript goes in as `input` (role/content items, assistant turns
included, which is what few-shot needs), the system prompt as `instructions`,
the cap as `max_output_tokens`, and reasoning depth as
`reasoning={"effort": …}` (`none|low|medium|high|xhigh|max`, default `medium`).
Completion state arrives as `status` + `incomplete_details.reason` and is mapped
onto the same `stop_reason` column the Anthropic path writes, so a truncated
generation reads `max_output_tokens` on both vendors.

Two vendor differences the code normalises, so the columns mean the same thing:

- **`input_tokens` is inclusive on OpenAI, exclusive on Anthropic.** OpenAI's
  figure is the whole prompt including its cached and written parts; Anthropic's
  is the uncached remainder. The provider splits OpenAI's total so
  `input_tokens` is the uncached part on both, and flags the row if the split
  does not reconcile.
- **Reasoning is on by default.** All three gpt-5.6 variants reason at effort
  `medium` unless told otherwise. Since 2026-08-31 the Advanced tier asks for
  `medium` too, so the three OpenAI tiers now differ by model size only, not
  by reasoning depth — the parameter is still sent explicitly there so the
  manifest records the depth instead of inheriting an undocumented default.
  The Standard and Efficient tiers reason without being asked — unlike their
  Claude counterparts, which have thinking off. **That asymmetry is a real
  threat to cross-vendor comparability**: either send
  `reasoning={"effort": "none"}` on those two tiers, or state in the thesis
  that "standard" means something different per vendor. Open decision — and
  since 2026-08-31 DeepSeek's Flash tier reasons too, so no vendor's cheap tier
  is reasoning-free except Anthropic's and Mistral's.

`supports_temperature` is `False` for all three OpenAI models: temperature
support on gpt-5.6 could not be confirmed from the docs, and the pipeline sends
no sampling parameter it has not verified.

### Nothing billed may go untracked

The cost model is checked against the API rather than assumed complete. After
every call the provider walks the vendor's usage object and reports any
**non-zero token- or request-bearing field it does not price** into
`untracked_usage`, printed as a warning during the run:

```
⚠  untracked billable usage: some_future_tier_input_tokens=999
   — cost_usd for this row is a lower bound
```

An empty `untracked_usage` is therefore a positive statement: every quantity the
vendor reported for that call was priced. A vendor adding a cache tier or a
server tool surfaces immediately instead of being silently worth zero. One case
is flagged deliberately: a cache write reported without its TTL split is charged
at the cheaper 5-minute rate and says so, so an unsplit 1-hour write can never be
quietly under-charged.

The second guard is the **service tier**. No provider in this project requests
one, so every call is served at the vendor's default and priced at the list
rate — which is an assumption, not a fact, until the response says so. Each
provider now reads the tier it was actually served by (Anthropic's and
Mistral's `service_tier`, OpenAI's on the response object, Gemini's
`traffic_type`) and reports anything that is not the list-rate tier. Batch stays
priced at 0.5x; priority, flex, scale and provisioned throughput bill at rates
this cost model has no figure for, so `cost_usd` for such a row is flagged as
not being the amount charged rather than quietly presented as if it were.

The third guard is arithmetic. Every vendor that states a grand total —
OpenAI, Mistral, DeepSeek and Gemini; Anthropic states none — has it checked
against the parts on every call. A mismatch means the response carries a token
class the split does not cover, exactly the way Gemini's `thoughts_token_count`
sits outside `candidates_token_count`, and it lands in `untracked_usage` rather
than being absorbed silently. Anthropic, which states no grand total, gets the
same treatment one level down: its `cache_creation_input_tokens` is checked
against the 5m + 1h split, so a third cache tier — the way the 1-hour tier
itself once arrived — cannot slip in unpriced. Where an earlier guard has
already fired on the same response, these checks stay quiet instead of restating
one anomaly twice.

## Prompting strategies

Five strategies are reproduced **verbatim** from Li et al. (2025), *"LLM-based
Business Process Models Generation from Textual Descriptions"* (IJCNLP-AACL,
523–533), two from Klievtsova et al., and three are author-supplied templates
with no published source on file. All live in `prompts.py`, identically to the
Anthropic-only build — the strategy text is vendor-agnostic. The **Quelle**
column records where each template's wording comes from, for the thesis
appendix.

| key                | description                                                        | Quelle |
|--------------------|---------------------------------------------------------------------|--------|
| `zero_shot`        | instruction + description only                                     | LLM-based Business Process Models Generation from Textual Descriptions, (Li et al.) — Sec. 4 (boxed Input/Output template), verbatim |
| `zero_shot_cot`    | zero-shot + "Please think about it step by step."                  | LLM-based Business Process Models Generation from Textual Descriptions, (Li et al.) — Sec. 4 (strategy #2), verbatim |
| `few_shot`         | 2 worked `(description → DOT)` exemplars, then the target          | LLM-based Business Process Models Generation from Textual Descriptions, (Li et al.) — Sec. 4, verbatim; exemplar DOT from the PMo ground truth (`cfg.FEW_SHOT_IDS`) |
| `few_shot_cot`     | few-shot where each exemplar carries the 4-step reasoning chain    | LLM-based Business Process Models Generation from Textual Descriptions, (Li et al.) — Sec. 4 + Appendix A.1 ("Steps of Chain_of_Thought"), verbatim |
| `tree_of_thought`  | single zero-shot turn with a self-contained 4-stage ToT reasoning scaffold — no exemplars needed | author-supplied template — no published source on file (ToT *concept* is external; this wording is not from Li et al.) |
| `role_prompt_1`    | single zero-shot turn opening with an "expert BPMN/DOT modeler" persona plus explicit node-labeling/edge/connectivity instructions — no exemplars needed | author-supplied template — no published source on file |
| `role_prompt_2`    | longer persona variant: same "process modeling expert" framing plus a full BPMN semantics briefing (gateway split/merge rules, event cardinality, error list) and a GraphViz syntax guide with a worked example — no exemplars needed | author-supplied template — no published source on file |
| `zero_shot_graph_type_tn_rules` | no persona — a zero-shot turn that pins the target notation with an explicit node/edge grammar (shapes, empty start/end labels, `seg_`/`meg_`/`spg_`/`mpg_` naming with per-type counters, edge labels for conditions) — no exemplars needed | Conversational Process Modeling: Can Generative AI Empower Domain Experts in Creating and Redesigning Process Models?, (Klievtsova et al.) — kept verbatim incl. original typos, **except** that description and briefing are separated by a line break instead of `. ` (author's instruction, 2026-08-27) |
| `zero_shot__short_bpmn_description` | counterpart to the row above — same frame, but the briefing covers BPMN *semantics* (event cardinality, task nature, gateway split/merge rules) and leaves the GraphViz notation implicit — no exemplars needed | Conversational Process Modeling: Can Generative AI Empower Domain Experts in Creating and Redesigning Process Models?, (Klievtsova et al.) — kept verbatim **except** that description and briefing are separated by a line break instead of `. ` (author's instruction, 2026-08-27) |
| ~~`zero_shot_system`~~ | **deactivated 2026-08-17** — zero-shot + Li et al.'s fine-tuning system prompt. Not selectable and rejected by `--strategies`; the template stays in `prompts.py` | LLM-based Business Process Models Generation from Textual Descriptions, (Li et al.) — Sec. 4 ("Fine-Tuned LLMs" training-example box), verbatim |

> **`zero_shot_system` is off.** It was the only strategy that sent a system
> prompt — every remaining one passes `system=None` — and the only one of Li et
> al.'s five that a run can no longer contain. Re-enable by adding the name back
> to `cfg.ALL_STRATEGIES` **and** `prompts.ALL_STRATEGIES` (two lists, not
> derived from one another). Its user turn is character-identical to
> `zero_shot`, so the pair would have isolated the effect of a system role with
> one variable — worth a sentence in the thesis on why it was dropped.

---

## Install

```bash
pip install -r requirements.txt
```

You don't need every vendor's SDK installed if you only ever run a subset —
but installing all four is simplest. Python 3.10+. The Graphviz `dot` binary
is **not** required (DOT validation uses `pydot`). `numpy`/`scipy` are needed
by the semantic dimension's optimal label matching. **Optional but
recommended**: the WordNet synonym term of semantic label similarity needs
the NLTK corpus once —

```bash
python -m nltk.downloader wordnet
```

— without it that term is 0 and every row records `sem_wordnet = 0` (see
*Semantic quality*). `networkx`, `scikit-learn` and `sentence-transformers`
stay out.

## Quickstart — offline demo (no API keys, no network)

```bash
python run.py --demo
```

Runs **every** vendor's models × the nine default strategies on **one** PMo item
using a **mock provider** — no keys are prompted for, no network calls are made.
The item generated for is `01`, which is also a few-shot exemplar — so the demo
walks the exemplar-swap path: the few-shot strategies show it `03`+`02` instead
of its own model, and the demo stays leakage-free.

## Real run

```bash
python run.py --limit 50
```

No `--dataset-dir` needed — it defaults to `cfg.PMO_DATASET_DIR`. Configure the
needed API keys before a live run (see *API keys* below); nothing is asked for
interactively.
This runs all enabled models across all vendors. To run only a subset of
vendors, or a PMo copy in another location:

```bash
python run.py --vendors anthropic,openai --limit 5
python run.py --dataset-dir /path/to/pmo-dataset --limit 5
```

or a subset of specific models by key:

```bash
python run.py --models anthropic_standard,openai_advanced --csv my_descriptions.csv
```

### API keys

**No API keys are included in this folder.** If `api_keys.py` is missing after
cloning the repository, copy `api_keys.example.py` to `api_keys.py` in this
directory. Before a live run, open **`api_keys.py`** and replace `None` for each
vendor you want to use with your own key in quotes. For example, change
`"anthropic": None` to `"anthropic": "<your Anthropic key>"`. The file has
these five slots:

```python
API_KEYS = {
    "anthropic": None,
    "openai": None,
    "mistral": None,
    "google": None,
    "deepseek": None,
}
```

Alternatively, leave a slot as `None` and set the corresponding environment
variable: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`,
`GOOGLE_API_KEY`, or `DEEPSEEK_API_KEY`. Resolution order per vendor is
`api_keys.py` → environment variable. If neither is set and the vendor is
live, the run aborts **before** any call with a message naming the vendor —
rather than prompting. Only vendors in `cfg.LIVE_VENDORS` need a key at all;
the rest are mocked and never call an API. Keys are held in memory only and
never printed (status output shows the *source*, e.g. `api_keys.py`, not the
value).

### The interactive setup

`python run.py` with no arguments walks five steps — providers, model tiers,
how many processes, strategies, **repetitions** — then shows the run plan and
asks before anything is spent. Input is always the PMo dataset; another
location or a CSV is a scripted run (`--dataset-dir` / `--csv`), which skips
the wizard.

Typing **`b`** at any prompt goes back one step, including from the run plan
itself — so a wrong choice spotted at the end costs one keystroke, not a
restart. Answers already given are kept; going back re-asks only that step.
The tier prompt lists only enabled models; a provider with none is skipped
cleanly instead of aborting the setup.

### Repetitions

Step 5 asks how often each `(process description × model × strategy)`
combination should be generated — **1 to 20**, default 1, the same thing
`--repetitions N` sets on a scripted run. Every repetition sends the identical
prompt with identical parameters; only the model's own non-determinism differs,
which is exactly what makes tokens, cost, latency and quality measurable as a
distribution instead of a single draw. The run plan multiplies its call count
by it, and the confirmation states the factor: **N repetitions cost N times as
much and take N times as long** — that, not a technical limit, is why 20 is the
ceiling.

What changes in the output:

- every row in `results.csv` / `results.jsonl` carries a 1-based `repetition`
  column, and `manifest.json` records `repetitions`;
- artefact names gain an `__r01`, `__r02`, … suffix — but **only when the run
  actually repeats something**, so single-repetition runs keep the exact file
  names every earlier run used;
- the key identifying one generation becomes
  `(item_id, model_key, strategy, repetition)` — `quality/score.py` scores and
  merges on all four (older runs without the column are keyed on the first
  three, as before);
- `aggregate.py` writes **`repetition_spread.csv`**: one row per
  `(item, model, strategy)` with mean/std/min/max over its repetitions for
  tokens, cost, both latencies and the four quality scores, plus
  `n_repetitions`. The summary tables pool items *and* repetitions and would
  otherwise average that spread away. `std` is empty for a cell that ran only
  once (a run stopped mid-sweep leaves some).

The repetition is the **outermost** loop: a full sweep of
`model × strategy × item` finishes before the next repetition starts. Running a
cell's repetitions back to back would measure the same network and API
conditions two or three times over and understate the spread; spreading them
across the run gives each repetition independent conditions. It also means a
run stopped early holds repetition 1 for everything, instead of some cells
repeated in full and others never attempted.

### Pausing or stopping a run from the keyboard

A live run prints its controls at the start and then watches the keyboard:

| key | effect |
|---|---|
| **Enter** | pause — the run stops before the *next* API call |
| **Enter** again | resume |
| **`end`** + Enter | stop now — the call in flight is dropped, not waited for |
| **`stop`** / **`quit`** / **`q`** + Enter | the same as `end`, for whichever word comes to hand |
| **`pause`** / **`resume`** + Enter | the explicit forms of the Enter toggle |
| **`?`** + Enter | print the list again |
| **Ctrl-C** | the same as `end`; press it twice to abort outright |

**Pausing** acts between calls: it takes effect before the next request, so a
call on the wire is never interrupted by it. `end` also releases a paused run,
so you never have to resume just to stop.

**Stopping does not wait.** The request in flight is dropped and the loop ends
at once, instead of sitting out a call that can take minutes on a reasoning
model. The trade-off is real and is recorded rather than hidden: the request was
already sent, so the vendor may still complete and **bill** it, while the run
keeps no reply, no tokens and no cost for it. That call gets a row of its own
with `generation_error` = *abandoned: stop requested while this call was in
flight …*, which is also why a stopped run's cost total is a **lower bound** on
what the account was charged.

**Ctrl-C is folded into the same stop.** The first press ends the run exactly as
`end` does, dropping the call in flight — it still scores, reports and writes its
manifest. The default handler is restored immediately, so a second press aborts
the hard way: a run wedged in a network read stays killable.

A stopped run is a **complete run directory, just shorter** — everything
generated so far is kept, scored and aggregated exactly as normal, and the
manifest records `stopped_early`, `n_generations_planned` and
`n_generations_completed` so a partial run can never be mistaken for the full
cross product it planned. This is the graceful alternative to Ctrl-C, which
loses the scoring and aggregation and leaves a half-written directory behind.

The keyboard half is on for **every** run, mocked ones included — a mock run is
where you try the keys out, and controls that exist only sometimes are controls
nobody trusts. It turns itself off where there is nobody to press a key: a run
whose stdin is not a terminal (piped, redirected, scheduled). Ctrl-C keeps
working there too, and the banner then promises only that.

> ⚠ If you enter keys in `api_keys.py`, they are stored in **plaintext**.
> `.gitignore` excludes the file from
> version control, but a zip of the folder, a backup, a screenshot or a shared
> screen carries it just as well — and anyone holding it can spend money on the
> account. Rotate an exposed key in the corresponding provider's console, and
> keep keys out of the thesis appendix.

## Run from a CSV of process descriptions

An alternative input to the dataset, for descriptions of your own: one row = one
process item, a `description` column is required, an optional `ground_truth`
column (path to a `.dot`/`.gv`) is carried through to the `ground_truth_path`
result column for later scoring, delimiter auto-sniffed. Drop the file in
`data/` and:

```bash
python run.py --provider mock --csv data/my_descriptions.csv   # offline preview
```

Few-shot exemplars still come from the PMo dataset in this mode — the CSV
replaces the *items*, not the examples.

### Useful flags

| flag                         | purpose                                                       |
|------------------------------|-----------------------------------------------------------------|
| `--vendors anthropic,openai` | restrict which vendors' models run (default: all)             |
| `--dataset-dir DIR`          | PMo location (root, or its `descriptions/`/`graphviz/`). Default: `cfg.PMO_DATASET_DIR` |
| `--csv FILE`                 | run on a CSV of descriptions (alternative to `--dataset-dir`) |
| `--models KEY,KEY`           | exact model keys, e.g. `anthropic_advanced,google_efficient` |
| `--strategies zero_shot,...` | subset of strategies                                          |
| `--few-shot-ids 01,02`       | which PMo items are shown as exemplars (still evaluated themselves) |
| `--few-shot-fallback-ids 03,04` | stand-in exemplars for the items that are exemplars themselves |
| `--few-shot-alt-ids 40,41`   | complete alternate exemplar set for `--few-shot-alt-for-ids` (empty string = off) |
| `--few-shot-alt-for-ids 01,02,03` | items that run on that set instead of on a per-slot swap |
| `--hold-out-few-shot-items`  | old behaviour: drop the exemplar items from the run (55 → 53)  |
| `--few-shot-files a.dot,b.dot` | exemplars from outside the dataset; overrides `--few-shot-ids` |
| `--limit N`                  | evaluate only the first N items per model × strategy           |
| `--repetitions N`            | generate each (item × model × strategy) N times, 1–20 (default 1). N× the calls, N× the cost — see *Repetitions* |
| `--temperature`              | sampling temperature (only applied to models with `supports_temperature=True` — Anthropic's reasoning-tier model rejects sampling params outright) |
| `--max-output-tokens`        | output cap                                                     |
| `--tag mylabel`              | label appended to the run-id folder                           |
| `--no-score`                 | skip the quality metrics (score later with `score_run.py`)    |
| `--no-report`                | skip building `results.html` (build it later with `score_run.py`) |
| `--no-csv`                   | skip the `csv/` export (write it later with `csv_export.py`)  |

### Tests

```bash
python -m pytest tests -q      # 265 tests, ~27 s
```

**Every test is offline.** The vendor SDKs are exercised by stubbing the client
and feeding the provider the SDK's own response types, so the code under test is
the code a live run executes — only the transport is replaced. The suite never
calls an API and never spends anything.

| file | covers |
|---|---|
| `tests/test_cost.py` | the cost formula over all billable token classes, promotional vs list rate, per-model cache rates, batch tier, unverified prices → `None` not `0.0`, and the untracked-usage guard |
| `tests/test_timing.py` | what `api_latency_s` means: the answering attempt only, with retries and backoff confined to `latency_s` |
| `tests/test_openai.py` | the Responses request shape (instructions / reasoning / few-shot transcript), the inclusive→exclusive token split, completion state, and the permanent-quota classification |
| `tests/test_report.py` | the HTML report builds from a synthetic run, renders missing metrics as "not computed", stays self-contained, and **leaks no API key** — synthetic keys check both the local-file and environment-variable paths |

Re-score or re-aggregate an existing run at any time — neither costs anything:

```bash
python score_run.py runs/<run_id>    # re-apply the quality metrics, then aggregate
python aggregate.py runs/<run_id>    # tables only
```

---

## Output layout

Identical to the Anthropic-only build, with two added columns: `vendor` and
`price_verified` (false for any row whose model has unconfirmed pricing —
`cost_usd` is `None`/empty for those rows rather than a fabricated number).

```
runs/<timestamp[_tag]>/
├── manifest.json        # resolved config, prompt sources, price sheets, repetitions, stopped_early
├── results.jsonl        # one JSON record per generation, flushed as it happens — the run's only data file
├── raw/        <id>__<model>__<strategy>[__rNN].txt         # verbatim model reply
│            <id>__<model>__<strategy>[__rNN].usage.json  # the vendor's usage object, verbatim
├── generated/  <id>__<model>__<strategy>[__rNN].gv          # extracted DOT
├── prompts/    <id>__<model>__<strategy>[__rNN].json        # the conditions of that one generation
├── results.html         # the report — aggregated per LLM and per strategy
├── request_ids.html     # one row per call; each request id unfolds into its conditions
└── csv/                 # the same results as tables, one file per table
```

### Traceable experimentation (O6)

Every generation writes a record of the conditions it ran under, next to the
reply it produced: the input (text, source file, sha-256), the prompt actually
sent (system + every turn, its sha-256, and the *template version* — a
fingerprint of the wording, which moves when a template is edited and stays put
when the item changes), the model identifier, the generation parameters, the
timestamps, the dated price sheet with the per-class rates the cost was computed
at, and the artefacts it produced. `results.jsonl` carries the handles onto all
of it — `prompt_file`, `prompt_sha256`, `prompt_template_version`,
`prompt_source`, `input_file`, `input_sha256`, `raw_output_file`,
`price_in_per_mtok`, `price_out_per_mtok`, `pricing_as_of` — so a row lifted out
of a CSV still identifies the exact prompt behind it.

`request_ids.html` is where that is read: **every request id opens**. Clicking
one shows the input, the prompt and its version, the model and its parameters,
the timestamps, the reply verbatim, the price list and the measurements derived
from them, in that order. Long texts are shown up to a limit and linked in full;
the exemplar turns of a few-shot prompt are abbreviated harder than the turn
carrying the item, because they repeat in every call of that strategy.

What this makes repeatable is the **experimental conditions**: the same
configuration can be sent again exactly as it was recorded. It is not a promise
of the same output — generation is stochastic, and the same prompt to the same
model at the same settings may answer differently.

A run made before this record existed still opens: the fold-out says the prompt
was not archived, reads the description back from the dataset (labelled as
today's file, not a copy taken at generation time), and fills everything else
from the result row, the reply archive and the manifest.

The `.usage.json` beside each reply is the vendor's own usage object as the SDK
returned it, archived unaltered. The results table carries the classes this
project prices and compares; the archive carries everything the API said —
Gemini's modality breakdowns (`prompt_tokens_details` and its three siblings)
above all, which text-only prompts can act on in no way but which are the one
part of Gemini's accounting no column holds. A run served by the mock provider
writes none: there was no API object, and an empty file would claim otherwise.

### Tokens & cost is one table per vendor

There is no single token table any more, in either view. The vendors do not
report the same quantities, so one grid could hold either every vendor or every
figure — it now holds every figure:

**By LLM.** The comparison table *is* the per-vendor tables: one table per
vendor, **one row per LLM of that vendor**, and every entry the **mean over all
of that model's calls in the run** (`n` beside the name says how many). Nothing
the old pooled table showed is lost — every model of the run has a row in one of
them. Three column groups:

| group | what it is |
|---|---|
| **calls** | `n` (the calls behind the mean) and `usage` (how many of them carried a usage object at all) |
| **as \<vendor\> reported it** | **every token figure that vendor's own API returned**, under its own name — `cache_creation.ephemeral_5m_input_tokens`, `prompt_cache_hit_tokens`, `candidates_token_count`, `output_tokens_details.reasoning_tokens`, … Different for every vendor, with no counterpart in any other table. A field the vendor never filled gets no column — it would be blank in every row |
| **cost** | `$ / call` (mean) and `$ total` for that model |

The classes this project prices are deliberately **not** a second column group
beside them. Checked against a live run, they are for four of the five vendors
the same numbers again under this project's names: only Anthropic's derived
`total_tokens` (it states none) and Gemini's thought-inclusive `output_tokens`
differ, and only once caching or reasoning is actually in play. Cost is still
computed from those classes, and a call's fold-out still breaks its cost down
class by class — that is where the mapping stays visible.

They do come back as columns in one case: a run with **no archived usage
objects at all** (a mock run, or one made before the archive existed) has no
vendor fields to show, so the recorded classes stand in, under a header that
says so. Note also that a vendor's field name can collide with this project's
name for a class without meaning the same thing — OpenAI reports an
`input_tokens` that *includes* its cache reads, where this project's excludes
them — so the two never share a key internally.

**By prompting strategy.** Each strategy gets the same per-vendor tables —
columns = that vendor's token figures, rows = that vendor's models, entries =
the mean over the calls **that strategy** made with that model. There is no
pooled table above them: the tokens block is the per-vendor tables in both
views. Reading one strategy's table against the same vendor's table under
another strategy is what shows what a prompt costs at that vendor; across two
vendors, only the cost columns can be read against each other.

Every row in every one of these tables unfolds into its calls; in the
per-strategy tables the fold-out filters on the strategy as well, so a row means
that model *within* that strategy.

The per-vendor half is **not comparable across vendors, and does not pretend to
be**. The APIs do not describe the same quantities:

| vendor | what only this vendor's usage object says |
|---|---|
| Anthropic | `cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` — cache writes split by TTL, because the two bill at 1.25× and 2× input. States **no grand total** |
| OpenAI | `input_tokens_details.cached_tokens` / `.cache_write_tokens` — cache figures reported *inside* an inclusive `input_tokens`, where Anthropic's `input_tokens` excludes them |
| Google | `thoughts_token_count` outside `candidates_token_count`, the `tool_use_prompt_token_count` breakdown, and per-modality prompt details (`prompt_tokens_details`, keyed here by modality: `prompt_tokens_details.TEXT.token_count`) |
| DeepSeek | `prompt_cache_hit_tokens` **and** `prompt_cache_miss_tokens` — the prompt split from the cache's side, which no other vendor states. It answers in OpenAI's dialect too: `prompt_tokens_details.cached_tokens` repeats the hit count under OpenAI's name. Same tokens, priced once off the pair; the two disagreeing is a note on the row |
| Mistral | `prompt_tokens`, `completion_tokens`, `total_tokens`, and the cached prompt share under any of the **four** names the Chat schema documents: `prompt_tokens_details.cached_tokens`, `prompt_token_details.cached_tokens`, `num_cached_tokens`, `cached_tokens`. The schema also carries `completion_tokens_details.reasoning_tokens`, `prompt_tokens_details.audio_tokens`, `prompt_tokens_details.messages[]`, `request_count` and — the only quantity any vendor here bills in something other than tokens or requests — `prompt_audio_seconds`. All are read; the archive/report preserves every name returned; pricing reconciles the cache aliases and counts the share once. Checked against the live API on 2026-08-29 (plain, reasoning on, reasoning off, and a 15k-token prompt sent twice to force a cache hit): every response carried the first three plus one cached alias and `service_tier`, on the wire as well as through the SDK — the rest are read so the first response to state one is measured rather than missed |

Three reading rules, enforced in the page rather than left to the eye:

- **A field the vendor named and left empty reads "not reported", never 0.**
  `thoughts_token_count: null` means the API measured nothing there — averaging
  it as zero would invent a measurement. Such a field gets no column in the
  table (it would be blank in every row); where it still shows is the fold-out
  of a call, which prints that call's usage object field for field.
- **Only cost is shaded.** Shading marks the best value in a column, relative
  to the other rows of the same table, and cheaper is better. Whether more of a
  vendor-specific token quantity is better or worse is not something this
  project can declare, so those columns carry no direction and stay flat.
- **Nothing in the vendor group is priced.** Cost is broken down class by class
  in each call's fold-out, off the priced classes; pricing a vendor field that
  is already inside another one (Gemini's REST tool-use prompt breakdown,
  OpenAI's cached tokens) would double-count.

Text fields of the usage object (`service_tier`, `traffic_type`,
`inference_geo`) are not token metrics and stay out of the columns; the ones
that matter for cost have their own column and their own guard (see *Nothing
billed may go untracked*). A run made before the archive existed, and every mock
run, has no usage objects at all: the token classes this project records stand
in, under a header that says why, instead of a table of zeros.

**No stored table is ever read back.** Quality is computed by the scoring pass
and handed straight to the report and to the CSV export, and both are rewritten
from that pass every time a run is scored — so nothing on disk can fall out of
step with the metric definitions, which is why there was no `results.csv` in the
first place. `results.jsonl` stays the raw generation log and is not touched by
scoring. The per-call dataset also travels *inside* `results.html` as embedded
JSON, so the aggregates stay re-derivable without the run directory. The page's
three tabs — **Quality**, **Token Usage & Cost**, **Latency** — are one flat row
of buttons, all of them on screen at all times.

### `csv/` — the same results as tables

For anything that reads tables rather than a page (R, pandas, a thesis table),
every run also carries a `csv/` folder, written by `csv_export.py`. It is an
export of the scoring pass, never an input to it: nothing in the pipeline reads
it back, and it is rewritten on every score. It is split on the seams the report
already has, one file per table, and its own `csv/README.md` lists them with row
counts:

| file(s) | one row per |
|---|---|
| `run_info.csv`, `models.csv`, `strategies.csv` | the run's settings and totals; each LLM; each strategy with the source its prompt is cited from |
| `by_model_{quality,tokens_cost,latency}.csv` | LLM, pooled over every strategy — the report's three panels of the **By LLM** view |
| `by_strategy_{quality,tokens_cost,latency}.csv` | strategy, pooled over every LLM |
| `calls.csv` | generation — the spine: what was asked, what came back, one headline figure per dimension |
| `calls_tokens_cost.csv`, `calls_latency.csv` | generation — every token class with the cost it contributed; the timings and attempts |
| `quality_{validity,syntactic,syntactic_bef4llm,semantic,pragmatic}.csv` | generation — one file per dimension, every metric column of the run in exactly one of them |
| `vendor_usage_<vendor>.csv` | generation — the usage object that vendor's own API returned, field for field, under its own name behind `native:` |
| `metrics_legend.csv` | metric column, with the label the report prints for it |
| `f08.csv`–`f20.csv` | thesis Figures 8–20, one compact file per figure; per-call observations for boxplots, pass shares with denominators for binary bars, and model × strategy means for heatmaps |

Every per-call file is keyed on `item_id, model_key, strategy, repetition`, so
any two of them join. Values are written at full precision, and **an empty cell
means "not measured", never 0** — the same rule the page renders as *not
computed*.

The numbered `f08.csv`–`f20.csv` files use the thesis figure numbers. Figures
8, 10–12, and 14–15 carry one measured call per row; cost is in US cents.
Figures 9 and 13 carry each pass share together with `n_assessed` and
`n_passed`. Figures 16–20 include all selected model × strategy cells,
including cells with no measured value, plus the measured sample count. The
`strategy_label` column supplies the readable strategy names used in the
figures; model rows retain both `model_key` and the run's `model_id`.

Re-score and re-render at any time — free, no API calls:

```bash
python score_run.py runs/<run_id>        # re-score against today's metrics, rewrite page + csv/
python results_report.py runs/<run_id>   # rebuild the page only
python csv_export.py runs/<run_id>       # rebuild the csv/ folder only (--all for every run)
```

---

## Quality metrics

Four dimensions, one module each in `quality/` — **all four are implemented**:

| dimension              | module                 | status                    | column prefix |
|------------------------|------------------------|---------------------------|---------------|
| Validity (DOT check)   | `quality/validity.py`  | **format validity, implemented** — Graphviz `nop -p` decides | `val_` |
| Syntactic quality      | `quality/syntactic.py` | **11 dictated checks + the Syntactical Correctness verdict + 4 size metrics + BEF4LLM's published metric set (`syntax_rules.py`)** | `syn_`, `syn_bef_` |
| Semantic quality       | `quality/semantic.py`  | **7 metrics, implemented** (BEF4LLM's published set; needs the ground truth) | `sem_` |
| Pragmatic quality      | `quality/pragmatic.py` | **14 metrics, implemented** (BEF4LLM's published set) | `prag_` |

Each module exposes `evaluate(...) -> dict` of prefixed columns; an empty dict
means the dimension contributes nothing. To add one, write the metrics and
return them — `quality/score.py` merges whatever comes back, and `aggregate.py`
averages any numeric `sem_`/`prag_`/`val_` column it finds without being edited.

### Validity — format validity (DOT check)

    DOT validity = Graphviz accepts the artefact as valid DOT language.

The judge is Graphviz itself, not a re-implementation of its grammar:
**`nop -p`**, which the Graphviz documentation describes as producing "no
output, just checks the input for valid DOT language". Exit code 0 = valid,
exit code 1 + a message on stderr = invalid. Same mechanism as
`echo 'digraph {}' | nop -p`, invoked per generated artefact.

| column            | meaning |
|-------------------|---------|
| `val_dot_valid`   | 1.0 valid / 0.0 invalid / **empty = could not be checked** (Graphviz missing, timeout, file gone) — never a silent zero |
| `val_dot_error`   | Graphviz's own message, e.g. `Error: <stdin>: syntax error in line 3`, or why no verdict was possible |
| `val_dot_checker` | which binary answered, with its version, e.g. `nop -p (graphviz 15.0.0)` |
| `val_score`       | the dimension headline — the mean over its implemented checks. The DOT check is currently the only member, so the two coincide; the name is what `aggregate.py` and the run report read and stays stable if a second check is added |

Scope, stated precisely because the thesis needs it: this measures **DOT
language validity, not process-model correctness**. `digraph { a -> a }` is
perfectly valid DOT and a nonsense BPMN model — that is what the syntactic,
semantic and pragmatic dimensions are for. It also judges the **extracted**
artefact (`generated/<...>.gv`), not the raw reply: markdown fences and prose
around the graph are stripped during generation by
`postprocess.extract_and_validate()`, whose `extract_parse_ok` / `extract_error`
columns cover that step (pydot-level). Measured against raw replies nearly
every model would "fail" for wrapping its answer in ``` fences, which is a
presentation convention, not a format error. The two signals stack:
`extract_parse_ok` says something DOT-shaped could be pulled out of the reply,
`val_dot_valid` says Graphviz accepts what was pulled out.

Two deliberate corrections on top of the bare exit code:

- **An empty document is not valid output.** Verified against Graphviz 15.0.0:
  `nop -p` accepts empty input, whitespace and comment-only files and exits 0,
  so a model that returned nothing would otherwise score 1.0. An artefact that
  declares no graph at all is therefore reported invalid, with `no graph
  declared in the artefact` as the reason, before Graphviz is asked.
- **A reply with no DOT block in it scores 0.0, a call that never returned
  scores nothing.** The first is a format failure and belongs in the metric;
  the second is a missing measurement (`val_dot_valid` empty). `score.py`
  separates them by whether the row carries a `generation_error`.

**Requires Graphviz on PATH** (`nop`, shipped with every Graphviz install —
`dot -V` to check). Point `$GRAPHVIZ_NOP` at the binary if it lives somewhere
unusual; on Windows the standard install directory is checked as a fallback.
Without it the dimension reports empty cells and the scoring pass prints one
`[quality] NOTE:` line saying so — a run never fails and never scores zeroes
over a missing tool.

Ground-truth calibration: the 55 PMo reference models are **55/55 valid**
(`val_score` 1.0). On this input that is a wiring check rather than a finding —
they are Graphviz files by construction — so anything below 1.0 there means
the checker is misconfigured, not that PMo is malformed.

### Syntactic quality — the eleven checks

Model-internal, so no ground truth is needed and every generation is scorable:

| column                        | check                                   | type |
|-------------------------------|-----------------------------------------|------|
| `syn_has_start_event`         | Existence of a start event              | pass/fail |
| `syn_has_end_event`           | Existence of an end event               | pass/fail |
| `syn_single_start_event`      | One start event per process             | pass/fail |
| `syn_single_end_event`        | One end event per process               | pass/fail |
| `syn_start_in0_out1`          | Start event: in = 0, out = 1            | pass/fail |
| `syn_tasks_labeled`           | Each observable task has a label        | pass/fail |
| `syn_task_in1_out1`           | Task: in = 1, out = 1                   | pass/fail |
| `syn_split_in1_outN`          | Split gateway: in = 1, out > 1          | pass/fail |
| `syn_join_inN_out1`           | Join gateway: in > 1, out = 1           | pass/fail |
| `syn_split_has_matching_join` | Split gateway has matching join gateway — **event-based gateways exempt** | pass/fail |
| `syn_sequence_flow_connection_rules` | **Sequence-flow connection rules** | **ratio** |

Every check yields a score in 0.0–1.0. The ten pass/fail checks score 1.0 or
0.0; **Sequence-flow connection rules** is a ratio and scores continuously.

- `syn_score` = **mean of all eleven check scores**
- `syn_checks_passed` = how many checks score a full 1.0

Diagnostic counts (`syn_n_start_events`, `syn_n_malformed_tasks`,
`syn_n_bad_splits`, `syn_n_gateways_degenerate`, `syn_n_gateways_mixed`,
`syn_n_unmatched_splits`,
`syn_n_event_based_splits`,
`syn_n_invalid_sequence_flows`, …) come along so a low score can be explained
without reopening the file.

### Syntactical Correctness — one yes/no beside the scores

`syn_correct` is a **verdict, not a score**: the model either obeys all five
rules below or it does not, and there is no partial credit.

| condition | column | Graphviz equivalent |
|---|---|---|
| Functions (tasks) have **exactly one** incoming and **one** outgoing arc | `syn_correct_functions` | — |
| Gateways have **at least one** incoming and **at least one** outgoing arc | `syn_correct_gateways` | — |
| There is **at least one start node and one end node** | `syn_correct_start_end` | — |
| The graph is **directed** | `syn_correct_directed` | `int agisdirected(Agraph_t *g)` |
| The graph is **coherent** (one connected component) | `syn_correct_connected` | `int isConnected(Agraph_t *g)` |

`no` is the default: all five conditions must hold for `yes`. Empty sets do not pass
vacuously — a model with no tasks, or with no start or end node, is `no` — with
one exception that is deliberate: a purely **sequential** process legitimately
has no gateway, so the gateway condition holds when there are none.

**Only sequence flows count for the two degree rules.** A message flow is not an
arc of the control flow, so an activity that also sends a message is still an
activity with one arc in and one out — degrees come from `seq_in_degree` /
`seq_out_degree` (deduplicated edges minus the dashed ones). This is the one
place the verdict counts differently from check 7 above, which counts every
edge; on a model that draws message flows the two can therefore disagree, and
that is deliberate. `syn_n_malformed_functions` / `syn_n_unwired_gateways` say
how many elements broke the rule.

The **coherence** condition is unaffected by that and still counts every edge:
it is a port of `isConnected()`, which is a statement about the file's graph —
and message flows are exactly what joins two pools, so dropping them would
declare every multi-participant model incoherent.

The last two are graph-level and are answered exactly as the two Graphviz C
functions answer them, in `quality/graph.py::is_directed` / `is_connected`:
**directed** is the `digraph` keyword the file opens with — the flag
`agisdirected()` returns — and without it nothing the other conditions check
means what it says, because there is no arrow direction to have. **Coherent** is
a search over every node of the file with edges followed in *both* directions,
which is what `isConnected()` does; `syn_n_components` reports the count, and a
model in two pieces is two processes, not one. The component count was checked
against Graphviz's own `ccomps` over 200 files — the 45 generated models of a
live run and all 155 PMo dataset models — and agrees on every one; the test
`test_connectivity_matches_graphviz_ccomps` keeps it that way where Graphviz is
installed.

It stays **out of `syn_score` and out of `CHECKS`**: folding a yes/no into a
mean would silently redefine a figure already in use, and the verdict overlaps
checks 1/2/7 by design (condition 1 *is* check 7; conditions 2 and 3 are looser
than checks 8/9 and equal to 1/2).

**It is drawn in the report again since 2026-08-26** (author's instruction).
`results_report.HIDDEN_COLUMNS` — the set that hid it — no longer names it, so
the verdict is back as a column in the Quality table and as the first block of a
call's fold-out.

`HIDDEN_COLUMNS` is now empty. The report shows all ten currently scored
BEF4LLM metrics, including `syn_bef_gateway_in_out_degree` (#15/#16 as this
port merges them, *Gateway in/out degree*).
The separate BEF4LLM metric `syn_bef_split_has_matching_join` (#9) was removed
from the scored set on **2026-09-24** at the user's request; it is no longer
scored or exported. The dictated `syn_split_has_matching_join` check remains.

The gateway-degree metric remains scored and travels per call in
`results.jsonl`, in `csv/quality_syntactic_bef4llm.csv` and in the dataset
embedded in the page. `HIDDEN_COLUMNS` can still hide a metric in the aggregate
table and a single metric row in the fold-out when needed.

**A page shows this as of the day it was written.** `results.html` is rendered
once per run and not touched again, so a run's page keeps whatever the set said
when it was last built; rebuilding it (`python score_run.py runs/<id>`) brings
it in line — and re-scores that run against today's metric definitions, which
for an older run is a change in its numbers, not only in its rows.

What the column looks like: it sits right after *Syntactic*. Per call the
answer is yes or no; a row pools several calls, so it shows **how many of them
said yes**, out of every call behind the row — `5/8`, read like the `valid`
column beside it. A call that produced no parseable model has no verdict and did
not pass, so it stays in the denominator; the cell's tooltip spells that out
(*"5 of 8 call(s) passed; 1 produced no parseable model and has no verdict"*). A
row with no verdict at all reads *not computed*, never `0/8`. The fold-out of a
single call shows the plain yes/no with its five conditions, so a `no` says
which rule broke.

### Sequence-flow connection rules

Adapted BEF4LLM definition. **Every directed DOT edge is one sequence flow.** A
flow is valid when its source is a StartEvent, Task, ExclusiveGateway or
ParallelGateway *and* its target is an EndEvent, Task, ExclusiveGateway or
ParallelGateway — so an EndEvent is never a valid source and a StartEvent is
never a valid target:

| | → Task | → Gateway | → End | → Start |
|---|---|---|---|---|
| **Start →** | ✅ | ✅ | ✅ | – |
| **Task →** | ✅ | ✅ | ✅ | ❌ |
| **Gateway →** | ✅ | ✅ | ✅ | ❌ |
| **End →** | ❌ | ❌ | – | ❌ |

```
score = valid sequence flows / all sequence flows
```

Reported alongside: `syn_n_sequence_flows`, `syn_n_valid_sequence_flows`,
`syn_n_invalid_sequence_flows`, and the two violation types split out as
`syn_n_flows_from_end_event` / `syn_n_flows_into_start_event`.

Three implementation decisions:

- **Edges are deduplicated.** The same flow written twice is one sequence flow,
  consistent with how node degrees are counted everywhere else in the module.
  `syn_n_duplicate_flows` records the difference (0 across all of PMo).
- **Every edge counts, message flows included** — as specified for this check
  ("treat every directed DOT edge as a sequence flow"). Note that **TNSF below
  does the opposite** and excludes message flows, also as specified. The two
  definitions therefore disagree by design; on the reference models the effect
  is small (0.9887 vs 0.9873, 3 items affected), but the divergence is real and
  worth resolving before the numbers go into the thesis.
- **Unclassified endpoints are not violations.** The rule reduces to two
  prohibitions, so an intermediate event or an unknown shape at either end does
  not make a flow invalid — punishing that would measure this scorer's
  classification gap, not the model. `syn_n_flows_unclassified_endpoint` keeps
  them visible.
- **A model with no edges scores 0.0**, rather than dividing by zero or
  crediting an empty model with a perfect ratio.

One limitation follows from event detection: a flow *out of* an end event can
only be detected when that end event is recognisable by **name**. An end event
identified by topology is a sink by definition, and since 2026-08-23 so is one
identified by its `doublecircle` shape — a doublecircle with an outgoing flow is
read as an intermediate event instead (point 3 of "Reading PMo"), which is why
this check no longer fires on the reference set's intermediate events.

The column key follows the `syn_` convention of its neighbours; the exact name
"Sequence-flow connection rules" lives in `syntactic.CHECK_LABELS` and is what
appears in all human-readable output.

### Size metrics (BEF4LLM)

Four size measures, each reported as a **raw value and a normalised score**.
They sit in the syntactic module but are **not** part of `syn_score` — see the
note below.

| raw column | score column | metric | definition |
|---|---|---|---|
| `syn_tnn` | `syn_tnn_score` | **TNN** | total number of nodes = \|FO\|, the flow objects: events + tasks + gateways |
| `syn_tng` | `syn_tng_score` | **TNG** | total number of gateways = \|G\|, any gateway type |
| `syn_tnsf` | `syn_tnsf_score` | **TNSF** | total number of sequence flows = \|F^S\|, message flows excluded |
| `syn_diameter` | `syn_diameter_score` | **Diameter** | max{\|p\|} — longest start-to-end path, counted in flow objects |

`syn_size_score` is the mean of the four scores. **TNMF (message flows) is
deliberately not implemented, and since 2026-08-16 no message-flow count is
reported at all** — see the note under the pragmatic size metrics below.

Scores use the descending normalisation `normdesc` in `quality/normalize.py`,
shared rather than repeated per metric:

```
1.0  if       x < t1        0.5   if t2 <= x < t3        0.0  if t4 <= x
0.75 if t1 <= x < t2        0.25  if t3 <= x < t4
```

| metric | t1 | t2 | t3 | t4 |
|---|---|---|---|---|
| TNN | 29.9 | 43.7 | 58.1 | 81.1 |
| TNG | 1.42 | 3.36 | 5.3 | 6.49 |
| TNSF | 19.4 | 34.8 | 50.2 | 74.8 |
| Diameter | 7.92 | 12.2 | 16.5 | 23.4 |

A value landing exactly on a threshold falls into the lower-scoring band, as
specified. An unmeasurable value stays `None` rather than becoming a `0.0` that
would read like a measured worst case.

**Counting is semantic, never presentational.** The element types come from
`graph.py`'s classification, the only semantic layer DOT offers. Pool and lane
anchors — PMo writes `"Pool_1" [shape=point, style=invis]` inside a
`cluster_Pool_1` subgraph — are classified as `markers` and excluded from
\|FO\|. Message flows (`[style=dashed, arrowhead=open]`) are excluded from
\|F^S\|. No layout data takes part in any count: no bounds, positions or
DI edges are read, and `shape`/`style` are consulted only as the element-type
carriers the DOT notation provides.

#### Diameter and cycles

A path is a non-empty sequence of flow objects connected by sequence flows from
a start event to an end event; `|p|` is the number of flow objects on it.

The project had **no existing path or cycle semantics** to reuse
(`reachable_from` is plain reachability, not a path definition), so — as the
formula leaves cycle handling open — `diameter()` counts **simple paths only**:
no node is revisited within one path. Without that rule a loop would make the
longest path unbounded. Longest-simple-path is NP-hard, so the search is also
capped at 200 000 expansions; if the cap is hit, the best length found so far is
returned and `syn_diameter_truncated` is set, rather than the run hanging on a
pathological model. Across all 55 reference models the cap is never reached.

A model with no start event, or none that reaches an end, scores a diameter
of 0.

#### Why these are not in `syn_score`

The eleven checks measure **correctness**; these four measure **size**, and
`normdesc` rewards smallness. Folding them into `syn_score` would make a large
but perfectly well-formed model look syntactically wrong, and would silently
redefine a number already in use. They carry their own aggregate,
`syn_size_score`. Add the four names to `syntactic.CHECKS` if that trade-off is
ever decided the other way.

Decisions worth knowing, all reversible in `quality/syntactic.py`:

- **Checks 1/3 and 2/4 overlap on purpose.** A model with two start events
  passes *existence* and fails *exactly one*, which separates "missing" from
  "too many" in the score.
- **`start_in0_out1` fails when there is no start event** (it cannot hold
  vacuously), and with several start events *all* must satisfy it.
- **`tasks_labeled` and `task_in1_out1` fail when the model has no tasks at
  all**, rather than passing over an empty set. The two gateway-degree checks
  do the opposite and pass vacuously — a purely sequential process with no
  gateways is legitimate, a process with no activities is not.
- **A gateway with in == out fails both degree checks.** It neither splits nor
  merges: a 1→1 pass-through or a 2→2 mixed split/join. It satisfies neither
  rule, so it counts against both, and `syn_n_gateways_degenerate` keeps that
  visible rather than letting it be double-counted invisibly.
- **"Matching join" is read as: a join gateway of the same type is reachable
  downstream of the split.** Type comes from the label/name (`X` vs `+`/`AND`,
  `seg_`/`spg_`, `XOR`/`OR`); when either side's type cannot be read, any join
  counts — an unreadable type is this scorer's gap, not the model's mistake.
  This is deliberately weaker than full block-structuredness: it does not
  require *every* branch to reconverge at *one* join.
- **A task's label** is its `label` attribute, or its node id when it has none
  (that is what DOT renders). A label that exists but names nothing — `Task_1`
  — is counted in `syn_n_placeholder_labels` but does *not* fail the check.
- **An unparseable generation gets `None`, not 0**, for every check. Producing
  no model at all is a *validity* failure, and averaging it in as a zero would
  blend two different failure modes. It is scored as one: `val_dot_valid` is
  0.0 for exactly these rows, with Graphviz's own error message next to it
  (see *Validity*), and `extract_parse_ok` carries the extraction-failure rate.
- The specified list has **no `end_in1_out0` counterpart** to check 5. Left as
  specified; add it to `CHECKS` if that asymmetry is not intended.

### Telling a split gateway from a join gateway

Checks 8–10 need this, and getting it wrong makes them meaningless: classifying
a gateway by its degree and then checking that degree would make both rules
**tautological** — every gateway would satisfy the rule it was sorted into.
`quality/graph.py::_gateway_role` therefore reads *declared intent* first:

1. name or label states the role — `SPLIT`/`FORK`, `JOIN`/`MERGE`/`SYNC`, or the
   `seg_`/`spg_` and `meg_`/`mpg_` convention from
   `zero_shot_graph_type_tn_rules`;
2. otherwise topology, as the two published metrics read GS and GJ: a gateway
   that **fans out** is a split, one that **fans in** is a join, and one that
   does both is in **both** sets;
3. neither → the 1→1 pass-through, which routes nothing: counted as degenerate
   (see above).

So a gateway explicitly named `AND_SPLIT_1` but wired with two incoming flows
does fail check 8, which is the point.

> **Gateways that fan in and out.** PMo draws seven (items 02, 07, 24, 29) and
> no 1→1 pass-through at all. Each is in GS **and** in GJ, so #15 asks whether
> exactly one flow enters it and #16 whether exactly one leaves it, and it
> fails both — which is what the two metrics are for. Reading such a gateway as
> "neither" put it outside both and let items 02 and 24 score 1.000 on each
> while every other reading fails them; that was the state until 2026-08-24.
>
> \|GS\| is therefore **248** with 241 conforming and \|GJ\| **223** with 216, the
> seven gateways being the whole gap on both sides, in items 02, 07, 24 and 29.
> They also enter \|GS\| for metric #9, whose denominator is 246 rather than
> 241. `syn_n_gateways_mixed` counts them, and is what lets such a gateway be
> the **partner of a split** in check 10 and metric #9 — it does merge.
>
> Their own code has **one** rule instead of these two: it walks every gateway
> and counts `(in > 1 ∧ out > 1) ∨ (in ≤ 1 ∧ out ≤ 1)` as the mistake, never
> asking whether the gateway splits or joins. The workbook shows that single
> number beside both metrics, marked as the one combined rule it is.

### Identifying start and end events

The crux of these checks, because no single signal is reliable across the
corpus: PMo's ground truth draws the end as `shape=circle, penwidth=4`, while
the prompt templates variously ask for `doublecircle`, or for `circle` with the
label "end". Scoring by one convention would measure *which template's notation
was encoded here*, not model quality. `quality/graph.py` therefore decides in
layers: `doublecircle` → **end if the node is a sink, intermediate otherwise**
(`syn_n_intermediate_events`); else a name/label keyword (`start…`, `end…`,
`stop…`); else topology (a source is a start, a sink is an end); else the event
stays **unclassified** rather than guessed at (`syn_n_events_unclassified`).
The sink condition is what keeps PMo's intermediate catching events out of the
end events without breaking the prompt templates' `doublecircle` = end
convention — see point 3 of "Reading PMo" above.

The topology step counts **sequence flows only**. Where a process begins is a
control-flow question and a message flow is not control flow: PMo's
`"scoring request received_1"` (item 23) is a *message start event* — no
incoming sequence flow, one outgoing, and an incoming message flow from another
pool. Counting every edge made it look like a node in the middle of a flow and
left it unclassified; four such events across items 23 and 24 were affected, and
with them the diameter and the connectedness of everything downstream. Fixing
this moved four figures in the tables above (start-event existence 54 → 55,
one-start-event 54 → 53, start degree 54 → 53, diameter min 0 → 4): items 23 and
24 legitimately have one start event **per pool**.

Check 3 counts **per process** since 2026-08-24, exactly as BEF4LLM's
`one_start_event_per_process` does — both read `syntax_rules.processes`, so the
two scorings cannot disagree on what a process is. It is 1 when every process
holds exactly one start event and 0 otherwise, which takes it from 53/55 to
**55/55**: items 23 and 24 have one start event in each of their two and three
processes and were failing the check for being correct. **Check 4 reads the same way** since the same day: 54/55, the one violation
being item 22 with four end events in one process. Note that the BEF4LLM port's
#4 is *not* per process — it divides by the end events, as the reference
implementation does (see the metric table above), so it cannot notice a process
with no end event; the dictated check can and does.

### Calibration against the ground truth

The PMo reference models themselves score **mean 0.944**, with 38 of 55 at
11/11. Per check, out of 55 items:

| check | full pass | mean |
|---|---|---|
| Existence of a start event | 55 | 1.000 |
| Existence of an end event | 55 | 1.000 |
| One start event per process | 55 | 1.000 |
| One end event per process | 54 | 0.982 |
| Start event: in = 0, out = 1 | 53 | 0.964 |
| Each observable task has a label | 55 | 1.000 |
| Task: in = 1, out = 1 | **43** | 0.782 |
| Split gateway: in = 1, out > 1 | 51 | 0.927 |
| Join gateway: in > 1, out = 1 | 51 | 0.927 |
| Split gateway has matching join gateway | 44 | 0.800 |
| Sequence-flow connection rules | 53 | 0.997 |

These are properties of the dataset, not scorer artefacts — spot-checked:
several items genuinely carry multiple end events (PMo's own README allows "one
or more"), and 35 tasks across 12 items sit on multiple in/out flows without an
intervening gateway (implicit merges, message flows in the multi-pool
Camunda-derived items 22–24).

The two end-event figures were **not** properties of the dataset until
2026-08-23: reading every `doublecircle` as an end event turned 17 intermediate
events into end events and cost "One end event per process" five models and the
sequence-flow rules six. The numbers above are the corrected ones — see point 3
of "Reading PMo".

Size metrics over the same 55 reference models — useful as the scale the
BEF4LLM thresholds are being applied to:

| metric | raw min | raw max | raw mean | score mean |
|---|---|---|---|---|
| TNN | 5 | 50 | 23.2 | 0.936 |
| TNG | 0 | 22 | 8.4 | **0.205** |
| TNSF | 4 | 60 | 27.3 | 0.741 |
| Diameter | 4 | 18 | 9.0 | 0.777 |
| `syn_size_score` | | | | **0.559** |

The TNG threshold band is the striking one: `t4 = 6.49`, while the reference
models average 8.4 gateways, so most of them score 0.0 on it. The published
thresholds are calibrated for smaller models than PMo contains — worth stating
explicitly before reporting size scores as a quality signal.

> ⚠ **Corrected 2026-08-23 — this figure used to be a notation artefact.**
> The sequence-flow score read 0.986 with 8 items violating, all of them the
> same way: a `doublecircle` node with an outgoing flow. In the prompt templates
> `doublecircle` *means* end event, and that is what generated models produce —
> but the Camunda-derived PMo items use it for **intermediate catching events**
> (item 24: `"Guest appears" -> "Hand over meal"`), reserving the thick single
> circle (`penwidth=4`) for the actual end. Those 17 flows were never broken.
> Since the shape decides "end" only for a sink, they are read as intermediate
> events and the figure is 0.997 over 53 clean items; the two remaining
> violations are in item 22 and are real. Any figure quoted from a run before
> that date carries the artefact.

**1.0 is therefore not the ceiling a generated model should be measured
against** — report this baseline alongside model scores, or a model that
reproduces its reference perfectly will still look imperfect. Reproduce with:

```bash
python -c "from quality import graph,syntactic; import glob,statistics; \
print(statistics.mean(syntactic.evaluate(graph.load(p))['syn_score'] \
for p in glob.glob(r'<PMO_DATASET_DIR>/graphviz/*.dot')))"
```

## Syntactic quality — BEF4LLM's published metric set

The eleven checks above were dictated rule by rule and are **pass/fail per
model**. `quality/syntax_rules.py` is the other thing: the syntactic dimension
of BEF4LLM **as the paper defines it** — Table 2, with the formulas in
Table A.15 — ported to DOT. Columns are prefixed `syn_bef_`, the headline is
`syn_bef_score`, and both scorings sit in the same `results.csv` row.

### ⚠ The paper and the supplied code define different metric sets

This is worth knowing before citing anything. `bef4llm/synactic_quality/synactic_quality_check.py`
does **not** implement the set in Table 2 / A.15:

| | paper (Table 2 / A.15) | supplied code |
|---|---|---|
| gateway degrees | **two** metrics: split `in=1 ∧ out>1` over \|GS\|, join `in>1 ∧ out=1` over \|GJ\| | **one** combined rule: a gateway is erroneous when `(in>1 ∧ out>1) ∨ (in≤1 ∧ out≤1)` |
| | *this port*: one combined metric since 2026-08-24; now uses the original predicate over all gateways, including its acceptance of `in=0, out>1` and `in>1, out=0` | |
| exception events | metric #14, `in=0 ∧ out=1` over \|dom(Excp)\| | folded into the intermediate-event rule |
| start/end degree | **ratios** over \|ES\| / \|EE\|, `out = 1` / `in = 1` exactly | whole-model 0/1 measures reading `out ≥ 1` / `in ≥ 1` |
| one start event per process | divides by the number of **processes** — the only form that notices a process with *no* start event | divides by the number of **start events** |
| split has matching join | over \|GS\| | over *all* gateways, plus unmatched joins as extra errors |
| | *this port*: ports the supplied code's gateway-order scan and single-path walk; event-based splits are included and may pair with exclusive joins. DOT type inference and edge order can differ from BPMN (see the 55-model comparison below). | |
| — | not present | adds `connected_nodes` and `event_gateway_predecessor_successor` |

The paper's own result tables (C.21/C.22) report the **code's** set, so the
published Qsyn figures — e.g. falcon3:10b at 0.9082 — are not comparable with
this module. The definition tables are what a thesis cites, so those are what
this implements.

### The ten scored metrics

Ten of the paper's sixteen, #15 and #16 counted as one, #5 present in an
adapted form, and #9 removed from this project's score on 2026-09-24 at the
user's request. Metric **6 (message-flow connection rules)** is **excluded**: its
formula quantifies over BPMN element types Graphviz cannot express — it needs
message start/intermediate/end events (`E_MS`, `E_MI`, `E_ME`), i.e. event
*definitions*, which have no DOT notation at all. It is neither computed nor
scored.

Metric **5 (sequence-flow connection rules)** was excluded for the same reason
until 2026-08-24 and is now scored in the part of it the notation does carry
(author's instruction): **no sequence flow into a start event and none out of an
end event**, 1 or 0 — `adapted_sequence_flow_connection_rules`, labelled
*Adapted Sequence-flow connection rules*. Counted over sequence flows only, so a
message start event receiving from another pool does not fail it; the dictated
check of the same name scores the same two prohibitions as a *ratio*, and counts
every edge.

Metrics **9**, **10**, **13** and **14** are not scored (#10/#13/#14 were
removed at the author's instruction on 2026-08-21; #9 was removed at the user's
request on 2026-09-24) and are struck through below. This is a deliberate
departure from the published set and must be stated wherever Qsyn is reported:
the divisor is **10, not 14** — four metrics removed, #15/#16 merged into one
and #5 back in adapted form — so a figure from this module is not
comparable with one over the paper's set, including figures this project
produced before those dates. On a 45-model live run the mean `syn_bef_score`
moved from **0.9014 to 0.8797**; #13 and #14 were near-vacuous in DOT (they
measured something on one model and on none), while #10 scored 0.944 on average,
so real signal left with it.

| # | column suffix | metric (the paper's wording) | formula |
|---|---|---|---|
| 1 | `existence_start_event` | Existence of a start event | `∃e ∈ ES` — Boolean |
| 2 | `existence_end_event` | Existence of an end event | `∃e ∈ EE` — Boolean |
| 3 | `one_start_event_per_process` | One start event per process | `\|{p ∈ P : \|ES_p\| = 1}\| / \|P\|` |
| 4 | `one_end_event_per_process` | One end event per process | `\|{p ∈ P : \|EE_p\| ≥ 1}\| / \|EE\|` — the reference implementation's count, **adopted 2026-08-24** |
| 5 | `adapted_sequence_flow_connection_rules` | **Adapted** Sequence-flow connection rules | no flow into a start event ∧ none out of an end event — Boolean, **added 2026-08-24** |
| 7 | `start_event_in_out_degree` | Start event: in = 0, out = 1 | `\|{e ∈ ES : in=0 ∧ out=1}\| / \|ES\|` |
| 8 | `end_event_in_out_degree` | End event: in = 1, out = 0 | `\|{e ∈ EE : in≥1 ∧ out=0}\| / \|EE\|` — `≥ 1` follows the reference implementation, **2026-08-24** |
| ~~9~~ | ~~`split_has_matching_join`~~ | ~~Split gateway has matching join gateway~~ | **removed from this pipeline's score on 2026-09-24**; the separate dictated `syn_split_has_matching_join` check remains |
| ~~10~~ | ~~`one_process_per_pool`~~ | ~~Exactly one process per pool~~ | **removed 2026-08-21** |
| 11 | `labeled_tasks` | Each observable task has a label | `\|{t ∈ T : label(t) ≠ ε}\| / \|T\|` |
| 12 | `task_in_out_degree` | Task: in = 1, out = 1 | `\|{t ∈ T : in=1 ∧ out=1}\| / \|T\|` |
| ~~13~~ | ~~`intermediate_event_in_out_degree`~~ | ~~Non-exception intermediate event: in = 1, out = 1~~ | **removed 2026-08-21** |
| ~~14~~ | ~~`exception_event_in_out_degree`~~ | ~~Exception event: in = 0, out = 1~~ | **removed 2026-08-21** |
| 15+16 | `gateway_in_out_degree` | Gateway in/out degree | `1 − mistakes / \|G\|`, where a mistake is `(in>1 ∧ out>1) ∨ (in≤1 ∧ out≤1)` — **merged 2026-08-24; original implementation adopted 2026-09-24**. Scored, exported and shown in `results.html` |

For the seven counting metrics the two counts behind the score are reported as
well (`syn_bef_<metric>_conforming`, `_covered`). They are there because a
**mean of per-model ratios is not the pooled element-level rate**: averaging
`syn_bef_task_in_out_degree` over a run weights a two-task model like a
forty-task one, while `Σ conforming / Σ covered` answers "what share of all
tasks was well-formed". Both are legitimate; the columns let the thesis pick
one knowingly.

### The scoring principles, all from the paper

1. **Boolean vs. counting metrics** (§4.1). Three metrics are Boolean and score 0
   or 1 — existence of a start event, an end event, and the adapted sequence-flow
   rule. The other seven are
   counting metrics: conforming elements over the elements the rule covers. The
   paper's own example: "a BPMN model with 8 labeled activities out of 10 would
   evaluate to 0.8".
2. **Arithmetic mean, equal weights** (§4.5, Eq. 4): "we sum the individual
   metric scores and divide by the number of metrics", with no weighting,
   because there is no empirical evidence that any metric should count more.

   ```
   syn_bef_score = Σ_m score(m) / 10
   ```
3. **A metric with nothing to measure scores 1.0.** The paper leaves the
   empty-denominator case open; 1.0 (rather than 0.0, which would read as a
   measured failure) matches the reference implementation and keeps the divisor
   at the full metric count. `syn_bef_metrics_applicable` records how many
   metrics the model actually triggered.
4. **`in(x)` and `out(x)` are sequence flows**, stated in the paper's notation:
   "in(x), out(x) incoming/outgoing **sequence flows** of x". A task that
   receives a message still has in-degree 1. The dictated checks count every
   edge, which is why the two scorings disagree on exactly the multi-pool items.

### Processes and pools

Three metrics (#3, #4, #10) need to know what a *process* and a *pool* are.

**P, the processes** — the **weakly-connected components of the sequence-flow
graph**, not the DOT clusters. BPMN forbids a sequence flow from crossing a
pool boundary, so a connected control-flow region *is* a process; and lanes,
which flows cross freely, do not split one. Reading the clusters instead would
mistake a lane for a pool — a generated model drawing `cluster_marketing` and
`cluster_tracking` as two lanes of one process scored 1/3 on "one start event
per process" for a notation choice rather than a modelling error. On the 55 PMo
reference models the two readings agree exactly (items 23 and 24 → 2 and 3
processes); across the generated corpus they differ on the lane-as-cluster
models. A collapsed black-box pool contributes no flow objects and therefore no
process, which is correct: it has no process inside.

**PO, the pools** — every `subgraph cluster_*`. `quality/graph.py` records, per
node, the **outermost** enclosing cluster as its pool and the **innermost** as
its lane (PMo item 21 nests `cluster_Lane_1..3` inside `cluster_Pool_1`; items
23/24/38 use three flat pools; the other 49 items have none).

`syn_bef_n_processes`, `syn_bef_n_pools`.

### One metric that cannot fail in DOT

**#14 (exception event)** is part of the published set and is kept so the
divisor stayed the full metric count, but it could never be triggered:
`dom(Excp)` holds the interrupting events — BPMN boundary events, attached to
an activity's border — and DOT has no notation for attachment, so the set is
always empty and the metric scored 1.0 by principle 3, contributing a constant
1/14 ≈ 0.071 to every model's score. **Resolved on 2026-08-21: removed**, along
with #13 and #10.

**#10 (exactly one process per pool)** *was* measurable under the component
reading: a pool violated it when the flow objects it drew belonged to two or
more disconnected control-flow regions — two processes in one pool, exactly what
the metric forbids. (A pool with no flow objects conformed: it holds no process,
not two.) In the paper's own results this metric is 1.0000 for every LLM; on PMo
it was 61/61. It was removed on 2026-08-21 together with #13 and #14, so none of
the three is computed any more; the PMo table below is the last measurement of
them and is kept as the record of what left.

### Two extra columns, deliberately outside the score

`syn_extra_connected_nodes` and `syn_extra_event_gateway_predecessor_successor`
are the two rules the reference *implementation* measures and the paper's
definition tables do not contain. They are reported because they are
informative — connectedness in particular catches genuinely broken models — but
they are **not** part of `syn_bef_score`, hence the different prefix. The
event-gateway one additionally drops the `messageEventDefinition` /
`timerEventDefinition` test on the successors, which DOT cannot express: a
successor only has to *be* an event, or a receiving task has to precede.

**Historical implementation of metric #9 (removed 2026-09-24).** The original
`pair_gateways` walks gateways in document order, counts every gateway in the
divisor, stops early on `len(pairs) * 2 == len(gateways)`, adds one mistake for
an unpaired split and another for the first unclaimed join, and follows one
arbitrary path (first successor, then first not-yet-stepped-on successor).
`syn_bef_split_has_matching_join` once reproduced this and agreed with the
reference column on 45/55 models. It is no longer part of this pipeline's
scored metrics or current calibration.

> The 45/55 comparison also had different gateway totals in some DOT and BPMN
> inputs. Because the original walk is order-sensitive, those differences
> combined representation and traversal effects. This historical comparison
> should not be read as a current pipeline metric result.
>
> What this metric therefore does **not** tell you is whether a split has a join:
> PMo's item 05 scores 0.5 although both of its splits are paired, because the
> walk leaves `ExclusiveGateway_2` through `Task_8` and never comes back to the
> loop merge `ExclusiveGateway_3`. For that question use the dictated check
> `syn_split_has_matching_join`, which is partner-aware (`graph.matching_joins`:
> the first point the branches meet again, or the gateway a loop's back edge
> runs into) and order-independent.

> Until 2026-08-24 this was mere reachability ("a join of the same type is
> reachable downstream"), which two reference models exposed:
>
> * item 30's `"How many people can play on that date?"` counted
>   `ExclusiveGateway_2` as its partner — six steps away in the beer branch,
>   both of whose inflows come from the beer decision — while its own branches
>   reconverge at the task `"Appoint a date"`, an implicit merge with no
>   gateway;
> * item 46's `"Is in stock?"` reached `ExclusiveGateway_2` **after** its
>   branches had already merged at the task
>   `"Check if the whole order is ready for shipment"`, through the outer loop,
>   and `ExclusiveGateway_1` through the same loop from behind.
>
> Hence both refinements: the reconvergence walk stops at the first meeting
> point, and the loop case stops at any further decision. Pooled 223/241 →
> 218/241, dictated check 50/55 → 44/55. A stricter variant without the loop
> case was measured too and rejected: it fails every loop split (41 of 241),
> whose merge legitimately sits in front of it.

The separate dictated check `syn_split_has_matching_join` exempts event-based
splits (author's instruction, 2026-08-23) and reports skipped splits in
`syn_n_event_based_splits`. BEF4LLM metric #9 was removed from the scored set on
2026-09-24, so its original implementation is not part of the current Qsyn.
Metrics #15/#16 check the degrees of every gateway, event-based ones included.

### Calibration against the ground truth

The current `syn_bef_score` calibration uses the ten-metric set after metric
#9 was removed on 2026-09-24: mean **0.9933**, with **42 of 55** models at 1.0.
These figures were refreshed by `score_pmo_dataset.py` on 2026-09-24.

| # | metric | perfect (of 55) | mean | pooled conforming / covered |
|---|---|---|---|---|
| 1 | Existence of a start event | 55 | 1.000 | Boolean |
| 2 | Existence of an end event | 55 | 1.000 | Boolean |
| 3 | One start event per process | 55 | 1.000 | 58 / 58 |
| 4 | One end event per process | 54 | 0.986 | 60 / 61 |
| 5 | Adapted Sequence-flow connection rules | 55 | 1.000 | Boolean |
| 7 | Start event: in = 0, out = 1 | 55 | 1.000 | 58 / 58 |
| 8 | End event: in = 1, out = 0 | 55 | 1.000 | 61 / 61 |
| ~~9~~ | ~~Split gateway has matching join gateway~~ | — | — | removed 2026-09-24 |
| ~~10~~ | ~~Exactly one process per pool~~ | — | — | removed 2026-08-21 |
| 11 | Each observable task has a label | 55 | 1.000 | 676 / 676 |
| 12 | Task: in = 1, out = 1 | 46 | 0.974 | **659 / 676** |
| ~~13~~ | ~~Non-exception intermediate event: in = 1, out = 1~~ | — | — | removed 2026-08-21 |
| ~~14~~ | ~~Exception event: in = 0, out = 1~~ | — | — | removed 2026-08-21 |
| 15+16 | Gateway in/out degree | 51 | 0.972 | 457 / 464 |

After adopting the original predicate, `score_pmo_dataset.py` was rerun on
2026-09-24: the DOT port and BEF4LLM's own BPMN implementation agree on all
55/55 models for gateway degree. The mean remains 0.9722.

Outside the score: `connected_nodes` 1276/1276, `event_gateway_predecessor_successor` 1/1.

The removed metric's old pooled counts and comparisons are historical only;
they are excluded from the current score and calibration.

Reproduce with:

```bash
python -c "from quality import graph,syntactic; import glob,statistics; \
print(statistics.mean(syntactic.evaluate(graph.load(p))['syn_bef_score'] \
for p in glob.glob(r'<PMO_DATASET_DIR>/graphviz/*.dot')))"
```

Three things this table says that the dictated eleven do not:

- **The task-degree figure is a 2.5 % element error rate, not a 22 % model
  failure rate.** 17 of 676 tasks sit on the wrong number of flows; the dictated
  `task_in1_out1` reports 43/55 models failing, which is the same models but
  reads as a far larger problem. Part of the gap is principle 4: counting
  sequence flows only stops a message flow from making a task look malformed.
- **Gateway degree uses the original predicate** since 2026-09-24: a gateway
  conforms unless both degrees are greater than 1 or both are at most 1. The
  original rule counts every gateway in its denominator and computes
  `1 − mistakes / gateways`; the DOT port now returns the equivalent
  `conforming / covered` ratio. Re-scoring the PMo set gave exact per-model
  agreement with the original on all 55 items.
- **Metric #8 no longer has a gap**: 61 of 61 end events since it reads `in ≥ 1`
  like the reference implementation (2026-08-24). The two it used to fail are
  items 35 and 42, whose single end event is reached by two flows without a
  gateway between them — an implicit merge, which the printed `in = 1` counts as
  a violation. Until 2026-08-23 this bullet read "19 of 78": 17 of those were
  the `doublecircle` notation clash, intermediate catching events counted as end
  events.

### Which of the two syntactic scorings goes in the thesis?

Nine of the eleven dictated checks map onto a currently scored published metric:

| dictated check | paper metric |
|---|---|
| `syn_has_start_event` | #1 Existence of a start event |
| `syn_has_end_event` | #2 Existence of an end event |
| `syn_single_start_event` | #3 One start event per process |
| `syn_single_end_event` | #4 One end event per process |
| `syn_start_in0_out1` | #7 Start event: in = 0, out = 1 |
| `syn_tasks_labeled` | #11 Each observable task has a label |
| `syn_task_in1_out1` | #12 Task: in = 1, out = 1 |
| `syn_split_in1_outN` | #15+16 Gateway in/out degree (one metric there, two checks here) |
| `syn_join_inN_out1` | #15+16 Gateway in/out degree (as above) |
| `syn_split_has_matching_join` | no BEF4LLM metric in this pipeline; #9 was removed 2026-09-24 |
| `syn_sequence_flow_connection_rules` | #5 — the ratio here, **Adapted Sequence-flow connection rules** (Boolean) there |

The published set adds four the dictated checks have no counterpart for: #8
(end-event degree — the missing mirror of check 5, noted as an open question
above), #10, #13 and #14. It is also per-process and element-weighted where the
dictated checks are neither.

Reporting both is fine as long as the difference is stated. Reporting one:
`syn_bef_score` is the defensible **reuse** claim (DC4) and the more
informative number; `syn_score` is the specification that was dictated. This is
an open decision — see below.

## Pragmatic quality — BEF4LLM's published metric set

*Can a human read and use the model?* Comprehensibility rather than
correctness. `quality/pragmatic.py` implements the pragmatic dimension **as the
paper defines it** — §4.2, the formulas in Table A.16 and the thresholds in
Table A.18 — on DOT. Where the paper and the supplied code
(`bef4llm/pragmatic_quality/`) disagree, the paper wins, and each case is
tabulated below.

All fourteen metrics are **model-internal**: no ground truth is consulted, so
every generation is scorable, including CSV generation-only runs. The
underlying definitions are Mendling (2008), *Metrics for Process Models*,
except cross-connectivity (Vanderfeesten et al., 2008, ref [53] in the paper).

### The fourteen metrics

| # | group | column | metric | formula (Table A.16) | direction |
|---|---|---|---|---|---|
| 1 | size | `prag_tnn` | **TNN** | \|FO\| | less |
| 2 | size | `prag_tng` | **TNG** | \|G\| | less |
| 3 | size | `prag_tnsf` | **TNSF** | \|F^S\| | less |
| 5 | size | `prag_diameter_nogw` | **Diameter** | max{\|p\| : p ∈ Paths}, gateways not counted — **the scored reading since 2026-08-29**; `prag_diameter` (BEF4LLM's traversal) and `syn_diameter` (every flow object) ride along unscored | less |
| 6 | density | `prag_density` | **Density** | \|F^S\| / (\|FO\|·(\|FO\|−1)) | less |
| 7 | density | `prag_agd` | **AGD** | Σ_{g∈G}(\|in(g)\|+\|out(g)\|) / \|G\| | less |
| 8 | density | `prag_cnc` | **CNC** | \|F^S\| / \|FO\| | less |
| 9 | connector interplay | `prag_gh` | **GH** | −Σ_{l∈{AND,XOR,OR}} p(l)·log₃ p(l) | less |
| 10 | connector interplay | `prag_cfc` | **CFC** | Σ_{G^S_AND} 1 + Σ_{G^S_XOR} \|out(g)\| + Σ_{G^S_OR} (2^\|out(g)\|−1) | less |
| 11 | connector interplay | `prag_cc` | **CC** | mean over ordered node pairs of the strongest path's value | **more** |
| 12 | partitionability | `prag_sequentiality` | **Sequentiality** | arcs between two non-connector nodes / \|arcs\|, message flows included on both sides — BEF4LLM's reading, see below | **more** |
| 13 | partitionability | `prag_separability` | **Separability** | \|{fo ∈ FO : fo is a cut vertex}\| / (\|FO\|−2) | **more** |
| 14 | partitionability | `prag_depth` | **Depth** | max{depth(fo) : fo ∈ FO} | less |
| 15 | concurrency | `prag_token_split` | **TS** | Σ_{g ∈ (G^S_AND ∪ G^S_OR)} (\|out(g)\|−1) | less |

The paper's notation fixes two things the DOT port has to respect: `in(x)` and
`out(x)` are **sequence flows only** ("in(x), out(x) incoming/outgoing sequence
flows of x"), and a **path** is "a non-empty sequence of flow objects connected
by sequence flows, from a start to an end event" — so the diameter counts
gateways along with everything else.

**#4 TNMF** (total number of message flows) is the one published metric left
out, by decision, and since 2026-08-16 **no message-flow count is reported
anywhere** — not as a score, not as a diagnostic. `prag_score` divides by **14**.

The reason is not that DOT cannot express a message flow: PMo writes them as
`[style=dashed, arrowhead=open]`, `graph.py` reads all 20, and that count agrees
with BEF4LLM's BPMN-derived TNMF on **55 of 55** models. The reason is that **no
prompt template in `prompts.py` prescribes the notation**, so a generated model
has no way to produce a message flow deliberately — any count over the generated
corpus would measure the prompt, not the model. Only items 23, 24 and 38 carry
message flows at all.

Detection stays and is load-bearing: `graph.is_message_flow` keeps these edges
out of |F^S|, out of every control-flow degree, and out of the
cross-connectivity graph. Removing it would silently inflate TNSF — item 23
would report 22 sequence flows instead of 16.

**Cyclicity** is a category the paper names and deliberately leaves empty — not
an omission of this port. §4.2: *"Cyclicity, frequently employed as a pragmatic
measure, is not included in the BEF4LLM framework because existing research does
not provide multiple thresholds for cyclicity metrics, which prevents
categorization of these metrics in a manner consistent with the other metrics
used in the framework."* The group is kept, empty, so the structure stays the
paper's.

### ⚠ Three misprints in the paper, corrected here

1. **Cross-connectivity's formula** is printed in Table A.16 as
   `normasc len(longest_loop)` — a cycle-length measure, which cannot be this
   metric: the row cites **[53] Vanderfeesten, Reijers, Mendling, van der Aalst,
   Cardoso, *On a quest for good process models: the cross-connectivity
   metric*** (2008), and §4.2 rules cyclicity out of the framework outright.
   The reference code does not compute [53] either — see
   *Cross-connectivity is the reference implementation's* below, which is what
   `prag_cc` reproduces at the author's instruction.
2. **Sequentiality's formula** is printed as
   `normdesc Σ_{g∈G}(|in(g)|+|out(g)|)/|G|` — AGD's formula, repeated from three
   rows above. Implemented per Mendling (2008). Its thresholds are listed in the
   "higher is better" direction, which AGD's formula would contradict, so the
   misprint is unambiguous.
3. **§4.2's examples of "greater is better"** name *token split* and
   *connectivity coefficient*. Both are complexity measures — more concurrent
   tokens and more arcs per node make a model harder to read, not easier — and
   Table A.16 marks both `normdesc`. The prose is the part that is wrong; the
   table, the thresholds and the semantics agree with each other.

### ⚠ Sequentiality counts message flows, because the reference does

`prag_sequentiality` divides **arcs between two non-connector nodes** by **all
arcs between flow objects** — message flows included on both sides (author's
decision, 2026-08-24). The reference implementation's own comment says the
opposite:

```python
# message flows are not taken into account, only sequence flows!
for edge in model.sequence_flow_id_edge_mapping:
```

but `collaboration_model.py::map_edges_to_id` fills that mapping from *every*
edge of the process graph:

```python
self.sequence_flow_id_edge_mapping = {
    data["id"]: (u, v) for u, v, data in self.process_graph.edges(data=True)}
```

so a message flow between two tasks counts as a sequential arc and the divisor
is the whole edge count. A message flow that ends at a collapsed black-box pool
has no node on the other side and is in neither count — here as there.

It shows only in the multi-pool models: item 23 becomes 9/19 = 0.474 instead of
6/16 = 0.375, item 24 becomes 33/37 = 0.892 instead of 24/28 = 0.857. The other
53 models draw no message flows. With this, `prag_sequentiality` matches
`bef_prag_sequentiality` on **55 of 55**; the raw mean moves 0.225 → 0.227 and
no banded score changes.

### ⚠ Cross-connectivity is the reference implementation's, not [53]'s

`prag_cc` reproduces what BEF4LLM's code computes (author's decision,
2026-08-24), and that is **not** the cross-connectivity of [53]:

```
CC = (|{(a, b) : a ≠ b, b reachable from a}| + |N|) / (|N| · (|N| − 1))
```

the share of ordered node pairs joined by a directed path, each node's pair
with itself counted as 1. The cause is one line in
`pragmatic_quality_metrics.py::cross_connectivity`:

```python
edges = graph.edges(data='id')   # 3-tuples (u, v, id)
for tuple in edges:
    if len(tuple) == 2:          # never true
        edge_dict[edge_id] = weight_edge
```

`edge_dict` stays empty for every model, so `max_weight_paths` multiplies
nothing and every path is worth 1; the node weights above it (1/d for XOR, 1
for AND, the OR term) are computed and discarded. Since networkx 3.3
`all_simple_paths(g, n, n)` yields the trivial path `[n]`, which is where the
`+ |N|` comes from. The expression above reproduces
`bef_prag_cross_connectivity` on **55 of 55** reference models to 1e-9.

**Consequences to state wherever this metric is reported.** The published
thresholds `(0.007996, 0.030407, 0.061814, 0.112903)` were calibrated for [53],
whose values on PMo run 0.016–0.5 (mean 0.111). This measure runs **0.217–0.921**
(mean 0.566), so every reference model — and in practice every generated one —
lands above the top band and scores **1.0**. `prag_cc_score` therefore does not
discriminate any more, and it lifted `prag_score` by about 0.016 at the time (0.626 → 0.641; it is 0.671 now, after the diameter change).
`pragmatic._node_weight` still holds [53]'s weights, unused, so switching back
is two lines.

### Scoring

Every metric is reported **raw and banded**:

| column | meaning |
|---|---|
| `prag_<metric>` | the raw value |
| `prag_<metric>_score` | its 0.0 / 0.25 / 0.5 / 0.75 / 1.0 band |
| `prag_group_<group>_score` | mean over that group's metrics |
| `prag_score` | **Qprag** — the mean over all 14 metric scores |

Banding uses the paper's Eq. 1 and Eq. 2 in `quality/normalize.py`. Both take
the thresholds **in Table A.18's ascending order** and use the same half-open
bands `[t_i, t_(i+1))`; they differ only in which end is the good end:

```
normdesc (lower is better)          normasc (higher is better)
  1.0  if       x <  t1               0.0  if       x <  t1
  0.75 if t1 <= x <  t2               0.25 if t1 <= x <  t2
  0.5  if t2 <= x <  t3               0.5  if t2 <= x <  t3
  0.25 if t3 <= x <  t4               0.75 if t3 <= x <  t4
  0.0  if t4 <= x                     1.0  if t4 <= x
```

The paper's worked example (§4.2) is reproduced exactly: a model with 45 nodes,
TNN thresholds 29.9 / 43.7 / 58.1 / 81.1, `43.7 ≤ 45 < 58.1` → **0.5**, group 3.

Aggregation follows §4.5 — *"we sum the individual metric scores and divide by
the number of metrics"*, equally weighted, because there is no empirical
evidence that any metric should count more. The divisor is therefore the **full
metric count, not the count of metrics that happened to be computable**: a
metric with nothing to measure (no gateways to average) scores 1.0 rather than
dropping out, so two models are always scored on the same scale. Its *raw*
column still reads `None`, so no measurement is stated that was not made, and
`prag_n_metrics_measured` records how many of the 14 were real.

Thresholds, Table A.18 verbatim:

| metric | t1 | t2 | t3 | t4 | ref |
|---|---|---|---|---|---|
| TNN | 29.9 | 43.7 | 58.1 | 81.1 | [48] |
| TNG | 1.42 | 3.36 | 5.3 | 6.49 | [48] |
| TNSF | 19.4 | 34.8 | 50.2 | 74.8 | [48] |
| Diameter | 7.92 | 12.2 | 16.5 | 23.4 | [48] |
| Density | 0.1361169 | 0.357143 | 0.741667 | 2.33333 | [51] |
| AGD | 3.67 | 3.88 | 4.06 | 4.18 | [49] |
| CNC | 0.37 | 0.9 | 1.43 | **4.18** | [50] |
| GH | 0.62 | 0.79 | 0.92 | 0.94 | [49] |
| CFC | 13 | 22 | 37 | 51 | [49] |
| CC | 0.007996 | 0.030407 | 0.061814 | 0.112903 | [51] |
| Sequentiality | 0.25 | 0.48 | 0.7 | 1.07 | [48] |
| Separability | 0.03 | 0.37 | 0.71 | 1.24 | [50] |
| Depth | 0.42 | 1.72 | 3.02 | 5.09 | [49] |
| TS | 0.12 | 0.21 | 0.6 | 1.36 | [50] |

> ⚠ **CNC's t4.** Table A.18 prints 4.18 — the same value as AGD's t4 in the row
> directly above, and out of step with this row's own 0.53 spacing. BEF4LLM's
> code has **2.28**. The published value is used, because that is what a thesis
> cites; nothing on PMo turns on it, since CNC never exceeds 1.43 on the
> reference models and no model lands between the two candidates.

Diagnostics come along so a band can be explained without reopening the file:
`prag_n_splits`, `prag_n_joins`, `prag_n_gw_exclusive` / `_parallel` /
`_inclusive` / `_eventbased`, `prag_n_gateways_type_defaulted`,
`prag_n_cut_vertices`, `prag_n_flows_outside_fo`, `prag_n_self_loops`,
`prag_diameter_truncated`, `prag_depth_truncated`.

### Reading gateway types

GH, CFC, CC and the token split all need to know what kind of gateway a diamond
is. `graph.py::resolve_gateway_type` decides in layers:

1. what the notation states — exclusive / parallel / inclusive, from the marker
   or the name;
2. **event-based** — a `label="E"` diamond or `EventBasedGateway` in the id;
   PMo's Camunda-derived items write exactly that;
3. otherwise **exclusive**. Not a guess: BPMN draws the exclusive gateway *with
   or without* the X marker, so PMo's `"recourse possible?" [shape=diamond]`
   **is** an exclusive gateway. `prag_n_gateways_type_defaulted` counts how
   often the rule fires — 99 of 464 gateways across the reference models.

The paper's connector vocabulary has only **three** classes: `p(l)` is the
"gateway-type share for l ∈ {AND, XOR, OR}", and the same three sets index CFC
and TS. An event-based gateway therefore contributes to the **XOR** share
rather than to a fourth one of its own — it routes the token to exactly one of
several catching events, which is XOR semantics, and it is how the code treats
it in CFC. The four-way reading survives only in the `prag_n_gw_*` diagnostics.

### Where this port diverges from BEF4LLM's code

Each case is commented at the metric in `quality/pragmatic.py`:

| what | BEF4LLM's code | here | why |
|---|---|---|---|
| **normalisation of a "higher is better" metric** | `get_rank` defaults `rank = 0`, so a value below *every* threshold never breaks the loop and scores **1.0 — the best** | reproduced since 2026-08-24 (`normalize.normasc`) | these columns are the comparable ones; the consequence is that `prag_sequentiality_score`, `prag_separability_score` and `prag_cc_score` are not quality signals — read the raw columns |
| **token split** | loops over the AND-splits **twice**, returning 2× (its docstring says AND/OR-*joins*), and never counts OR-splits | reproduced since 2026-08-24 (55/55) | this column is the comparable one; Table A.16's formula would be half of it, plus the OR-splits |
| **depth** | a join never cancels a split (`maxcounter` only ever rises from 0), subtrees are *added* on, out-depth walks **forward** from the starts, and one `visited` set is shared across the traversal | reproduced since 2026-08-24 (54/55; item 37's `complexGateway` is invisible in DOT) | this column is the comparable one — it counts the splits along a path, not how deeply the model nests |
| **cross-connectivity** | never applies its arc weights at all (see below), so every path is worth 1 | the same computation, expressed directly | reproduced on purpose — `prag_cc` is meant to be the comparable column here |
| **separability** | cut vertices over the whole graph, message flows included | over the sequence-flow graph | a message flow merges two pools into one component and hides cut vertices on both sides |
| **GH** | counts a gateway only if out > 1, else in > 1; counts event-based as a fourth class; skips a `complexGateway` entirely | reproduced since 2026-08-24 — same filter, same four classes (54/55; item 37's complex gateway is a bare diamond in DOT and cannot be told from an exclusive one) | this column is the comparable one; the paper's three-class reading is no longer computed |
| **splits/joins** | anything with out > 1 is a split | `graph.py::_gateway_role` — declared intent first, topology second | a malformed `AND_SPLIT_1` wired 2→1 must fail as a split, not be counted as a join (see *Telling a split gateway from a join gateway*) |
| **diameter** | counts only non-gateways, and its backward `countpaths` returns from inside the loop three times, so it reports the first backward path to a source rather than the longest | `prag_diameter` reproduces exactly that (55/55); `prag_diameter_nogw` is the longest gateway-free path, `syn_diameter` the longest path over every flow object | reproduced on purpose — this column stays the comparable one, but **since 2026-08-29 it no longer feeds `prag_score`**: `prag_diameter_nogw` does (author's instruction). The defect is not neutral in a score that rewards smallness — on a generated model it read a diameter of 1 against a longest path of 9 and handed it the full 1.00 |
| **empty defining set** | raw 0, which bands to the best score | raw `None`, score 1.0, divisor unchanged | a 0 there is not a measurement; the score is the same either way |

Two things the port does **not** carry over, because DOT has no such notion:
the subprocess expansion (`add_subproccess_nodes_to_graph`) and the
per-participant pool decomposition. The paper excludes subprocesses from the
framework anyway ("they introduce hierarchical structures that complicate
metrics such as size"), along with artifacts, data objects and groups.

`networkx` is **not** re-added as a dependency: the two graph algorithms this
needed — articulation points (iterative Tarjan) and the max-product path
(Dijkstra over −log weights) — are ~40 lines of stdlib in `pragmatic.py`, and
both are verified against brute force on all 55 reference models.

### Calibration against the ground truth

The PMo reference models score **`prag_score` 0.698**:

| metric | raw min | raw max | raw mean | score mean | not measurable |
|---|---|---|---|---|---|
| TNN | 5 | 50 | 23.200 | 0.936 | 0 |
| TNG | 0 | 22 | 8.436 | **0.205** | 0 |
| TNSF | 4 | 60 | 27.345 | 0.741 | 0 |
| Diameter | 4 | 18 | 9.018 | 0.777 | 0 |
| Density | 0.024 | 0.200 | 0.061 | 0.986 | 0 |
| AGD | 3 | 4.667 | 3.205 | 0.954 | 1 |
| CNC | 0.8 | 1.429 | 1.154 | 0.505 | 0 |
| GH | 0 | 0.921 | 0.456 | 0.909 | 0 |
| CFC | 0 | 21 | 7.927 | 0.959 | 0 |
| CC | 0.217 | 0.921 | 0.566 | **1.000** | 0 |
| Sequentiality | 0.023 | 1 | 0.227 | **0.800** | 0 |
| Separability | 0.08 | 1 | 0.438 | 0.400 | 0 |
| Depth | 0 | 11 | 3.927 | 0.377 | 0 |
| Token split | 0 | 12 | 3.527 | **0.218** | 0 |

| group | score mean |
|---|---|
| Size | 0.665 |
| Density | 0.815 |
| Connector interplay | 0.956 |
| Partitionability | 0.526 |
| Cyclicity | — (empty by the paper's decision) |
| Concurrency | 0.218 |
| Other metrics | — (empty) |
| **`prag_score`** | **0.698** |

Reproduce with:

```bash
python -c "from quality import graph,pragmatic; import glob,statistics; \
print(statistics.mean(pragmatic.evaluate(graph.load(p))['prag_score'] \
for p in glob.glob(r'<PMO_DATASET_DIR>/graphviz/*.dot')))"
```

> ⚠ **Three caveats before any of these numbers go into the thesis.**
>
> 1. **This is not a correctness measure**, and the paper says so itself: *"its
>    score decreases as a process model becomes larger and more complex.
>    However, 'simpler' is not always better."* Eleven of the fourteen metrics
>    score smallness or simplicity, so a degenerate model scores high — an empty
>    graph gets `prag_score` 1.0. Report it *next to* `syn_score`, never instead
>    of it. Concretely: the 11 real (non-mock) generations currently sitting in
>    `runs/` average TNN 12.7 and `prag_score` ≈ 0.83, against the reference
>    models' TNN 23.2 and 0.698 — they score higher because they are half the
>    size, which is the metric working as designed, not the models being better.
>    (The remaining ~810 files in `runs/` are three-node mock-provider stubs;
>    ignore them, and re-derive this line from a real run before quoting it.)
> 2. **The thresholds are calibrated for smaller models than PMo contains.**
>    Sequentiality (t4 = 1.07) and separability (t4 = 1.24) are ratios in
>    [0, 1], so their top band is unreachable and 0.75 is the effective ceiling;
>    TNG's t4 = 6.49 while PMo averages 8.4 gateways, so most reference models
>    score 0.0 on it. That is why the three bolded rows are low — a property of
>    the bands, not of the models.
> 3. **1.0 is not the ceiling to measure against.** Report generated models
>    against the 0.698 baseline above, as with `syn_score`.

### Size is reported twice

BEF4LLM files TNN / TNG / TNSF / diameter under **pragmatic** quality; this
project implemented them earlier in `quality/syntactic.py`. Both read the same
`ProcessGraph` primitives, so they agree by construction — verified equal on
all 55 reference models. TNMF, BEF4LLM's fifth size metric, is left out of
both:

| syntactic | pragmatic |
|---|---|
| `syn_tnn`, `syn_tng`, `syn_tnsf`, `syn_diameter` | `prag_tnn`, `prag_tng`, `prag_tnsf`, `prag_diameter` (three different diameter readings — see below) |
| `syn_size_score` (mean of 4) | `prag_group_size_score` (mean of the same 4 — identical, 0.571 on the reference models) |
| `syn_diameter_nogw` (+ `_score`, gateway-skipping convention, unscored) | `prag_diameter_nogw` (+ `_score`, identical by construction — **this is the scored diameter since 2026-08-29**) |

Neither set feeds `syn_score`. Which prefix survives into the thesis tables is
an open decision (see below) — nothing breaks either way, but the same number
under two names in one table needs a sentence of explanation or one of them
removed.

## Semantic quality — BEF4LLM's published metric set

`quality/semantic.py` — the semantic dimension of the BEF4LLM paper (§4.3,
formulas in **Table A.17**), ported from the paper's definitions like the
other two BEF4LLM dimensions. Semantic quality asks whether the generated
model *says the same thing* as the reference: the candidate model is compared
against the PMo ground-truth `.dot` of its item, so this is the **only
dimension that needs `ground_truth_path`** — in CSV generation-only mode every
`sem_` column is `None` (not comparable ≠ maximally dissimilar).

Five of the paper's seven metrics, in two of the three similarity groups of
Dijkman et al. (2011) — the causal-footprint overlap and the dependency-graph
overlap were dropped on 2026-08-26, at the author's instruction, see
`quality/semantic.py::METRICS`. With that second behavioural metric the
**behaviour group disappeared**: the semantic dimension carries no behavioural
component. Three metrics that had also been dropped have since returned: the
context similarity on 2026-08-30, the graph-edit distance on 2026-09-08 — which
brought the **graph-structure group** back with it — and common nodes and edges
on 2026-09-09. Each metric is a similarity in 0–1 (no
raw/banded split — the paper: "Each metric is already scaled to the interval
[0,1]"):

| column | metric | group | in words |
|---|---|---|---|
| `sem_label_sim_syntactic` | Syntactic label similarity | natural language | matched node pairs, scored by Levenshtein distance on the label strings |
| `sem_label_sim_semantic` | Semantic label similarity | natural language | same, scored on stemmed word sets: exact overlap ×1.0, WordNet synonyms ×0.75 |
| `sem_label_sim_context` | Context similarity | natural language | how much of the semantic-label matching survives one step out from the pair, gateways skipped over |
| `sem_graph_edit_distance` | Graph-edit distance | graph structure | networkx' `optimize_edit_paths` over the two contracted skeletons, normalised against the trivial edit script |
| `sem_common_nodes_edges` | Common nodes and edges | graph structure | matched nodes and matched skeleton edges over all nodes and edges of both sides; halves in `sem_common_nodes` / `sem_common_edges` |

### Common nodes and edges (metric 5): dropped 2026-08-29, restored 2026-09-09

It was dropped the day its reproduction settled what their column is: their
`common_percentage_similarity` is called with `edges=False` and their matching
binds every node at threshold 0.0, so **their value is 1.0 for any two models**.
That has since been confirmed on **155 pairs across three pairings** — 55
unrelated offset pairs, 45 basic-vs-full, 55 manipulated-vs-original — constant
throughout. A figure that cannot distinguish two unrelated processes from two
nearly identical ones carries no information, and this port's own reading had
nothing left to be validated against. The reproduction column went with it.

**It is back since 2026-09-09** at the author's instruction, as the printed
formula over nodes *and* edges — `quality/semantic.py::_common_nodes_edges`:

    sem_common_nodes_edges = 1 − (unmatched nodes + unmatched edges)
                                 / (|FOc| + |FOg| + |Fc| + |Fg|)

read through M^opt_Sem, so it costs nothing beyond the matching metric 2 already
computes. The two halves are reported separately as `sem_common_nodes` and
`sem_common_edges` — **the metric is not their mean**, it weighs each by how
many elements it ranges over.

> ⚠ **What validates it, and what does not.** There is no informative reference
> column to reproduce, so the `55/55 exact` standard the label similarities meet
> is unreachable here. What stands in its place is `pmo_common_edge.py`, which
> scores each manipulated PMo model against its original and checks the result
> against `PMo_manipulated/manipulations.json`. **Half of that check is real:**
> the 19 rename-only items keep the edge half at exactly 1.0 — a label change
> does not leak into the edge term. **Half is vacuous:** `sem_common_nodes` is
> 1.0 on all 55 models, so "the node half must stay 1.0" is satisfied whatever
> the metric does. Those manipulations never change a node count, and the metric
> asks whether a node is matched, never how well — so a rename with any
> surviving word overlap keeps its node bound. Consequence: on that corpus the
> metric is carried entirely by its edge half and is **blind to all 46 renames**
> in the rename-only items. Exercising the node half needs a manipulation the
> dataset does not contain — adding or deleting a flow object, or renaming to a
> label with no word in common. The script flags any half that never varies, so
> this does not read as 55 passes.
>
> **That limitation is the manipulated corpus's, not the metric's.** On real
> generations both halves carry signal: over the 25217 scored models of run
> 20260831_182408, `sem_common_nodes` means 0.7194 across **419 distinct
> values** and `sem_common_edges` 0.3401. Generated models differ from the
> ground truth in node count and contain genuinely unmatchable nodes, which is
> exactly what the manipulations never produce.

`sem_score` is the mean over **five** metrics now and the graph-structure group
over two, so no semantic figure is comparable with one produced before this
date.

### The graph-edit distance (metric 4): dropped 2026-08-29, restored 2026-09-08

It was dropped right after metric 5 — and with it the **graph-structure
group**, which then had no member left. The reason was never the metric but the
*reproduction* of BEF4LLM's version of it: 8 of 55 pairs exact and 24 within
0.01 once the comparison ran over identical graphs with disjoint ids, the
loosest of the three reproductions, because their `sbv` term divides
contracted-node similarities by *uncontracted* node counts, and those differ
between DOT and BPMN wherever a model carries pool anchors (items 23, 24, 38).

**It is back since 2026-09-08** at the author's instruction, as
`quality/semantic.py::_graph_edit_distance` on networkx. That reopens nothing:
what returns is **this port's own reading** — a real edit distance on the two
contracted skeletons — and not their `1 - avg(snv, sev, sbv)`, which despite
the name is a quota of unmatched nodes and edges plus the mean label distance
of the matched pairs. A real edit distance has no `sbv` term and needs no
matching of ours, so the DOT-versus-BPMN question that removed the reproduction
does not arise for it. It is also the one semantic metric that is **reflexive**:
a model against itself scores exactly 1.0, where the two label similarities put
a self-match at ≈0.478 for their shared divisor.

Two nodes substitute for free when the label *and* the element kind agree, two
edges when the flow type does; the result is normalised against the trivial edit
script (delete every node and edge of one side, insert every node and edge of
the other), so `1.0` means the same graph:

    sem_graph_edit_distance = 1 − ops / (|V_c| + |V_r| + |E_c| + |E_r|)

> ⚠ **Read `sem_ged_truncated` before quoting a value.** The exact distance is
> exponential and does not finish on models this size, so the search is cut by
> networkx' own `timeout` (`_GED_BUDGET_S`, five seconds). A cut value is an
> upper bound on the distance — a *lower* bound on the similarity — and depends
> on how fast the machine was, so it is not reproducible; an untruncated one is
> exact. **On real generations truncation is the common case:** on
> `runs/20260824_235554`, 43 of 55 pairs were cut, mean 4.01 s per pair. Raising
> the budget does not buy this back — at 60 s half of a 12-pair sample was still
> cut, for eight times the wall time. The manipulated pairs are the easy case
> (3 of 55 cut), the two sides there being nearly the same graph.
>
> Two consequences for a table: a mean over `sem_graph_edit_distance` mixes
> exact values with bounds, and `sem_score` inherits both. Report the truncation
> rate beside any figure that uses them.

The budget is also what this metric costs a run: it is scored inline on the
generation worker like every other, so ~4 s per generation is the order of
magnitude to plan with (~29 h of CPU across the workers on a run the size of
`20260831_182408`, spread over its parallelism).

Plus `sem_group_natural_language_score`, `sem_group_graph_structure_score`,
`sem_score` (Qsem — mean over all five, same fixed-divisor convention as
`prag_score`: an unmeasurable metric enters at 1.0, `sem_n_metrics_measured`
counts the measured ones), and diagnostics
(`sem_n_label_nodes_*`, `sem_n_nodes_matched`, `sem_n_skeleton_edges_*`,
`sem_n_edges_matched`, `sem_wordnet`, `sem_ged_operations`,
`sem_ged_truncated`, `sem_common_nodes`, `sem_common_edges`).

**No semantic figure is comparable with one produced between 2026-08-29 and
2026-09-08** — `sem_score` runs over a different metric set again. Re-run
`score_pmo_dataset.py` and `score_pmo_pairs.py` before quoting either.

### How the comparison works

- **Compared nodes** are the non-gateway flow objects (Table A.17's
  denominators quantify over `τ(n) ∉ G`): tasks, start/end/intermediate
  events. Pools and lanes are not flow objects and stay out (the supplied
  code adds them; the paper's formula does not).
- **The matching is an optimal bipartite matching** ("M^opt … maximize
  similarity"), computed exactly per criterion with
  `scipy.optimize.linear_sum_assignment`; zero-similarity pairs are dropped.
  The semantic-label matching M^opt_Sem is the node equivalence for the
  structural and behavioural metrics (Table A.17 writes plain `M` there; the
  supplied code chooses the same).
- **Structure is compared on the contracted graph**: gateways (and pool
  anchors / unclassifiable nodes) are skipped over, `task → gateway → task`
  becomes `task → task`, edge identity is (mapped source, mapped target,
  flow type). Without this, every gateway of both models would count as
  unmatched forever — gateways are excluded from label matching — and even a
  model identical to the ground truth could not reach 1.0.
- **The ε-label rule.** `label=""` is how PMo *and two prompt templates*
  write BPMN's canonically unlabeled start/end events, and for two ε labels
  the printed formulas are 0/0. Resolved by element kind: ε ↔ ε scores 1.0
  when both nodes are the same kind (start/end/task/intermediate event),
  else 0. The reference implementation scores ε ↔ ε as 0, which makes every
  correctly-unlabeled model structurally unmatchable at its start and end
  events. Without this rule, 14 of the 55 PMo models don't reach 1.0
  against **themselves** (worst 0.66).

### ⚠ Where this port diverges from BEF4LLM's code (the paper wins)

| | paper (Table A.17) | supplied code |
|---|---|---|
| matching | optimal bipartite | greedy in file order, later pairs overwrite one-sidedly |
| semantic label sim | (2·wi·overlap + ws·synonyms) / (\|w1\|+\|w2\|) | overlap + 0.75·synonyms, divided by max(\|w1\|,\|w2\|) |
| context sim denominator | 2·√(\|c1\|·\|c2\|) per direction | max(\|c1_in\|,\|c2_in\|) + max(\|c1_out\|,\|c2_out\|), pooled |
| label metrics over | non-gateway flow objects | + pools and named lanes |

The paper's own Qsem results were produced by the code, so — as with the
syntactic set — `sem_score` is **not comparable with the published Qsem
numbers** without stating that the definitions differ.

### Weights, words and WordNet

The paper names the label-similarity weights (wi, ws) but never assigns
them; this port uses Dijkman et al.'s **wi = 1.0, ws = 0.75**, which the
supplied code also uses. The text layer lives in `quality/textsim.py` and is
deterministic stdlib — alphanumeric tokenisation, an embedded English
stopword list, the original **Porter (1980) stemmer** (implemented there, not
NLTK's extended variant), Levenshtein — so the same labels score identically
on every machine. The one exception is the **synonym term** of
`sem_label_sim_semantic`: it needs WordNet, via

```bash
pip install nltk
python -m nltk.downloader wordnet
```

If NLTK or the corpus is missing the term is 0 and the row records
`sem_wordnet = 0`, so a degraded score is never silent. English only — the
PMo corpus is English.

### Calibration against the ground truth

Self-comparison (every PMo model against itself): **`sem_score` = 0.8261**,
with metrics 4–7 at 1.0 on all 55 models — the ε-label rule in the node
matching is what closes the last 14 — and metrics 1 and 2 at 0.4782, because
they are scored the reference implementation's way since 2026-08-24 (its
divisor has no factor 2 and an unlabelled node contributes nothing, so a
perfect match reads 0.5). Read the four structural metrics against a ceiling of
1.0 and the two label metrics against 0.5.

> **Metric 3, context similarity, is not computed.** Dropped at the author's
> instruction on 2026-08-24: the published formula scores a self-comparison
> 1.0 where the reference implementation scores about 0.48, and reproducing
> theirs stalls at 42 of 55 models because their greedy node matching depends
> on the order the nodes stand in the file — `.bpmn` and `.dot` order them
> differently. Eight variants were measured before dropping it (formula,
> matching, neighbourhood traversal, node order, BPMN input); none reached 55.
> `sem_score` therefore divides by **six**, and the natural-language group by
> two — neither is comparable with a figure produced before that date.

The discrimination floor — each model against the *next item's* model, i.e.
against the wrong process:

| | mean | max |
|---|---|---|
| `sem_score` (54 neighbour pairs) | **0.183** | 0.898 |

The 0.898 pair is items 52/53, which genuinely are the same dismissal
process modelled twice with different party abbreviations — the metric is
answering correctly. With the behavioural metrics gone, the label
similarities carry most of the floor.

Runtime: ~0.4 s for the largest PMo pair, well under 100 ms typically —
re-scoring a full 6.7k-generation run stays in the minutes.

### Running the metrics

Scoring is a **separate pass over a run directory** — no API calls, nothing
regenerated, nothing spent — so metrics can change and be re-applied freely:

```bash
python score_run.py runs/<run_id>          # (re-)score a finished run
python run.py --limit 5                    # a run scores itself at the end
python run.py --limit 5 --no-score         # …unless you skip it
```

It writes `quality.csv` and merges those columns into `results.csv`
(idempotent — re-running replaces them). `results.jsonl` is left alone as the
raw generation log. The two handles it works from, present in every row:

| column              | meaning                                                        |
|---------------------|----------------------------------------------------------------|
| `generated_gv`      | path of the extracted DOT, relative to the run dir. Empty = nothing extractable was produced (scored as unevaluable, never dropped) |
| `ground_truth_path` | absolute path of the PMo `.dot` for that item. Empty in CSV generation-only mode |

Read the *PMo DOT quirks* warning under **Dataset** before writing a new
dimension — both quirks silently produce zeroes rather than errors.

### Timing

How long a prompt took from being sent to the answer arriving is recorded per
row in `results.csv` / `results.jsonl`, in seven columns measured around the
vendor SDK call in `providers.py` (`CallTiming`) and in `pipeline.py`:

| column | meaning |
|--------|---------|
| `api_latency_s` | **the round trip of the prompt** — from just before the request is handed to the vendor SDK until the complete reply object is back. Measured on the attempt that actually answered, so retries and their backoff are *not* in it. Empty when the generation failed: no reply ever arrived |
| `request_sent_at` | wall-clock ISO-8601 timestamp (local time with UTC offset, ms resolution) of that request going out |
| `response_received_at` | same for the moment the reply was complete. The difference reproduces `api_latency_s`, and both put a run on a timeline (rate-limit windows, time of day) |
| `latency_s` | the same span **plus** failed attempts and the sleeps between them. Identical to `api_latency_s` unless `api_attempts > 1`. On a generation that failed outright, this is the time until the last attempt gave up |
| `api_attempts` | how many attempts the answer took (1 = first try) |
| `retry_wait_s` | seconds spent in backoff sleeps. Together with `api_attempts` it explains any gap between the two spans above |
| `call_wall_s` | the same call one level out, in `pipeline.py`: adds the provider wrapper's own work (kwargs assembly, usage extraction). The gap to `latency_s` is this project's overhead, not the vendor's — typically < 1 ms |

Which one belongs in the thesis: **`api_latency_s`** is the model's response
time; `latency_s` is what a user of the pipeline waits for. Report the first
and mention the second where they differ (i.e. where the vendor throttled).

Two caveats:

- The calls are **non-streaming**, so "arrival" means the complete reply, not
  the first token — there is no time-to-first-token figure here, and latency
  therefore scales with `output_tokens`.
- The vendor SDKs retry **transparently inside a single call** (Anthropic's
  client defaults to 2 internal retries), so a rare `api_latency_s` outlier can
  still contain a retry this project cannot see. Build the client with
  `max_retries=0` in `providers.py` if a run has to time single HTTP requests
  exactly; the pipeline's own retry loop then handles every attempt visibly and
  countably in `api_attempts`.

`is_mock=True` rows carry stub timings of a local string operation
(microseconds), never measurements — filter them out before reporting.

Aggregated per `(model, strategy)`: `api_latency_s` mean/median/**min/max** and
`api_attempts` mean/max in `summary_by_model_strategy.csv`, plus
`pivot_api_latency.csv` next to the existing `pivot_latency.csv`.

---

## Architecture

| module                   | responsibility                                                              |
|--------------------------|------------------------------------------------------------------------------|
| `config.py`              | models (4 vendors × 3 tiers), prices, strategies, run settings, paths      |
| `prompts.py`             | verbatim Li et al. templates; exemplar loading; message assembly (vendor-agnostic) |
| `dataset.py`             | `ProcessItem` + loaders for PMo `.txt`/`.dot` pairs and CSV descriptions    |
| `providers.py`           | `AnthropicProvider` / `OpenAIProvider` / `MistralProvider` / `GoogleProvider` / `MockProvider` |
| `postprocess.py`         | extract & validate DOT from a raw reply                                     |
| `pipeline.py`            | the `model × strategy × item` loop; dispatches each model to its vendor's provider |
| `quality/graph.py`       | DOT parsing + BPMN element classification, shared by all dimensions        |
| `quality/normalize.py`   | the paper's Eq. 1 / Eq. 2 banding + the Table A.18 thresholds              |
| `quality/syntactic.py`   | syntactic quality — the eleven dictated checks + the BEF4LLM size metrics |
| `quality/syntax_rules.py`| syntactic quality — BEF4LLM's published metric set (paper Table 2 / A.15): 14 metrics scored as conforming/covered |
| `quality/pragmatic.py`   | pragmatic quality — BEF4LLM's published metric set (paper Table A.16 / A.18): 14 metrics in 7 groups |
| `quality/semantic.py`    | semantic quality — BEF4LLM's published metric set (paper Table A.17): 7 similarity metrics in 3 groups, candidate vs ground truth |
| `quality/textsim.py`     | label-text primitives for the semantic dimension: tokeniser, embedded stopwords, Porter stemmer, Levenshtein, optional WordNet synonyms |
| `quality/validity.py`    | validity — format validity of the artefact, decided by Graphviz `nop -p` |
| `quality/score.py`       | runs the dimensions over one model or a whole run directory                |
| `score_run.py`           | CLI to (re-)score a finished run without regenerating anything             |
| `aggregate.py`           | the two aggregations the report is built on: one row per LLM, one per strategy |
| `results_report.py`      | renders `results.html` from the scored frame                                |
| `csv_export.py`          | writes the run's `csv/` folder — the same results as tables, one file per table |
| `runcontrol.py`          | keyboard pause/stop while a run is in flight                               |
| `run.py`                 | CLI; resolves vendors/models and loads API keys from `api_keys.py`         |
| `api_keys.py`            | empty local key slots; optional plaintext values are git-ignored and never printed |
| `api_keys.example.py`    | versioned template for `api_keys.py`; contains no keys |

### Design Science criteria (mapped in code comments)

- **DC1 Reproducibility** — central config + a full manifest per run.
- **DC2 Faithful prompts** — Li et al. templates reproduced verbatim, shared across all vendors.
- **DC3 Token/cost transparency** — explicit flags when usage isn't reported,
  when a vendor doesn't separately report reasoning/thinking tokens, and when
  a price isn't verified (`cost_usd=None` rather than a guess).
- **DC4 Metric reuse** — the metric layer was rebuilt from scratch as
  `quality/`, one module per dimension, but three of the four dimensions are
  now ports rather than own constructions: **pragmatic quality** takes
  BEF4LLM's metric set, group structure and published thresholds, **semantic
  quality** takes its seven-metric similarity suite (Table A.17), and
  **syntactic quality** additionally carries BEF4LLM's published metric set
  and its element-ratio scoring (`syntax_rules.py`) next to the eleven
  dictated checks. All three are ported from the **paper's definition
  tables**; where the supplied code contradicts them, that is tabulated
  above. **Validity** reuses a tool rather than a metric set: Graphviz's own
  `nop -p`, the check its documentation names for valid DOT language.
- **DC5 Extensibility** — `Provider` is an abstraction; a 5th vendor (e.g. a
  local model) is one more subclass plus `ModelSpec` entries.
- **DC6 Robustness** — retries with backoff per vendor; a failed generation or
  an unparseable reply still produces a row, it is never dropped.
- **DC7 Separation of concerns** — raw replies are always persisted for audit.

---

## Open decisions (carried for the supervisor)

1. **Strategy set** — *decided 2026-08-17:* the nine-strategy thesis set runs;
   `zero_shot_system` is deactivated (template kept, see *Prompting
   strategies*). What still needs an answer is the write-up: it is the only one
   of Li et al.'s five the pipeline no longer runs, so the thesis must either
   justify the omission or stop claiming the paper's set is reproduced in full.
2. **Model IDs & prices** — confirm the exact Efficient / Standard / Advanced
   IDs and current per-token prices for OpenAI, Mistral, and Google in
   `config.py` (Anthropic's are already confirmed). Google's Advanced tier runs
   on a **preview** id, `gemini-3.1-pro-preview`, which is the only Pro-class
   text model the Gemini API offers — confirm that a preview model is acceptable
   for a reproducible result, or drop the tier to a Flash model and say so.
3. **Few-shot exemplars** — items `01`/`02` are the current default
   (`cfg.FEW_SHOT_IDS`), with `03`/`04` as stand-ins and `40`/`41` as the
   complete alternate set for items `01`–`03` (`cfg.FEW_SHOT_ALT_*`). They are
   evaluated like every other item, but neither pair is a randomised or
   otherwise justified choice — confirm which items should serve as exemplars,
   and whether the three items that run on the alternate set are comparable
   enough to the other 52 to be pooled with them (they see two exemplars of the
   same size and shape, but not the *same two*, so a per-item effect cannot be
   ruled out by construction).
4. **Quality metrics** — all four dimensions exist now (see *Quality
   metrics*), so a run answers the cost/token question plus format validity,
   syntactic conformance, pragmatic comprehensibility and semantic similarity
   to the reference. Two calls inside the syntactic dimension are worth
   confirming: unparseable generations score `None` rather than 0, and there
   is no `end_in1_out0` counterpart to the start-event degree check. In the
   validity dimension, confirm the two judgement calls documented there: an
   artefact declaring no graph counts as invalid (Graphviz alone would accept
   it), and a reply that arrived without any DOT block scores 0.0 while a call
   that never returned scores nothing at all.
5. **Sampling / budget** — 55 items × 9 strategies × 14 models is ~6.9k
   generations; check that against the ~€200 budget before a full live run
   (`--limit` is the lever).
6. **Reasoning-token comparability** — Anthropic, OpenAI, Mistral, and Google
   each report (or don't report) thinking/reasoning tokens differently; see
   `reasoning_note` per row before treating `output_tokens` as apples-to-apples
   across vendors for the Advanced tier.
7. **Two syntactic scorings** — *decided 2026-08-18: the thesis reports
   `syn_bef_score`*, BEF4LLM's published metric set (Table 2 / A.15, with
   project exclusions #9, #10, #13 and #14; ten metrics scored, element ratios,
   per-process). It is citable and shares a framework with the
   semantic and pragmatic dimensions, which come from the same paper. The
   eleven dictated pass/fail checks (`syn_score`) are **still computed and
   still in `results.jsonl`** — they name *which* rule a model broke, which a
   ratio cannot — but they no longer appear as a second headline in
   `results.html`. That matters: on the 90-call run the two sets do not rank
   the strategies the same way, so reporting both would mean reporting two
   orders. **Resolved through 2026-09-24**: metrics #9, #10, #13 and #14 were
   removed by decision, so Qsyn is now Σ score / 10. Fidelity to the
   published set was given up knowingly — say so wherever the figure appears,
   and do not compare it with a Qsyn computed over the paper's fourteen.
8. **Size reported under two prefixes** — TNN/TNG/TNSF/diameter appear as both
   `syn_*` and `prag_*` (identical values; BEF4LLM files them under pragmatic
   quality, this project implemented them first under syntactic). Decide which
   set goes into the thesis tables, and whether the other is dropped from
   `quality/syntactic.py` — see *Size is reported twice*.
9. **Pragmatic thresholds** — the published bands (Table A.18) are calibrated
   for smaller models than PMo contains: sequentiality's and separability's top
   band is mathematically unreachable, and most reference models score 0.0 on
   TNG. Report `prag_score` against the 0.698 reference baseline, or re-band
   those metrics on PMo and say so. Confirm also which CNC t4 to use — the
   paper prints 4.18 (identical to AGD's t4 one row above), the reference code
   has 2.28; nothing on PMo turns on it.
10. **Semantic-port interpretation calls** — three places where Table A.17
   under-determines the implementation, each documented in
   `quality/semantic.py` and worth a supervisor nod: (a) the **ε-label rule**
   (two empty labels score 1.0 iff same element kind — the formula is 0/0
   there, and the reference code's 0 punishes the notation two prompt
   templates prescribe); (b) **wi = 1.0 / ws = 0.75** taken from Dijkman
   et al., since the paper never assigns its weights; (c) the **synonym term
   depends on WordNet** being installed — rows record `sem_wordnet`, and a
   thesis run should be scored with it present. Related: `sem_score` is not
   comparable with the paper's published Qsem (its numbers come from the
   code's diverging metric definitions — see the divergence table).
