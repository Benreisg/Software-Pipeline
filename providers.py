"""
providers.py — per-vendor LLM providers (usage + cost capture, retries) and offline mock
============================================================================================
Vendor-neutral interface: `Provider.generate(model, system, messages, settings)`
where `messages` is a list of {"role": "user"|"assistant", "content": str}
turns and `system` is an optional system-prompt string. Each provider adapts
that shape to its own SDK.

⚠ VERIFICATION STATUS (see also config.py):
  - AnthropicProvider was built against the current Claude API docs and is
    believed correct.
  - OpenAIProvider was rebuilt on 2026-08-17 against the current OpenAI docs
    and verified against the installed SDK (openai 3.2.0): it uses the
    **Responses API**, and its request parameters, usage fields and completion
    state were each checked against the SDK's own types rather than recalled.
    Model IDs and prices in config.py come from the price sheet of the same
    day. It has not yet made a paid call — smoke-test with `--limit 1`.
  - MistralProvider's usage accounting was verified against the official Chat
    endpoint reference on 2026-08-20 and re-checked against the schema and the
    live API on 2026-08-29. The schema is wider than the responses: it
    documents four names for the cached prompt count, a reasoning breakdown, an
    audio duration, a per-message prompt breakdown and a request count, while
    every observed chat completion carried only prompt/completion/total, one
    cached alias and the service tier. All of them are read regardless — see
    `_MISTRAL_ACCOUNTED`. GoogleProvider
    remains best-effort. Before a real run, smoke-test with `--limit 1` and
    read the traceback if it 404s/400s — that is almost always a stale model id
    or a renamed parameter, not a logic bug.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config as cfg


@dataclass
class CallTiming:
    """When one generation request went out and how long the answer took.

    `api_latency_s` is the figure "how long did this prompt take": the round
    trip of the attempt that actually returned the reply — from just before the
    request is handed to the vendor SDK until the complete response object is
    back. Retries in front of it and the backoff sleeps between them are NOT
    in it; `total_s` is that wider span, and the two are equal whenever
    `attempts == 1`, which is the normal case.

    ⚠ The vendor SDKs retry transparently inside a single call (Anthropic's
    client defaults to 2 internal retries), so a rare `api_latency_s` outlier
    can still hide a retry this layer never sees. Construct the client with
    `max_retries=0` if a run has to time single HTTP requests exactly — the
    retry loop below then handles every attempt visibly.
    """
    api_latency_s: float = 0.0
    total_s: float = 0.0
    attempts: int = 1
    retry_wait_s: float = 0.0
    request_sent_at: str = ""       # ISO-8601 local time w/ offset, start of the answering attempt
    response_received_at: str = ""  # ISO-8601 local time w/ offset, moment the reply was complete

    def as_fields(self) -> Dict[str, Any]:
        """The timing half of a GenerationResult's kwargs."""
        return {
            "latency_s": self.total_s,
            "api_latency_s": self.api_latency_s,
            "api_attempts": self.attempts,
            "retry_wait_s": self.retry_wait_s,
            "request_sent_at": self.request_sent_at,
            "response_received_at": self.response_received_at,
        }


@dataclass
class GenerationResult:
    text: str
    # ── what the vendor reported, verbatim ──────────────────────────────────
    # The prompt-side count exactly as the API returned it, before this module
    # normalises anything: Anthropic's `input_tokens`, OpenAI's `input_tokens`,
    # Mistral's and DeepSeek's `prompt_tokens`, Gemini's `prompt_token_count`.
    # Kept so every row can be checked against the vendor's own dashboard, and
    # so the normalisation below stays auditable instead of lossy.
    # ⚠ Never priced, and NOT comparable across vendors: Anthropic reports the
    # uncached remainder here while the others report the whole prompt. The
    # column that means one thing everywhere is `input_tokens`.
    reported_input_tokens: int = 0
    # The vendor's own grand total, verbatim: OpenAI's, Mistral's and DeepSeek's
    # `total_tokens`, Gemini's `total_token_count`. Never priced — `total_tokens`
    # below is this project's own sum, and the two are cross-checked per call so
    # a class the split does not cover shows up as `untracked_usage`.
    # 0 means the vendor states no total at all, which is a fact about the API
    # rather than a gap: Anthropic's Messages usage carries none.
    reported_total_tokens: int = 0
    # Anthropic's `cache_creation_input_tokens`, verbatim: the flat total of all
    # cache writes on this call, which the API states *alongside* its 5m/1h
    # split rather than instead of it. Anthropic is the only vendor here that
    # reports both, so it is the only one that fills this — OpenAI's single
    # write tier is already kept verbatim in `cache_write_tokens`, and DeepSeek
    # charges for no writes at all. Never priced: the two split fields are, and adding
    # this would charge every write twice. 0 means no writes, or a vendor that
    # states no such total.
    reported_cache_creation_tokens: int = 0
    # The vendor's own completion count, verbatim: `output_tokens` on Anthropic
    # and OpenAI, `completion_tokens` on Mistral and DeepSeek, and — the reason
    # this field exists — `candidates_token_count` on Gemini, which counts ONLY
    # the answer. `output_tokens` below is the billable figure and adds Gemini's
    # thoughts to it, so on that one vendor the two differ by exactly
    # `thinking_tokens`.
    reported_output_tokens: int = 0
    # Gemini's `tool_use_prompt_token_count`, kept verbatim. Google's REST
    # contract treats it as a tool-use prompt breakdown and excludes it from the
    # documented total formula; some SDK contracts describe it as a separate
    # input class included in `total_token_count`. The Google provider reconciles
    # the response before deciding whether it must also enter `input_tokens`.
    # Never price this informational column directly. 0 on every other vendor.
    tool_use_prompt_tokens: int = 0
    # ── billable token classes (see config.TokenUsage / config.cost_usd) ──
    # input_tokens is the UNCACHED prompt only, as the vendors report it;
    # billable_input_tokens is the whole prompt the invoice charges for.
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0             # cache READ tokens, billed at 0.10x input
    # Cache WRITES at the standard 1.25x-input rate: OpenAI's own
    # `cache_write_tokens`, and Anthropic's default (5-minute TTL) writes.
    # Deliberately not named after a TTL — only Anthropic tiers its writes by
    # one. It is NOT the sum of all writes: Anthropic's 1-hour tier bills at 2x
    # and is counted separately below (Anthropic's own
    # `cache_creation_input_tokens` *is* that sum, and is split before it lands
    # in these two fields).
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0     # cache WRITE, Anthropic's 1-hour TTL, billed at 2x input
    billable_input_tokens: int = 0     # input + cache read + cache writes
    thinking_tokens: Optional[int] = None  # informational: the share of output_tokens spent thinking
    web_search_requests: int = 0       # server-side tool calls, priced per request
    # Mistral's `prompt_audio_seconds`: audio handed to the model, billed by
    # duration rather than by token. It is the only quantity any vendor here
    # bills in a unit other than tokens or requests, and this project has no
    # audio rate, so a non-zero value always raises an `untracked_usage` note —
    # the column exists so the quantity is *recorded* rather than only
    # described in that note. 0.0 on every text call, which is all of them.
    audio_input_seconds: float = 0.0
    service_tier: str = ""             # "standard" | "priority" | "batch" — batch is half price
    total_tokens: int = 0
    cost_usd: Optional[float] = None  # None when the model's prices aren't verified (see config.py)
    cost_basis: str = ""              # which rate applied: "list" | "intro" | "unverified" (+"batch")
    untracked_usage: str = ""         # non-empty = the vendor billed something this cost model misses
    # Timing — see CallTiming. latency_s covers the whole provider call
    # (retries and backoff included), api_latency_s only the request that
    # produced this reply.
    latency_s: float = 0.0
    api_latency_s: float = 0.0
    api_attempts: int = 1
    retry_wait_s: float = 0.0
    request_sent_at: str = ""
    response_received_at: str = ""
    # Vendor-side handles for one call, so a row in this project can be traced
    # back to a request in the vendor's own logs and dashboards.
    # `request_id` is the HTTP request id the SDK reads from the response header
    # (`request-id` / `x-request-id`) — the identifier vendor support asks for.
    # `response_id` is the object id in the payload (`msg_…`, `resp_…`).
    request_id: str = ""
    response_id: str = ""
    # The vendor's complete usage object as JSON, exactly as the SDK returned
    # it. Written to `raw/<tag>.usage.json`, never into the results table — the
    # normalised columns above are what makes two vendors comparable, and they
    # can only hold what all of them state. `results_report.py` reads the
    # archive back and shows it per model and per call, which is where the
    # fields only one vendor reports (Anthropic's TTL split, Gemini's modality
    # breakdown, DeepSeek's cache hits and misses) become visible without
    # pretending they line up with anyone else's.
    raw_usage: str = ""
    stop_reason: Optional[str] = None
    token_reporting: str = "reported"  # "reported" | "estimated"
    tokens_estimated: bool = False
    reasoning_note: str = ""           # explicit DC3 note on how (or whether) reasoning tokens are reported


class Provider:
    name = "base"

    def generate(self, model: "cfg.ModelSpec", system, messages, settings: "cfg.RunSettings") -> GenerationResult:
        raise NotImplementedError

    def check_connection(self, model_id: str):
        """Lightweight, free, read-only call to verify network connectivity,
        auth, and that `model_id` is valid — run once per vendor before a
        real run so a broken connection fails fast with one clear message
        instead of exhausting retries on every single generation. Returns
        (ok, message)."""
        raise NotImplementedError


def _estimate_tokens(text: str) -> int:
    # crude fallback (~4 chars/token); only used by the mock provider, which
    # never calls a real API and always flags tokens_estimated=True.
    return max(1, len(text) // 4)


# Error codes that arrive as a 429 but will never succeed on retry: the account
# is out of credit or over its plan quota. Distinguishing them from a genuine
# rate limit is what keeps a run against an unfunded account from spending its
# whole wall-clock in backoff.
_PERMANENT_QUOTA_CODES = ("insufficient_quota", "credit_balance_exhausted",
                          "billing_not_active", "account_deactivated")


def _is_permanent_quota_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    code = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or error.get("type") or "")
    haystack = f"{code} {exc}".lower()
    return any(c in haystack for c in _PERMANENT_QUOTA_CODES)


def is_fatal_account_error(message: str) -> bool:
    """True when a generation failed for a reason no later call can recover
    from — no credit, dead billing, revoked key. `pipeline.py` stops the run on
    one of these instead of repeating it for every remaining generation."""
    text = (message or "").lower()
    return any(c in text for c in _PERMANENT_QUOTA_CODES) or "authentication" in text


def _request_ids(response: Any) -> Dict[str, str]:
    """The vendor's own handles for this call.

    `_request_id` is set on the response object by the SDK from the HTTP
    `request-id` / `x-request-id` header — public despite the underscore, and
    the identifier vendor support asks for. It only exists on a response that
    actually came over the wire, so it is read defensively: a stub or a mocked
    path simply yields an empty string rather than raising.

    The response's own id is `id` on Anthropic and OpenAI and `response_id` on
    Gemini, which sets no request-id header at all — there the response id is
    the only handle a call can be traced by.
    """
    return {
        "request_id": str(getattr(response, "_request_id", "") or ""),
        "response_id": str(getattr(response, "id", None)
                           or getattr(response, "response_id", None) or ""),
    }


def _usage_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """A vendor usage object as a plain nested dict, or None if there is none.

    Every vendor SDK here returns a pydantic model, so `model_dump` is the path
    that actually runs; `mode="json"` resolves the enums Gemini puts in its
    modality breakdowns. The attribute sweep is the fallback for a stub or a
    future SDK that hands back something plainer.
    """
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump(mode="json")
        except TypeError:      # pydantic v1, or a model_dump without the kwarg
            return usage.model_dump()
    return {k: getattr(usage, k, None) for k in dir(usage) if not k.startswith("_")}


def _usage_json(usage: Any) -> str:
    """The vendor's usage object as JSON, for the run's raw archive. Never
    raises: an object that will not serialise is worth less than the call it
    describes, so it degrades to an empty string rather than losing the row."""
    data = _usage_dict(usage)
    if data is None:
        return ""
    try:
        return json.dumps(data, indent=2, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 — the archive is a nice-to-have
        return ""


# The service tiers that bill at the list rate this cost model prices with —
# one vendor's spelling per entry. `batch` is priced separately (0.5x, see
# TokenUsage.batch); everything else a vendor can report — priority, flex,
# scale, fast, provisioned throughput — bills at a rate this project has no
# figure for, so it is reported rather than silently priced as standard.
#
# None of them can occur in this project's runs: no provider requests a tier, so
# every vendor serves the default. That is exactly why the check is cheap and
# worth having — a tier appearing anyway means an assumption broke.
_LIST_RATE_TIERS = {
    "", "standard",                      # Anthropic, Mistral
    "default", "auto",                   # OpenAI
    "unspecified", "on_demand", "traffic_type_unspecified",  # Google
}


def _service_tier_note(tier: Any) -> str:
    """A warning when a call was served by a tier the cost model cannot price."""
    name = str(tier or "").strip().lower().rsplit(".", 1)[-1]
    if name in _LIST_RATE_TIERS or name == "batch":
        return ""
    return (f"served by service_tier={str(tier)!r}, which is not the list-rate "
            f"tier this cost model prices — the vendor bills it at a different "
            f"rate (priority is dearer, flex cheaper), so cost_usd is not the "
            f"amount charged")


# The name fragments that mark a usage field as a quantity a vendor can bill
# for. "token" and "request" were the whole list while tokens and server-tool
# calls were the only things any vendor here charged for — but Mistral's Chat
# schema also carries `prompt_audio_seconds`, a billed quantity whose name says
# nothing about tokens, and a guard that only knows the word "token" lets a
# whole unit of billing through unnoticed. Matching on the *unit* rather than on
# a list of known field names is what lets the check survive a field nobody has
# seen yet, which is the entire point of having it.
_BILLABLE_NAME_PARTS = (
    "token", "request", "second", "duration", "minute", "hour",
    "char", "byte", "image", "page", "pixel", "call", "count", "credit",
)


def _is_billable_name(path: str) -> bool:
    """Whether a usage field's own name says it counts something billable.

    Only the last segment is read: `prompt_tokens_details.cached_tokens` is
    named by `cached_tokens`, and a parent whose name happens to contain
    "token" must not make every field underneath it look like a token count.
    """
    leaf = path.rsplit(".", 1)[-1].lower()
    return any(part in leaf for part in _BILLABLE_NAME_PARTS)


def _unaccounted_usage_fields(usage: Any, accounted: set) -> str:
    """Any non-zero billable field in the vendor's usage object that the cost
    model does not price. Returns "" when the accounting is complete.

    This is the standing check behind the rule that no billed quantity may go
    untracked: vendors add usage fields (a new cache tier, a new server tool, a
    whole new unit of billing) without notice, and a field this project has
    never heard of would otherwise be silently worth zero. Flagged fields land
    in `untracked_usage` on the row and are printed once per run — loudly,
    because the cost in that column is then known to be too low.

    Lists are walked as well as objects: a breakdown a vendor sends as a list
    (Mistral's per-message prompt detail, Gemini's modality split) carries
    billable counts too. An entry's position is not part of its identity — the
    same class lands at a different index on the next call — so every entry
    collapses onto one `name[]` path that an accounted set can name once
    instead of once per index.
    """
    data = _usage_dict(usage)
    if data is None:
        return ""

    found: List[str] = []

    def walk(path: str, node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{path}.{key}" if path else str(key), value)
        elif isinstance(node, list):
            for entry in node:
                walk(f"{path}[]", entry)
        elif (isinstance(node, (int, float)) and not isinstance(node, bool)
                and node and path not in accounted and _is_billable_name(path)):
            found.append(f"{path}={node}")

    walk("", data)
    return ", ".join(sorted(found))


# Every usage path the Anthropic cost model reads and prices. `cache_creation.*`
# is the 5m/1h split of `cache_creation_input_tokens` — both are listed so the
# guard above does not report the same tokens twice, and only one of them is
# charged (see AnthropicProvider). `output_tokens_details.thinking_tokens` is a
# breakdown of `output_tokens`, already billed at the output rate.
# The SDK computes an expected duration of 3600s * max_tokens / 128_000 and
# refuses a non-streaming request whose result could exceed ten minutes. That
# puts the ceiling at 128_000 * 10/60 tokens; above it, stream.
ANTHROPIC_NONSTREAMING_MAX_TOKENS = 128_000 * 10 // 60

_ANTHROPIC_ACCOUNTED = {
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation.ephemeral_5m_input_tokens",
    "cache_creation.ephemeral_1h_input_tokens",
    "output_tokens_details.thinking_tokens",
    "server_tool_use.web_search_requests",
    "server_tool_use.web_fetch_requests",
}


# Every usage path documented for Chat usage at
# https://docs.mistral.ai/api/endpoint/chat, read off the schema the installed
# SDK generates from it (`UsageInfoDollarDefs`, mistralai 2.9.3; re-checked
# 2026-08-29). The schema is wider than any chat completion actually returns:
# against the live API on 2026-08-29 — plain, reasoning on, reasoning off, and
# a 15k-token prompt sent twice to provoke a cache hit — every response carried
# exactly `prompt_tokens`, `completion_tokens`, `total_tokens`,
# `prompt_tokens_details.cached_tokens` and `service_tier`, on the wire as well
# as through the SDK. The rest are read anyway, because a field that appears
# the day Mistral starts sending it must not be discovered by a wrong invoice:
#   * the cached share has four documented names, which are aliases rather than
#     four classes — `_mistral_cache_counts` reconciles them so the count is
#     priced exactly once, and the raw archive keeps every name;
#   * `completion_tokens_details.reasoning_tokens` is a breakdown of the
#     completion count, so it is informational here (`thinking_tokens`) and
#     billed at the output rate wherever it lands;
#   * `prompt_audio_seconds` and `…_details.audio_tokens` are billed at rates
#     this project has no figure for, so they raise explicit notes.
_MISTRAL_ACCOUNTED = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    # The cached share of the prompt, under all four names the schema uses.
    "prompt_tokens_details.cached_tokens",
    "prompt_token_details.cached_tokens",
    "num_cached_tokens",
    "cached_tokens",
    # A breakdown *of* completion_tokens: reasoning is inside the completion
    # count and billed at the output rate, never beside it. Priced already —
    # `thinking_tokens` carries it as information, not as a class of its own.
    "completion_tokens_details.reasoning_tokens",
    # A breakdown *of* prompt_tokens, one entry per message of the prompt.
    # `usage_count` says how often a message was reused, not what it cost.
    "prompt_tokens_details.messages[].total_tokens",
    "prompt_tokens_details.messages[].usage_count",
    "prompt_token_details.messages[].total_tokens",
    "prompt_token_details.messages[].usage_count",
    # Audio input. Both are billed, neither at the text rate this project
    # prices, so each raises its own explicit note in MistralProvider rather
    # than the generic "not priced" line — they are listed here so the same
    # quantity is not reported twice.
    "prompt_tokens_details.audio_tokens",
    "prompt_token_details.audio_tokens",
    "prompt_audio_seconds",
    # How many requests this usage object covers. 1 on a chat completion; a
    # higher figure means the row is not one call, which the provider says
    # explicitly.
    "request_count",
}

# Where a cached-prompt count can appear. Mistral's schema exposes the same
# quantity under four names — three of them nested, one at the top level. They
# are aliases, not four classes: `_mistral_cache_counts` reconciles them so the
# count is priced exactly once, and the raw archive keeps every name.
_MISTRAL_CACHE_PATHS = (
    ("prompt_tokens_details", "prompt_tokens_details.cached_tokens"),
    ("prompt_token_details", "prompt_token_details.cached_tokens"),
)
_MISTRAL_CACHE_TOP_LEVEL = ("num_cached_tokens", "cached_tokens")


def _mistral_cache_counts(usage: Any) -> Dict[str, int]:
    """Every cached-token alias actually present in a Mistral usage object.

    Reading the dumped object is intentional: the installed SDK declares only
    part of the current Chat usage schema and keeps newer fields as pydantic
    extras. Attribute access works for some SDK versions but silently misses
    those extras in others.
    """
    data = _usage_dict(usage) or {}
    counts: Dict[str, int] = {}
    for container_name, path in _MISTRAL_CACHE_PATHS:
        container = data.get(container_name)
        if (isinstance(container, dict)
                and container.get("cached_tokens") is not None):
            counts[path] = int(container["cached_tokens"])
    for name in _MISTRAL_CACHE_TOP_LEVEL:
        if data.get(name) is not None:
            counts[name] = int(data[name])
    return counts


def _mistral_detail(data: Dict[str, Any], field: str) -> Any:
    """One field of the prompt-detail object, under either of its two names.

    The schema spells the container `prompt_tokens_details` in some places and
    `prompt_token_details` in others. Whichever arrives, the field inside means
    the same thing, so both are read and the first one that carries a value
    wins — an absent container and an absent field are the same answer here.
    """
    for container_name in ("prompt_tokens_details", "prompt_token_details"):
        container = data.get(container_name)
        if isinstance(container, dict) and container.get(field) is not None:
            return container[field]
    return None


def _field(container: Any, name: str) -> int:
    """A token count out of a usage sub-object, whether it is a model or a dict.

    Vendors differ, and one vendor differs from itself: a field the SDK declares
    comes back as an object, an undeclared extra comes back as a plain dict.
    Attribute access alone would read the dict as zero.
    """
    if container is None:
        return 0
    if isinstance(container, dict):
        return int(container.get(name) or 0)
    return int(getattr(container, name, 0) or 0)


def _mistral_text(content: Any) -> Tuple[str, int]:
    """The answer text from a Mistral message, and how much thinking came with it.

    With `reasoning_effort` set, `message.content` is not a string but a list of
    chunks — a `ThinkChunk` carrying the reasoning followed by a `TextChunk`
    carrying the answer. Returning the list unchanged would put the model's
    internal monologue into the generated DOT file, so the thinking is dropped
    from the text and only measured.
    """
    if content is None:
        return "", 0
    if isinstance(content, str):
        return content, 0

    answer: List[str] = []
    thinking_chars = 0
    for chunk in content:
        kind = getattr(chunk, "type", "") or type(chunk).__name__
        if "think" in str(kind).lower():
            for part in (getattr(chunk, "thinking", None) or []):
                thinking_chars += len(str(getattr(part, "text", part) or ""))
            continue
        piece = getattr(chunk, "text", None)
        if piece:
            answer.append(str(piece))
    return "".join(answer), thinking_chars


# Every usage path the Google cost model reads and prices. Gemini's accounting
# differs from the other vendors in three ways that matter for cost:
#
#   * The REST contract defines `total_token_count` as prompt + candidates +
#     thoughts. Thinking tokens are NOT inside `candidates_token_count`, but
#     they are billed at the output rate — so they must be added, not merely
#     reported.
#   * `cached_content_token_count` is the cached share *of* prompt_token_count,
#     so the uncached remainder is the difference.
#   * `tool_use_prompt_token_count` is a tool-use prompt breakdown. Google's
#     current REST reference does not add it separately in the total formula,
#     while SDK schemas do. The provider uses the returned total to support both
#     shapes without double-counting it.
#
# The `*_details` entries are modality breakdowns of the counts above: Gemini
# sends each as a list of `{modality, token_count}`, and every entry is part of
# the count it breaks down, already priced. They are named here so the guard
# does not report the same tokens a second time — `[]` stands for any position
# in the list, because an entry's index is not part of its identity. The
# modality itself is checked separately by `_google_modality_warning`: Gemini
# charges more for audio input than for text, so *which* modality the tokens
# were is a cost fact the count alone does not carry.
_GOOGLE_ACCOUNTED = {
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "cached_content_token_count",
    "tool_use_prompt_token_count",
    "total_token_count",
    "prompt_tokens_details[].token_count",
    "candidates_tokens_details[].token_count",
    "cache_tokens_details[].token_count",
    "tool_use_prompt_tokens_details[].token_count",
}

# Input modalities billed at the plain text rate. Anything else on a Gemini call
# (audio above all) costs more per token, so the cost figure would be too low.
_GOOGLE_TEXT_MODALITIES = {"TEXT", "MODALITY_UNSPECIFIED"}


def _google_modality_warning(usage: Any) -> str:
    """Any input modality on this call that is not priced at the text rate."""
    offenders: List[str] = []
    for field in ("prompt_tokens_details", "cache_tokens_details",
                  "tool_use_prompt_tokens_details"):
        for entry in (getattr(usage, field, None) or []):
            modality = getattr(entry, "modality", None)
            name = getattr(modality, "value", None) or str(modality or "")
            tokens = getattr(entry, "token_count", 0) or 0
            if tokens and name.upper() not in _GOOGLE_TEXT_MODALITIES:
                offenders.append(f"{field}:{name}={tokens}")
    return ", ".join(sorted(offenders))


def _retry_loop(call, max_retries: int, is_retryable):
    """Shared retry-with-backoff scaffold. `call()` performs one attempt;
    `is_retryable(exc)` decides whether to back off and retry or re-raise.

    Returns `(result, CallTiming)`. The clock is started per attempt rather
    than around the loop, so the reported round trip belongs to the request
    that actually answered — a rate-limit retry inflates `total_s`, not
    `api_latency_s`."""
    last_exc: Optional[Exception] = None
    loop_start: Optional[float] = None
    retry_wait = 0.0
    for attempt in range(max_retries):
        sent_at = datetime.now().astimezone()
        t0 = time.monotonic()
        if loop_start is None:
            # Anchored on the first attempt's clock, not on entry, so that a
            # single-attempt call has total_s == api_latency_s exactly rather
            # than differing by the bookkeeping above.
            loop_start = t0
        try:
            result = call()
        except Exception as exc:  # noqa: BLE001 — narrowed per-vendor by is_retryable
            last_exc = exc
            if not is_retryable(exc) or attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            retry_wait += wait
            time.sleep(wait)
            continue
        t1 = time.monotonic()
        received_at = datetime.now().astimezone()
        return result, CallTiming(
            api_latency_s=t1 - t0,
            total_s=t1 - loop_start,
            attempts=attempt + 1,
            retry_wait_s=retry_wait,
            request_sent_at=sent_at.isoformat(timespec="milliseconds"),
            response_received_at=received_at.isoformat(timespec="milliseconds"),
        )
    raise last_exc  # pragma: no cover


# ══════════════════════════════════════════════════════════════════════════
# Anthropic — verified
# ══════════════════════════════════════════════════════════════════════════
class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._max_retries = 5

    def generate(self, model: cfg.ModelSpec, system, messages: List[Dict[str, str]],
                 settings: cfg.RunSettings) -> GenerationResult:
        import anthropic

        kwargs = dict(model=model.model_id, max_tokens=settings.max_output_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        if model.thinking:
            # Opus 5: adaptive thinking only — `budget_tokens` is rejected with a
            # 400 on this generation, and sampling params are rejected outright.
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": model.effort or "high"}
        elif model.supports_temperature and settings.temperature is not None:
            kwargs["temperature"] = settings.temperature

        def is_retryable(exc):
            return isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)) or (
                isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500
            )

        # Above ~21k output tokens the SDK refuses a non-streaming request
        # outright — it computes an expected duration of 3600s * max_tokens /
        # 128_000 and will not wait more than ten minutes for one response. The
        # reasoning tiers need a budget well past that, so a large request is
        # streamed and reassembled. `get_final_message()` returns the same
        # Message object `create()` would have, so everything below is unchanged.
        if settings.max_output_tokens > ANTHROPIC_NONSTREAMING_MAX_TOKENS:
            def send():
                with self._client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
                    # A streamed message carries no `_request_id`; the header is
                    # on the HTTP response the stream was read from.
                    headers = getattr(getattr(stream, "response", None), "headers", None)
                    if headers is not None and not getattr(message, "_request_id", None):
                        try:
                            message._request_id = headers.get("request-id") or ""
                        except Exception:  # noqa: BLE001 — id is nice to have, not required
                            pass
                    return message
        else:
            def send():
                return self._client.messages.create(**kwargs)

        response, timing = _retry_loop(
            send, self._max_retries, is_retryable
        )

        if response.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")

        usage = response.usage
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Cache writes: the API reports a flat total and, when it can, the split
        # by TTL — which matters, because the two TTLs bill at different
        # multipliers (1.25x vs 2x input). Prefer the split; fall back to the
        # flat total charged at the 5-minute rate, which is the default TTL and
        # the cheaper of the two — noted in untracked_usage so an unsplit 1-hour
        # write is never quietly under-charged.
        cache_creation = getattr(usage, "cache_creation", None)
        write_total = getattr(usage, "cache_creation_input_tokens", 0) or 0
        if cache_creation is not None:
            write_5m = getattr(cache_creation, "ephemeral_5m_input_tokens", 0) or 0
            write_1h = getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0
        else:
            write_5m, write_1h = write_total, 0

        details = getattr(usage, "output_tokens_details", None)
        thinking_tokens = getattr(details, "thinking_tokens", None) if details else None
        server_tools = getattr(usage, "server_tool_use", None)
        web_searches = getattr(server_tools, "web_search_requests", 0) or 0 if server_tools else 0
        service_tier = getattr(usage, "service_tier", None) or "standard"

        token_usage = cfg.TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cached,
            cache_write_tokens=write_5m, cache_write_1h_tokens=write_1h,
            web_search_requests=web_searches, batch=(service_tier == "batch"),
        )
        cost, cost_basis = cfg.cost_usd(model, token_usage)

        untracked = ", ".join(filter(None, [
            _unaccounted_usage_fields(usage, _ANTHROPIC_ACCOUNTED),
            _service_tier_note(service_tier),
        ]))
        # Anthropic states the writes total itself, so the split can be checked
        # against it rather than trusted. A mismatch means this response carries
        # a cache tier beyond the two TTLs — which is exactly how the 1-hour tier
        # arrived — and the cost below would then be missing it. Only reachable
        # when a split object exists; the unsplit case is the branch below.
        if cache_creation is not None and write_total \
                and write_total != write_5m + write_1h:
            untracked = ", ".join(filter(None, [
                untracked,
                f"cache write split does not reconcile: Anthropic reports "
                f"cache_creation_input_tokens={write_total}, "
                f"ephemeral_5m={write_5m} + ephemeral_1h={write_1h} = "
                f"{write_5m + write_1h}",
            ]))
        if cache_creation is None and write_total:
            untracked = ", ".join(filter(None, [
                untracked,
                f"cache_creation_input_tokens={write_total} charged at the 5m rate "
                f"(no TTL split reported; a 1h write would cost 1.6x more)",
            ]))

        if model.thinking:
            reasoning_note = (
                f"thinking enabled (adaptive, effort={model.effort or 'high'}); "
                f"thinking_tokens={thinking_tokens}, billed inside output_tokens per "
                "usage.output_tokens_details"
                if thinking_tokens is not None else
                f"thinking enabled (adaptive, effort={model.effort or 'high'}); no "
                "thinking_tokens in usage.output_tokens_details for this response — "
                "the tokens are still billed as part of output_tokens"
            )
        else:
            reasoning_note = ""

        return GenerationResult(
            text=text, reported_input_tokens=input_tokens,
            reported_output_tokens=output_tokens,
            reported_cache_creation_tokens=write_total, raw_usage=_usage_json(usage),
            # No reported_total_tokens: the Messages usage object states no
            # grand total, so there is nothing to keep or to check against.
            input_tokens=input_tokens, output_tokens=output_tokens,
            cached_tokens=cached,
            cache_write_tokens=write_5m, cache_write_1h_tokens=write_1h,
            billable_input_tokens=token_usage.billable_input_tokens,
            thinking_tokens=thinking_tokens, web_search_requests=web_searches,
            service_tier=service_tier, total_tokens=token_usage.total_tokens,
            cost_usd=cost, cost_basis=cost_basis, untracked_usage=untracked,
            **timing.as_fields(), **_request_ids(response),
            stop_reason=response.stop_reason, token_reporting="reported", tokens_estimated=False,
            reasoning_note=reasoning_note,
        )

    def check_connection(self, model_id: str):
        import anthropic
        try:
            self._client.models.retrieve(model_id)
            return True, "ok"
        except anthropic.AuthenticationError as exc:
            return False, f"authentication failed — check the API key ({exc})"
        except anthropic.NotFoundError:
            return False, f"model id {model_id!r} not found — check config.py"
        except anthropic.APIConnectionError as exc:
            return False, f"connection error — {exc}"
        except anthropic.APIStatusError as exc:
            return False, f"API error {exc.status_code} — {exc.message}"
        except Exception as exc:  # noqa: BLE001
            return False, f"unexpected error — {exc}"


# ══════════════════════════════════════════════════════════════════════════
# OpenAI — verified against the SDK and the current docs (2026-08-17)
# ══════════════════════════════════════════════════════════════════════════
# Built on the **Responses API** (`client.responses.create`), not Chat
# Completions, for two reasons that both matter to this thesis:
#
#  1. **Cost completeness.** OpenAI bills cache writes at 1.25x the uncached
#     input rate, and only the Responses usage object reports them
#     (`input_tokens_details.cache_write_tokens`). Chat Completions reports
#     `prompt_tokens_details.cached_tokens` alone, so a run on that endpoint
#     would pay for writes it cannot see — exactly the untracked cost this
#     project forbids.
#  2. **Comparability.** OpenAI documents the Responses API as the recommended
#     endpoint for reasoning models, and states that Chat Completions can yield
#     lower model performance. Measuring gpt-5.6 through the weaker endpoint
#     would understate it against Claude.
#
# Verified against openai 3.2.0: `responses.create(model, input, instructions,
# max_output_tokens, reasoning, temperature, store, stream)`, `response.status`
# / `.incomplete_details.reason` / `.output_text`, and a usage object of exactly
# {input_tokens, input_tokens_details{cached_tokens, cache_write_tokens},
#  output_tokens, output_tokens_details{reasoning_tokens}, total_tokens}.
_OPENAI_ACCOUNTED = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_tokens_details.cached_tokens",
    "input_tokens_details.cache_write_tokens",
    "output_tokens_details.reasoning_tokens",
}

# Every usage path the DeepSeek cost model reads and prices — and nothing else.
# DeepSeek speaks OpenAI's dialect but its usage object is its own: it replaces
# OpenAI's `prompt_tokens_details` with `prompt_cache_hit_tokens` /
# `prompt_cache_miss_tokens`, its authoritative split of the prompt (both listed
# so the guard does not report the same prompt twice under two names).
#
# ⚠ `prompt_tokens_details.*` is deliberately NOT listed. DeepSeek documents no
# such object and charges nothing for cache writes — only hits and misses (API
# reference, checked 2026-08-20). Listing it would silence the guard for a
# field this vendor is not known to have; leaving it out means that if one ever
# appears, it is reported instead of quietly priced at an invented rate.
_DEEPSEEK_ACCOUNTED = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    # The same cached tokens a second time, under OpenAI's name: DeepSeek
    # answers in both dialects, and a live run returns
    # `prompt_tokens_details.cached_tokens == prompt_cache_hit_tokens` on every
    # call that hit the cache. They are priced once, as cache reads off the
    # hit/miss pair — listing the alias here is what stops the guard reporting
    # tokens that *are* accounted for. Silence is not blind: a disagreement
    # between the two names is a note of its own (see DeepSeekProvider).
    "prompt_tokens_details.cached_tokens",
    "completion_tokens_details.reasoning_tokens",
}

# `prompt_tokens_details.cache_write_tokens` is deliberately absent: DeepSeek
# bills hits and misses only, so a write count arriving there is a change in
# the billing model and must be flagged, not priced.


def _deepseek_alias_cached(usage: Any) -> Optional[int]:
    """`prompt_tokens_details.cached_tokens`, or None if the call did not carry
    it. None and 0 are different findings — absent field versus reported zero.
    """
    container = getattr(usage, "prompt_tokens_details", None) if usage else None
    if container is None:
        return None
    value = (container.get("cached_tokens") if isinstance(container, dict)
             else getattr(container, "cached_tokens", None))
    return None if value is None else int(value)


def _is_retryable_openai_error(exc) -> bool:
    """Retry policy for anything speaking the OpenAI dialect — the SDK itself
    and DeepSeek through it.

    A 429 is usually "slow down", but it is also what an exhausted balance
    returns, and that never clears by waiting.
    """
    import openai
    if isinstance(exc, openai.RateLimitError):
        return not _is_permanent_quota_error(exc)
    if isinstance(exc, openai.APIConnectionError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: Optional[str] = None):
        import openai
        self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        self._max_retries = 5

    def generate(self, model: cfg.ModelSpec, system, messages: List[Dict[str, str]],
                 settings: cfg.RunSettings) -> GenerationResult:
        import openai

        # The Responses API takes the transcript as `input` (role/content items,
        # assistant turns included — that is what the few-shot strategies need)
        # and the system prompt as the separate `instructions` field.
        kwargs: Dict[str, Any] = dict(
            model=model.model_id,
            input=list(messages),
            max_output_tokens=settings.max_output_tokens,
        )
        if system:
            kwargs["instructions"] = system
        if model.thinking:
            # effort: none | low | medium (default) | high | xhigh | max
            kwargs["reasoning"] = {"effort": model.effort or "high"}
        elif model.supports_temperature and settings.temperature is not None:
            kwargs["temperature"] = settings.temperature

        def is_retryable(exc):
            # A 429 is usually "slow down", but OpenAI also returns it for an
            # exhausted balance — a permanent condition. Retrying that burns
            # ~15s of backoff per call and, over a full run, hours of nothing.
            if isinstance(exc, openai.RateLimitError):
                return not _is_permanent_quota_error(exc)
            if isinstance(exc, openai.APIConnectionError):
                return True
            return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500

        response, timing = _retry_loop(
            lambda: self._client.responses.create(**kwargs), self._max_retries, is_retryable
        )

        text = getattr(response, "output_text", "") or ""

        # The Responses API reports the tier that served the call on the response
        # itself, not in usage — `default` unless one was requested, which this
        # project never does. Read so the row states it and so a `flex`,
        # `priority` or `scale` call cannot be priced as if it were standard.
        service_tier = getattr(response, "service_tier", None) or "default"

        usage = response.usage
        reported_input = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        in_details = getattr(usage, "input_tokens_details", None)
        out_details = getattr(usage, "output_tokens_details", None)
        cached = getattr(in_details, "cached_tokens", 0) or 0 if in_details else 0
        cache_write = getattr(in_details, "cache_write_tokens", 0) or 0 if in_details else 0
        reasoning_tokens = getattr(out_details, "reasoning_tokens", None) if out_details else None

        # `input_tokens` is the whole prompt **including** its cached and
        # written parts (unlike Anthropic, where it is the uncached remainder).
        # Splitting it here keeps the two vendors' columns meaning the same
        # thing and stops the same tokens being priced twice.
        uncached = reported_input - cached - cache_write
        reconciles = uncached >= 0
        if not reconciles:
            # The split does not fit inside the reported total: rather than
            # silently pick one reading, price the safer (larger) one and say so.
            uncached = reported_input

        token_usage = cfg.TokenUsage(
            input_tokens=uncached, output_tokens=output_tokens,
            cache_read_tokens=cached, cache_write_tokens=cache_write,
        )
        cost, cost_basis = cfg.cost_usd(model, token_usage)

        untracked = ", ".join(filter(None, [
            _unaccounted_usage_fields(usage, _OPENAI_ACCOUNTED),
            _service_tier_note(service_tier),
        ]))
        if not reconciles:
            untracked = ", ".join(filter(None, [
                untracked,
                f"cached({cached}) + cache_write({cache_write}) exceed "
                f"input_tokens({reported_input}); priced the full input at the "
                f"uncached rate, cost may be overstated",
            ]))

        # OpenAI states the total itself, so it can be checked rather than
        # trusted: input + output must account for it. A mismatch means this
        # response carries a token class the split above does not cover — the
        # same guard Mistral, DeepSeek and Gemini already get. Skipped when the
        # cache split already failed, so one anomaly is not reported twice.
        reported_total = getattr(usage, "total_tokens", 0) or 0
        if reconciles and reported_total and reported_total != reported_input + output_tokens:
            untracked = ", ".join(filter(None, [
                untracked,
                f"token split does not reconcile: OpenAI reports "
                f"total_tokens={reported_total}, input_tokens={reported_input} + "
                f"output_tokens={output_tokens} = {reported_input + output_tokens}",
            ]))

        if model.thinking:
            reasoning_note = (
                f"reasoning effort={model.effort or 'high'}; reasoning_tokens="
                f"{reasoning_tokens}, billed inside output_tokens per "
                "usage.output_tokens_details"
                if reasoning_tokens is not None else
                f"reasoning effort={model.effort or 'high'}; no reasoning_tokens in "
                "usage.output_tokens_details for this response — check the SDK version"
            )
        else:
            reasoning_note = ("gpt-5.6 reasons by default at effort 'medium'; this tier "
                              "sends no reasoning parameter, so that default applies")

        # Responses reports completion state as status + incomplete_details,
        # where the Messages API uses stop_reason. Map onto the same column.
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        stop_reason = getattr(incomplete, "reason", None) if incomplete else None
        stop_reason = stop_reason or ("end_turn" if status == "completed" else str(status))

        return GenerationResult(
            text=text, reported_input_tokens=reported_input,
            reported_total_tokens=reported_total, reported_output_tokens=output_tokens,
            raw_usage=_usage_json(usage), service_tier=service_tier,
            input_tokens=uncached, output_tokens=output_tokens,
            cached_tokens=cached, cache_write_tokens=cache_write,
            billable_input_tokens=token_usage.billable_input_tokens,
            thinking_tokens=reasoning_tokens, total_tokens=token_usage.total_tokens,
            cost_usd=cost, cost_basis=cost_basis, untracked_usage=untracked,
            **timing.as_fields(), **_request_ids(response),
            stop_reason=stop_reason, token_reporting="reported",
            tokens_estimated=False, reasoning_note=reasoning_note,
        )

    def check_connection(self, model_id: str):
        import openai
        try:
            self._client.models.retrieve(model_id)
            return True, "ok"
        except openai.AuthenticationError as exc:
            return False, f"authentication failed — check the API key ({exc})"
        except openai.NotFoundError:
            return False, f"model id {model_id!r} not found — check config.py"
        except openai.APIConnectionError as exc:
            return False, f"connection error — {exc}"
        except openai.APIStatusError as exc:
            return False, f"API error {exc.status_code} — {exc.message}"
        except Exception as exc:  # noqa: BLE001
            return False, f"unexpected error — {exc}"


# ══════════════════════════════════════════════════════════════════════════
# Mistral — usage verified against the Chat endpoint docs (2026-08-20)
# ══════════════════════════════════════════════════════════════════════════
class MistralProvider(Provider):
    name = "mistral"

    def __init__(self, api_key: Optional[str] = None):
        try:
            from mistralai import Mistral  # newer SDK versions re-export here
        except ImportError:
            from mistralai.client import Mistral  # mistralai 2.x — not re-exported at top level
        self._client = Mistral(api_key=api_key) if api_key else Mistral()
        self._max_retries = 5

    def generate(self, model: cfg.ModelSpec, system, messages: List[Dict[str, str]],
                 settings: cfg.RunSettings) -> GenerationResult:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        kwargs = dict(model=model.model_id, messages=chat_messages,
                      max_tokens=settings.max_output_tokens)
        if model.thinking:
            # Mistral's reasoning switch. `prompt_mode="reasoning"` is the other
            # one, but Medium 3.5 rejects it with a 400 ("not enabled for this
            # model"), so effort is what this tier uses.
            kwargs["reasoning_effort"] = model.effort or "high"
        else:
            if model.reasoning_off:
                # Small 4 is reasoning-capable and documents
                # `reasoning_effort="none"` as "the model thinks minimally and
                # the thinking chunk is omitted". What happens when the
                # parameter is omitted entirely is NOT documented, so a tier
                # that is meant not to reason says so rather than inheriting an
                # undocumented default — the same trap DeepSeek's toggle was.
                kwargs["reasoning_effort"] = "none"
            if model.supports_temperature and settings.temperature is not None:
                kwargs["temperature"] = settings.temperature

        response, timing = _retry_loop(
            lambda: self._client.chat.complete(**kwargs), self._max_retries,
            self._is_retryable,
        )

        choice = response.choices[0]
        text, thinking_chars = _mistral_text(choice.message.content)
        usage = response.usage

        # Read the dumped object rather than attributes: the installed SDK
        # declares only five of the schema's usage fields and keeps everything
        # else as pydantic extras, which attribute access misses on some SDK
        # versions. `_usage_dict` sees declared fields and extras alike, so a
        # field Mistral adds tomorrow is read by the same code as the rest.
        usage_data = _usage_dict(usage) or {}

        def count(field: str) -> int:
            return int(usage_data.get(field) or 0)

        prompt_tokens = count("prompt_tokens")
        output_tokens = count("completion_tokens")
        # The Chat schema documents four names for the same cached prompt
        # count. Preserve every one in raw_usage/results.html, then use one
        # reconciled value here so aliases can never be billed twice.
        cache_counts = _mistral_cache_counts(usage)
        cache_values = set(cache_counts.values())
        cache_aliases_reconcile = len(cache_values) <= 1
        reported_cached = next(iter(cache_counts.values()), 0)
        cache_fits_prompt = 0 <= reported_cached <= prompt_tokens
        cached = (reported_cached
                  if cache_aliases_reconcile and cache_fits_prompt else 0)

        # Mistral follows the OpenAI convention: prompt_tokens is the whole
        # prompt and cached_tokens is the share of it served from cache, so the
        # uncached remainder is the difference. Counting both would bill twice.
        # If aliases disagree, or a cache count cannot fit inside the prompt,
        # price the full prompt as uncached. That is the conservative result:
        # cached input is cheaper, so guessing a cached share could understate
        # the charge.
        input_tokens = prompt_tokens - cached
        service_tier = str(usage_data.get("service_tier") or "standard")

        # The quantities the schema documents beside the token counts. None of
        # them has been observed on a chat completion; each is read so that the
        # first response to carry one is measured rather than missed.
        # Reasoning is a breakdown *of* the completion count — informational,
        # already billed at the output rate. None (not 0) when absent: a vendor
        # that states no figure has not measured zero thinking.
        completion_details = usage_data.get("completion_tokens_details")
        reasoning_tokens = (
            int(completion_details["reasoning_tokens"])
            if isinstance(completion_details, dict)
            and completion_details.get("reasoning_tokens") is not None else None)
        audio_seconds = float(usage_data.get("prompt_audio_seconds") or 0.0)
        audio_tokens = int(_mistral_detail(usage_data, "audio_tokens") or 0)
        request_count = usage_data.get("request_count")

        token_usage = cfg.TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cached, batch=service_tier == "batch",
        )
        cost, cost_basis = cfg.cost_usd(model, token_usage)

        notes = []
        untracked = _unaccounted_usage_fields(usage, _MISTRAL_ACCOUNTED)
        if untracked:
            notes.append(f"usage fields not priced: {untracked}")
        if not cache_aliases_reconcile:
            aliases = ", ".join(f"{path}={value}"
                                for path, value in cache_counts.items())
            notes.append(
                f"cached-token aliases do not reconcile: {aliases}; priced the "
                f"full prompt as uncached, which may overstate the cost")
        elif not cache_fits_prompt:
            notes.append(
                f"cached token count {reported_cached} does not fit inside "
                f"prompt_tokens={prompt_tokens}; priced the full prompt as "
                f"uncached, which may overstate the cost")
        if audio_seconds:
            notes.append(
                f"prompt_audio_seconds={audio_seconds}: audio input is billed by "
                f"duration at a rate this cost model has no figure for, so "
                f"cost_usd is a floor rather than the amount charged")
        if audio_tokens:
            notes.append(
                f"audio_tokens={audio_tokens} of the prompt were audio, priced "
                f"here at the text input rate — Mistral charges audio input at "
                f"its own rate, so cost_usd is understated")
        if request_count is not None and int(request_count) > 1:
            notes.append(
                f"request_count={int(request_count)}: this usage object covers "
                f"more than one request, so the row's token counts and cost are "
                f"not those of a single call")
        tier_note = _service_tier_note(service_tier)
        if tier_note:
            notes.append(tier_note)
        reported_total = count("total_tokens")
        if reported_total and reported_total != prompt_tokens + output_tokens:
            notes.append(
                f"token split does not reconcile: Mistral reports "
                f"total_tokens={reported_total}, prompt_tokens={prompt_tokens} + "
                f"completion_tokens={output_tokens} = {prompt_tokens + output_tokens}")

        if model.thinking:
            # The schema documents `completion_tokens_details.reasoning_tokens`,
            # but no chat completion observed here has carried it (checked live
            # on 2026-08-29 with reasoning_effort=high). Say which of the two
            # happened on *this* call rather than asserting either as a
            # standing fact about the vendor: the thinking is inside
            # completion_tokens and billed at the output rate either way, and
            # without the figure its share is only bounded by the characters of
            # thinking text that came back.
            reasoning_note = (
                f"reasoning_effort={model.effort or 'high'}; Mistral reported "
                f"completion_tokens_details.reasoning_tokens={reasoning_tokens}, "
                f"billed inside completion_tokens at the output rate"
                if reasoning_tokens is not None else
                f"reasoning_effort={model.effort or 'high'}; the Chat schema "
                f"documents completion_tokens_details.reasoning_tokens but this "
                f"response carried none, so thinking is billed inside "
                f"completion_tokens without a separate figure "
                f"({thinking_chars} characters of thinking text returned)"
            )
        elif model.reasoning_off:
            reasoning_note = (
                'reasoning_effort="none" — this model is reasoning-capable and '
                "Mistral documents no default for an omitted parameter, so the "
                "tier asks for no thinking explicitly"
                + (f"; {thinking_chars} characters of thinking text came back "
                   "anyway, billed inside completion_tokens" if thinking_chars else "")
            )
        else:
            reasoning_note = ""

        return GenerationResult(
            text=text, reported_input_tokens=prompt_tokens,
            reported_total_tokens=reported_total, reported_output_tokens=output_tokens,
            raw_usage=_usage_json(usage),
            input_tokens=input_tokens, output_tokens=output_tokens,
            cached_tokens=cached,
            billable_input_tokens=token_usage.billable_input_tokens,
            thinking_tokens=reasoning_tokens,   # None unless the vendor states one
            total_tokens=token_usage.total_tokens,
            audio_input_seconds=audio_seconds,
            service_tier=service_tier,
            cost_usd=cost, cost_basis=cost_basis, untracked_usage="; ".join(notes),
            **timing.as_fields(), **_request_ids(response),
            stop_reason=getattr(choice, "finish_reason", None), token_reporting="reported",
            tokens_estimated=False, reasoning_note=reasoning_note,
        )

    @staticmethod
    def _is_retryable(exc) -> bool:
        """Retry rate limits and server faults; give up on a rejected request.

        A 400 ("Reasoning prompt mode is not enabled for this model") or a 401
        will fail identically on every attempt — retrying only burns time.
        """
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(code, int):
            return code == 429 or code >= 500
        text = str(exc).lower()
        if any(m in text for m in ("status 400", "status 401", "status 403", "status 404")):
            return False
        return True

    def check_connection(self, model_id: str):
        # mistralai's exception hierarchy isn't verified in this session —
        # report whatever comes back rather than assuming specific classes.
        try:
            self._client.models.retrieve(model_id=model_id)
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, f"error — {exc}"


# ══════════════════════════════════════════════════════════════════════════
# DeepSeek — live, prices verified 2026-08-18
# ══════════════════════════════════════════════════════════════════════════
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(Provider):
    """DeepSeek speaks the OpenAI Chat Completions dialect, so it rides the
    `openai` SDK with a different base URL — but its accounting is its own.

    Three departures from OpenAI's shape decide how the cost is computed:

      * `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` are DeepSeek's
        own split of the prompt, and the only one it documents. There is no
        `prompt_tokens_details` object and no charge for cache writes.
      * A cache hit costs about a thirtieth of a miss, not the usual tenth
        ($0.014 against $0.44 per Mtok on Flash, $0.044 against $1.32 on Pro,
        peak rates as published on 2026-08-20), so misreading the split is
        expensive in a way it is not elsewhere. ⚠ These rates are four days
        old. The peak/off-peak table is what api-docs.deepseek.com/quick_start/
        pricing states verbatim on 2026-08-20 — the strings 0.14, 0.0028, 0.28
        and 0.435 do not appear on it. The sheet it replaced was flat
        ($0.14/$0.0028/$0.28 on Flash, $0.435/$0.003625/$0.87 on Pro) and is
        still what many third-party trackers show; the changeover date,
        2026-08-16 16:00 UTC, comes from secondary sources rather than from the
        page itself. Anything measured before that moment was billed flat and
        cannot be re-priced with the numbers in config.py.
      * Both models reason by default at effort `high`. Thinking is switched
        with the `thinking` object, NOT with `reasoning_effort`, whose accepted
        values are `low | high | max` — DeepSeek ignores what it does not
        recognise instead of erroring, so `reasoning_effort="none"` would leave
        a tier reasoning while the row claimed otherwise.

    Request shape and usage fields verified against the API reference and the
    Thinking Mode guide, 2026-08-20.
    """
    name = "deepseek"

    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self._max_retries = 5

    def generate(self, model: cfg.ModelSpec, system, messages: List[Dict[str, str]],
                 settings: cfg.RunSettings) -> GenerationResult:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        kwargs: Dict[str, Any] = dict(
            model=model.model_id, messages=chat_messages,
            max_tokens=settings.max_output_tokens,
        )
        # Thinking is switched with the `thinking` object, which the OpenAI SDK
        # can only pass through `extra_body` — it is not part of OpenAI's own
        # schema. `reasoning_effort` is, so it stays a top-level argument; that
        # split is exactly what DeepSeek's own OpenAI-SDK example shows.
        #
        # ⚠ `reasoning_effort="none"`, the shape this used until 2026-08-20, is
        # NOT one of DeepSeek's accepted values (`low | high | max`), and this
        # API ignores what it does not recognise instead of erroring — so the
        # tier meant to have reasoning off kept reasoning and kept billing it.
        if model.reasoning_off:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        elif model.thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = model.effort or "high"
        # Temperature only reaches a model that is not thinking. DeepSeek states
        # that temperature has no effect in thinking mode and ignores it without
        # erroring — sending it anyway would put a sampling setting in the
        # manifest that the run never actually had.
        if not model.thinking and model.supports_temperature \
                and settings.temperature is not None:
            kwargs["temperature"] = settings.temperature

        response, timing = _retry_loop(
            lambda: self._client.chat.completions.create(**kwargs),
            self._max_retries, _is_retryable_openai_error,
        )

        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        usage = response.usage

        def count(field: str) -> int:
            return getattr(usage, field, 0) or 0 if usage else 0

        prompt_tokens = count("prompt_tokens")
        output_tokens = count("completion_tokens")
        hit = count("prompt_cache_hit_tokens")
        miss = count("prompt_cache_miss_tokens")

        # The hit/miss pair is the whole prompt: miss is the uncached part, hit
        # the cached one, and DeepSeek defines prompt_tokens as their sum.
        # Reading anything else would be guessing at a field this API does not
        # document. If the pair is missing altogether, the prompt is priced as
        # entirely uncached — a miss costs ~30x a hit, so that is the direction
        # that overstates the cost rather than hiding it, and the row says so.
        if hit or miss:
            cached, input_tokens = hit, miss
        elif prompt_tokens:
            cached, input_tokens = 0, prompt_tokens
        else:
            cached, input_tokens = 0, 0

        # None and 0 are different findings: 0 means DeepSeek reported no
        # reasoning on this call, None that it reported no such field at all.
        # Collapsing them would make "the switch worked" indistinguishable from
        # "the API said nothing" — the very thing the 2026-08-20 toggle bug hid.
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning_tokens = _field(details, "reasoning_tokens") if details is not None else None

        # No cache_write class: DeepSeek charges for hits and misses only.
        token_usage = cfg.TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cached,
        )
        cost, cost_basis = cfg.cost_usd(model, token_usage)

        notes = []
        untracked = _unaccounted_usage_fields(usage, _DEEPSEEK_ACCOUNTED)
        if untracked:
            notes.append(f"usage fields not priced: {untracked}")
        if prompt_tokens and not (hit or miss):
            notes.append(
                f"no cache split reported: prompt_cache_hit_tokens and "
                f"prompt_cache_miss_tokens are both absent or zero while "
                f"prompt_tokens={prompt_tokens} — priced the full prompt as "
                f"uncached, which may overstate the cost")
        elif prompt_tokens and hit + miss != prompt_tokens:
            notes.append(
                f"cache split does not reconcile: prompt_cache_hit_tokens={hit} + "
                f"prompt_cache_miss_tokens={miss} != prompt_tokens={prompt_tokens}")
        # The OpenAI-dialect alias of the same figure. Equal is the normal case
        # and says nothing; a disagreement means one of the two names no longer
        # describes what this cost model prices.
        alias_cached = _deepseek_alias_cached(usage)
        if alias_cached is not None and alias_cached != hit:
            notes.append(
                f"cached-token names disagree: prompt_tokens_details.cached_tokens="
                f"{alias_cached} != prompt_cache_hit_tokens={hit} — priced "
                f"DeepSeek's own hit/miss split")
        reported_total = count("total_tokens")
        if reported_total and reported_total != prompt_tokens + output_tokens:
            notes.append(
                f"token split does not reconcile: DeepSeek reports "
                f"total_tokens={reported_total}, prompt_tokens={prompt_tokens} + "
                f"completion_tokens={output_tokens} = {prompt_tokens + output_tokens}")

        if reasoning_tokens is None:
            counted = ("no reasoning-token count in completion_tokens_details for "
                       "this response — any thinking is still billed inside "
                       "completion_tokens")
        else:
            counted = (f"reasoning_tokens={reasoning_tokens}, billed inside "
                       "completion_tokens and returned separately in "
                       "message.reasoning_content")

        if model.reasoning_off:
            # Whether the switch actually took effect is only visible in the
            # usage object, so the note says which of the three it was: no
            # figure at all, a confirmed zero, or thinking that was billed
            # anyway. The distinction is the check on this vendor's off switch.
            if reasoning_tokens is None:
                outcome = ("; no reasoning-token count in completion_tokens_details "
                           "for this response, so the switch cannot be confirmed "
                           "from usage")
            elif reasoning_tokens:
                outcome = (f"; residual reasoning_tokens={reasoning_tokens} were "
                           "still billed")
            else:
                outcome = "; reasoning_tokens=0 confirms it took effect"
            reasoning_note = (
                'thinking={"type": "disabled"} — DeepSeek reasons by default, so '
                "this tier switches it off explicitly" + outcome
            )
        else:
            reasoning_note = (
                f'thinking={{"type": "enabled"}} with reasoning_effort='
                f'"{model.effort or "high"}"; {counted}'
            )

        return GenerationResult(
            text=text, reported_input_tokens=prompt_tokens,
            reported_total_tokens=reported_total, reported_output_tokens=output_tokens,
            raw_usage=_usage_json(usage),
            input_tokens=input_tokens, output_tokens=output_tokens,
            cached_tokens=cached,
            billable_input_tokens=token_usage.billable_input_tokens,
            thinking_tokens=reasoning_tokens,   # 0 is a measurement, None is silence
            total_tokens=token_usage.total_tokens,
            cost_usd=cost, cost_basis=cost_basis, untracked_usage="; ".join(notes),
            **timing.as_fields(), **_request_ids(response),
            stop_reason=getattr(choice, "finish_reason", None),
            token_reporting="reported", tokens_estimated=False,
            reasoning_note=reasoning_note,
        )

    def check_connection(self, model_id: str):
        try:
            self._client.models.retrieve(model_id)
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, f"error — {exc}"


# ══════════════════════════════════════════════════════════════════════════
# Google (Gemini) — live, prices verified 2026-08-20
# ══════════════════════════════════════════════════════════════════════════
# The only SDK here that ships **no default timeout**: `HttpOptions.timeout` is
# None, so a stalled response leaves the request open indefinitely — nothing
# raises, `_retry_loop` never gets its exception, and the run sits on one call
# for as long as the socket stays up.
#
# **120 s, not the 600 s the anthropic and openai clients use**, because this
# endpoint fails in two different ways. Under load Gemini answers `503
# UNAVAILABLE — "this model is currently experiencing high demand"` in about
# two seconds, which the retry loop handles; but the same overloaded model also
# just stops answering, and then the deadline is the only thing that ends the
# attempt. Measured on 2026-08-29 against an overloaded `gemini-3.7-flash`: a
# one-word prompt returned 503 in 1.7 s on one attempt and stalled into a
# 504 DEADLINE_EXCEEDED at 59 s on the next, while `gemini-3.6-flash` answered
# in 2.1 s throughout. A stall must therefore cost a fraction of a run, not ten
# minutes of it: no call in this project has legitimately needed two minutes,
# and the deadline error is retryable (`GoogleProvider._is_retryable`), so a
# stalled attempt is cut and retried rather than waited out.
GOOGLE_TIMEOUT_MS = 120_000


class GoogleProvider(Provider):
    name = "google"

    def __init__(self, api_key: Optional[str] = None):
        from google import genai
        from google.genai import types

        http = types.HttpOptions(timeout=GOOGLE_TIMEOUT_MS)
        self._client = (genai.Client(api_key=api_key, http_options=http) if api_key
                        else genai.Client(http_options=http))
        self._max_retries = 5

    @staticmethod
    def _is_retryable(exc) -> bool:
        """Retry the transient Gemini failures, give up on the rest.

        503 UNAVAILABLE ("high demand") is common enough on the Flash models to
        be the main reason this exists; 429 and 5xx are the usual suspects.
        A 400 or 404 is a broken request or a model this key cannot reach —
        retrying those only burns time.
        """
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code == 429 or code >= 500
        text = str(exc).lower()
        if any(m in text for m in ("not_found", "invalid_argument", "permission_denied")):
            return False
        return any(m in text for m in ("unavailable", "resource_exhausted",
                                       "internal", "deadline", "timeout", "503", "429"))

    def generate(self, model: cfg.ModelSpec, system, messages: List[Dict[str, str]],
                 settings: cfg.RunSettings) -> GenerationResult:
        from google.genai import types

        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

        config_kwargs = {"max_output_tokens": settings.max_output_tokens}
        if system:
            config_kwargs["system_instruction"] = system
        if model.thinking:
            # thinking_level: MINIMAL | LOW | MEDIUM | HIGH. Sent only when the
            # model spec names one — otherwise Gemini's own default applies.
            thinking_kwargs = {"include_thoughts": True}
            if model.effort:
                thinking_kwargs["thinking_level"] = model.effort.upper()
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)
        elif model.supports_temperature and settings.temperature is not None:
            config_kwargs["temperature"] = settings.temperature
        gen_config = types.GenerateContentConfig(**config_kwargs)

        response, timing = _retry_loop(
            lambda: self._client.models.generate_content(
                model=model.model_id, contents=contents, config=gen_config),
            self._max_retries, self._is_retryable,
        )

        text = response.text or ""
        usage = response.usage_metadata

        def count(field: str) -> int:
            return getattr(usage, field, 0) or 0 if usage else 0

        prompt_tokens = count("prompt_token_count")
        candidates_tokens = count("candidates_token_count")
        thoughts_tokens = count("thoughts_token_count")
        cached = count("cached_content_token_count")
        tool_prompt_tokens = count("tool_use_prompt_token_count")

        # Thinking tokens sit outside candidates_token_count but are billed at
        # the output rate, so billable output is the sum of the two. The current
        # REST contract defines total_token_count as prompt + thoughts +
        # candidates, which makes tool_use_prompt_token_count a prompt
        # breakdown. SDK schemas instead describe it as an extra term in the
        # total. Reconcile the actual response so either representation is
        # measured once, never twice.
        reported_total = count("total_token_count")
        documented_total = prompt_tokens + candidates_tokens + thoughts_tokens
        separate_tool_prompt_tokens = (
            tool_prompt_tokens
            if tool_prompt_tokens
            and reported_total == documented_total + tool_prompt_tokens
            else 0
        )
        output_tokens = candidates_tokens + thoughts_tokens
        input_tokens = max(0, prompt_tokens - cached) + separate_tool_prompt_tokens

        token_usage = cfg.TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cached,
        )
        cost, cost_basis = cfg.cost_usd(model, token_usage)

        # Three independent checks that the cost above is complete.
        notes = []
        untracked = _unaccounted_usage_fields(usage, _GOOGLE_ACCOUNTED)
        if untracked:
            notes.append(f"usage fields not priced: {untracked}")
        modality = _google_modality_warning(usage)
        if modality:
            notes.append(f"non-text input modality billed above the text rate: {modality}")
        if reported_total and reported_total != token_usage.total_tokens:
            # Gemini states the total itself; a mismatch means a token class
            # exists on this response that the split above does not cover.
            notes.append(
                f"token split does not reconcile: Gemini reports "
                f"total_token_count={reported_total}, priced "
                f"{token_usage.total_tokens} "
                f"(prompt={prompt_tokens}, cached={cached}, tool_prompt="
                f"{tool_prompt_tokens}, separately_counted_tool_prompt="
                f"{separate_tool_prompt_tokens}, candidates={candidates_tokens}, "
                f"thoughts={thoughts_tokens})")
        # Current responses report service_tier; older google-genai releases
        # expose traffic_type instead. Preserve whichever the response supplies.
        tier = getattr(usage, "service_tier", None) if usage else None
        if not tier:
            tier = getattr(usage, "traffic_type", None) if usage else None
        tier_name = getattr(tier, "value", None) or (str(tier) if tier else "")
        tier_note = _service_tier_note(tier_name)
        if tier_note:
            notes.append(tier_note)

        reasoning_note = (
            f"thinking_level={model.effort.upper() if model.effort else 'default'}; "
            f"thoughts_token_count={thoughts_tokens}, reported outside "
            "candidates_token_count by Gemini and billed at the output rate, so "
            "added to output_tokens here"
            if model.thinking else
            # Not a footnote: every Gemini 3.x model reasons by default, so a
            # tier with thinking off still produces billable thinking tokens.
            f"thinking not requested, but Gemini 3.x reasons by default — "
            f"thoughts_token_count={thoughts_tokens} billed at the output rate"
        )

        stop_reason = None
        if getattr(response, "candidates", None):
            stop_reason = str(getattr(response.candidates[0], "finish_reason", "")) or None

        return GenerationResult(
            text=text, reported_input_tokens=prompt_tokens,
            reported_total_tokens=reported_total,
            # Gemini's own output count is the answer alone; the thoughts it
            # bills at the output rate are added into output_tokens, not here.
            reported_output_tokens=candidates_tokens,
            tool_use_prompt_tokens=tool_prompt_tokens, raw_usage=_usage_json(usage),
            input_tokens=input_tokens, output_tokens=output_tokens,
            cached_tokens=cached,
            billable_input_tokens=token_usage.billable_input_tokens,
            thinking_tokens=thoughts_tokens, total_tokens=token_usage.total_tokens,
            cost_usd=cost, cost_basis=cost_basis,
            untracked_usage="; ".join(notes),
            service_tier=tier_name,
            **timing.as_fields(), **_request_ids(response),
            stop_reason=stop_reason, token_reporting="reported", tokens_estimated=False,
            reasoning_note=reasoning_note,
        )

    def check_connection(self, model_id: str):
        # google-genai's exception hierarchy isn't verified in this session —
        # report whatever comes back rather than assuming specific classes.
        try:
            self._client.models.get(model=model_id)
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, f"error — {exc}"


# ══════════════════════════════════════════════════════════════════════════
# Mock — offline, no network, no API key
# ══════════════════════════════════════════════════════════════════════════
class MockProvider(Provider):
    """Offline provider for --demo: deterministic, no network, no API key.
    Echoes a syntactically valid (if trivial) DOT graph so the rest of the
    pipeline — extraction, persistence, aggregation — can be exercised end to
    end without spending anything, for any vendor's models."""

    name = "mock"

    def generate(self, model: cfg.ModelSpec, system, messages: List[Dict[str, str]],
                 settings: cfg.RunSettings) -> GenerationResult:
        # Timed like a real call so the columns are always populated — but this
        # measures a local string operation (microseconds), not a network round
        # trip. Every mock row is flagged is_mock=True for exactly that reason.
        sent_at = datetime.now().astimezone()
        t0 = time.monotonic()
        dot = (
            "digraph mock {\n"
            '  START_NODE [label="" shape=circle width=0.3]\n'
            '  "do the described task" [shape=box]\n'
            '  END_NODE [label="" shape=doublecircle width=0.2]\n'
            '  START_NODE -> "do the described task"\n'
            '  "do the described task" -> END_NODE\n'
            "}\n"
        )
        text = f"Output:\n{dot}"
        elapsed = time.monotonic() - t0
        timing = CallTiming(
            api_latency_s=elapsed, total_s=elapsed, attempts=1, retry_wait_s=0.0,
            request_sent_at=sent_at.isoformat(timespec="milliseconds"),
            response_received_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
        )

        input_tokens = _estimate_tokens(system or "") + sum(_estimate_tokens(m["content"]) for m in messages)
        output_tokens = _estimate_tokens(text)

        return GenerationResult(
            text=text, reported_input_tokens=input_tokens,
            reported_total_tokens=input_tokens + output_tokens,
            reported_output_tokens=output_tokens,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cached_tokens=0, billable_input_tokens=input_tokens,
            total_tokens=input_tokens + output_tokens, cost_usd=0.0, cost_basis="mock",
            **timing.as_fields(), stop_reason="end_turn", token_reporting="estimated",
            tokens_estimated=True, reasoning_note="",
        )

    def check_connection(self, model_id: str):
        return True, "ok (mock — no network involved)"


_VENDOR_PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "mistral": MistralProvider,
    "google": GoogleProvider,
    "deepseek": DeepSeekProvider,
}


def get_provider(vendor: str, api_key: Optional[str] = None) -> Provider:
    if vendor == "mock":
        return MockProvider()
    cls = _VENDOR_PROVIDERS.get(vendor)
    if cls is None:
        raise ValueError(f"Unknown vendor {vendor!r}. Known: {list(_VENDOR_PROVIDERS)} (or 'mock')")
    return cls(api_key=api_key)
