"""Request timing — what `api_latency_s` means, and what it must exclude.

`api_latency_s` is the round trip of the attempt that actually answered:
request handed to the SDK until the complete reply is back. A rate-limit retry
in front of it belongs in `latency_s`, never in the round trip — otherwise the
figure reported as the model's response time is really a measure of throttling.
"""
from __future__ import annotations

import time
import types

import httpx
import pytest

import anthropic
import config as cfg
from providers import AnthropicProvider

MODEL = cfg.model_by_key("anthropic_standard")
SLEEP = 0.30


def _reply():
    return types.SimpleNamespace(
        stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text="Output:\ndigraph g { a -> b }")],
        usage=types.SimpleNamespace(input_tokens=111, output_tokens=222,
                                    cache_read_input_tokens=0),
    )


class _SlowClient:
    """Sleeps SLEEP per attempt; the first `fail_first` attempts raise a
    retryable connection error."""

    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self.fail_first = fail_first
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        time.sleep(SLEEP)
        if self.calls <= self.fail_first:
            raise anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        return _reply()


def _generate(fail_first: int, settings):
    provider = AnthropicProvider(api_key="sk-test-not-used")
    provider._client = _SlowClient(fail_first=fail_first)
    return provider.generate(MODEL, None, [{"role": "user", "content": "hi"}], settings)


def test_clean_call_reports_the_round_trip(settings):
    gen = _generate(0, settings)
    assert gen.api_latency_s == pytest.approx(SLEEP, abs=0.05)
    assert gen.api_attempts == 1
    assert gen.retry_wait_s == 0.0


def test_single_attempt_makes_both_spans_identical(settings):
    """So a reader can trust 'latency_s == api_latency_s unless api_attempts > 1'."""
    gen = _generate(0, settings)
    assert gen.latency_s == gen.api_latency_s


def test_retry_inflates_the_wider_span_only(settings):
    gen = _generate(1, settings)
    assert gen.api_latency_s == pytest.approx(SLEEP, abs=0.05)   # answering attempt only
    assert gen.api_attempts == 2
    assert gen.retry_wait_s == 1.0                                # first backoff step
    assert gen.latency_s > gen.api_latency_s + 1.0                # failed attempt + backoff


def test_timestamps_bracket_the_answering_attempt(settings):
    gen = _generate(0, settings)
    assert gen.request_sent_at and gen.response_received_at
    assert gen.request_sent_at < gen.response_received_at
