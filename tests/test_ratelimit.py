"""Tests for the token pacer and retry (src/ratelimit.py).

This module exists because scoring used to fire eight requests at once into an
8,000-tokens-per-minute ceiling, and the rejected ones were dropped rather than
retried — which is what produced "N jobs couldn't be scored".

So the behaviours worth pinning are: it holds requests back before they are
rejected, it discovers the real ceiling instead of assuming one, it honours the
delay the server asks for, and it never retries something a retry cannot fix.
"""

import time
from types import SimpleNamespace

import pytest
from groq import APIStatusError, RateLimitError

import src.ratelimit as rl


def _err(cls, status, headers=None):
    """Build a groq error without going near the network."""
    resp = SimpleNamespace(status_code=status, headers=headers or {}, request=None)
    e = cls.__new__(cls)
    e.status_code = status
    e.response = resp
    e.message = "test"
    e.body = None
    e.request_id = None
    return e


# ── The budget ───────────────────────────────────────────────────────────────
def test_a_request_inside_the_budget_does_not_wait():
    b = rl.TokenBudget(tokens_per_minute=100_000)
    t0 = time.monotonic()
    b.acquire(1_000)
    assert time.monotonic() - t0 < 0.1


def test_spending_the_budget_makes_the_next_request_wait(monkeypatch):
    """The whole point: hold the request back rather than have it rejected.

    Asserts the *decision* to sleep rather than a real elapsed time — the wait is
    up to a full minute, and the short-timeout escape hatch deliberately returns
    immediately instead, so neither is observable as wall-clock in a test.

    Requests stay well under the budget so they don't trip the
    larger-than-the-whole-budget fallback; queueing is what's under test.
    """
    b = rl.TokenBudget(tokens_per_minute=1_000)
    portion = b.budget // 3
    for _ in range(3):
        b.acquire(portion)  # window now effectively full

    class _Slept(Exception):
        pass

    slept: list[float] = []

    def _stub(seconds):
        slept.append(seconds)
        raise _Slept  # break the loop; the decision is all we need to see

    monkeypatch.setattr(rl.time, "sleep", _stub)

    with pytest.raises(_Slept):
        b.acquire(portion)

    assert slept[0] > 0, "a full budget must delay the next request"


def test_a_request_larger_than_the_whole_budget_is_not_deadlocked():
    """No amount of waiting creates room for it, so let it through.

    It will probably be rejected, and the retry path handles that — which is a
    better outcome than a search that hangs forever.
    """
    b = rl.TokenBudget(tokens_per_minute=1_000)
    t0 = time.monotonic()
    b.acquire(50_000)
    assert time.monotonic() - t0 < 0.1


def test_the_budget_only_spends_part_of_the_stated_ceiling():
    """Estimates are approximate and other callers share the account."""
    b = rl.TokenBudget(tokens_per_minute=10_000)
    assert b.budget < 10_000


# ── Learning the tier ────────────────────────────────────────────────────────
def test_the_real_ceiling_is_read_off_the_response():
    """This is what makes adding a payment method a no-code change."""
    b = rl.TokenBudget()
    assert b.limit == rl._DEFAULT_TPM

    b.observe_headers({"x-ratelimit-limit-tokens": "250000"})

    assert b.limit == 250_000


def test_a_response_without_the_header_changes_nothing():
    b = rl.TokenBudget(tokens_per_minute=8_000)
    b.observe_headers({})
    b.observe_headers(None)
    b.observe_headers({"x-ratelimit-limit-tokens": "not-a-number"})
    assert b.limit == 8_000


# ── Sizing decisions follow the ceiling ──────────────────────────────────────
def test_a_small_ceiling_serialises_and_a_large_one_parallelises():
    small = rl.TokenBudget(tokens_per_minute=8_000)
    large = rl.TokenBudget(tokens_per_minute=250_000)

    assert small.suggested_workers(6_000) == 1, \
        "extra threads on a throttled tier only queue against the same budget"
    assert large.suggested_workers(6_000, hard_cap=8) == 8


def test_a_planned_request_always_fits_the_ceiling():
    """Regression: a batch was sized off input alone and got 413'd every time.

    Groq counts input *and* the output reservation against the per-minute limit
    — "Limit 8000, Requested 9942" is input + max_completion_tokens. Sizing on
    input only produced a single request that could never succeed, no matter how
    patiently it was paced.
    """
    per_job, overhead = 810, 1_430

    for tpm in (8_000, 30_000, 250_000):
        b = rl.TokenBudget(tokens_per_minute=tpm)
        batch, max_out = b.plan_request(per_job, overhead, hard_cap=12)
        requested = overhead + batch * per_job + max_out

        assert batch >= 1
        assert requested <= b.budget, (
            f"TPM {tpm}: planned request of {requested} exceeds budget {b.budget}"
        )
        assert requested < tpm, f"TPM {tpm}: {requested} would be rejected outright"


def test_a_small_ceiling_forces_a_smaller_batch_than_a_large_one():
    small = rl.TokenBudget(tokens_per_minute=8_000)
    large = rl.TokenBudget(tokens_per_minute=250_000)

    per_job, overhead = 810, 1_430
    assert small.plan_request(per_job, overhead, hard_cap=12)[0] < 12
    assert large.plan_request(per_job, overhead, hard_cap=12)[0] == 12


def test_fits_rejects_a_request_larger_than_the_budget():
    """What tells the scorer to split rather than send something doomed."""
    b = rl.TokenBudget(tokens_per_minute=8_000)
    assert b.fits(5_000)
    assert not b.fits(9_942)  # the exact size that produced the 413


# ── Retry ────────────────────────────────────────────────────────────────────
def test_a_rate_limit_is_retried_and_then_succeeds(monkeypatch):
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _err(RateLimitError, 429)
        return "ok"

    assert rl.call_with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_the_servers_requested_delay_is_honoured(monkeypatch):
    """retry-after is authoritative — the server knows when the window resets."""
    slept = []
    monkeypatch.setattr(rl.time, "sleep", slept.append)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _err(RateLimitError, 429, {"retry-after": "7"})
        return "ok"

    rl.call_with_retry(flaky)

    assert slept and 7.0 <= slept[0] <= 8.1, f"expected ~7s, slept {slept}"


def test_a_bad_key_is_not_retried(monkeypatch):
    """401 fails identically four times — retrying only delays the real error."""
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise _err(APIStatusError, 401)

    with pytest.raises(APIStatusError):
        rl.call_with_retry(bad)
    assert calls["n"] == 1, "a client error must fail fast, not four times slowly"


def test_a_server_error_is_retried(monkeypatch):
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _err(APIStatusError, 503)
        return "ok"

    assert rl.call_with_retry(flaky) == "ok"


def test_persistent_rate_limiting_eventually_raises(monkeypatch):
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)

    def always():
        raise _err(RateLimitError, 429)

    with pytest.raises(RateLimitError):
        rl.call_with_retry(always, attempts=3)


# ── Estimation ───────────────────────────────────────────────────────────────
def test_token_estimate_tracks_length():
    assert rl.estimate_tokens("") >= 1
    assert rl.estimate_tokens("a" * 4000) == pytest.approx(1000, rel=0.1)
