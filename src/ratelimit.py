"""Send scoring requests at a rate the account can actually absorb.

The problem this exists for: scoring a 100-job chunk fans out ~17 requests, and
``rank_jobs`` used to fire eight of them at once. On Groq's free tier the ceiling
is 8,000 tokens *per minute*, so the first wave alone was several times over
budget. The rejected requests were dropped — nothing retried them — which is
exactly the "N jobs couldn't be scored" warning on the Job Search page.

Two mechanisms, deliberately separate:

**A pacer** that estimates what a request will cost and waits until this minute's
budget can absorb it. Prevention, and the reason most 429s stop happening.

**A retry** that honours ``retry-after`` when one gets through anyway. Cure, for
the cases the estimate got wrong or another tab spent the budget.

The budget is *discovered*, not configured. Groq reports the real ceiling in
``x-ratelimit-limit-tokens`` on every response, so the first reply teaches the
pacer which tier the key is on. That is what makes adding a payment method a
no-code change: the same path sees 250,000/min instead of 8,000 and stops
throttling.
"""

import logging
import os
import random
import threading
import time

log = logging.getLogger(__name__)

# What we assume before any response has told us otherwise. Groq's published free
# tier for openai/gpt-oss-120b. Guessing low is the safe direction: guess too high
# and the first wave is rejected, which is the bug this module exists to fix.
_DEFAULT_TPM = int(os.environ.get("GROQ_ASSUMED_TPM", "8000"))

# Spend only part of the stated ceiling. Token counts here are estimates from
# character length, other tabs and the résumé parser draw on the same account, and
# Groq enforces over shorter windows than a full minute.
_SAFETY = float(os.environ.get("GROQ_BUDGET_SAFETY", "0.85"))

_MAX_ATTEMPTS = int(os.environ.get("GROQ_MAX_ATTEMPTS", "4"))


def estimate_tokens(text: str) -> int:
    """Rough token count for a prompt.

    Deliberately arithmetic rather than a tokenizer: this is a scheduling hint,
    not billing. ~4 characters per token holds well enough for English prose, and
    the safety margin above absorbs the error. Adding a tokenizer dependency to
    decide how long to sleep would be a poor trade.
    """
    return max(1, len(text) // 4)


class TokenBudget:
    """A token bucket over a sliding minute, sized from the observed ceiling.

    Thread-safe: ``rank_jobs`` calls ``acquire`` from worker threads, and the
    whole point is that they queue against one shared budget rather than each
    deciding independently that there is room.
    """

    def __init__(self, tokens_per_minute: int | None = None):
        self._limit = tokens_per_minute or _DEFAULT_TPM
        self._spent: list[tuple[float, int]] = []  # (timestamp, tokens)
        self._lock = threading.Lock()
        self._learned = tokens_per_minute is not None

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def budget(self) -> int:
        return max(1, int(self._limit * _SAFETY))

    def observe_headers(self, headers) -> None:
        """Learn the real ceiling from a response.

        Called after every successful request. The first one that carries the
        header replaces the assumed free-tier default with the truth, so a paid
        key stops being throttled without anyone editing config.
        """
        if headers is None:
            return
        try:
            raw = headers.get("x-ratelimit-limit-tokens")
            if raw is None:
                return
            limit = int(str(raw).strip())
        except (AttributeError, TypeError, ValueError):
            return
        if limit > 0 and limit != self._limit:
            log.info("rate limit discovered: %s tokens/min (was %s)", limit, self._limit)
            with self._lock:
                self._limit = limit
                self._learned = True

    def _spent_in_window(self, now: float) -> int:
        self._spent = [(t, n) for t, n in self._spent if now - t < 60.0]
        return sum(n for _, n in self._spent)

    def acquire(self, tokens: int, timeout: float = 300.0) -> None:
        """Block until ``tokens`` fit in the current minute, then record them.

        A single request larger than the whole budget is let through rather than
        deadlocking — it will probably 429, and the retry path handles that. The
        alternative is waiting forever for room that can never exist.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                used = self._spent_in_window(now)
                if tokens >= self.budget or used + tokens <= self.budget:
                    self._spent.append((now, tokens))
                    return
                oldest = self._spent[0][0] if self._spent else now
                wait = max(0.05, 60.0 - (now - oldest))

            if time.monotonic() + wait > deadline:
                log.warning("token budget wait exceeded %.0fs; sending anyway", timeout)
                with self._lock:
                    self._spent.append((time.monotonic(), tokens))
                return
            time.sleep(min(wait, 5.0))

    def suggested_workers(self, per_request_tokens: int, hard_cap: int = 8) -> int:
        """How many requests can sensibly be in flight at once.

        On a throttled tier, extra threads do not make anything faster — they
        just queue against the same budget while multiplying the chance of a
        burst 429. On a large ceiling, concurrency is the whole win.
        """
        if per_request_tokens <= 0:
            return 1
        return max(1, min(hard_cap, self.budget // per_request_tokens))

    def plan_request(
        self,
        per_job_tokens: int,
        overhead: int,
        out_per_job: int = 400,
        out_slack: int = 500,
        hard_cap: int = 12,
    ) -> tuple[int, int]:
        """Pick ``(batch_size, max_completion_tokens)`` that fit one minute's budget.

        **Groq counts the output reservation against the per-minute limit**, not
        just the input. A 413 reads:

            Limit 8000, Requested 9942

        where 9942 is ``input + max_completion_tokens``. Sizing the batch off input
        alone is what produced that error: a single request cannot fit in the whole
        minute, so no amount of pacing helps and the request fails every time.

        Both numbers are therefore solved together:

            overhead + batch*per_job + (batch*out_per_job + out_slack) <= budget

        ``out_per_job`` is the JSON a scored job produces — two ten-item skill
        lists, six integers and a sentence. ``out_slack`` covers the wrapper.
        """
        if per_job_tokens <= 0:
            return hard_cap, hard_cap * out_per_job + out_slack

        room = self.budget - overhead - out_slack
        batch = int(room // (per_job_tokens + out_per_job))
        batch = max(1, min(hard_cap, batch))
        return batch, batch * out_per_job + out_slack

    def fits(self, tokens: int) -> bool:
        """Whether a single request of this size can ever succeed."""
        return tokens <= self.budget


def call_with_retry(fn, *, budget: TokenBudget | None = None, attempts: int = _MAX_ATTEMPTS):
    """Run ``fn``, retrying rate limits with the delay the server asks for.

    ``retry-after`` is authoritative — the server knows when the window resets and
    guessing shorter just burns another attempt. Jitter is added so parallel
    batches that were rejected together do not all return at the same instant and
    trip the limit again.

    Only 429 and 5xx are retried. A bad key or an unknown model will fail exactly
    the same way four times, so retrying those wastes the user's time and hides
    the real error behind a delay.
    """
    from groq import APIConnectionError, APIStatusError, RateLimitError

    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except RateLimitError as e:
            last = e
            delay = _retry_after(e, attempt)
            log.info("rate limited, retrying in %.1fs (attempt %d/%d)",
                     delay, attempt + 1, attempts)
        except APIConnectionError as e:
            last = e
            delay = _backoff(attempt)
        except APIStatusError as e:
            if e.status_code < 500:
                raise  # 400/401/403/404 will not improve on a retry
            last = e
            delay = _backoff(attempt)

        if attempt == attempts - 1:
            break
        time.sleep(delay)

    raise last


def _retry_after(err, attempt: int) -> float:
    """The server's requested delay, falling back to exponential backoff."""
    try:
        headers = getattr(getattr(err, "response", None), "headers", None)
        if headers is not None:
            raw = headers.get("retry-after")
            if raw is not None:
                return min(60.0, float(str(raw).strip())) + random.uniform(0, 1.0)
    except (TypeError, ValueError):
        pass
    return _backoff(attempt)


def _backoff(attempt: int) -> float:
    return min(30.0, (2 ** attempt)) + random.uniform(0, 1.0)
