"""Tests for the persistent score cache (src/score_cache.py + rank_jobs wiring).

The cache is the difference between re-searching a role costing a full 100-job
scoring run and costing nothing, so what matters is:

1. A second identical search makes **no API calls at all**.
2. Changing the résumé, or the rubric, invalidates — stale scores are worse than
   no scores, because nothing in the UI says which rules produced a number.
3. It never caches a failure, and never breaks a search when it is broken itself.
4. jd_flags survives a cache hit — it is a security signal, not a nicety.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import src.job_matcher as jm
import src.profile_manager as pm
import src.score_cache as sc
from src.job_matcher import rank_jobs


_COMP = {
    "ats_coverage": 70,
    "matched_skills": ["Python"],
    "missing_skills": ["Kubernetes"],
    "title_fit": 80,
    "seniority_fit": 60,
    "education_fit": 100,
    "knockouts": [],
    "injections": [],
    "reason": "ok",
}


def _job_ids(kwargs) -> list[int]:
    return [
        int(s.split("=")[1].split()[0])
        for m in kwargs["messages"]
        for s in m["content"].split("### job_id")
        if s.startswith("=")
    ]


def _reply(comps):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"results": comps})),
            finish_reason="stop",
        )]
    )


class _Client:
    """Fake Groq client.

    Scoring calls go through ``with_raw_response`` so the pacer can read the
    rate-limit headers, so the fake exposes both surfaces. ``self.calls`` records
    every request either way, which is what the cache assertions count.
    """

    def __init__(self):
        self.calls = []
        self.make_comps = lambda ids: [{**_COMP, "job_id": i} for i in ids]

        def _create(**kwargs):
            self.calls.append(kwargs)
            return _reply(self.make_comps(_job_ids(kwargs)))

        def _raw_create(**kwargs):
            parsed = _create(**kwargs)
            return SimpleNamespace(headers={}, parse=lambda: parsed)

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=_create,
                with_raw_response=SimpleNamespace(create=_raw_create),
            )
        )


@pytest.fixture
def client(monkeypatch):
    c = _Client()
    monkeypatch.setattr(jm, "_get_client", lambda: c)
    return c


def _df(n=3):
    return pd.DataFrame([
        {"title": f"Engineer {i}", "company": f"Co{i}",
         "location": "Boston, MA", "description": f"desc {i}"}
        for i in range(n)
    ])


RESUME = {"summary": "Backend engineer", "skills": ["Python"]}


# ── The headline behaviour ───────────────────────────────────────────────────
def test_second_identical_search_makes_no_api_calls(client):
    """The whole point: re-searching a role you already scored is free."""
    first = rank_jobs(_df(), RESUME)
    assert client.calls, "first search must actually score"
    n_first = len(client.calls)

    second = rank_jobs(_df(), RESUME)

    assert len(client.calls) == n_first, "a fully cached search must call nothing"
    assert second["match_score"].notna().all(), "cached jobs still need scores"
    pd.testing.assert_series_equal(
        first["match_score"], second["match_score"], check_names=False
    )


def test_fully_cached_search_does_not_report_failure(client):
    """Regression: `if succeeded == 0: raise` fired on a perfect result.

    A fully cached search dispatches zero batches, so `succeeded` is 0 while every
    job has a score. The guard has to test that batches were actually attempted.
    """
    rank_jobs(_df(), RESUME)
    client.calls.clear()

    out = rank_jobs(_df(), RESUME)  # must not raise

    assert not client.calls
    assert out["match_score"].notna().all()


def test_partial_cache_scores_only_the_misses(client):
    rank_jobs(_df(2), RESUME)
    client.calls.clear()

    rank_jobs(_df(4), RESUME)  # 2 known, 2 new

    sent = [
        s.split("=")[1].split()[0]
        for kw in client.calls for m in kw["messages"]
        for s in m["content"].split("### job_id") if s.startswith("=")
    ]
    assert len(sent) == 2, f"only the two unseen jobs should be sent, got {sent}"


# ── Invalidation ─────────────────────────────────────────────────────────────
def test_a_different_resume_invalidates(client):
    rank_jobs(_df(), RESUME)
    client.calls.clear()

    rank_jobs(_df(), {"summary": "Totally different", "skills": ["Go"]})

    assert client.calls, "a changed résumé must re-score, not serve old numbers"


def test_reuploading_the_same_resume_keeps_the_cache(client):
    """save_resume INSERTs a new row every upload.

    Keying on resume.id would therefore discard every cached score whenever the
    same file is uploaded again. Keying on the rendered profile text does not.
    """
    rank_jobs(_df(), RESUME)
    client.calls.clear()

    rank_jobs(_df(), dict(RESUME))  # equal content, different object

    assert not client.calls


def test_a_changed_rubric_invalidates(client, monkeypatch):
    rank_jobs(_df(), RESUME)
    client.calls.clear()

    monkeypatch.setattr(jm, "_SCORING_INSTRUCTIONS", jm._SCORING_INSTRUCTIONS + " Also weigh X.")
    rank_jobs(_df(), RESUME)

    assert client.calls, "scores from an older rubric must not be served silently"


# ── Safety ───────────────────────────────────────────────────────────────────
def test_failures_are_never_cached(monkeypatch):
    class _Boom:
        def __init__(self):
            def _create(**kw):
                raise RuntimeError("rate limited")
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=_create,
                    with_raw_response=SimpleNamespace(create=_create),
                )
            )

    monkeypatch.setattr(jm, "_get_client", lambda: _Boom())
    with pytest.raises(RuntimeError):
        rank_jobs(_df(), RESUME)

    assert sc.get_many(["co0|engineer 0|boston"], "x", "y") == {}, \
        "a transient failure must not freeze into a permanent no-score"


def test_an_unusable_database_does_not_break_a_search(client, monkeypatch):
    """The cache is an optimisation. Its failure mode is 'call the API'.

    Pointed at a broken engine, both halves must degrade quietly and the search
    must still produce scores.
    """
    monkeypatch.setattr(pm, "ENGINE", None)  # any use raises

    out = rank_jobs(_df(), RESUME)

    assert client.calls, "with no usable cache, everything should be scored"
    assert out["match_score"].notna().all(), "a broken cache must not cost scores"


def test_put_many_swallows_its_own_errors(monkeypatch):
    monkeypatch.setattr(pm, "ENGINE", None)  # any use will raise
    sc.put_many([("k", _COMP)], "r", "v")  # must not raise


def test_get_many_swallows_its_own_errors(monkeypatch):
    monkeypatch.setattr(pm, "ENGINE", None)
    assert sc.get_many(["k"], "r", "v") == {}


# ── The security signal has to survive a cache hit ───────────────────────────
def test_injections_survive_a_cache_hit(client, monkeypatch):
    """jd_flags drives the red outline on a trapped posting.

    A cached job that lost its injections would render as clean, which is worse
    than not caching at all.
    """
    client.make_comps = lambda ids: [
        {**_COMP, "job_id": i, "injections": ["ignore previous instructions"]}
        for i in ids
    ]

    first = rank_jobs(_df(1), RESUME)
    assert "ignore previous instructions" in first["jd_flags"].iloc[0]

    client.calls.clear()
    second = rank_jobs(_df(1), RESUME)

    assert not client.calls, "should be a cache hit"
    assert "ignore previous instructions" in second["jd_flags"].iloc[0], \
        "a cached trapped posting must keep its warning"


# ── TTL ──────────────────────────────────────────────────────────────────────
def test_expired_entries_are_not_served(client, monkeypatch):
    rank_jobs(_df(1), RESUME)
    client.calls.clear()

    monkeypatch.setattr(sc, "_TTL_DAYS", -1)  # everything is now stale
    rank_jobs(_df(1), RESUME)

    assert client.calls, "a stale posting must be re-read, not served forever"


# ── Applied jobs: reuse the score already recorded ───────────────────────────
def test_applied_jobs_reuse_their_saved_score(client):
    """A role already applied to should not spend a batch slot.

    saved_jobs kept the score from when the application went out. Re-grading it
    pays full price to re-decide something that is already decided.
    """
    job = {"title": "Engineer 0", "company": "Co0",
           "location": "Boston, MA", "url": "u", "match_score": 88,
           "match_reason": "strong overlap"}
    pm.mark_job_applied(job)

    out = rank_jobs(_df(1), RESUME)

    assert not client.calls, "an applied job must not be sent to the API"
    assert out["match_score"].iloc[0] == 88, "the recorded score should be shown"
    assert out["match_reason"].iloc[0] == "strong overlap"


def test_applied_job_without_a_score_is_still_scored(client):
    """Saved before scoring existed → score it, don't show a permanent blank."""
    pm.mark_job_applied({"title": "Engineer 0", "company": "Co0",
                         "location": "Boston, MA", "url": "u"})

    out = rank_jobs(_df(1), RESUME)

    assert client.calls, "a scoreless applied job still needs grading"
    assert out["match_score"].notna().all()


def test_a_saved_jobs_score_is_not_written_into_the_cache(client):
    """The two stores mean different things.

    A saved_jobs score was produced by whatever rubric and résumé were current
    then. Copying it into score_cache would launder it into looking like a score
    this rubric produced, and it would then outlive the applied status.
    """
    pm.mark_job_applied({"title": "Engineer 0", "company": "Co0",
                         "location": "Boston, MA", "url": "u", "match_score": 88})

    rank_jobs(_df(1), RESUME)

    assert sc.get_many(["co0|engineer 0|boston"], sc.resume_key("x"), "v") == {}


# ── 413: a request too large to ever succeed ─────────────────────────────────
def test_an_oversized_batch_is_split_rather_than_failing(client, monkeypatch):
    """Regression: a batch bigger than the whole per-minute ceiling 413'd forever.

    Groq counts input + max_completion_tokens against the limit, so an oversized
    request is rejected every time no matter how it is paced:

        Limit 8000, Requested 9942

    The fix is to notice before sending and score the halves instead.
    """
    import src.ratelimit as rl

    # A ceiling so low that a multi-job batch cannot possibly fit.
    monkeypatch.setattr(rl, "_DEFAULT_TPM", 3_000)
    # The pacer would genuinely wait out a window at this ceiling; the subject
    # here is the split decision, not the sleeping.
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)

    out = rank_jobs(_df(4), RESUME)

    assert out["match_score"].notna().all(), "splitting must still score every job"
    sizes = [len(_job_ids(kw)) for kw in client.calls]
    assert sizes, "something must have been sent"
    assert max(sizes) < 4, f"batch should have been split, sent {sizes}"


def test_planned_requests_fit_the_ceiling(client, monkeypatch):
    """Every request actually dispatched must be small enough to be accepted."""
    import src.ratelimit as rl

    monkeypatch.setattr(rl, "_DEFAULT_TPM", 8_000)
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)
    budget = rl.TokenBudget()

    rank_jobs(_df(6), RESUME)

    for kw in client.calls:
        sent = sum(rl.estimate_tokens(m["content"]) for m in kw["messages"])
        requested = sent + kw["max_completion_tokens"]
        assert requested <= budget.budget, (
            f"request of {requested} exceeds the {budget.budget} budget — this is "
            f"the shape that produced the 413"
        )
