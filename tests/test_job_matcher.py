"""Tests for ATS-style résumé→job scoring (src/job_matcher.py).

These pin the behavior of the matching rework:

1. The scorer reads the *full* job description (the old 300-char cap hid the
   Requirements section), and asks for it deterministically (temperature=0).
2. The final score is a fixed weighted blend of the model's component sub-scores,
   and a hard-requirement knockout caps it.
3. The candidate profile carries enough résumé detail (skills + several bullets)
   for keyword matching.
4. A failing batch yields score=None (not a fake 0/100) without aborting the rest.

The Groq client is faked — no network, no GROQ_API_KEY required.
"""

import json
import re
from types import SimpleNamespace

import pandas as pd
import pytest

import src.job_matcher as jm
from src.job_matcher import _build_candidate_profile, _blended_score, rank_jobs


# ── Fake Groq client ─────────────────────────────────────────────────────────
class _FakeClient:
    """Stand-in for groq.Groq. Records every create() call and returns whatever
    the supplied handler produces for that call's kwargs."""

    def __init__(self, handler):
        self.calls: list[dict] = []

        def _create(**kwargs):
            self.calls.append(kwargs)
            content = handler(kwargs)
            msg = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


_DEFAULT_COMP = {
    "ats_coverage": 70,
    "matched_skills": ["Python"],
    "missing_skills": ["Kubernetes"],
    "title_fit": 80,
    "seniority_fit": 60,
    "education_fit": 100,
    "knockouts": [],
    "reason": "ok",
}


def _ids_from_prompt(kwargs) -> list[int]:
    content = kwargs["messages"][0]["content"]
    return [int(m) for m in re.findall(r"job_id=(\d+)", content)]


def _make_handler(per_id: dict | None = None, fail_batch_with: int | None = None):
    """Build a handler that replies with a component object per job_id in the
    prompt. ``per_id`` overrides fields for specific job_ids; ``fail_batch_with``
    makes the batch containing that job_id raise (deterministic under the parallel
    scoring, where call *order* is not guaranteed) to exercise error isolation."""

    def handler(kwargs) -> str:
        ids = _ids_from_prompt(kwargs)
        if fail_batch_with is not None and fail_batch_with in ids:
            raise RuntimeError("boom")
        out = []
        for jid in ids:
            comp = dict(_DEFAULT_COMP)
            if per_id and jid in per_id:
                comp.update(per_id[jid])
            comp["job_id"] = jid
            out.append(comp)
        return json.dumps(out)

    return handler


@pytest.fixture
def patch_client(monkeypatch):
    def _install(handler):
        client = _FakeClient(handler)
        monkeypatch.setattr(jm, "_get_client", lambda: client)
        return client

    return _install


# ── _build_candidate_profile ─────────────────────────────────────────────────
def test_profile_includes_skills_and_more_than_four_bullets():
    resume = {
        "summary": "Backend engineer.",
        "skills": ["Python", "AWS", "PostgreSQL"],
        "experience": [
            {
                "title": "SWE",
                "company": "Acme",
                "duration": "3y",
                "bullets": [f"bullet{i}" for i in range(1, 8)],  # 7 bullets
            }
        ],
    }
    profile = _build_candidate_profile(resume)
    assert "Skills: Python, AWS, PostgreSQL" in profile
    # Was capped at 4; now keeps up to 6 — bullet5 must appear, bullet7 must not.
    assert "bullet5" in profile
    assert "bullet7" not in profile


# ── _blended_score ───────────────────────────────────────────────────────────
def test_blended_score_weighted_formula():
    comp = {"ats_coverage": 80, "title_fit": 60, "seniority_fit": 70, "education_fit": 100}
    # 0.50*80 + 0.15*60 + 0.20*70 + 0.15*100 = 40 + 9 + 14 + 15 = 78
    assert _blended_score(comp) == 78


def test_blended_score_defaults_missing_components():
    # coverage missing -> 0; fits missing -> neutral 50.
    # 0.50*0 + 0.15*50 + 0.20*50 + 0.15*50 = 0 + 7.5 + 10 + 7.5 = 25
    assert _blended_score({}) == 25


def test_knockout_caps_score():
    comp = {
        "ats_coverage": 100,
        "title_fit": 100,
        "seniority_fit": 100,
        "education_fit": 100,
        "knockouts": ["requires 10+ yrs (résumé shows ~5)"],
    }
    assert _blended_score(comp) == jm._KNOCKOUT_CAP == 40


# ── rank_jobs (faked client) ─────────────────────────────────────────────────
def _jobs_df(descriptions):
    return pd.DataFrame(
        {
            "title": [f"Job {i}" for i in range(len(descriptions))],
            "company": [f"Co {i}" for i in range(len(descriptions))],
            "description": descriptions,
        }
    )


def test_full_description_reaches_prompt_with_temperature_zero(patch_client):
    sentinel = "SENTINEL_KW_BEYOND_300"
    long_desc = ("A" * 350) + sentinel + ("B" * 100)  # sentinel sits past char 300
    client = patch_client(_make_handler())

    rank_jobs(_jobs_df([long_desc]), {"summary": "x"})

    prompt = client.calls[0]["messages"][0]["content"]
    assert sentinel in prompt, "Requirements text past char 300 must reach the model"
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["max_tokens"] == jm._MAX_TOKENS


def test_missing_skills_and_blended_score_columns(patch_client):
    patch_client(_make_handler(per_id={0: {"missing_skills": ["Kubernetes", "gRPC"]}}))

    out = rank_jobs(_jobs_df(["a backend role"]), {"summary": "x"})

    row = out.iloc[0]
    assert row["missing_skills"] == ["Kubernetes", "gRPC"]
    assert row["matched_skills"] == ["Python"]
    assert row["ats_coverage"] == 70
    # default comp -> 0.50*70 + 0.15*80 + 0.20*60 + 0.15*100 = 74
    assert row["match_score"] == 74


def test_batch_failure_yields_none_without_aborting(patch_client):
    # 6 jobs -> batch size 5 -> two batches. The batch holding job 0 raises (5 jobs),
    # the other (job 5) still scores.
    client = patch_client(_make_handler(fail_batch_with=0))

    out = rank_jobs(_jobs_df([f"desc {i}" for i in range(6)]), {"summary": "x"})

    assert client.calls and len(out) == 6
    scored = out["match_score"].notna().sum()
    assert scored == 1, "only the surviving batch (1 job) should be scored"
    # The failed jobs carry a None score (not 0) and an explanatory reason.
    failed = out[out["match_score"].isna()]
    assert len(failed) == 5
    assert failed["match_reason"].str.contains("scoring error").all()


def test_all_batches_failing_raises(patch_client):
    def _always_raise(kwargs):
        raise RuntimeError("bad key")

    patch_client(_always_raise)
    with pytest.raises(RuntimeError):
        rank_jobs(_jobs_df(["a", "b"]), {"summary": "x"})


def test_empty_df_short_circuits():
    # No client needed; empty input returns unchanged without calling Groq.
    out = rank_jobs(pd.DataFrame(), {"summary": "x"})
    assert out.empty
