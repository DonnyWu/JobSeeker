"""Shared fixtures.

The one thing here is DB isolation, and it is a safety net rather than a
convenience. ``test_profile_manager`` and ``test_job_search_page`` already point
``profile_manager.ENGINE`` at a tmp file themselves, because they always talked to
SQLite. ``test_job_matcher`` never did — until scoring gained a score cache, at
which point ``rank_jobs`` reads and writes the database on every call.

Without an autouse fixture that redirect is easy to forget, and forgetting it
means the suite quietly writes cached scores into the developer's real
``data/jobseeker.db``. The failure is invisible: tests pass, and the app starts
serving scores invented by a test run.

Tests that want their own engine still monkeypatch it; this just guarantees the
default is never the real file.
"""

import pytest
from sqlalchemy import create_engine

import src.profile_manager as pm
import src.ratelimit as rl


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point every test at a throwaway SQLite file, schema already created."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(pm, "ENGINE", engine)
    pm.init_db()
    yield engine


@pytest.fixture(autouse=True)
def _no_throttling(monkeypatch):
    """Give the pacer an effectively unlimited budget in tests.

    ``TokenBudget.acquire`` really does sleep when a minute's budget is spent —
    that is the entire point of it in production. Under the assumed free-tier
    ceiling of 8,000 tokens/min, a suite that scores a few dozen fake jobs blocks
    for real minutes waiting out windows that only exist because the default is
    pessimistic.

    Tests about scoring should not be tests about wall-clock, so the ceiling is
    raised here. The pacer's own arithmetic is covered directly in
    ``test_ratelimit.py`` with an explicit budget, where sleeping is the subject
    rather than an obstacle.
    """
    monkeypatch.setattr(rl, "_DEFAULT_TPM", 10_000_000)
