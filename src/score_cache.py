"""Remember scores we already paid for.

Scoring is the expensive half of a search — every job is a Groq call, and the
free tier allows 200,000 tokens a day. Job boards return substantially the same
postings day to day, so without a cache a second search for the same role re-buys
scores that have not changed.

The cache is keyed on ``(job_key, resume_key, scorer_version)`` — see the
``score_cache`` DDL in :mod:`src.profile_manager` for why each part is there.

Three rules this module follows, each learned the hard way somewhere else in this
codebase:

1. **The engine is resolved at call time**, never bound at import. Tests
   monkeypatch ``profile_manager.ENGINE``; a module-level ``from ... import
   ENGINE`` would capture the real ``data/jobseeker.db`` and the monkeypatch would
   silently not apply — the tests would pass while writing to the developer's
   database.
2. **Every entry point swallows its own exceptions.** A missing table, a locked
   file, a corrupt payload: none of those are reasons a job search should fail.
   The cache is an optimisation, so its failure mode is "call the API", never
   "show the user an error".
3. **Callers use it from the main thread only.** ``rank_jobs`` scores batches on a
   thread pool; SQLite plus worker threads is a locking problem with nothing to
   gain, so reads happen before the pool and writes after it.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from src import profile_manager

log = logging.getLogger(__name__)

# How long a cached score stays trustworthy. job_signature() deliberately ignores
# the description, so an employer editing the posting under the same title would
# otherwise be served the old score indefinitely. Three weeks is long enough that
# day-to-day re-searching always hits, short enough that a rewritten posting gets
# re-read within a normal job hunt.
_TTL_DAYS = int(os.environ.get("SCORE_CACHE_TTL_DAYS", "21"))

# SQLite's default host-parameter ceiling is 999. Chunk well under it so a large
# result set doesn't turn a cache lookup into an OperationalError.
_MAX_PARAMS = 500


def resume_key(candidate_profile: str) -> str:
    """Identity of the résumé *as the model sees it*.

    Deliberately derived from the rendered profile text rather than ``resume.id``:
    ``save_resume`` INSERTs a new row on every upload, so an id-based key would
    throw the entire cache away when the same file is uploaded twice. The profile
    string is what actually reaches the prompt, so hashing it invalidates when —
    and only when — the prompt would differ.
    """
    return hashlib.sha256(candidate_profile.encode("utf-8")).hexdigest()[:16]


def scorer_version(*parts: str) -> str:
    """Identity of the scoring rubric.

    Without this, editing the instructions or the weights would keep serving
    scores computed under the previous rubric, with nothing in the UI to indicate
    the numbers came from different rules.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")  # length-delimit so ("ab","c") != ("a","bc")
    return h.hexdigest()[:12]


def get_many(
    job_keys: list[str], rkey: str, version: str
) -> dict[str, dict]:
    """Return ``{job_key: component_dict}`` for whatever is cached and still fresh.

    Missing keys simply don't appear — the caller scores those. Never raises.
    """
    if not job_keys:
        return {}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=_TTL_DAYS)).isoformat()
    found: dict[str, dict] = {}

    try:
        engine = profile_manager.ENGINE  # resolved now, not at import — see docstring
        with engine.connect() as conn:
            for start in range(0, len(job_keys), _MAX_PARAMS):
                chunk = job_keys[start : start + _MAX_PARAMS]
                placeholders = ", ".join(f":k{i}" for i in range(len(chunk)))
                params = {f"k{i}": k for i, k in enumerate(chunk)}
                params.update(rk=rkey, ver=version, cutoff=cutoff)
                rows = conn.execute(
                    text(
                        f"SELECT job_key, payload FROM score_cache "
                        f"WHERE job_key IN ({placeholders}) "
                        f"AND resume_key = :rk AND scorer_version = :ver "
                        f"AND scored_at >= :cutoff"
                    ),
                    params,
                ).fetchall()
                for job_key, payload in rows:
                    try:
                        found[job_key] = json.loads(payload)
                    except (TypeError, ValueError):
                        # A corrupt row is just a miss; the job gets re-scored.
                        continue
    except Exception as e:  # noqa: BLE001 - the cache must never break a search
        log.debug("score cache read failed, scoring everything: %s", e)
        return {}

    return found


def put_many(
    scored: list[tuple[str, dict]], rkey: str, version: str
) -> None:
    """Store freshly computed components. Never raises.

    ``scored`` is ``[(job_key, component_dict), ...]``. Callers must not pass
    error markers — a failed batch has no score worth remembering, and caching one
    would make a transient rate limit look like a permanent zero.
    """
    if not scored:
        return

    now = datetime.now(timezone.utc).isoformat()
    try:
        engine = profile_manager.ENGINE
        with engine.connect() as conn:
            for job_key, comp in scored:
                conn.execute(
                    text(
                        "INSERT OR REPLACE INTO score_cache "
                        "(job_key, resume_key, scorer_version, payload, scored_at) "
                        "VALUES (:jk, :rk, :ver, :payload, :at)"
                    ),
                    {
                        "jk": job_key,
                        "rk": rkey,
                        "ver": version,
                        "payload": json.dumps(comp),
                        "at": now,
                    },
                )
            conn.commit()
    except Exception as e:  # noqa: BLE001
        log.debug("score cache write failed, scores are still correct: %s", e)


def purge_expired() -> int:
    """Drop rows past the TTL. Returns how many went. Never raises."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_TTL_DAYS)).isoformat()
    try:
        engine = profile_manager.ENGINE
        with engine.connect() as conn:
            n = conn.execute(
                text("DELETE FROM score_cache WHERE scored_at < :cutoff"),
                {"cutoff": cutoff},
            ).rowcount
            conn.commit()
            return int(n or 0)
    except Exception as e:  # noqa: BLE001
        log.debug("score cache purge failed: %s", e)
        return 0
