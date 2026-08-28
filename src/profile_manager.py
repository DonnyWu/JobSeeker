import os

from sqlalchemy import create_engine, text

# The job identity key now lives in src.jobkey so the scraper can dedupe rows
# without importing this (database-bound) module. Re-exported under its original
# name because this module's callers and tests already import it from here.
from src.jobkey import job_signature

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jobseeker.db")
ENGINE = create_engine(f"sqlite:///{os.path.abspath(DB_PATH)}")

_DDL = """
CREATE TABLE IF NOT EXISTS profile (
    id              INTEGER PRIMARY KEY,
    name            TEXT,
    email           TEXT,
    phone           TEXT,
    current_company TEXT,
    linkedin        TEXT,
    portfolio       TEXT,
    github          TEXT
);

CREATE TABLE IF NOT EXISTS resume (
    id          INTEGER PRIMARY KEY,
    file_name   TEXT,
    raw_text    TEXT,
    skills      TEXT,
    experience  TEXT,
    education   TEXT,
    summary     TEXT,
    uploaded_at TEXT
);

CREATE TABLE IF NOT EXISTS saved_jobs (
    id           INTEGER PRIMARY KEY,
    title        TEXT,
    company      TEXT,
    location     TEXT,
    url          TEXT,
    company_url  TEXT,
    match_score  INTEGER,
    match_reason TEXT,
    source       TEXT,
    posted_at    TEXT,
    status       TEXT DEFAULT 'saved',
    outcome         TEXT,
    interview_stage TEXT,
    applied_at      TEXT
);

CREATE TABLE IF NOT EXISTS search_prefs (
    id           INTEGER PRIMARY KEY,
    query        TEXT,
    location     TEXT,
    time_filter  TEXT,
    is_remote    INTEGER,
    min_score    INTEGER,
    distance     INTEGER
);

CREATE TABLE IF NOT EXISTS score_cache (
    job_key        TEXT NOT NULL,
    resume_key     TEXT NOT NULL,
    scorer_version TEXT NOT NULL,
    payload        TEXT NOT NULL,
    scored_at      TEXT NOT NULL,
    PRIMARY KEY (job_key, resume_key, scorer_version)
);
"""
# ^ Every part of that key earns its place:
#
#   job_key        jobkey.job_signature(company, title, city) — the same identity
#                  the scraper dedupes on and saved_jobs recognises across
#                  sessions. Not a new notion of "same job".
#   resume_key     a hash of the *rendered candidate profile*, not resume.id.
#                  save_resume always INSERTs a new row, so an id-based key would
#                  discard the whole cache every time the same file is re-uploaded.
#                  The profile text is literally what goes into the prompt, so
#                  hashing it invalidates exactly when the prompt changes.
#   scorer_version a hash of the instructions, the data guard and _JD_CHARS.
#                  Without it, editing the rubric would keep serving scores
#                  computed under the old one, forever, with nothing to show why.
#
# Note for anyone extending _DDL: init_db() splits it on ";", so no statement may
# contain an embedded semicolon.


def init_db():
    with ENGINE.connect() as conn:
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()

        # Lightweight migration: add columns missing from older databases.
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(resume)")).fetchall()]
        if "total_years_experience" not in cols:
            conn.execute(text("ALTER TABLE resume ADD COLUMN total_years_experience REAL"))
            conn.commit()

        sj_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(saved_jobs)")).fetchall()]
        if "job_key" not in sj_cols:
            conn.execute(text("ALTER TABLE saved_jobs ADD COLUMN job_key TEXT"))
            conn.commit()

        # Post-application outcome tracking (added to older databases).
        for col in ("outcome", "interview_stage", "applied_at"):
            if col not in sj_cols:
                conn.execute(text(f"ALTER TABLE saved_jobs ADD COLUMN {col} TEXT"))
                conn.commit()

        # Search radius slider (added to older databases).
        sp_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(search_prefs)")).fetchall()]
        if "distance" not in sp_cols:
            conn.execute(text("ALTER TABLE search_prefs ADD COLUMN distance INTEGER"))
            conn.commit()


def get_profile() -> dict:
    with ENGINE.connect() as conn:
        row = conn.execute(text("SELECT * FROM profile LIMIT 1")).fetchone()
    if row is None:
        return {}
    return dict(row._mapping)


def save_profile(data: dict):
    with ENGINE.connect() as conn:
        existing = conn.execute(text("SELECT id FROM profile LIMIT 1")).fetchone()
        if existing:
            conn.execute(
                text(
                    "UPDATE profile SET name=:name, email=:email, phone=:phone, "
                    "current_company=:current_company, linkedin=:linkedin, "
                    "portfolio=:portfolio, github=:github WHERE id=:id"
                ),
                {**data, "id": existing[0]},
            )
        else:
            conn.execute(
                text(
                    "INSERT INTO profile (name, email, phone, current_company, linkedin, portfolio, github) "
                    "VALUES (:name, :email, :phone, :current_company, :linkedin, :portfolio, :github)"
                ),
                data,
            )
        conn.commit()


def get_search_prefs() -> dict:
    """Return the last-used Job Search inputs, or {} if none saved yet."""
    with ENGINE.connect() as conn:
        row = conn.execute(text("SELECT * FROM search_prefs LIMIT 1")).fetchone()
    if row is None:
        return {}
    return dict(row._mapping)


def save_search_prefs(data: dict):
    """Upsert the singleton row of Job Search inputs (role, location, filters)."""
    data = {"distance": None, **data}  # tolerate callers that omit the radius
    with ENGINE.connect() as conn:
        existing = conn.execute(text("SELECT id FROM search_prefs LIMIT 1")).fetchone()
        if existing:
            conn.execute(
                text(
                    "UPDATE search_prefs SET query=:query, location=:location, "
                    "time_filter=:time_filter, is_remote=:is_remote, "
                    "min_score=:min_score, distance=:distance WHERE id=:id"
                ),
                {**data, "id": existing[0]},
            )
        else:
            conn.execute(
                text(
                    "INSERT INTO search_prefs "
                    "(query, location, time_filter, is_remote, min_score, distance) "
                    "VALUES (:query, :location, :time_filter, :is_remote, :min_score, :distance)"
                ),
                data,
            )
        conn.commit()


def save_resume(file_name: str, raw_text: str, parsed: dict):
    import json
    from datetime import datetime

    tye = parsed.get("total_years_experience")
    try:
        tye = float(tye) if tye is not None and str(tye).strip() != "" else None
    except (TypeError, ValueError):
        tye = None

    with ENGINE.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO resume (file_name, raw_text, skills, experience, education, summary, "
                "total_years_experience, uploaded_at) "
                "VALUES (:file_name, :raw_text, :skills, :experience, :education, :summary, "
                ":total_years_experience, :uploaded_at)"
            ),
            {
                "file_name": file_name,
                "raw_text": raw_text,
                "skills": json.dumps(parsed.get("skills", [])),
                "experience": json.dumps(parsed.get("experience", [])),
                "education": json.dumps(parsed.get("education", [])),
                "summary": parsed.get("summary", ""),
                "total_years_experience": tye,
                "uploaded_at": datetime.utcnow().isoformat(),
            },
        )
        conn.commit()


def get_latest_resume() -> dict:
    """The parsed résumé, without the raw text.

    Named columns rather than SELECT *: ``raw_text`` holds the entire extracted
    document — up to 20,000 characters — and nothing reads it back. It is written
    for provenance, and ``parse_resume`` takes it as an argument at upload time
    rather than fetching it here.

    That matters because the Job Search page calls this on *every* Streamlit
    rerun, which is every pagination click, filter toggle and expander. SELECT *
    dragged the whole document out of SQLite each time, largely to answer
    ``bool(resume)``.
    """
    import json

    with ENGINE.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, file_name, skills, experience, education, summary, "
                "uploaded_at, total_years_experience "
                "FROM resume ORDER BY id DESC LIMIT 1"
            )
        ).fetchone()
    if row is None:
        return {}
    r = dict(row._mapping)
    for field in ("skills", "experience", "education"):
        try:
            r[field] = json.loads(r[field] or "[]")
        except Exception:
            r[field] = []
    return r


def _coerce_score(v):
    """Coerce a match score (which may be a float, NaN, pd.NA, or None) to int|None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return int(round(f))


def _job_payload(job: dict) -> dict:
    """Normalize a job dict (results-row or Apply-page record) to saved_jobs columns."""
    return {
        "title": job.get("title", "") or "",
        "company": job.get("company", "") or "",
        "location": job.get("location", "") or "",
        "url": job.get("url") or job.get("job_url") or job.get("company_url") or "",
        "company_url": job.get("company_url", "") or "",
        "match_score": _coerce_score(job.get("match_score")),
        "match_reason": job.get("match_reason", "") or "",
        "source": job.get("source") or job.get("site") or "",
        "posted_at": str(job.get("posted_at") or job.get("date_posted") or ""),
    }


def _upsert_job(job: dict, status: str):
    from datetime import datetime

    payload = _job_payload(job)
    payload["job_key"] = job_signature(
        payload["company"], payload["title"], payload["location"]
    )
    with ENGINE.connect() as conn:
        existing = conn.execute(
            text("SELECT id, status, applied_at FROM saved_jobs WHERE job_key=:k LIMIT 1"),
            {"k": payload["job_key"]},
        ).fetchone()
        if existing:
            # Never downgrade an already-applied job back to 'saved'.
            payload["status"] = "applied" if existing[1] == "applied" else status
            # Stamp the application date the first time a job becomes 'applied'.
            applied_at = existing[2]
            if payload["status"] == "applied" and not applied_at:
                applied_at = datetime.utcnow().isoformat()
            payload["applied_at"] = applied_at
            conn.execute(
                text(
                    "UPDATE saved_jobs SET title=:title, company=:company, location=:location, "
                    "url=:url, company_url=:company_url, match_score=:match_score, "
                    "match_reason=:match_reason, source=:source, posted_at=:posted_at, "
                    "status=:status, job_key=:job_key, applied_at=:applied_at WHERE id=:id"
                ),
                {**payload, "id": existing[0]},
            )
        else:
            payload["status"] = status
            payload["applied_at"] = (
                datetime.utcnow().isoformat() if status == "applied" else None
            )
            conn.execute(
                text(
                    "INSERT INTO saved_jobs (title, company, location, url, company_url, "
                    "match_score, match_reason, source, posted_at, status, job_key, applied_at) "
                    "VALUES (:title, :company, :location, :url, :company_url, "
                    ":match_score, :match_reason, :source, :posted_at, :status, :job_key, :applied_at)"
                ),
                payload,
            )
        conn.commit()


def save_job(job: dict):
    """Upsert a job as 'saved' (won't downgrade one already marked 'applied')."""
    _upsert_job(job, "saved")


def mark_job_applied(job: dict):
    """Upsert a job and mark it 'applied'."""
    _upsert_job(job, "applied")


def unmark_job_applied(job):
    """Revert an applied job back to 'saved'. Accepts a job dict or a job_key str."""
    key = job if isinstance(job, str) else job_signature(
        job.get("company", ""), job.get("title", ""), job.get("location", "")
    )
    with ENGINE.connect() as conn:
        conn.execute(
            text("UPDATE saved_jobs SET status='saved' WHERE job_key=:k"), {"k": key}
        )
        conn.commit()


def get_applied_keys() -> set:
    """Return the set of job_keys the user has marked as applied."""
    with ENGINE.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT job_key FROM saved_jobs "
                "WHERE status='applied' AND job_key IS NOT NULL"
            )
        ).fetchall()
    return {r[0] for r in rows}


def get_applied_scores() -> dict:
    """Return ``{job_key: (match_score, match_reason)}`` for applied jobs.

    Applied jobs come back in later searches for the same role, and re-scoring
    them costs a full slot in the batch — for a decision that has already been
    made. The score recorded when the job was saved is the one that was true when
    you applied, which is arguably the more useful number to show anyway.

    Only rows that actually carry a score are returned; a job saved before scoring
    existed should still be scored normally rather than shown as blank forever.
    """
    with ENGINE.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT job_key, match_score, match_reason FROM saved_jobs "
                "WHERE status='applied' AND job_key IS NOT NULL "
                "AND match_score IS NOT NULL"
            )
        ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def get_applied_jobs() -> list[dict]:
    """Return full records for every job marked 'applied', newest application first.

    Each dict includes the post-application fields (outcome, interview_stage,
    applied_at) so the Applied tab and future analytics can read them directly.
    """
    with ENGINE.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM saved_jobs WHERE status='applied' "
                "ORDER BY applied_at DESC, id DESC"
            )
        ).fetchall()
    return [dict(r._mapping) for r in rows]


# Outcomes a user can record after applying (drives the Applied-tab buttons).
APPLICATION_OUTCOMES = ("interview", "offer", "accepted", "declined")


def update_application_outcome(job_key: str, outcome: str, interview_stage: str | None = None):
    """Set the post-application outcome for an applied job.

    Only updates interview_stage when it is explicitly provided, so recording an
    outcome doesn't wipe a previously entered stage.
    """
    params = {"k": job_key, "outcome": outcome}
    sql = "UPDATE saved_jobs SET outcome=:outcome"
    if interview_stage is not None:
        sql += ", interview_stage=:interview_stage"
        params["interview_stage"] = interview_stage
    sql += " WHERE job_key=:k"
    with ENGINE.connect() as conn:
        conn.execute(text(sql), params)
        conn.commit()


def update_job_company_url(job_id: int, company_url: str):
    with ENGINE.connect() as conn:
        conn.execute(
            text("UPDATE saved_jobs SET company_url=:url WHERE id=:id"),
            {"url": company_url, "id": job_id},
        )
        conn.commit()
