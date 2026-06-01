import os
from sqlalchemy import create_engine, text

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
    status       TEXT DEFAULT 'saved'
);
"""


def init_db():
    with ENGINE.connect() as conn:
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
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


def save_resume(file_name: str, raw_text: str, parsed: dict):
    import json
    from datetime import datetime

    with ENGINE.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO resume (file_name, raw_text, skills, experience, education, summary, uploaded_at) "
                "VALUES (:file_name, :raw_text, :skills, :experience, :education, :summary, :uploaded_at)"
            ),
            {
                "file_name": file_name,
                "raw_text": raw_text,
                "skills": json.dumps(parsed.get("skills", [])),
                "experience": json.dumps(parsed.get("experience", [])),
                "education": json.dumps(parsed.get("education", [])),
                "summary": parsed.get("summary", ""),
                "uploaded_at": datetime.utcnow().isoformat(),
            },
        )
        conn.commit()


def get_latest_resume() -> dict:
    import json

    with ENGINE.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM resume ORDER BY id DESC LIMIT 1")
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


def save_job(job: dict):
    with ENGINE.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO saved_jobs (title, company, location, url, company_url, "
                "match_score, match_reason, source, posted_at, status) "
                "VALUES (:title, :company, :location, :url, :company_url, "
                ":match_score, :match_reason, :source, :posted_at, :status)"
            ),
            job,
        )
        conn.commit()


def update_job_company_url(job_id: int, company_url: str):
    with ENGINE.connect() as conn:
        conn.execute(
            text("UPDATE saved_jobs SET company_url=:url WHERE id=:id"),
            {"url": company_url, "id": job_id},
        )
        conn.commit()
