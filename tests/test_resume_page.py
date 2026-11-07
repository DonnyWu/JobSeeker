"""Tests for the Resume page (pages/2_Resume.py).

Driven through Streamlit's AppTest harness with a simulated upload, so these
exercise the real page. Text extraction and the Groq call are monkeypatched
wherever a test is about the page rather than the parser, so nothing touches the
network.

The page used to save each upload as uploads/<name the browser sent>. That name
is attacker-controlled — Streamlit only checks the extension — so an absolute path
or ../ steps in it wrote the file anywhere on disk. The first test pins that shut.

conftest's autouse fixture points profile_manager.ENGINE at a tmp file, so the
real data/jobseeker.db is untouched.
"""

import pytest
from streamlit.testing.v1 import AppTest

import src.profile_manager as pm
import src.resume_parser as rp

PAGE = "pages/2_Resume.py"

_PARSED = {
    "summary": "Backend engineer.",
    "total_years_experience": 6,
    "skills": ["Python"],
    "experience": [],
    "education": [],
}


def _upload(name: str, content: bytes = b"%PDF-1.4 fake") -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.run()
    at.file_uploader[0].upload(name, content, "application/pdf")
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture
def fake_extract(monkeypatch):
    monkeypatch.setattr(rp, "extract_text", lambda file_bytes, filename: "Jane Doe\nEngineer")


@pytest.fixture
def fake_parse(monkeypatch):
    monkeypatch.setattr(rp, "parse_resume", lambda raw_text: dict(_PARSED))


# ── The upload never lands on disk ───────────────────────────────────────────
def test_upload_name_is_never_used_as_a_path(tmp_path, fake_extract, fake_parse):
    """An absolute path as the file name made os.path.join discard uploads/ entirely."""
    target = tmp_path / "escaped.pdf"
    # Drop the drive letter: Streamlit happens to reject "C:\..." (it reads the
    # colon as an NTFS stream), but "\Users\..." passes its check and Windows
    # fills the drive back in. On POSIX there is no drive, so this is the full path.
    _upload(str(target)[len(target.drive):])

    assert not target.exists()
    # The résumé still went where it belongs: the database.
    assert pm.get_latest_resume()["summary"] == "Backend engineer."


def test_successful_upload_is_saved_and_shown(fake_extract, fake_parse):
    at = _upload("resume.pdf")

    assert [s.value for s in at.success] == ["Resume parsed and saved!"]
    # The file name heads the file card now, rather than a "Latest resume:" heading.
    assert any("**resume.pdf**" in m.value for m in at.markdown)
    assert not any("Latest resume" in h.value for h in at.subheader)


def test_old_word_doc_is_not_offered():
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.run()
    assert at.file_uploader[0].allowed_type == [".pdf", ".docx"]


# ── Failures show a message, not a traceback ─────────────────────────────────
def test_unreadable_file_is_refused_before_groq(monkeypatch):
    calls = []
    monkeypatch.setattr(rp, "parse_resume", lambda raw_text: calls.append(raw_text))

    at = _upload("resume.pdf", b"this is not a pdf")

    assert [e.value for e in at.error] == [
        "Couldn't read that file. Please upload a valid PDF or DOCX."
    ]
    assert calls == []
    assert pm.get_latest_resume() == {}


def test_missing_groq_key_names_the_fix(monkeypatch, fake_extract):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    at = _upload("resume.pdf")

    assert len(at.error) == 1
    assert "GROQ_API_KEY" in at.error[0].value
    assert pm.get_latest_resume() == {}


def test_groq_failure_keeps_its_details_off_the_page(monkeypatch, fake_extract):
    """Groq's error text carries the organization ID; it belongs in the terminal."""

    def _boom(raw_text):
        raise Exception("Rate limit reached for organization org_SECRET123")

    monkeypatch.setattr(rp, "parse_resume", _boom)

    at = _upload("resume.pdf")

    assert len(at.error) == 1
    assert "request to Groq failed" in at.error[0].value
    assert "org_SECRET123" not in at.error[0].value


# ── Parse once; show what was extracted ──────────────────────────────────────
@pytest.fixture
def counted_parse(monkeypatch):
    calls = []

    def _parse(raw_text):
        calls.append(raw_text)
        return dict(_PARSED)

    monkeypatch.setattr(rp, "parse_resume", _parse)
    return calls


def _resume_rows() -> int:
    from sqlalchemy import text

    with pm.ENGINE.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM resume")).scalar()


def test_a_rerun_does_not_parse_the_same_upload_again(fake_extract, counted_parse):
    """The file stays in the uploader across reruns, and every click is a rerun.
    Each one used to repeat the Groq call and insert another copy."""
    at = _upload("resume.pdf")
    assert len(counted_parse) == 1

    at.run()
    at.run()

    assert len(counted_parse) == 1
    assert _resume_rows() == 1
    assert not at.success          # the message belongs to the upload, not every rerun


def test_replacing_the_file_parses_the_new_one(fake_extract, counted_parse):
    at = _upload("old.pdf")
    at.file_uploader(key="resume_upload").upload("new.pdf", b"%PDF-1.4 other",
                                                 "application/pdf")
    at.run()
    assert not at.exception, at.exception
    assert len(counted_parse) == 2
    assert pm.get_latest_resume()["file_name"] == "new.pdf"


def test_no_resume_shows_the_upload_card_and_the_trust_note():
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert [h.value for h in at.subheader] == ["Upload your resume"]
    assert any("never saved to disk" in c.value for c in at.caption)
    assert not at.metric


def test_years_of_experience_are_shown(fake_extract, fake_parse):
    at = _upload("resume.pdf")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Experience": "6 yrs", "Skills": "1", "Roles": "0",
                       "Education": "0"}
    assert any(":gray-badge[Python]" in m.value for m in at.markdown)
