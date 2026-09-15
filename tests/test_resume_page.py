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
    assert "Latest resume: resume.pdf" in [h.value for h in at.subheader]


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
