"""Tests for the role-tailored AI company summary (src/company_insights.py).

These pin the best-effort contract that replaced the old Indeed/Glassdoor scraper:

1. The web-search model (``groq/compound``) is tried first → ``source == "web"``.
2. If the web-search call fails, it falls back to ``llama-3.3-70b-versatile`` →
   ``source == "general"``.
3. If everything fails (or the API key is missing), it returns an empty summary
   instead of raising, so the UI can degrade gracefully.

Pure/in-memory: the Groq client is monkeypatched, so no network or GROQ_API_KEY is
required.
"""

import pytest

import src.company_insights as ci
from src.company_insights import company_summary, _prompt, _WEB_MODEL, _TEXT_MODEL


# ── Fake Groq client ─────────────────────────────────────────────────────────
class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _Resp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


class _FakeClient:
    """Records each create() call and replies per-model from ``replies``.

    A reply value may be a string (returned as content) or an Exception instance
    (raised) to simulate a model being unavailable.
    """

    def __init__(self, replies: dict):
        self._replies = replies
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, *, model, messages, **kw):
        self.calls.append({"model": model, "messages": messages})
        reply = self._replies.get(model)
        if isinstance(reply, Exception):
            raise reply
        return _Resp(reply)


def _patch(monkeypatch, replies: dict) -> _FakeClient:
    client = _FakeClient(replies)
    monkeypatch.setattr(ci, "_get_client", lambda: client)
    return client


# ── company_summary ──────────────────────────────────────────────────────────
def test_web_model_used_first(monkeypatch):
    client = _patch(monkeypatch, {_WEB_MODEL: "**What people like** — great team"})
    out = company_summary("Acme", "Software Engineer II")
    assert out == {"summary": "**What people like** — great team", "source": "web"}
    # Only the web model is called when it succeeds — no wasted fallback call.
    assert [c["model"] for c in client.calls] == [_WEB_MODEL]


def test_falls_back_to_text_model(monkeypatch):
    client = _patch(
        monkeypatch,
        {_WEB_MODEL: RuntimeError("compound unavailable"), _TEXT_MODEL: "general summary"},
    )
    out = company_summary("Acme", "Software Engineer II")
    assert out == {"summary": "general summary", "source": "general"}
    assert [c["model"] for c in client.calls] == [_WEB_MODEL, _TEXT_MODEL]


def test_blank_web_response_falls_back(monkeypatch):
    # An empty/whitespace web reply is treated as a miss and falls through.
    client = _patch(monkeypatch, {_WEB_MODEL: "   ", _TEXT_MODEL: "general summary"})
    out = company_summary("Acme", "Software Engineer")
    assert out["source"] == "general"
    assert [c["model"] for c in client.calls] == [_WEB_MODEL, _TEXT_MODEL]


def test_both_models_fail_returns_empty(monkeypatch):
    _patch(
        monkeypatch,
        {_WEB_MODEL: RuntimeError("x"), _TEXT_MODEL: RuntimeError("y")},
    )
    assert company_summary("Acme", "Software Engineer") == {"summary": "", "source": ""}


def test_missing_api_key_returns_empty(monkeypatch):
    # _get_client raises when GROQ_API_KEY is unset — must not propagate.
    def _boom():
        raise RuntimeError("GROQ_API_KEY is not set.")

    monkeypatch.setattr(ci, "_get_client", _boom)
    assert company_summary("Acme", "Software Engineer") == {"summary": "", "source": ""}


def test_empty_company_skips_client(monkeypatch):
    client = _patch(monkeypatch, {_WEB_MODEL: "should not be used"})
    assert company_summary("", "Software Engineer") == {"summary": "", "source": ""}
    assert client.calls == []  # no API call made for a blank company


# ── _prompt ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("web", [True, False])
def test_prompt_includes_company_role_and_salary(web):
    p = _prompt("Acme", "Software Engineer II", web=web)
    assert "Acme" in p
    assert "Software Engineer II" in p
    assert "salary" in p.lower()


def test_web_prompt_mentions_web_sources_text_prompt_does_not():
    web = _prompt("Acme", "Software Engineer", web=True)
    text = _prompt("Acme", "Software Engineer", web=False)
    assert "Glassdoor" in web and "web" in web.lower()
    assert "Glassdoor" not in text


def test_prompt_defaults_blank_title_to_this_role():
    assert "this role" in _prompt("Acme", "", web=True)
