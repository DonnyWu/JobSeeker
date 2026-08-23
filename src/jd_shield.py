"""Defenses against hidden instructions planted in scraped job descriptions.

Job boards serve postings as HTML, and jobspy converts that HTML to Markdown.
The conversion throws away the *styling* that hid a span — white-on-white text,
a 1px font, ``display:none`` — but keeps the *words*. So a "trap" an employer
buried for AI screeners reaches us as ordinary-looking description text, and
today gets concatenated straight into a Groq prompt where the model has no way
to tell it apart from our own instructions.

This module is the boundary every description crosses before it reaches a model:
:func:`inspect` returns the cleaned text plus human-readable flags describing
anything that reads like an instruction to a machine rather than a description
of a job.

Pure string work — no I/O, no network — so it is fast enough to run across a
whole result set and testable without a Groq key.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

__all__ = ["ShieldResult", "sanitize", "sanitize_field", "inspect", "shield_frame"]


# Characters that render as nothing: zero-width spaces and joiners, directional
# marks, bidi embedding/override, word joiner and invisible operators, the BOM,
# and the soft hyphen. They survive the HTML -> Markdown conversion as ordinary
# characters, so a payload spelled with them is invisible on screen and
# perfectly legible to a model.
#
# Built from codepoint ranges rather than written as a string literal on
# purpose: literal invisible characters cannot be seen in source, and editors
# and tooling mangle them silently.
_INVISIBLE = frozenset(
    list(range(0x200B, 0x2010))    # ZWSP, ZWNJ, ZWJ, LRM, RLM
    + list(range(0x202A, 0x202F))  # LRE, RLE, PDF, LRO, RLO
    + list(range(0x2060, 0x2065))  # word joiner, invisible operators
    + [0xFEFF, 0x00AD]             # BOM / ZWNBSP, soft hyphen
)

# Straight and curly quotes, assembled by codepoint for the same reason.
_QUOTES = "".join(chr(c) for c in (0x22, 0x27, 0x2018, 0x2019, 0x201C, 0x201D))
_Q = "[" + re.escape(_QUOTES) + "]"

# A long run of blank lines is how trap text gets pushed below the visible fold.
_BLANK_RUN = re.compile(r"\n{3,}")

# Any whitespace run. Title and company are single-line values by nature, so
# every newline and tab in them collapses to one space.
_WHITESPACE = re.compile(r"\s+")


# Genuine postings describe a role. They do not address a machine, refer to
# earlier instructions, or try to set a score. Each entry is (pattern, label);
# the label is what the user sees, so it is phrased plainly.
#
# Every pattern here is deliberately narrow. A flag paints the job card red, and
# a warning that fires on ordinary postings gets ignored within a week — so the
# bar is "no plausible job description says this", not "this looks suspicious".
# In particular: "as an AI engineer" and "if you are an AI enthusiast" are normal
# phrases in ML postings and must not match, which is why the AI-addressing
# patterns require a model-ish noun or an immediately following clause break.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?"
            r"(?:the\s+|your\s+|these\s+)?"
            r"(?:previous|prior|above|earlier|preceding)\b",
            re.I,
        ),
        "tries to override earlier instructions",
    ),
    (
        re.compile(
            r"\b(?:ignore|disregard|reveal|print|output|repeat|leak)\s+"
            r"(?:the\s+|your\s+)?system\s+prompt\b",
            re.I,
        ),
        "refers to a system prompt",
    ),
    (
        re.compile(
            r"\bif\s+you(?:'re|\s+are)\s+(?:an?\s+)?"
            r"(?:ai|a\.i\.|llm|language\s+model|chatbot|bot)\s*[,:;.]",
            re.I,
        ),
        "addresses an AI directly",
    ),
    (
        re.compile(
            r"\bif\s+you(?:'re|\s+are)\s+(?:an?\s+)?(?:ai|llm)\s+"
            r"(?:model|assistant|system|agent|screener|reading|processing|parsing)\b",
            re.I,
        ),
        "addresses an AI directly",
    ),
    (
        re.compile(
            r"\bas\s+an?\s+(?:ai\s+(?:language\s+model|model|assistant|system)"
            r"|llm|language\s+model)\b",
            re.I,
        ),
        "addresses an AI directly",
    ),
    (
        re.compile(
            r"\b(?:rate|score|rank|mark|grade|classify)\s+(?:this|the)\s+"
            r"(?:candidate|applicant|application|resume|r[eé]sum[eé])\b",
            re.I,
        ),
        "tries to set the candidate's score",
    ),
    (
        re.compile(
            r"\bflag\s+(?:this|the)\s+(?:candidate|applicant|application)\b",
            re.I,
        ),
        "asks for the application to be flagged",
    ),
]

# Canary markers: "include the word X". Only the *quoted* form is matched, and
# that is a deliberate trade-off.
#
# A canary token is arbitrary ("pomegranate"), so the employer has to quote it to
# be unambiguous. A legitimate instruction uses a word that means something in
# context and leaves it bare — "please mention the word referral in your
# application if a current employee referred you". Structurally those two are
# identical; quoting is the only signal separating them.
#
# So an unquoted canary slips past this regex. It is still fenced by the data
# guard in job_matcher, and the model is asked to report injections it notices —
# two of the three layers still apply. Flagging every posting that mentions a
# referral code or a subject-line keyword would paint ordinary job cards red,
# and a warning that fires on ordinary postings stops being read.
_CANARY_QUOTED = re.compile(
    r"\b(?:include|insert|add|use|mention|say|write|begin\s+with|start\s+with)\s+"
    r"(?:the\s+)?(?:word|phrase|term|keyword)s?\b[:\s]*"
    + _Q + r"([^" + re.escape(_QUOTES) + r"\n]{1,40})" + _Q,
    re.I,
)


@dataclass
class ShieldResult:
    """Outcome of inspecting one job description.

    ``text`` is always safe to interpolate into a prompt; ``flags`` is empty for
    an ordinary posting.
    """

    text: str
    flags: list[str] = field(default_factory=list)

    @property
    def is_flagged(self) -> bool:
        return bool(self.flags)


def sanitize(raw) -> str:
    """Return ``raw`` cleaned of anything that hides text from a human reader.

    Accepts whatever the results frame holds: a missing description arrives as
    ``NaN`` (a float), and ``str(NaN)`` is the literal string ``"nan"`` — which
    is what currently reaches the model. Anything that is not a real string
    becomes ``""``.
    """
    if not isinstance(raw, str):
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(ch for ch in text if ord(ch) not in _INVISIBLE)
    return _BLANK_RUN.sub("\n\n", text).strip()


def sanitize_field(raw, limit: int = 200) -> str:
    """Return a short single-line field (``title``, ``company``) safe to embed.

    Same NFKC normalization and invisible-character stripping as :func:`sanitize`,
    plus two rules that only make sense for a one-line value:

    * **All** whitespace collapses to single spaces. A newline in a job *title* is
      never legitimate, and it is exactly how forged prompt structure gets in — a
      fake ``### job_id=`` header, or a fence marker sitting on its own line.
    * The result is length-capped. A 40,000-character "company name" is not a
      company name; it is an attempt to push the real instructions out of the
      model's attention.
    """
    text = _WHITESPACE.sub(" ", sanitize(raw)).strip()
    return text[:limit].strip()


def _canary_flags(text: str) -> list[str]:
    """Labels for any magic word the posting asks to have echoed back."""
    flags: list[str] = []
    for match in _CANARY_QUOTED.finditer(text):
        token = match.group(1).strip().strip(_QUOTES).strip()
        if token:
            flags.append(f"asks for the word {token!r} to be included")
    return flags


def inspect(raw) -> ShieldResult:
    """Clean ``raw`` and flag anything addressed to a machine rather than a person.

    The returned ``text`` is what should be sent to a model — never ``raw``.
    """
    text = sanitize(raw)
    if not text:
        return ShieldResult(text="")

    flags: list[str] = []
    for pattern, label in _PATTERNS:
        if pattern.search(text):
            flags.append(label)
    flags.extend(_canary_flags(text))

    # Two patterns share the "addresses an AI directly" label, and a posting can
    # repeat a canary; keep first-seen order but show each label once.
    return ShieldResult(text=text, flags=list(dict.fromkeys(flags)))


# Fields scraped off the posting page. The description is the obvious one, but the
# title and company come off the *same* page, put there by the *same* party — so
# they are untrusted for exactly the same reason, and get inspected too.
_UNTRUSTED_FIELDS = ("title", "company")


def shield_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the shield's output to a whole results frame.

    Adds two columns:

    * ``_jd_text`` — the cleaned description, and the only description text that
      should ever reach a prompt.
    * ``jd_flags`` — everything suspicious found across the description *and* the
      title and company, deduplicated.

    Called at the scrape boundary rather than inside scoring, so the flags exist
    whether or not the jobs were ever scored: a posting is no less trapped for the
    user not having uploaded a résumé yet.

    Pure string work with no API call, so it costs milliseconds across a whole
    result set, and it is idempotent — re-running it on an already-shielded frame
    produces the same two columns.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    texts: list[str] = []
    flag_lists: list[list[str]] = []

    for record in out.to_dict("records"):
        described = inspect(record.get("description"))
        flags = list(described.flags)
        for name in _UNTRUSTED_FIELDS:
            flags.extend(inspect(sanitize_field(record.get(name))).flags)
        texts.append(described.text)
        # Order-preserving dedup: the same label can arrive from two fields.
        flag_lists.append(list(dict.fromkeys(flags)))

    out["_jd_text"] = texts
    out["jd_flags"] = flag_lists
    return out
