"""The stable identity key for a job posting.

Lives here rather than in :mod:`src.profile_manager` because two very different
layers need it and only one of them talks to the database: ``profile_manager``
uses it to recognize a saved/applied job across sessions, and ``job_scraper``
uses it to collapse the same role scraped from several boards into one row. The
scraper has no business importing SQLAlchemy to normalize a string.

Same split as ``jd_shield.fence``, which moved out of ``job_matcher`` once
``company_insights`` needed it too — ``profile_manager`` re-exports
``job_signature`` under its original name so existing importers are unaffected.
"""

import re


def job_signature(company: str = "", title: str = "", location: str = "") -> str:
    """Stable key for a job: normalized company|title|city.

    Used to recognize the same role across searches/boards. City is the first
    comma-part of location so "Boston, MA" and "Boston, MA, US" match, while a
    different city does not.
    """
    def _norm(s) -> str:
        return re.sub(r"\s+", " ", str(s or "").strip().lower())

    city = _norm(str(location or "").split(",")[0])
    return f"{_norm(company)}|{_norm(title)}|{city}"
