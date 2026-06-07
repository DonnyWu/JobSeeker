import re

import pandas as pd


HOURS_OLD_MAP = {
    "Last 6 hours": 6,
    "Last 24 hours": 24,
    "Last 3 days": 72,
    "Last week": 168,
    "Last month": 720,
}

# Abbreviation → full name for all US states + DC. Used to filter scraped jobs
# down to the searched location, since the job boards (esp. Google) loosely
# interpret location and leak out-of-state / remote results.
_STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
_STATE_NAME = {name.lower(): abbr for abbr, name in _STATE_ABBR.items()}


def _build_targets(location: str) -> tuple[set[str], set[str]]:
    """From a typed location, return (names, abbrevs) to match row locations against.

    ``names`` are substring-matched (cities, full state names); ``abbrevs`` are
    matched as whole tokens (2-letter state codes) so "MA" can't match "Miami".
    """
    names: set[str] = set()
    abbrevs: set[str] = set()
    for part in (p.strip() for p in location.split(",")):
        if not part:
            continue
        pl = part.lower()
        if len(part) == 2 and part.upper() in _STATE_ABBR:        # "MA"
            abbrevs.add(pl)
            names.add(_STATE_ABBR[part.upper()].lower())
        elif pl in _STATE_NAME:                                    # "Massachusetts"
            names.add(pl)
            abbrevs.add(_STATE_NAME[pl].lower())
        elif len(pl) >= 3:                                         # city / other
            names.add(pl)
    return names, abbrevs


def _row_matches(row_loc: str, names: set[str], abbrevs: set[str]) -> bool:
    rl = (row_loc or "").lower()
    if not rl:
        return False
    tokens = set(t for t in re.split(r"[^a-z]+", rl) if t)
    if abbrevs & tokens:
        return True
    return any(n in rl for n in names)


def _filter_by_location(jobs: pd.DataFrame, location: str) -> pd.DataFrame:
    """Keep only jobs in the searched location, plus any explicitly remote roles."""
    if not location or not location.strip() or "location" not in jobs.columns:
        return jobs
    names, abbrevs = _build_targets(location)
    if not names and not abbrevs:
        return jobs

    loc = jobs["location"].fillna("").astype(str)
    local = loc.apply(lambda x: _row_matches(x, names, abbrevs))
    # Decide "remote" from the location text only. jobspy's is_remote flag is
    # keyword-driven and over-eager — it flags on-site/hybrid out-of-state roles
    # that merely mention "remote", which is what let e.g. Maine jobs leak through.
    remote = loc.str.contains(
        r"\bremote\b|\banywhere\b|work from home|\bwfh\b", case=False, regex=True, na=False
    )
    return jobs[local | remote]


def scrape_jobs(
    query: str,
    location: str,
    hours_old_label: str = "Last 24 hours",
    results_wanted: int = 50,
    is_remote: bool = False,
) -> pd.DataFrame:
    from jobspy import scrape_jobs as _scrape

    hours_old = HOURS_OLD_MAP.get(hours_old_label, 24)

    jobs = _scrape(
        site_name=["linkedin", "indeed", "glassdoor", "zip_recruiter", "google"],
        search_term=query,
        location=location,
        is_remote=is_remote,
        hours_old=hours_old,
        results_wanted=results_wanted,
        country_indeed="USA",
    )

    if jobs is None or jobs.empty:
        return pd.DataFrame()

    # The boards (especially Google) leak out-of-state and remote results, so
    # constrain to the searched location. Skip for "Remote only" searches, where
    # every result is meant to be remote regardless of HQ location.
    if not is_remote:
        jobs = _filter_by_location(jobs, location)
        if jobs.empty:
            return pd.DataFrame()

    keep = [
        "title", "company", "location", "job_url", "date_posted", "site", "description",
        # Extra company/job fields jobspy returns for free — used by the
        # "More about the company" dropdown on the Job Search page.
        "job_url_direct", "job_level",
        "company_industry", "company_url", "company_url_direct", "company_logo",
        "company_num_employees", "company_revenue", "company_description",
        "company_rating", "company_reviews_count",
        "min_amount", "max_amount", "currency", "interval",
    ]
    existing = [c for c in keep if c in jobs.columns]
    return jobs[existing].dropna(subset=["title", "company"]).reset_index(drop=True)
