import re

import pandas as pd

# State maps + offline geocoding live in src.geo (the lower-level module); import
# them here so location parsing stays in one place and there's no import cycle.
from src.geo import _STATE_ABBR, _STATE_NAME, geocode, haversine_miles


HOURS_OLD_MAP = {
    "Last 6 hours": 6,
    "Last 24 hours": 24,
    "Last 3 days": 72,
    "Last week": 168,
    "Last month": 720,
}

# A row is treated as remote purely from its location *text*. jobspy's is_remote
# flag is keyword-driven and over-eager (it flags on-site/hybrid out-of-state
# roles that merely mention "remote"), which is what let out-of-state jobs leak.
_REMOTE_RE = r"\bremote\b|\banywhere\b|work from home|\bwfh\b"


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


def _remote_mask(loc: pd.Series) -> pd.Series:
    """Boolean mask of rows whose location text marks them remote (see _REMOTE_RE)."""
    return loc.str.contains(_REMOTE_RE, case=False, regex=True, na=False)


def _filter_by_location(jobs: pd.DataFrame, location: str) -> pd.DataFrame:
    """Keep only jobs in the searched location, plus any explicitly remote roles."""
    if not location or not location.strip() or "location" not in jobs.columns:
        return jobs
    names, abbrevs = _build_targets(location)
    if not names and not abbrevs:
        return jobs

    loc = jobs["location"].fillna("").astype(str)
    local = loc.apply(lambda x: _row_matches(x, names, abbrevs))
    return jobs[local | _remote_mask(loc)]


def _filter_by_radius(jobs: pd.DataFrame, location: str, radius_miles: int) -> pd.DataFrame:
    """Keep jobs within ``radius_miles`` of the typed location — across state lines.

    Geocodes the typed location and each job's "City, ST" to coordinates and keeps
    rows inside the radius. This is a precise *superset* of :func:`_filter_by_location`:

    * Remote-by-text rows are always kept (same as the text filter).
    * Rows we can geocode are kept iff their distance is ``<= radius_miles``.
    * Rows we *can't* geocode (bare state, "United States", small towns missing
      from the dataset) fall back to in-state text matching, preserving the
      out-of-state anti-leak guarantee.

    If the typed location itself can't be geocoded (e.g. a bare state like "MA"),
    a radius is meaningless, so we defer entirely to :func:`_filter_by_location`.
    """
    if not location or not location.strip() or "location" not in jobs.columns:
        return jobs
    origin = geocode(location)
    if origin is None:
        return _filter_by_location(jobs, location)

    names, abbrevs = _build_targets(location)
    loc = jobs["location"].fillna("").astype(str)

    def _within(text: str) -> bool:
        pt = geocode(text)
        if pt is not None:
            return haversine_miles(origin, pt) <= radius_miles
        return _row_matches(text, names, abbrevs)  # ungeocodable -> in-state fallback

    return jobs[loc.apply(_within) | _remote_mask(loc)]


def scrape_jobs(
    query: str,
    location: str,
    hours_old_label: str = "Last 24 hours",
    results_wanted: int = 50,
    is_remote: bool = False,
    distance_miles: int = 50,
) -> pd.DataFrame:
    from jobspy import scrape_jobs as _scrape

    hours_old = HOURS_OLD_MAP.get(hours_old_label, 24)

    jobs = _scrape(
        site_name=["linkedin", "indeed", "glassdoor", "zip_recruiter", "google"],
        search_term=query,
        location=location,
        # Server-side radius (honored by Indeed/LinkedIn/Glassdoor/ZipRecruiter)
        # gives the boards a first-pass net; _filter_by_radius then trims to the
        # exact mileage client-side. 0 means "exact city only".
        distance=max(distance_miles, 1),
        is_remote=is_remote,
        hours_old=hours_old,
        results_wanted=results_wanted,
        country_indeed="USA",
    )

    if jobs is None or jobs.empty:
        return pd.DataFrame()

    # The boards (especially Google) leak out-of-state and remote results, so
    # constrain to within the chosen radius of the searched location. Skip for
    # "Remote only" searches, where every result is meant to be remote regardless
    # of HQ location.
    if not is_remote:
        jobs = _filter_by_radius(jobs, location, distance_miles)
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
