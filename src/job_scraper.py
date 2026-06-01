import pandas as pd


HOURS_OLD_MAP = {
    "Last 6 hours": 6,
    "Last 24 hours": 24,
    "Last 3 days": 72,
    "Last week": 168,
    "Last month": 720,
}


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

    keep = ["title", "company", "location", "job_url", "date_posted", "site", "description"]
    existing = [c for c in keep if c in jobs.columns]
    return jobs[existing].dropna(subset=["title", "company"]).reset_index(drop=True)
