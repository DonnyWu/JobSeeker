import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import contextmanager
from itertools import zip_longest

import pandas as pd

# State maps + offline geocoding live in src.geo (the lower-level module); import
# them here so location parsing stays in one place and there's no import cycle.
from src.geo import (
    _STATE_ABBR,
    _STATE_NAME,
    geocode,
    haversine_miles,
    parse_locations,
)

# Same key the Apply/Job Search pages use to recognize a saved job, reused here to
# recognize one role posted to several boards. Imported from src.jobkey rather
# than src.profile_manager so scraping doesn't drag in the database layer.
from src.jobkey import job_signature


# Scraped one at a time so a single board's failure can't sink the whole search —
# see the comment in scrape_jobs.
log = logging.getLogger(__name__)

_ALL_SITES = ["linkedin", "indeed", "glassdoor", "zip_recruiter", "google"]

# Which boards to actually ask. Overridable because boards block people for
# reasons that have nothing to do with this app and everything to do with the
# network they are on: streamlit.detached.err.log shows ZipRecruiter answering 403
# and Glassdoor 400 on every run for months here, while both work fine elsewhere.
#
# Every board is scraped before any scoring starts, so one that reliably returns
# nothing is pure wait. Dropping it is deliberately a *choice* rather than
# something the app decides on its own: a board that starts working again would
# never be retried, and the user would have no way to find out.
#
#     SCRAPE_BOARDS=linkedin,indeed,google
_SITES = [
    s.strip().lower()
    for s in os.environ.get("SCRAPE_BOARDS", ",".join(_ALL_SITES)).split(",")
    if s.strip()
] or list(_ALL_SITES)

# How long the whole scrape may take before the boards still running are written
# off. Generous, because a board is walked once per location in sequence and a
# multi-city search legitimately takes a while — this is a stop for a hang, not a
# performance budget.
_BOARD_TIMEOUT = float(os.environ.get("SCRAPE_TIMEOUT_SECONDS", "90"))

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


def _location_mask(loc: pd.Series, location: str, radius_miles: int | None) -> pd.Series:
    """Boolean mask of rows matching *one* searched location.

    ``radius_miles=None`` matches on text alone. Otherwise the location is
    geocoded and rows are kept by true distance, falling back to in-state text
    matching for rows we can't geocode (bare states, "United States", small
    towns missing from the dataset) so the out-of-state anti-leak guarantee
    survives. A location that won't geocode *itself* (a bare "MA") has no centre
    to measure from, so a radius is meaningless and it falls back to text too.
    """
    names, abbrevs = _build_targets(location)
    if not names and not abbrevs:
        # Nothing recognisable to match on, so this location constrains nothing.
        return pd.Series(True, index=loc.index)

    origin = geocode(location) if radius_miles is not None else None
    if origin is None:
        return loc.apply(lambda text: _row_matches(text, names, abbrevs))

    def _within(text: str) -> bool:
        pt = geocode(text)
        if pt is not None:
            return haversine_miles(origin, pt) <= radius_miles
        return _row_matches(text, names, abbrevs)  # ungeocodable -> in-state fallback

    return loc.apply(_within)


def _keep_matching(jobs: pd.DataFrame, locations, radius_miles: int | None) -> pd.DataFrame:
    """Keep jobs matching *any* searched location, plus explicitly remote roles.

    The union is the point of multi-location search: someone who listed New
    York, Boston and San Francisco wants a job sitting in any one of them, so
    the per-location masks are OR-ed rather than intersected (which would keep
    only jobs in all three at once — i.e. nothing).
    """
    locs = parse_locations(locations)
    if not locs or "location" not in jobs.columns:
        return jobs

    loc = jobs["location"].fillna("").astype(str)
    mask = _remote_mask(loc)
    for one in locs:
        mask |= _location_mask(loc, one, radius_miles)
    return jobs[mask]


def _filter_by_location(jobs: pd.DataFrame, locations) -> pd.DataFrame:
    """Keep only jobs in the searched locations, plus any explicitly remote roles."""
    return _keep_matching(jobs, locations, None)


def _filter_by_radius(jobs: pd.DataFrame, locations, radius_miles: int) -> pd.DataFrame:
    """Keep jobs within ``radius_miles`` of any searched location — across state lines.

    A precise *superset* of :func:`_filter_by_location`: remote-by-text rows are
    always kept, geocodable rows are kept if they sit inside the radius of at
    least one location, and ungeocodable rows fall back to in-state text
    matching.
    """
    return _keep_matching(jobs, locations, radius_miles)


def _interleave_by_location(jobs: pd.DataFrame) -> pd.DataFrame:
    """Round-robin the rows across the locations that were searched.

    The Job Search page scores only the first chunk of this frame and leaves the
    rest behind the "More Jobs" button, so plain board order would spend that
    whole first batch on whichever city the boards answered for first — you'd
    add three cities and see one. Taking a row from each location in turn puts
    every city into the first batch.
    """
    if jobs.empty or "search_location" not in jobs.columns:
        return jobs
    groups = [g.index.tolist() for _, g in jobs.groupby("search_location", sort=False)]
    if len(groups) < 2:
        return jobs
    order = [i for rank in zip_longest(*groups) for i in rank if i is not None]
    return jobs.loc[order].reset_index(drop=True)


def _dedupe_across_boards(jobs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse one role posted to several boards into a single row.

    jobspy dedupes *within* a board (each scraper tracks its own ``seen_ids``) but
    never *across* them, so a role listed on LinkedIn, Indeed and Google arrives
    as three rows. Keyed on :func:`job_signature` — normalized company|title|city.

    The copy that survives is the *richest* one, not the first one. That
    distinction matters: ``linkedin_fetch_description`` is left off below (it costs
    an extra request per job behind jobspy's delay band, which is unaffordable at
    this result count), so LinkedIn rows arrive with no description at all. Keeping
    whichever row happened to come first would sometimes hand the scorer an empty
    job description while a complete copy of the same posting sat in the discard
    pile. Rows are ranked by whether they carry a description, then by its length,
    then by how many fields are filled in — the last tiebreak favours the boards
    that also return salary and company details.

    Returns the deduped frame plus how many rows were dropped, so the caller can
    explain the smaller count instead of looking like it lost results.
    """
    if jobs.empty:
        return jobs, 0

    keys = [
        job_signature(r.get("company"), r.get("title"), r.get("location"))
        for r in jobs.to_dict("records")
    ]
    if "description" in jobs.columns:
        desc_len = jobs["description"].fillna("").astype(str).str.len()
    else:
        desc_len = pd.Series(0, index=jobs.index)

    # Sorting on several columns goes through np.lexsort, which is stable, so rows
    # tied on all three rank terms keep the order the boards returned them in —
    # the pick stays deterministic instead of varying run to run.
    deduped = (
        jobs.assign(
            _key=keys,
            _has_desc=(desc_len > 0).astype(int),
            _desc_len=desc_len,
            _filled=jobs.notna().sum(axis=1),
        )
        .sort_values(["_has_desc", "_desc_len", "_filled"], ascending=False)
        .drop_duplicates("_key", keep="first")
        .drop(columns=["_key", "_has_desc", "_desc_len", "_filled"])
        .sort_index()  # restore the original board/date ordering of the survivors
    )
    return deduped.reset_index(drop=True), len(jobs) - len(deduped)


class _BoardErrorRecorder(logging.Handler):
    """Records which boards jobspy logged an error for.

    Needed because a blocked board usually doesn't raise. jobspy reports a
    Glassdoor 400 or a ZipRecruiter 403 by logging it and handing back an empty
    list, which is indistinguishable from "this board had no jobs for you" — so
    a board that blocked every single request would quietly shrink your results
    while the app reported every board healthy. Its loggers are named
    ``JobSpy:<Board>`` and have ``propagate`` off, so we attach to them directly
    rather than watching the root logger.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.boards: set[str] = set()  # set.add is atomic; boards are scraped in threads

    def emit(self, record: logging.LogRecord) -> None:
        # "JobSpy:ZipRecruiter" -> "ziprecruiter", matching _SITES' "zip_recruiter"
        # once its underscore is removed.
        _, _, board = record.name.partition(":")
        if board:
            self.boards.add(board.replace("_", "").lower())

    def logged_error(self, site: str) -> bool:
        return site.replace("_", "").lower() in self.boards


@contextmanager
def _record_board_errors():
    """Attach a :class:`_BoardErrorRecorder` to jobspy's per-board loggers."""
    # Import first so jobspy's own create_logger() runs and installs its console
    # handler: it only does that when the logger has no handlers yet, so
    # attaching ours first would silence its logging entirely.
    import jobspy  # noqa: F401

    recorder = _BoardErrorRecorder()
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("JobSpy:")
    ]
    for lg in loggers:
        lg.addHandler(recorder)
    try:
        yield recorder
    finally:
        for lg in loggers:
            lg.removeHandler(recorder)


def _scrape_one_board(site: str, kwargs: dict) -> pd.DataFrame | None:
    """Scrape a single board, returning ``None`` if it failed.

    Boards fail constantly and in two different ways: most log the problem and hand
    back a short list (a 403, a 429 mid-pagination), but some raise instead —
    Google throws ``RetryError`` once it starts serving its "sorry" page. Only the
    raising kind needs catching here; the quiet kind just looks like a thin result.
    """
    from jobspy import scrape_jobs as _scrape

    try:
        jobs = _scrape(site_name=[site], **kwargs)
    except Exception:
        return None
    return pd.DataFrame() if jobs is None else jobs


def _scrape_board(
    site: str, locations: list[str], kwargs: dict
) -> tuple[list[pd.DataFrame], bool]:
    """Scrape one board once per location, walking the locations in sequence.

    Returns the frames that came back, plus whether *every* location failed for
    this board. jobspy takes a single location per call, so several locations
    mean several calls — made one after another rather than in parallel, because
    firing every board x location combination at once is exactly what trips the
    boards' rate limiters (see the RetryError note in :func:`_scrape_one_board`).
    Serialising them means each board sees the same request pattern it saw when
    there was only one location, just repeated.

    A board counts as failed only when it failed for *all* locations: answering
    for New York but refusing for Boston is not a blocked board, and naming it
    as one would send the user chasing a problem that isn't there.
    """
    frames: list[pd.DataFrame] = []
    failures = 0
    for loc in locations:
        df = _scrape_one_board(site, {**kwargs, "location": loc})
        if df is None:
            failures += 1
        elif not df.empty:
            # Tag the rows so _interleave_by_location can round-robin them.
            frames.append(df.assign(search_location=loc))
    return frames, failures == len(locations)


def _with_meta(jobs: pd.DataFrame, merged: int, failed: list[str]) -> pd.DataFrame:
    """Attach the counts the Job Search page reports back to the user."""
    jobs.attrs["duplicates_merged"] = merged
    jobs.attrs["boards_failed"] = failed
    return jobs


def scrape_jobs(
    query: str,
    locations,
    hours_old_label: str = "Last 24 hours",
    # Per *board, per location* — jobspy runs every site with this cap, so five
    # boards across three cities can return fifteen times this number before
    # deduping.
    #
    # Back to 50 from the 150 that shipped with pagination. That depth was never
    # reaching the user: the Job Search page scores _SCORE_CHUNK (100) jobs per
    # click regardless of how many were scraped, so the extra 500 rows a
    # single-city search pulled at 150 just sat in `pending_df`. What they did
    # cost was wall-clock — locations are walked in sequence per board (see
    # _scrape_board), so the scrape is the slowest part of a search and it scaled
    # straight off this number.
    #
    # 50 still fills a 100-job chunk comfortably (five boards ≈ 250 rows for one
    # city before dedupe) while cutting the pre-scoring wait roughly threefold.
    results_wanted: int = 50,
    is_remote: bool = False,
    distance_miles: int = 50,
) -> pd.DataFrame:
    """Scrape every board for every location and return one merged frame.

    ``locations`` takes a list of locations, or a single string for the
    one-location case.
    """
    hours_old = HOURS_OLD_MAP.get(hours_old_label, 24)
    # Falling back to [""] keeps what an empty location box has always meant:
    # let the boards search nationwide rather than refusing to search at all.
    locs = parse_locations(locations) or [""]

    common = dict(
        search_term=query,
        # Server-side radius (honored by Indeed/LinkedIn/Glassdoor/ZipRecruiter)
        # gives the boards a first-pass net; _filter_by_radius then trims to the
        # exact mileage client-side. 0 means "exact city only". ``location`` is
        # filled in per call by _scrape_board.
        distance=max(distance_miles, 1),
        is_remote=is_remote,
        hours_old=hours_old,
        results_wanted=results_wanted,
        country_indeed="USA",
    )

    # One jobspy call per board instead of one call for all five. jobspy threads the
    # boards internally but collects them with a bare ``future.result()``, so a
    # single board raising takes the whole search down and every other board's
    # results are lost with it — Google in particular raises RetryError once its
    # rate limiter trips, which asking each board for 150 results makes far easier
    # to hit. Isolating the calls means a blocked board costs you that board only.
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    rows_from: dict[str, int] = {}
    raised_everywhere: dict[str, bool] = {}
    with _record_board_errors() as errors:
        # Deliberately NOT `with ThreadPoolExecutor(...)`. The context manager's
        # __exit__ always calls shutdown(wait=True), which blocks until every
        # worker finishes — including the hung one. A `with` block here would
        # catch the timeout, log it, and then block anyway on the way out,
        # reinstating the exact hang this guard exists to prevent. (Measured: a
        # board hanging 8s returned in 8.7s against a 0.4s timeout.)
        #
        # Future.cancel() is no help either: it only drops work that has not
        # started, and cannot interrupt a thread already blocked in a socket read.
        pool = ThreadPoolExecutor(max_workers=len(_SITES))
        timed_out = False
        try:
            futures = {pool.submit(_scrape_board, s, locs, common): s for s in _SITES}
            # A board that *fails* is already handled — _scrape_board absorbs it.
            # A board that *hangs* is what this catches: the other boards' results
            # are kept and the stalled one is reported like any other failure.
            try:
                for future in as_completed(futures, timeout=_BOARD_TIMEOUT):
                    site = futures[future]
                    board_frames, all_failed = future.result()
                    rows_from[site] = sum(len(f) for f in board_frames)
                    raised_everywhere[site] = all_failed
                    frames.extend(board_frames)
            except FuturesTimeout:
                timed_out = True
                stalled = [s for f, s in futures.items() if not f.done()]
                for site in stalled:
                    raised_everywhere[site] = True
                    rows_from.setdefault(site, 0)
                log.warning("board(s) timed out after %ss: %s",
                            _BOARD_TIMEOUT, ", ".join(stalled))
        finally:
            # wait=False only on the timeout path, so a thread stuck in a socket
            # read cannot hold the search open. It keeps running and is abandoned;
            # a normal run still joins its workers cleanly.
            pool.shutdown(wait=not timed_out, cancel_futures=timed_out)

    # A board is called out only when it produced nothing at all *and* something
    # went wrong — it either raised every time or logged an error. A board that
    # answered for one location but not another isn't blocked, and a board that
    # simply had no matching jobs isn't either; neither should send the user
    # chasing a problem that doesn't exist.
    failed = [
        site
        for site in _SITES
        if not rows_from.get(site)
        and (raised_everywhere.get(site) or errors.logged_error(site))
    ]

    if not frames:
        return _with_meta(pd.DataFrame(), 0, failed)

    # Drop all-NA columns per board before concatenating: boards fill different
    # subsets of jobspy's columns, and pandas warns (and will change dtypes in a
    # future release) when an all-NA frame decides a column's type. Same guard
    # jobspy applies internally when it merges its own per-board frames.
    jobs = pd.concat(
        [f.dropna(axis=1, how="all") for f in frames], ignore_index=True
    )

    # The boards (especially Google) leak out-of-state and remote results, so
    # constrain to within the chosen radius of the searched location. Skip for
    # "Remote only" searches, where every result is meant to be remote regardless
    # of HQ location.
    if not is_remote:
        jobs = _filter_by_radius(jobs, locs, distance_miles)
        if jobs.empty:
            return _with_meta(pd.DataFrame(), 0, failed)

    keep = [
        "title", "company", "location", "job_url", "date_posted", "site", "description",
        # Extra company/job fields jobspy returns for free — used by the
        # "More about the company" dropdown on the Job Search page.
        "job_url_direct", "job_level",
        "company_industry", "company_url", "company_url_direct", "company_logo",
        "company_num_employees", "company_revenue", "company_description",
        "company_rating", "company_reviews_count",
        "min_amount", "max_amount", "currency", "interval",
        # Which of the searched locations turned this row up — drives the
        # round-robin below so the first scored batch spans every city.
        "search_location",
    ]
    existing = [c for c in keep if c in jobs.columns]
    jobs = jobs[existing].dropna(subset=["title", "company"])

    # Dedupe here, at the scrape boundary, rather than in the page: everything
    # downstream (the shield, then scoring) runs per row, so collapsing duplicates
    # first is what stops us paying Groq to score the same posting three times.
    jobs, merged = _dedupe_across_boards(jobs)
    jobs = _interleave_by_location(jobs)
    return _with_meta(jobs, merged, failed)
