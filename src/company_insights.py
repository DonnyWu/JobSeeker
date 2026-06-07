"""Best-effort company ratings & reviews from Indeed and Glassdoor.

There is no free reviews API for either board, and their live review pages are
bot-protected (Cloudflare / login walls). So everything here is *best-effort*:

- The deep-link builders always return a usable URL (role-filtered where the board
  supports it) so the user can click through even when scraping fails.
- ``fetch_company_insights`` tries to pull an overall rating + a few review snippets
  but NEVER raises — on any network/parse failure it degrades to a links-only result.
- ``summarize_reviews`` only summarizes snippets we actually retrieved, so the AI
  summary is grounded in real text rather than invented.
"""

import json
import re
from urllib.parse import quote_plus

import httpx

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 6.0
_MAX_SNIPPETS = 3
_SNIPPET_CHARS = 320


# ── URL builders (always reliable) ───────────────────────────────────────────
def _indeed_slug(company: str) -> str:
    # Indeed company paths keep the original casing, drop most punctuation, and
    # join words with hyphens, e.g. "Home Depot" -> "Home-Depot".
    cleaned = re.sub(r"[^A-Za-z0-9 &.\-]", "", company).strip()
    return re.sub(r"\s+", "-", cleaned)


def indeed_reviews_url(company: str, title: str = "") -> str:
    url = f"https://www.indeed.com/cmp/{_indeed_slug(company)}/reviews"
    if title:
        url += f"?fjobtitle={quote_plus(title)}"
    return url


def glassdoor_reviews_url(company: str, title: str = "") -> str:
    # Glassdoor's reviews keyword search resolves for arbitrary company names.
    # (Role filtering needs an employer id, so the role is conveyed in UI text.)
    return (
        "https://www.glassdoor.com/Reviews/company-reviews.htm"
        f"?sc.keyword={quote_plus(company)}"
    )


def _empty(url: str) -> dict:
    return {"rating": None, "count": None, "snippets": [], "url": url}


# ── Parsing helpers ──────────────────────────────────────────────────────────
def _rating_from_jsonld(html: str) -> tuple[float | None, int | None]:
    """Pull aggregateRating ratingValue / ratingCount from any JSON-LD block."""
    for block in re.findall(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
    ):
        try:
            data = json.loads(block.strip())
        except Exception:
            continue
        for node in data if isinstance(data, list) else [data]:
            if not isinstance(node, dict):
                continue
            agg = node.get("aggregateRating", node)
            if isinstance(agg, dict) and agg.get("ratingValue") is not None:
                try:
                    rating = float(agg["ratingValue"])
                except (TypeError, ValueError):
                    continue
                count = agg.get("ratingCount") or agg.get("reviewCount")
                try:
                    count = int(count) if count is not None else None
                except (TypeError, ValueError):
                    count = None
                return rating, count
    return None, None


def _rating_from_text(soup) -> float | None:
    m = re.search(r"(\d\.\d)\s*(?:out of 5|/\s*5|stars)", soup.get_text(" "), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _snippets(soup, selectors: list[str]) -> list[str]:
    out: list[str] = []
    for sel in selectors:
        for el in soup.select(sel):
            text = " ".join(el.get_text(" ").split())
            if len(text) >= 40:
                out.append(text[:_SNIPPET_CHARS])
            if len(out) >= _MAX_SNIPPETS:
                return out
    return out


def _fetch_board(url: str, snippet_selectors: list[str]) -> dict:
    """Fetch one reviews page and extract rating/count/snippets. Never raises."""
    result = _empty(url)
    try:
        from bs4 import BeautifulSoup

        resp = httpx.get(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
        if resp.status_code != 200 or not resp.text:
            return result
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        rating, count = _rating_from_jsonld(html)
        if rating is None:
            rating = _rating_from_text(soup)
        result["rating"] = rating
        result["count"] = count
        result["snippets"] = _snippets(soup, snippet_selectors)
    except Exception:
        # Blocked, timed out, or markup changed — fall back to links only.
        pass
    return result


# ── Public API ───────────────────────────────────────────────────────────────
def fetch_company_insights(company: str, title: str = "") -> dict:
    """Best-effort ratings + review snippets from Indeed and Glassdoor.

    Always returns a dict shaped like::

        {"indeed":    {"rating", "count", "snippets", "url"},
         "glassdoor": {"rating", "count", "snippets", "url"}}

    with ``url`` always populated. Never raises.
    """
    if not company:
        return {"indeed": _empty(""), "glassdoor": _empty("")}

    indeed = _fetch_board(
        indeed_reviews_url(company, title),
        ['[data-testid="reviewDescription"]', '[itemprop="reviewBody"]', "div.cmp-Review-text"],
    )
    glassdoor = _fetch_board(
        glassdoor_reviews_url(company, title),
        ['[data-test="review-text"]', 'span[data-test="reviewLink"]', "p.reviewBody"],
    )
    return {"indeed": indeed, "glassdoor": glassdoor}


def summarize_reviews(snippets: list[str], company: str, title: str) -> str:
    """Summarize *real* scraped review snippets in 2-3 sentences via Groq.

    Returns "" when there are no snippets (so callers can show links only instead
    of fabricating sentiment). Never raises.
    """
    snippets = [s for s in (snippets or []) if s and s.strip()]
    if not snippets:
        return ""
    try:
        from src.job_matcher import _get_client

        client = _get_client()
        joined = "\n".join(f"- {s}" for s in snippets[: _MAX_SNIPPETS * 2])
        prompt = (
            "Summarize what current/former employees say about working as a "
            f"\"{title}\" at {company}, based ONLY on these review excerpts. "
            "Write 2-3 neutral sentences covering common positives and negatives. "
            "Do not invent details beyond the excerpts.\n\n"
            f"Review excerpts:\n{joined}"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""
