import re
from urllib.parse import urlparse


def _get_company_domain(company: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", company).strip().lower().replace(" ", "")
    return cleaned


def find_company_job_url(company: str, title: str, fallback_url: str) -> str:
    try:
        from googlesearch import search

        query = f"{company} {title} careers apply"
        company_domain = _get_company_domain(company)

        for url in search(query, num_results=5, sleep_interval=1):
            parsed = urlparse(url)
            netloc = parsed.netloc.lower().replace("www.", "")
            # Accept URLs that appear to be on the company's own domain
            if company_domain[:6] in netloc or any(
                kw in parsed.path.lower() for kw in ["careers", "jobs", "apply", "job"]
            ):
                # Exclude aggregator sites
                aggregators = {"linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com", "google.com"}
                if not any(agg in netloc for agg in aggregators):
                    return url
    except Exception:
        pass

    return fallback_url
