import json
import os

from groq import Groq
import pandas as pd


def _get_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file and restart the app.")
    return Groq(api_key=key)


def _build_candidate_profile(resume: dict) -> str:
    """Build a text profile from whatever resume sections are available.

    Missing/empty sections are skipped so a resume without (say) a summary or
    education section still produces a usable profile.
    """
    parts: list[str] = []

    summary = (resume.get("summary") or "").strip()
    if summary:
        parts.append(f"Summary: {summary}")

    yoe = resume.get("total_years_experience")
    if yoe is not None and str(yoe).strip() != "":
        try:
            parts.append(f"Total professional experience: ~{float(yoe):g} years")
        except (TypeError, ValueError):
            pass

    skills = resume.get("skills") or []
    if skills:
        parts.append("Skills: " + ", ".join(str(s) for s in skills))

    experience = resume.get("experience") or []
    exp_lines = []
    for e in experience:
        if not isinstance(e, dict):
            continue
        header = " — ".join(
            str(e[k]) for k in ("title", "company", "duration") if e.get(k)
        )
        bullets = "; ".join(str(b) for b in (e.get("bullets") or [])[:4])
        line = header + (f": {bullets}" if bullets else "")
        if line:
            exp_lines.append(f"- {line}")
    if exp_lines:
        parts.append("Experience:\n" + "\n".join(exp_lines))

    education = resume.get("education") or []
    edu_lines = []
    for ed in education:
        if not isinstance(ed, dict):
            continue
        line = " — ".join(
            str(ed[k]) for k in ("degree", "institution", "year") if ed.get(k)
        )
        if line:
            edu_lines.append(f"- {line}")
    if edu_lines:
        parts.append("Education:\n" + "\n".join(edu_lines))

    return "\n\n".join(parts)


def _score_batch(client: Groq, candidate_profile: str, jobs: list[dict]) -> list[dict]:
    job_list_text = "\n".join(
        f"{i+1}. job_id={j['job_id']} | Title: {j['title']} | Company: {j['company']} | "
        f"Description snippet: {str(j.get('description', ''))[:300]}"
        for i, j in enumerate(jobs)
    )

    prompt = (
        "You are a job-fit evaluator. Given the candidate's profile below, score each job "
        "0–100 for fit. Weigh the candidate's summary, skills, experience, and education "
        "against each job, AND factor seniority/level alignment: compare the candidate's total "
        "years of experience (use 'Total professional experience' if given, otherwise infer it "
        "from the Experience section) against the experience level each job requires. "
        "Significantly lower the score for jobs that require substantially MORE experience than "
        "the candidate has (e.g. Senior/Staff/Principal/Lead/Director/Manager roles for a "
        "mid-level candidate), and also lower it for jobs clearly below the candidate's level. "
        "Mention the level fit in the reason. Return ONLY valid JSON array (no markdown):\n"
        '[{"job_id": ..., "score": ..., "reason": "one sentence"}]\n\n'
        f"Candidate profile:\n{candidate_profile}\n\n"
        f"Jobs:\n{job_list_text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def generate_why_interested(resume: dict, job: dict) -> str:
    """Generate a short, first-person "Why do you want to work here?" answer
    tailored to the candidate's résumé and this specific job.

    Returns one concise paragraph (~3-4 sentences). Raises RuntimeError if the
    GROQ_API_KEY is missing (same contract as scoring) so the caller can surface it.
    """
    client = _get_client()
    candidate_profile = _build_candidate_profile(resume)

    title = job.get("title", "")
    company = job.get("company", "")
    description = str(job.get("description", ""))[:1500]

    prompt = (
        "Write a first-person answer to the interview/application question "
        f'"Why do you want to work here?" for the {title} role at {company}. '
        "Use ONE concise paragraph of 3-4 sentences. Connect the candidate's actual "
        "background and skills to specifics of this role and company. Be genuine and "
        "specific — avoid generic filler, clichés, and flattery. Do not invent facts "
        "about the candidate. Return only the paragraph, no preamble or quotes.\n\n"
        f"Candidate profile:\n{candidate_profile}\n\n"
        f"Job title: {title}\n"
        f"Company: {company}\n"
        f"Job description:\n{description}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def rank_jobs(jobs_df: pd.DataFrame, resume: dict) -> pd.DataFrame:
    """Score and sort jobs against the resume.

    Raises RuntimeError if scoring fails for every batch (e.g. a bad/missing
    GROQ_API_KEY) so the caller can surface a real error instead of silently
    showing 0/100 for everything.
    """
    if jobs_df.empty:
        return jobs_df

    client = _get_client()
    candidate_profile = _build_candidate_profile(resume)

    records = jobs_df.to_dict("records")
    for i, r in enumerate(records):
        r["job_id"] = i

    scores: list[dict] = []
    errors: list[str] = []
    succeeded = 0
    batch_size = 10
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            scores.extend(_score_batch(client, candidate_profile, batch))
            succeeded += 1
        except Exception as e:
            errors.append(str(e))
            for job in batch:
                # None (not 0) so a scoring failure isn't mistaken for a real "no match"
                scores.append(
                    {"job_id": job["job_id"], "score": None, "reason": f"scoring error: {e}"}
                )

    if succeeded == 0:
        raise RuntimeError(errors[0] if errors else "scoring failed")

    score_map = {s["job_id"]: s for s in scores}
    jobs_df = jobs_df.copy()
    jobs_df["match_score"] = pd.to_numeric(
        pd.Series([score_map.get(i, {}).get("score") for i in range(len(jobs_df))]),
        errors="coerce",
    )
    jobs_df["match_reason"] = [score_map.get(i, {}).get("reason", "") for i in range(len(jobs_df))]

    return jobs_df.sort_values(
        "match_score", ascending=False, na_position="last"
    ).reset_index(drop=True)
