import json
import os

from groq import Groq
import pandas as pd


def _get_client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _score_batch(client: Groq, resume_summary: str, skills: list, jobs: list[dict]) -> list[dict]:
    job_list_text = "\n".join(
        f"{i+1}. job_id={j['job_id']} | Title: {j['title']} | Company: {j['company']} | "
        f"Description snippet: {str(j.get('description', ''))[:300]}"
        for i, j in enumerate(jobs)
    )

    prompt = (
        "You are a job-fit evaluator. Given the candidate's resume summary and skills, "
        "score each job 0–100 for fit. Return ONLY valid JSON array (no markdown):\n"
        '[{"job_id": ..., "score": ..., "reason": "one sentence"}]\n\n'
        f"Candidate summary: {resume_summary}\n"
        f"Skills: {', '.join(skills)}\n\n"
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


def rank_jobs(jobs_df: pd.DataFrame, resume: dict) -> pd.DataFrame:
    if jobs_df.empty:
        return jobs_df

    client = _get_client()
    summary = resume.get("summary", "")
    skills = resume.get("skills", [])

    records = jobs_df.to_dict("records")
    for i, r in enumerate(records):
        r["job_id"] = i

    scores: list[dict] = []
    batch_size = 10
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            batch_scores = _score_batch(client, summary, skills, batch)
            scores.extend(batch_scores)
        except Exception:
            for job in batch:
                scores.append({"job_id": job["job_id"], "score": 0, "reason": "scoring unavailable"})

    score_map = {s["job_id"]: s for s in scores}
    jobs_df = jobs_df.copy()
    jobs_df["match_score"] = [score_map.get(i, {}).get("score", 0) for i in range(len(jobs_df))]
    jobs_df["match_reason"] = [score_map.get(i, {}).get("reason", "") for i in range(len(jobs_df))]

    return jobs_df.sort_values("match_score", ascending=False).reset_index(drop=True)
