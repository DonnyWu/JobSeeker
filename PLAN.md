# JobSeeker — AI-Powered Job Matching & Auto-Apply App

## Context
Build a Streamlit web app that parses the user's resume, scrapes major job boards for matching roles, ranks them using Claude AI, and auto-fills job applications via Playwright. The goal is to eliminate manual job hunting: the tool reads your resume, finds the best-fit roles across every major board, and pre-fills application forms with your saved profile.

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | `streamlit` |
| Job scraping | `python-jobspy` (LinkedIn, Indeed, Glassdoor, ZipRecruiter, Google Jobs) |
| Resume parsing | `pdfplumber` (PDF) + `python-docx` (DOCX) |
| AI matching & parsing | `anthropic` (Claude Sonnet 4.6) |
| Browser auto-fill | `playwright` |
| Local storage | `sqlite3` via `sqlalchemy` |
| Data wrangling | `pandas` |

---

## Project Structure

```
JobSeeker/
├── app.py                   # Streamlit entrypoint, sidebar nav
├── requirements.txt
├── .env.example             # ANTHROPIC_API_KEY placeholder
├── data/
│   └── jobseeker.db        # SQLite: profile + saved jobs
├── uploads/                 # Resume files stored here
└── src/
    ├── resume_parser.py     # Extract text → Claude → structured skills/experience JSON
    ├── job_scraper.py       # python-jobspy wrapper; maps time-filter UI → hours_old
    ├── job_matcher.py       # Score each job vs resume with Claude, return ranked list
    ├── company_finder.py    # Search web to find job on company's own careers page
    ├── autofill.py          # Playwright: open URL, detect form fields, fill profile
    └── profile_manager.py  # CRUD for user profile in SQLite
```

---

## Pages (Streamlit multi-page)

### Page 1 — Profile (`pages/1_Profile.py`)
- Form fields: **Name, Email, Phone, Current Company, LinkedIn URL, Portfolio URL, GitHub URL**
- "Save Profile" → persists to SQLite via `profile_manager.py`
- Pre-populates from DB on load so user only fills once

### Page 2 — Resume (`pages/2_Resume.py`)
- File uploader: accepts `.pdf` or `.docx`
- On upload: `resume_parser.py` extracts raw text → sends to Claude with a structured extraction prompt
- Claude returns JSON: `{ skills, experience, education, summary }`
- Parsed result stored in DB and shown in an expandable preview panel

### Page 3 — Job Search (`pages/3_Job_Search.py`)
- Inputs: Job title / keywords, Location (city or "Remote"), Remote toggle
- **Time filter dropdown** → maps to `hours_old` for jobspy:

  | UI label | `hours_old` value |
  |---|---|
  | Last 6 hours | 6 |
  | Last 24 hours | 24 |
  | Last 3 days | 72 |
  | Last week | 168 |
  | Last month | 720 |

- On "Search": `job_scraper.py` → `job_matcher.py` → ranked results table
- Results table columns: Title, Company, Match Score, Source, Posted, Actions
- Per-row action buttons:
  - **"Find on Company Site"** → `company_finder.py` replaces URL with company careers link
  - **"Auto-Apply"** → passes URL to `autofill.py` via Playwright

### Page 4 — Apply (`pages/4_Apply.py`)
- Shows selected job details + profile preview side-by-side
- "Launch Auto-Fill" button: Playwright opens the URL in a headed Chromium window
  - Detects input fields by `name`, `aria-label`, `placeholder`, and adjacent `<label>` text
  - Fills matched fields with profile data (name, email, phone, LinkedIn, etc.)
  - **Pauses for user to review before they click Submit** (safety checkpoint — never auto-submits)

---

## Key Implementation Details

### Resume Parsing (`src/resume_parser.py`)
```python
# pdfplumber for PDF, python-docx for DOCX → raw text
# Claude prompt:
# "Extract from this resume: skills (list), work_experience
#  (list of {company, title, duration, bullets}), education (list),
#  and a 3-sentence professional summary. Return valid JSON only."
```

### Job Scraping (`src/job_scraper.py`)
```python
from jobspy import scrape_jobs
jobs = scrape_jobs(
    site_name=["linkedin", "indeed", "glassdoor", "zip_recruiter", "google"],
    search_term=query,
    location=location,
    hours_old=hours_old,
    results_wanted=50,
    country_indeed="USA"
)
# Returns a pandas DataFrame
```

### Job Matching (`src/job_matcher.py`)
- Batch jobs into groups of 10 (token-limit management)
- Claude prompt: "Given this resume summary and skills, score each job 0–100 for fit. Return JSON: `[{job_id, score, reason}]`"
- Sort final list by score descending; display top matches first

### Company Site Finder (`src/company_finder.py`)
- Use `googlesearch-python` or DuckDuckGo to search: `[company] [job title] careers apply`
- Return the first result that matches the company's own domain
- Fall back to original jobspy URL if nothing credible found within 3 results

### Auto-Fill (`src/autofill.py`)
```python
# Playwright headed Chromium
# Common field mapping:
#   name/aria-label/placeholder contains "first" → profile.first_name
#   ... "last"                                  → profile.last_name
#   type="email" / label "email"                → profile.email
#   type="tel"  / label "phone"                 → profile.phone
#   label "linkedin"                            → profile.linkedin
#   label "github"                              → profile.github
#   label "portfolio" / "website"               → profile.portfolio
# After filling all detected fields → page.pause() for user review
```

---

## Database Schema (`data/jobseeker.db`)

```sql
-- Single-row user profile
CREATE TABLE profile (
  id              INTEGER PRIMARY KEY,
  name            TEXT,
  email           TEXT,
  phone           TEXT,
  current_company TEXT,
  linkedin        TEXT,
  portfolio       TEXT,
  github          TEXT
);

-- Parsed resume (most recent is active)
CREATE TABLE resume (
  id          INTEGER PRIMARY KEY,
  file_name   TEXT,
  raw_text    TEXT,
  skills      TEXT,       -- JSON array
  experience  TEXT,       -- JSON array of objects
  education   TEXT,       -- JSON array
  summary     TEXT,
  uploaded_at TEXT
);

-- Jobs found + applied
CREATE TABLE saved_jobs (
  id           INTEGER PRIMARY KEY,
  title        TEXT,
  company      TEXT,
  location     TEXT,
  url          TEXT,
  company_url  TEXT,
  match_score  INTEGER,
  match_reason TEXT,
  source       TEXT,
  posted_at    TEXT,
  status       TEXT DEFAULT 'saved'   -- 'saved' | 'applied'
);
```

---

## Setup & First Run

```bash
# From C:\Users\Donny\OneDrive\Documents\Git\JobSeeker
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
copy .env.example .env        # then add your ANTHROPIC_API_KEY
streamlit run app.py
```

---

## Dependencies (`requirements.txt`)

```
streamlit
anthropic
python-jobspy
pdfplumber
python-docx
playwright
sqlalchemy
pandas
googlesearch-python
python-dotenv
```

---

## Verification Checklist

1. **Profile page** — fill all fields → Save → refresh → fields still populated
2. **Resume page** — upload a PDF → parsed skills/experience appear in preview
3. **Job Search** — search "Software Engineer" in "New York, NY" → ranked results appear
4. **Time filter** — toggle between filters → result count changes
5. **Find on Company Site** — URL updates to company's own careers page
6. **Auto-Apply** — Playwright opens browser, Name/Email/Phone fields filled, app pauses before submit
