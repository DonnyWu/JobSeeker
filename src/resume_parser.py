import io
import json
import os

from groq import Groq


def _extract_text_pdf(file_bytes: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _extract_text_docx(file_bytes: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _extract_text_pdf(file_bytes)
    if ext in (".docx", ".doc"):
        return _extract_text_docx(file_bytes)
    raise ValueError(f"Unsupported file type: {ext}")


def parse_resume(raw_text: str) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = (
        "You are a resume parser. Given the resume text below, extract the following fields "
        "and return ONLY valid JSON (no markdown, no explanation):\n"
        '{\n'
        '  "skills": ["list of skills"],\n'
        '  "experience": [\n'
        '    {"company": "...", "title": "...", "duration": "...", "bullets": ["..."]}\n'
        '  ],\n'
        '  "education": [\n'
        '    {"institution": "...", "degree": "...", "year": "..."}\n'
        '  ]\n'
        '}\n\n'
        "Resume:\n"
        f"{raw_text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)
