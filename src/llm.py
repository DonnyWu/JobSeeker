"""Which Groq models the app talks to.

One place, because the same model id used to be hardcoded in three files
(``job_matcher``, ``resume_parser``, ``company_insights``). Groq retires model
ids on its own schedule, and when ``llama-3.3-70b-versatile`` stopped answering
it broke scoring, résumé parsing, and company summaries separately — three edits
for one upstream change.

Both ids are environment-overridable, so a retirement is a one-line ``.env``
change and a restart rather than a code edit:

    GROQ_TEXT_MODEL=openai/gpt-oss-120b
    GROQ_WEB_MODEL=groq/compound

To see what the account can actually use::

    curl -s https://api.groq.com/openai/v1/models \\
      -H "Authorization: Bearer $GROQ_API_KEY"
"""

import os

# General chat/JSON model: scoring, résumé extraction, and the non-web company
# summary fallback.
#
# The default has to be a model that actually answers, or a fresh clone hits the
# same 404 with no clue why. gpt-oss-120b was picked over the other two general
# models the account can reach because every caller here does ``json.loads`` on
# the reply: qwen3.6-27b emits a ``<think>`` block ahead of its JSON and breaks
# that parse, and gpt-oss-20b is the smaller sibling.
TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")

# Web-search model for company summaries. Kept separate because it is the one
# path where the model fetches live pages, which is why company_insights fences
# its scraped input.
WEB_MODEL = os.environ.get("GROQ_WEB_MODEL", "groq/compound")
