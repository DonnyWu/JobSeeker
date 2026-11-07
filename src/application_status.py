"""The vocabulary for where a job application stands.

Lives here rather than in :mod:`src.profile_manager` for the same reason
``job_signature`` moved into :mod:`src.jobkey`: two layers need it and only one of
them talks to the database. The Job History page needs the labels and colours, and
a future export (Excel, or another database) needs the stored values — neither
should have to import the other, and neither should own the list.

Note the two different notions of "status" in ``saved_jobs``, which are easy to
confuse:

    status   which list the job is in — 'saved' or 'applied'. Written by
             ``save_job`` / ``mark_job_applied``.
    outcome  what happened *after* applying — the values in this module. Written
             by the status control on the My Jobs page.

FUTURE: exporting this record to a spreadsheet or syncing it to another database.
Three things here are deliberately shaped for that and are painful to retrofit:

  * ``normalize`` folds NULL, blank and retired values into a canonical one, so an
    export never has to special-case a row written by an older version.
  * ``PENDING`` is *stored*, not implied by NULL, so every exported row has a real
    value in every cell.
  * ``DISPLAY`` holds the human labels, so a spreadsheet's dropdown can be built
    from the same table the UI renders and the two cannot drift apart.

The seam itself is ``profile_manager.to_application_record``.
"""

PENDING = "pending"
IN_PROCESS = "in_process"
ACCEPTED = "accepted"
REJECTED = "rejected"

#: Every value that may be written to ``saved_jobs.outcome``, in the order the
#: buttons are rendered — roughly the order an application moves through.
APPLICATION_STATUSES = (PENDING, IN_PROCESS, ACCEPTED, REJECTED)

#: The vocabulary this replaced (commit 2ef18fe: interview/offer/accepted/declined).
#: A database written by that version still reads correctly through ``normalize``,
#: which is why no backfill migration is needed.
_LEGACY = {
    "interview": IN_PROCESS,
    "offer": IN_PROCESS,
    "declined": REJECTED,
}

#: value -> (label, emoji, hex colour). The colour paints the accent bar down the
#: left edge of a card on the My Jobs page; the label is the text badge beside it,
#: which is deliberately redundant with the bar so the status is never conveyed by
#: colour alone. Pending is blue rather than orange so it can't be mistaken for
#: In-process — the two used to be orange and yellow, which read as the same colour.
#: (src/ui.py maps each status to a badge colour and icon for the page itself.)
DISPLAY = {
    PENDING: ("Pending", "🔵", "#3B82F6"),
    IN_PROCESS: ("In-process", "🟠", "#F59E0B"),
    ACCEPTED: ("Accepted", "🟢", "#22C55E"),
    REJECTED: ("Rejected", "🔴", "#EF4444"),
}


def normalize(value) -> str:
    """Coerce a stored ``outcome`` to one of :data:`APPLICATION_STATUSES`.

    Everything that reads the column goes through here, which is what makes
    "pending" the default state: the 25 rows applied for before this feature
    existed hold NULL, and a row written by the previous version holds a retired
    value like 'interview'. Both answer this question without a data migration.

    An unrecognized value is pending rather than an error — a status is a note to
    yourself about a job hunt, and refusing to render the list because one cell is
    odd would be the worse failure.
    """
    v = str(value or "").strip().lower()
    if v in DISPLAY:
        return v
    return _LEGACY.get(v, PENDING)


def label(value) -> str:
    """Human-readable name for a status, e.g. ``"In-process"``."""
    return DISPLAY[normalize(value)][0]


def badge(value) -> str:
    """Emoji + label as plain text, e.g. ``"🟠 In-process"``.

    The page itself uses :func:`src.ui.status_badge`; this stays for plain-text
    output (a log line, a future export) where markdown badges mean nothing.
    """
    _label, emoji, _color = DISPLAY[normalize(value)]
    return f"{emoji} {_label}"
