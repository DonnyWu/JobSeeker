# archive/

Code that is switched off but not thrown away.

The app's navigation is the list returned by `src.ui.app_pages()`, which the router
in `app.py` hands to `st.navigation`. Streamlit no longer discovers pages from the
`pages/` folder on its own, so a page is part of the app only if that list names it.
The files kept here are out of that list as well as out of `pages/`, so nothing in
here is imported or executed.

## `4_Apply.py`

The old **Apply** page: a one-job view with a **Launch Auto-Fill** button that opened
a visible Chromium window, typed your profile into the application form, and paused
so you could review before submitting.

It was retired because auto-fill only handles plain text inputs. No dropdowns, no
checkboxes, no file uploads — so your résumé never got attached, and on a real ATS
like Workday or Greenhouse it filled the easy fields and left the rest. It was
occupying a top-level page slot without doing the job.

The engine it drives, `src/autofill.py`, is untouched and still in place.

**To bring it back:** `git mv archive/4_Apply.py pages/4_Apply.py`, then add an
`st.Page` entry for it in `src.ui.app_pages()`. Two things then need attention — the file still renders an "Applied" tab that has since been replaced
by `pages/4_Job_History.py`, and Job Search no longer hands it a job (the button that
used to do that, `st.session_state["apply_job"]` plus `st.switch_page`, is now a
plain **Save**).
