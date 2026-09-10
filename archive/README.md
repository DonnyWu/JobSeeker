# archive/

Code that is switched off but not thrown away.

Streamlit builds the sidebar from whatever it finds in `pages/`, so the only way to
take a page out of the navigation while keeping it is to move the file out of that
folder. Nothing in here is imported or executed.

## `4_Apply.py`

The old **Apply** page: a one-job view with a **Launch Auto-Fill** button that opened
a visible Chromium window, typed your profile into the application form, and paused
so you could review before submitting.

It was retired because auto-fill only handles plain text inputs. No dropdowns, no
checkboxes, no file uploads — so your résumé never got attached, and on a real ATS
like Workday or Greenhouse it filled the easy fields and left the rest. It was
occupying a top-level page slot without doing the job.

The engine it drives, `src/autofill.py`, is untouched and still in place.

**To bring it back:** `git mv archive/4_Apply.py pages/4_Apply.py`. Two things then
need attention — the file still renders an "Applied" tab that has since been replaced
by `pages/4_Job_History.py`, and Job Search no longer hands it a job (the button that
used to do that, `st.session_state["apply_job"]` plus `st.switch_page`, is now a
plain **Save**).
