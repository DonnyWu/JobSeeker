import asyncio
from typing import Optional


FIELD_MAP = [
    (["first name", "first_name", "firstname"], "first_name"),
    (["last name", "last_name", "lastname"], "last_name"),
    (["full name", "full_name", "name"], "name"),
    (["email", "e-mail"], "email"),
    (["phone", "telephone", "mobile", "cell"], "phone"),
    (["linkedin", "linkedin url", "linkedin profile"], "linkedin"),
    (["github", "github url", "github profile"], "github"),
    (["portfolio", "website", "personal site"], "portfolio"),
    (["current company", "employer", "company"], "current_company"),
]


def _get_field_value(key: str, profile: dict) -> Optional[str]:
    if key == "first_name":
        name = profile.get("name", "")
        return name.split()[0] if name else ""
    if key == "last_name":
        name = profile.get("name", "")
        parts = name.split()
        return parts[-1] if len(parts) > 1 else ""
    return profile.get(key, "")


async def _autofill_async(url: str, profile: dict):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url, timeout=30000)

        inputs = await page.query_selector_all("input:not([type='hidden']):not([type='submit']):not([type='button'])")

        for inp in inputs:
            input_type = (await inp.get_attribute("type") or "").lower()
            name_attr = (await inp.get_attribute("name") or "").lower()
            aria_label = (await inp.get_attribute("aria-label") or "").lower()
            placeholder = (await inp.get_attribute("placeholder") or "").lower()

            # Try to find adjacent label text
            label_text = ""
            try:
                inp_id = await inp.get_attribute("id")
                if inp_id:
                    label = await page.query_selector(f"label[for='{inp_id}']")
                    if label:
                        label_text = (await label.inner_text() or "").lower()
            except Exception:
                pass

            combined = f"{name_attr} {aria_label} {placeholder} {label_text} {input_type}"

            value = None
            for keywords, profile_key in FIELD_MAP:
                if any(kw in combined for kw in keywords):
                    value = _get_field_value(profile_key, profile)
                    break

            if value:
                try:
                    await inp.fill(value)
                except Exception:
                    pass

        # Pause so the user can review before submitting
        await page.pause()
        await browser.close()


def run_autofill(url: str, profile: dict):
    asyncio.run(_autofill_async(url, profile))
