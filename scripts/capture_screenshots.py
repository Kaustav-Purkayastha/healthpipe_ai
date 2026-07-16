#!/usr/bin/env python3
"""
scripts/capture_screenshots.py — Capture README screenshots from the live Flask app.

Prerequisites (one-time):
    python -m pip install playwright
    python -m playwright install chromium

Run (Flask server must be running on port 8501):
    python scripts/capture_screenshots.py

Saves PNGs to docs/screenshots/.
"""

import asyncio
from pathlib import Path

BASE_URL = "http://localhost:8501"
OUT_DIR = Path(__file__).parent.parent / "docs" / "screenshots"
VIEWPORT = {"width": 1440, "height": 900}

DEMO_QUESTION = "What are the top 10 states with the highest mental health issue prevalence?"
DEMO_SQL = (
    "SELECT state, diabetes_prevalence\n"
    "FROM reporting_state_health\n"
    "ORDER BY diabetes_prevalence DESC\n"
    "LIMIT 10;"
)


async def capture_all() -> None:
    from playwright.async_api import async_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport=VIEWPORT)
        page = await ctx.new_page()

        # ── 1. Login page ──────────────────────────────────────────────────────
        await page.goto(BASE_URL)
        await page.wait_for_load_state("networkidle")
        await page.screenshot(path=str(OUT_DIR / "01_login.png"))
        print("OK 01_login.png")

        # ── 2. Dashboard – hero stats row ──────────────────────────────────────
        await page.goto(f"{BASE_URL}/pages/dashboard.html")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(OUT_DIR / "02_dashboard.png"))
        print("OK 02_dashboard.png")

        # ── 3. Dashboard – quality scorecard (first run auto-selected) ─────────
        # The page calls selectRun(RUNS[0]) automatically; wait for #pane-quality
        await page.wait_for_selector("#pane-quality:not(:empty)", timeout=8000)
        await page.wait_for_timeout(500)
        await page.evaluate("document.getElementById('pane-quality').scrollIntoView()")
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(OUT_DIR / "03_dashboard_quality.png"), full_page=True)
        print("OK 03_dashboard_quality.png")

        # ── 4. Dashboard – privacy audit (scroll to bottom) ────────────────────
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(600)
        await page.screenshot(path=str(OUT_DIR / "04_dashboard_audit.png"))
        print("OK 04_dashboard_audit.png")

        # ── 5. Onboard – Step 1 (no source chosen yet) ────────────────────────
        await page.goto(f"{BASE_URL}/pages/onboard.html")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(OUT_DIR / "05_onboard_step1.png"))
        print("OK 05_onboard_step1.png")

        # ── 6. Onboard – Step 2 revealed (Public APIs selected) ───────────────
        await page.click("button.lane-card[data-lane='api']")
        await page.wait_for_timeout(1200)
        await page.screenshot(path=str(OUT_DIR / "06_onboard_api.png"), full_page=True)
        print("OK 06_onboard_api.png")

        # ── 7. Ask the data – blank (table picker + starter questions) ─────────
        await page.goto(f"{BASE_URL}/pages/chat.html")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(OUT_DIR / "07_chat.png"))
        print("OK 07_chat.png")

        # ── 8. Ask the data – result with chart ────────────────────────────────
        try:
            table_sel = page.locator("#table-select")
            await table_sel.wait_for(state="visible", timeout=6000)
            options = await table_sel.evaluate(
                "el => Array.from(el.options).map(o => o.value).filter(v => v)"
            )
            if options:
                await table_sel.select_option(options[0])
                await page.wait_for_timeout(800)

            await page.fill("#composer-input", DEMO_QUESTION)
            await page.click("#send-btn")
            # AI response can take up to 45s on the local model
            await page.wait_for_timeout(45000)
            await page.screenshot(
                path=str(OUT_DIR / "08_chat_result.png"), full_page=True
            )
            print("OK 08_chat_result.png")
        except Exception as exc:
            print(f"SKIP 08_chat_result.png: {exc}")

        # ── 9. US States Health Mart ───────────────────────────────────────────
        await page.goto(f"{BASE_URL}/pages/mart.html")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(3000)
        await page.screenshot(path=str(OUT_DIR / "09_mart.png"), full_page=True)
        print("OK 09_mart.png")

        # ── 10. SQL Console – prefilled sample query ───────────────────────────
        await page.goto(f"{BASE_URL}/pages/sql.html")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(1500)
        try:
            await page.fill("#sql-editor", DEMO_SQL)
        except Exception:
            pass
        await page.screenshot(path=str(OUT_DIR / "10_sql.png"))
        print("OK 10_sql.png")

        await browser.close()

    print(f"\nAll screenshots saved to: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(capture_all())
