"""Capture README screenshots from a running app.

Start the app first (streamlit run app.py), then:
    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py [--url http://localhost:8501]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parents[1] / "assets"


async def settle(page, ms: int = 2500):
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(ms)


async def scroll_to(page, text: str):
    await page.get_by_text(text, exact=True).first.evaluate("el => el.scrollIntoView({block: 'start'})")
    await page.wait_for_timeout(800)


async def main(url: str):
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1040}, device_scale_factor=2)
        await page.goto(url)
        await page.wait_for_selector("text=Data quality", timeout=180_000)
        await settle(page)

        await page.get_by_role("tab", name="Forecast").click()
        await settle(page)
        await page.screenshot(path=str(OUT / "screenshot_forecast.png"))
        await scroll_to(page, "Model comparison")
        await page.screenshot(path=str(OUT / "screenshot_model_comparison.png"))

        await page.get_by_role("tab", name="Recommendation").click()
        await settle(page)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.screenshot(path=str(OUT / "screenshot_recommendation.png"))

        await page.get_by_role("tab", name="All products").click()
        await settle(page, 1000)
        await page.get_by_role("button", name="Analyse all products").click()
        await page.wait_for_selector("text=Download this table", timeout=300_000)
        await settle(page, 1500)
        await page.screenshot(path=str(OUT / "screenshot_portfolio.png"))

        # Validator demo with the deliberately messy sample file.
        await page.get_by_text("Sample data with errors (validator demo)").click()
        await page.wait_for_selector("text=exact duplicate", state="attached", timeout=180_000)
        await page.get_by_role("tab", name="Data check").click()
        await settle(page)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.screenshot(path=str(OUT / "screenshot_validation.png"))
        await browser.close()
    print(f"Screenshots written to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    asyncio.run(main(ap.parse_args().url))
