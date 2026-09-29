"""Record a silent walkthrough video of the app with on-screen captions.

Start the app first (streamlit run app.py), then:
    python scripts/record_demo.py
Writes assets/demo_walkthrough.webm (and .mp4 if ffmpeg is installed).
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parents[1] / "assets"
URL = "http://localhost:8501"
SIZE = {"width": 1440, "height": 900}

CAPTION_JS = """
(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);max-width:980px;' +
      'background:rgba(11,11,11,0.86);color:#fff;font:500 20px/1.4 system-ui,sans-serif;padding:14px 22px;' +
      'border-radius:10px;z-index:999999;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,0.25)';
    document.body.appendChild(el);
  }
  el.textContent = text;
}
"""


async def caption(page, text, seconds):
    await page.evaluate(CAPTION_JS, text)
    await page.wait_for_timeout(int(seconds * 1000))


async def scroll_to_text(page, text):
    await page.get_by_text(text, exact=True).first.evaluate("el => el.scrollIntoView({behavior: 'smooth', block: 'start'})")
    await page.wait_for_timeout(900)


async def main():
    OUT.mkdir(exist_ok=True)
    tmp = OUT / "_video"
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport=SIZE, record_video_dir=str(tmp), record_video_size=SIZE)
        page = await ctx.new_page()
        await page.goto(URL)
        await page.wait_for_selector("text=Data quality", timeout=180_000)
        await page.wait_for_timeout(2500)

        await caption(page, "DemandPilot: upload daily sales, forecast demand, then decide what to reorder.", 5)
        await caption(page, "Step 1: the validator checks the file. Here is a messy export with deliberate errors.", 3)
        await page.get_by_text("Sample data with errors (validator demo)").click()
        await page.wait_for_timeout(3500)
        await caption(page, "Bad dates, negative units, duplicates and a too-short product are all reported, never fixed silently.", 6)

        await page.get_by_text("Sample data", exact=True).click()
        await page.wait_for_timeout(3000)
        await page.get_by_role("tab", name="Forecast").click()
        await page.wait_for_timeout(1500)
        await caption(page, "Step 2: a 14-day forecast with an 80% range built from the model's own backtest errors.", 6)
        await scroll_to_text(page, "Model comparison")
        await caption(page, "Four models, including a seasonal-naive baseline, are refitted before each of 6 rolling windows. No future data leaks in.", 7)

        await page.get_by_role("tab", name="Recommendation").click()
        await page.wait_for_timeout(1200)
        await page.get_by_role("tab", name="Recommendation").evaluate("el => el.scrollIntoView({block: 'start'})")
        await caption(page, "Step 3: the decision. Stock is below the reorder point, so it recommends ordering now.", 6)
        lead = page.get_by_label("Supplier lead time (days) · from data")
        await lead.fill("10")
        await lead.press("Enter")
        await page.wait_for_timeout(1500)
        await caption(page, "What if the supplier takes 10 days instead of 5? Safety stock and the order quantity update instantly.", 6)
        slider = page.get_by_role("slider").first
        await slider.focus()
        for _ in range(6):
            await slider.press("ArrowRight")
            await page.wait_for_timeout(150)
        await page.wait_for_timeout(1500)
        await caption(page, "Raising the service level from 95% to 98% adds more safety stock. Every step of the formula is shown.", 6)
        await page.get_by_text("Explanation", exact=True).first.evaluate("el => el.scrollIntoView({block: 'start'})")
        await page.wait_for_timeout(600)
        await caption(page, "The explanation is written from computed numbers. An optional language model cannot add new figures.", 7)

        await page.get_by_role("tab", name="All products").click()
        await page.wait_for_timeout(800)
        await page.get_by_role("tab", name="All products").evaluate("el => el.scrollIntoView({block: 'start'})")
        await page.evaluate(CAPTION_JS, "Step 4: every product ranked by stockout and overstock risk, so a manager knows where to look first.")
        await page.get_by_role("button", name="Analyse all products").click()
        await page.wait_for_selector("text=Download this table", timeout=300_000)
        await page.wait_for_timeout(6000)

        await page.get_by_role("tab", name="Download").click()
        await page.wait_for_timeout(1000)
        await caption(page, "Step 5: download the forecast, the recommendation or a printable report. Code, tests and evaluation are on GitHub.", 6)
        await caption(page, "Built with Python, LightGBM, statsmodels and Streamlit. Sample data is synthetic.", 4)
        video_path = await page.video.path()
        await ctx.close()
        await browser.close()

    webm = OUT / "demo_walkthrough.webm"
    shutil.move(video_path, webm)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Wrote {webm}")
    if shutil.which("ffmpeg"):
        mp4 = OUT / "demo_walkthrough.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "26", "-movflags", "+faststart", str(mp4)], check=True)
        print(f"Wrote {mp4}")


if __name__ == "__main__":
    asyncio.run(main())
