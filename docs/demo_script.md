# Two-minute demo script

Record at 1440 × 900 or 1920 × 1080 with the app running locally or on the live link. Hide bookmarks and notifications. Speak slowly. Timings are a guide.

A silent, captioned screen recording of this same flow is included as `assets/demo_walkthrough.mp4` (regenerate with `python scripts/record_demo.py`). You can post it as it is, or record your own voice over it.

| Time | On screen | What to say |
|---|---|---|
| 0:00 to 0:12 | App open on **Data check** with the sample data | "This is DemandPilot. It helps a small retailer decide what to reorder. You upload daily sales and it forecasts demand and turns that into an order recommendation." |
| 0:12 to 0:30 | Sidebar: select **Sample data with errors**. Show the yellow warning box. Open **Expected CSV format** | "Real exports are messy. Here the validator found invalid dates, negative units, duplicates, and a product with only three weeks of history. It reports every change instead of fixing things silently." |
| 0:30 to 0:55 | Switch back to **Sample data**. Open **Forecast**. Point at the chart, then scroll to **Model comparison** | "For each product it compares four models, including a seasonal-naive baseline, on rolling 30-day backtests. Each model is refitted using only data before each window, so there is no leakage from the future. The shaded band is an 80% range, and the app shows how often that range was actually right in the backtest." |
| 0:55 to 1:25 | Open **Recommendation**. Point at order quantity, reorder point, safety stock, stockout chance. Then change **Supplier lead time** from 5 to 10 and **service level** to 0.99 | "This is the decision. Stock is below the reorder point, so it recommends ordering now. The table on the right shows every step of the calculation. If the supplier gets slower, or I want a higher service level, the safety stock and order quantity update immediately." |
| 1:25 to 1:40 | Scroll to **Explanation** | "The explanation is generated from computed values by a template. A language model is optional, and if it changes any number or the recommended action, its answer is rejected." |
| 1:40 to 1:52 | **All products** tab, click **Analyse all products** | "This view ranks every product by stockout and overstock risk, so a manager knows where to look first." |
| 1:52 to 2:00 | **Download** tab, click **Report (HTML)** | "Results can be downloaded as CSV or a printable report. The code, tests and evaluation are all on GitHub." |

## Tips

- Run the app once before recording so the models are cached and nothing lags on camera.
- If you are asked about accuracy, the honest line is: "On synthetic data, on a holdout that model selection never saw, error at a 14-day horizon went from 39% of volume for the seasonal-naive baseline to 26%."
