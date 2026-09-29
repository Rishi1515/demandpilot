# Résumé entry and interview notes

Every number below was produced by this repository's own code (`scripts/evaluate.py`, `pytest`). If you change the data or models, rerun both and update the numbers.

## Résumé entry

**DemandPilot: AI Demand Forecasting and Inventory Decision Assistant** · Python, LightGBM, statsmodels, Streamlit, Plotly

- Built an end-to-end forecasting app that validates product-level sales data, compares a seasonal-naive baseline, moving average, Holt-Winters and LightGBM on rolling-origin backtests, and turns the selected forecast into safety-stock, reorder-point and order-quantity recommendations.
- On a holdout period never used for model selection (8 synthetic SKUs, 2 years of daily sales), reduced pooled WAPE from 39.0% to 25.9% at a 14-day horizon (34% relative) versus a seasonal-naive baseline, beating it for every product at 7, 14 and 30 days.
- Added lag, rolling and calendar features with tests that check no future data leaks into training, uncertainty ranges and lead-time safety stock built from backtest errors with their holdout coverage reported, and scenario controls for stock, lead time and service level.
- Wrote a template-based explanation layer with an optional LLM rephrase that is rejected unless it keeps the computed numbers and recommended action; covered validation, inventory formulas, edge cases and leakage checks with a 53-test pytest suite.

**Shorter two-line version**

- Built a demand-forecasting and reorder-recommendation app (LightGBM, Holt-Winters, Streamlit) with leakage-checked rolling backtests and transparent safety-stock logic.
- Cut forecast error (WAPE) from 39.0% to 25.9% versus a seasonal-naive baseline on an untouched holdout of synthetic retail data at a 14-day horizon.

Add the GitHub link and the live app link once deployed.

## Rules for talking about it

- Always say **synthetic data** when you quote the WAPE figures.
- Say "relative" when you quote 34%: error went from 39.0% to 25.9% of volume.
- Be ready to say that all three smoothed models beat the baseline by a similar margin, and that per-product selection was only slightly better than using LightGBM everywhere.
- Do not say it saved money, was used by a business, or is production ready.
- "Four models compared, the simplest one that is close to the best is chosen" is accurate. "LightGBM beat everything" is not.

## Interview talking points

**Why not a random train/test split?**
In time series, a random split puts future days in the training set. The model then learns trends and seasons it could not know at forecast time, so the score is inflated. I used an expanding-window backtest: refit before each cutoff, forecast the next window, move on. I also hid future promotion flags from the model in testing because it would not have them in real use.

**Why report on a separate holdout?**
If you pick the model that did best on some windows and quote its score on those windows, you are rewarding luck. I chose models on 6 earlier 30-day windows and reported on 4 later ones.

**Why might a simple model be preferable?**
In my results the moving average, Holt-Winters and LightGBM were within about two points of each other, and at 7 days a moving average for every product did as well as anything. When accuracy is tied, the simpler model is easier to explain, faster and less likely to break on new data. My selection rule prefers a simpler model if it is within 2% of the best.

**How does forecast error affect inventory?**
Safety stock is z times the spread of the total forecast error over the lead time. I measure that spread by summing the model's real backtest errors over the lead time, rather than using the textbook daily error × √(lead time), which assumes errors on different days are independent. A worse forecast directly means more stock held. On the holdout my method covered demand in all 32 windows, while the √ rule covered 31 or 32 of 32 with 31% to 43% less stock, so mine was more cautious than needed on this data. That is a real trade-off I would test with more data.

**How do you stop the language model from making things up?**
It is optional. It only gets text built from computed values, and its answer is rejected unless it has exactly the same numbers in the same order and the same recommended action. The forecast and the order quantity are never produced by the language model. The check cannot catch every change of meaning, so the numbers and action are always shown separately.

**What does the project not prove?**
It does not show accuracy on a real shop, because the data is synthetic. Sales are treated as demand, so stockouts hide true demand. Lead time is fixed. The prediction ranges were too narrow on the holdout (73% to 77% average coverage for an 80% range, 46% for one product), and the app reports its own coverage.

**What would you do next?**
Add a promotion calendar, train one global model across products, estimate lead-time error directly from backtests, and test on a public retail dataset.

**How does this connect to MITB AI?**
The model was the easy part. The harder parts were evaluating it honestly, turning its uncertainty into a business decision, and presenting that to someone who is not technical. That mix of AI and business judgement is what I want to study further.
