# DemandPilot: technical write-up

## The problem

A small retailer usually decides what to reorder from gut feel and last week's sales. That leads to two kinds of mistakes: running out of fast movers, and sitting on too much of slow ones. DemandPilot tries to help with that decision, not just draw a forecast. The question it answers is "should I order this product today, and how much?"

## Data

I used a synthetic dataset so that anyone can run the project without downloading anything. It has 8 products over 730 days. I built in the patterns real shops see: weekends busier than weekdays, a summer peak for water and sunscreen, a festive spike for a gift hamper, a payday bump at the start of the month for rice, a slow decline for hand wash, a slow and patchy seller (olive oil, with zero-sale days), and random promotion blocks that lift sales by 25% to 70%. Daily counts are drawn from a negative binomial distribution, so they are noisier than a Poisson process, like real retail data.

Because I generated the data, I do not claim anything about real-world accuracy. The point is to show a pipeline that is built and evaluated properly.

## Validation

Real exports are messy, so validation is its own layer. The validator drops rows with invalid dates, empty or non-numeric units and negative units, removes exact duplicates (unless a transaction_id keeps them apart), and sums rows that share a product and date. It never fixes anything silently: every change is counted and shown to the user. A product needs at least 8 weeks of history and 28 days with sales rows, because a time-aware backtest needs enough past data to mean anything. Short histories get shorter backtest windows and a capped forecast horizon, with a note saying so.

Every product's series is extended to the last date in the file. Without that, a product that stopped selling (often because it ran out of stock) would be forecast from an old date without any warning.

## Models

I compared four models:

- **Seasonal naive**: repeat last week. This is the baseline.
- **28-day moving average**: a flat forecast at the recent average.
- **Holt-Winters**: exponential smoothing with a damped additive trend and weekly seasonality, fitted on the most recent year.
- **LightGBM**: gradient-boosted trees with a Poisson objective, trained on lags, rolling statistics, calendar fields, promotion and price. It forecasts recursively, feeding each prediction back in as the next day's lag.

The model is chosen by WAPE over 30-day backtest windows. If a simpler model is within 2% (relative) of the best, the simpler one wins, because when scores are that close the simpler model is easier to explain and less likely to break.

An early version selected the model separately for each display horizon (7, 14 or 30 days). A review pointed out that this meant the order recommendation could change just because the user looked at a different chart length. Now the backtest always uses the longest window the history allows, the model is chosen on that, and the horizon only changes what is shown.

## Evaluation and leakage

Forecasting has a trap that normal machine learning does not: if you split the data randomly, the model trains on days after the ones it is tested on. It then "knows" about trends and seasons it would not know in real life, and the accuracy looks better than it is.

So every evaluation here is time-aware. The backtest cuts the history at several points. At each cut, every model is refitted on data before the cut and forecasts the next 7, 14 or 30 days. It does not see anything from the test window, including promotion flags. This matches real use, where future promotions would be unknown to the model unless someone entered them.

There is a second, subtler problem. If I pick the best model on some windows and then report its score on those same windows, the number is optimistic, because I chose the model that happened to do well there. The evaluation script avoids this by splitting time into two parts: 6 windows of 30 days for choosing the model, then 4 later windows for reporting. Selection never sees the reporting windows.

One honest caveat: while building the app I compared two ways of building prediction ranges on backtests that ran over the full history, which overlaps the later holdout period. So the choice of range method was not made completely blind to the holdout. The model choice itself was.

Tests back these claims up. One test changes every value after day 100 and checks that features for earlier days do not change. Another replaces the final test window's actual sales with 5,000 and checks that the forecasts for that window stay exactly the same.

## Results

On the untouched holdout, pooled across all 8 products:

| Horizon | Selected per product | Seasonal naive | Relative reduction |
|---|---|---|---|
| 7 days | 27.0% | 40.9% | 34.0% |
| 14 days | 25.9% | 39.0% | 33.6% |
| 30 days | 27.7% | 37.9% | 27.0% |

The selected model beat the baseline for every product at every horizon (24 of 24).

Full tables are in `reports/evaluation_summary.md`.

Three findings surprised me a little:

1. **All smoothed models did about equally well.** Moving average, Holt-Winters and LightGBM were within about two points of each other. Most of the improvement over the baseline comes from averaging out noise instead of copying one week.
2. **Per-product selection helped only a little.** At 14 days it scored 25.9% against 26.5% for using LightGBM everywhere, and at 7 days a plain moving average everywhere did just as well. With 4 holdout windows that gap is within noise. If I kept working on this, I would try a single global LightGBM trained across all products, which would also help new products with short histories.
3. **The prediction ranges were too narrow.** An 80% range should contain the actual value 80% of the time. On the holdout it did so 73% to 77% of the time on average, and only 46% for one product at 7 days. Scaling errors by the demand level at each cutoff makes the ranges widen in busy seasons, but they are still short. The app shows the measured coverage under the chart so the user is not misled.
4. **The safety stock was cautious.** I measure lead-time error by summing the model's real backtest errors over the lead time, instead of the textbook daily error × √(lead time), which assumes errors on different days are independent. On the holdout, demand stayed within forecast plus a 95% safety stock in all 32 windows. The √ rule also covered almost every window (31 or 32 of 32) with 31% to 43% less stock. So on this data my method holds more stock than it needs to. I kept it because it does not rely on the independence assumption, but a real deployment would need more windows to settle which is better.

## From forecast to decision

The inventory layer uses standard formulas on purpose, so a manager or an interviewer can check them by hand:

- safety stock = z × (standard deviation of the total forecast error over the lead time)
- reorder point = expected lead-time demand + safety stock
- if stock plus units on order is at or below the reorder point, order up to the level that covers the lead time plus the review period

The error comes from the model's own backtest, so a less accurate forecast automatically leads to more safety stock. That link is the part I find most interesting: the quality of the model has a direct cost in inventory.

The app also estimates the chance of running out during the lead time and the expected units short, using the normal loss function. The formulas live in `src/inventory.py` with their own unit tests, separate from the interface, and the user can change stock, lead time and service level and see the result update straight away.

## Explanation layer

The explanation is written by a fixed template from computed values. I made the language model optional and kept it on a short leash: it only gets the template text, and its answer is rejected unless it states exactly the same numbers in the same order, as digits, and keeps the recommended action. Otherwise the template is shown. My first version only checked that each number also appeared somewhere in the template, which would have let the model swap two numbers; the review caught that. The check still cannot catch every change in meaning, so the numbers and the action are always shown separately on screen. The core app does not need any API key.

## Limitations

- Sales are treated as demand, but a product that was out of stock could not sell, so real demand was higher.
- Missing days are treated as zero sales.
- Lead-time error is estimated from a limited number of backtest windows, and on the holdout it looked cautious.
- Stockout probabilities use a normal approximation, which is rough for slow sellers.
- Lead time is fixed. Supplier delays are not modelled.
- Promotions and price changes in the future are not planned into the forecast.
- One model per product, so products do not learn from each other.

## What I would do next

1. Let users enter a promotion calendar, since retailers usually know it in advance.
2. Train one global LightGBM across products.
3. Use more backtest windows, or a model of lead-time error, to calibrate safety stock more tightly.
4. Use sales and stock history together to correct for lost sales during stockouts.
5. Test on a public retail dataset with its licence and attribution kept.

## Lessons learned

- A strong baseline is humbling and useful. It stopped me from assuming the most complex model would win.
- The honest evaluation took more thought than the models did. Separating selection from reporting changed how I talk about the results.
- Showing uncertainty is not enough. You also have to check whether the uncertainty is right, and say so when it is not.
- The business layer is where a forecast becomes useful. Most of the questions a user would ask are about the order quantity, not the model.
