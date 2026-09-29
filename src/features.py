"""Time-series features for the gradient-boosting model.

Leakage rule: every feature for day t is built only from values observed
before day t. Lags and rolling statistics are computed on the series
shifted by one day, so the target on day t never feeds its own features.
Calendar fields, the promotion flag and the price are allowed on day t
because a retailer knows them in advance (and in forecasting we assume
no promotion and the last known price, see `future_frame`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
WINDOWS = (7, 28)
MAX_LOOKBACK = max(max(LAGS), max(WINDOWS)) + 1


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for every row of a daily frame.

    `frame` must have a DatetimeIndex and a `units_sold` column. Rows whose
    target is unknown (future rows) may hold NaN in `units_sold`.
    """
    y = frame["units_sold"].astype(float)
    past = y.shift(1)  # only information up to yesterday
    idx = frame.index
    feats = pd.DataFrame(index=idx)

    for lag in LAGS:
        feats[f"lag_{lag}"] = y.shift(lag)
    for w in WINDOWS:
        feats[f"roll_mean_{w}"] = past.rolling(w, min_periods=max(2, w // 2)).mean()
        feats[f"roll_std_{w}"] = past.rolling(w, min_periods=max(2, w // 2)).std()
    feats["trend_ratio_7_28"] = feats["roll_mean_7"] / (feats["roll_mean_28"] + 1e-6)
    # Same weekday average over the last four weeks captures weekly shape.
    feats["same_weekday_mean_4"] = pd.concat([y.shift(k) for k in (7, 14, 21, 28)], axis=1).mean(axis=1)

    feats["day_of_week"] = idx.dayofweek
    feats["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    feats["day_of_month"] = idx.day
    feats["week_of_year"] = idx.isocalendar().week.to_numpy().astype(int)
    feats["month"] = idx.month
    feats["day_of_year"] = idx.dayofyear

    if "promotion" in frame.columns:
        feats["promotion"] = frame["promotion"].astype(float)
    if "unit_price" in frame.columns:
        price = frame["unit_price"].astype(float)
        feats["unit_price"] = price
        feats["price_vs_28d"] = price / (price.shift(1).rolling(28, min_periods=1).mean() + 1e-9)
    return feats


def future_frame(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rows for the next `horizon` days with the assumptions used in forecasting.

    Future promotions are unknown, so they are set to 0. Price is held at
    the last regular (non-promotional) price. The same assumptions are used in backtesting,
    so the reported accuracy reflects how the model is really used.
    """
    start = history.index.max() + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=horizon, freq="D", name="date")
    fut = pd.DataFrame({"units_sold": np.nan}, index=idx)
    if "promotion" in history.columns:
        fut["promotion"] = 0.0
    if "unit_price" in history.columns:
        price = history["unit_price"]
        if "promotion" in history.columns:  # a promotional price should not be carried into "no promotion" days
            regular = price[history["promotion"] <= 0].dropna()
            price = regular if not regular.empty else price
        last = price.dropna()
        fut["unit_price"] = float(last.iloc[-1]) if not last.empty else np.nan
    return fut
