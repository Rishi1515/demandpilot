"""Forecasting models behind one small common interface.

Each model is fitted on a daily history frame (DatetimeIndex, a
`units_sold` column and optional `promotion` / `unit_price`) and returns
a point forecast for the next `horizon` days. Models are listed from
simplest to most complex; model selection uses that order to prefer a
simpler model when accuracy is effectively tied.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .features import MAX_LOOKBACK, future_frame, make_features


class Forecaster:
    name = "base"
    label = "Base"
    short_label = "base"
    complexity = 0
    description = ""

    def fit(self, history: pd.DataFrame) -> "Forecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"
    label = "Seasonal naive (last week repeated)"
    short_label = "seasonal naive"
    complexity = 0
    description = "Each future day equals the same weekday last week. This is the baseline every other model must beat."

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, history):
        self._last = history["units_sold"].to_numpy(dtype=float)[-self.season:]
        return self

    def predict(self, horizon):
        reps = int(np.ceil(horizon / self.season))
        return np.tile(self._last, reps)[:horizon]


class MovingAverage(Forecaster):
    name = "moving_average"
    label = "28-day moving average"
    short_label = "28-day moving average"
    complexity = 1
    description = "A flat forecast equal to the average of the last 28 days. Simple and stable, ignores weekly shape."

    def __init__(self, window: int = 28):
        self.window = window

    def fit(self, history):
        self._level = float(history["units_sold"].tail(self.window).mean())
        return self

    def predict(self, horizon):
        return np.full(horizon, self._level)


class HoltWinters(Forecaster):
    name = "holt_winters"
    label = "Holt-Winters exponential smoothing"
    short_label = "Holt-Winters"
    complexity = 2
    description = (
        "Classical exponential smoothing with a damped trend and a weekly seasonal pattern. "
        "Interpretable: level, trend and weekday effects are updated as new data arrives."
    )

    def fit(self, history):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # The last year is enough to learn level, trend and weekly shape,
        # and keeps fitting fast inside the app.
        y = history["units_sold"].astype(float).tail(365).reset_index(drop=True)
        seasonal = "add" if len(y) >= 3 * 7 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal=seasonal,
                seasonal_periods=7 if seasonal else None,
                initialization_method="estimated",
            )
            self._fit = model.fit(optimized=True)
        return self

    def predict(self, horizon):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = np.asarray(self._fit.forecast(horizon), dtype=float)
        return np.clip(np.nan_to_num(fc, nan=0.0), 0, None)


class GradientBoosting(Forecaster):
    name = "lightgbm"
    label = "LightGBM with lag and calendar features"
    short_label = "LightGBM"
    complexity = 3
    description = (
        "Gradient-boosted trees trained on lags, rolling statistics, calendar fields, promotion and price. "
        "Forecasts are made one day at a time, feeding each prediction back in as the next day's lag."
    )

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, history):
        import lightgbm as lgb

        self._history = history.copy()
        feats = make_features(history)
        mask = feats["lag_7"].notna()
        self._columns = list(feats.columns)
        self._model = None
        if history.loc[mask, "units_sold"].sum() <= 0:
            return self  # nothing to learn from: predict zeros
        self._model = lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1,
        )
        self._model.fit(feats.loc[mask, self._columns], history.loc[mask, "units_sold"])
        return self

    def predict(self, horizon):
        if self._model is None:
            return np.zeros(horizon)
        work = pd.concat([self._history.tail(MAX_LOOKBACK + 7), future_frame(self._history, horizon)])
        n_hist = len(work) - horizon
        preds = np.empty(horizon)
        for i in range(horizon):
            pos = n_hist + i
            window = work.iloc[max(0, pos - MAX_LOOKBACK - 1) : pos + 1]
            x = make_features(window).iloc[[-1]][self._columns]
            p = max(0.0, float(self._model.predict(x)[0]))
            preds[i] = p
            work.iloc[pos, work.columns.get_loc("units_sold")] = p
        return preds

    def feature_importance(self) -> pd.Series | None:
        if self._model is None:
            return None
        imp = pd.Series(self._model.booster_.feature_importance("gain"), index=self._columns)
        return (imp / imp.sum()).sort_values(ascending=False)


MODEL_REGISTRY = {
    SeasonalNaive.name: SeasonalNaive,
    MovingAverage.name: MovingAverage,
    HoltWinters.name: HoltWinters,
    GradientBoosting.name: GradientBoosting,
}
BASELINE = SeasonalNaive.name


def make_model(name: str) -> Forecaster:
    return MODEL_REGISTRY[name]()


def model_label(name: str, short: bool = False) -> str:
    cls = MODEL_REGISTRY[name]
    return cls.short_label if short else cls.label
