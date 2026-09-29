"""Time-aware evaluation, model selection and forecast intervals.

Backtesting uses an expanding window with rolling forecast origins:

    fold 1: train [ ........ ]  test [ h days ]
    fold 2: train [ ............ ]  test [ h days ]
    fold 3: train [ ................ ]  test [ h days ]

Each model is refitted on data strictly before the fold's cutoff and then
forecasts the next h days in one go, with no access to actual values from
the test window (not even promotion flags). This mirrors how the model is
used for real. A random train/test split would let the model learn from
days that come after the days it is tested on, which inflates accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import BASELINE, MODEL_REGISTRY, make_model

DEFAULT_MODELS = tuple(MODEL_REGISTRY.keys())
MIN_TRAIN_DAYS = 42
PARSIMONY_TOLERANCE = 0.02  # prefer a simpler model within 2% (relative) of the best WAPE
INTERVAL_LEVEL = 0.80
MAX_INTERNAL_HORIZON = 90  # forecast length kept for inventory calculations
BACKTEST_HORIZON = 30  # backtest window length when history allows


# ----------------------------------------------------------------------------- metrics

def mae(actual, forecast) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: total absolute error / total actual demand."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = np.abs(a).sum()
    return float(np.abs(a - f).sum() / denom) if denom > 0 else float("nan")


def smape(actual, forecast) -> float:
    """Symmetric MAPE in [0, 2]; days where both values are zero count as zero error."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = np.abs(a) + np.abs(f)
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(a - f) / np.where(denom == 0, 1, denom))
    return float(np.mean(ratio))


def bias(actual, forecast) -> float:
    """Mean forecast minus actual. Positive means the model over-forecasts."""
    return float(np.mean(np.asarray(forecast) - np.asarray(actual)))


def score(actual, forecast) -> dict:
    return {
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "WAPE": wape(actual, forecast),
        "sMAPE": smape(actual, forecast),
        "Bias": bias(actual, forecast),
    }


# ----------------------------------------------------------------------------- splits

def rolling_origin_cutoffs(n: int, horizon: int, n_folds: int, min_train: int = MIN_TRAIN_DAYS) -> list[int]:
    """Index positions where each fold's test window starts (training uses rows before it)."""
    cutoffs = [n - horizon * (n_folds - k) for k in range(n_folds)]
    return [c for c in cutoffs if c >= min_train]


def feasible_folds(n: int, horizon: int, wanted: int, min_train: int = MIN_TRAIN_DAYS) -> int:
    return max(0, min(wanted, (n - min_train) // horizon))


# ----------------------------------------------------------------------------- backtest

def backtest(daily: pd.DataFrame, horizon: int, n_folds: int = 6,
             models: tuple[str, ...] = DEFAULT_MODELS) -> pd.DataFrame:
    """Long table of every backtest forecast: fold, model, date, step, actual, forecast.

    A model that fails to fit on a fold (for example on an all-zero history)
    is left out of that fold; the names of failed models are stored in
    `result.attrs["failed"]`.
    """
    n = len(daily)
    cutoffs = rolling_origin_cutoffs(n, horizon, n_folds)
    rows, failed = [], set()
    for fold, cut in enumerate(cutoffs, start=1):
        train = daily.iloc[:cut]
        test = daily.iloc[cut : cut + horizon]
        assert train.index.max() < test.index.min(), "training data must end before the test window"
        level = demand_level(train)
        for name in models:
            try:
                fc = make_model(name).fit(train).predict(horizon)
            except Exception:  # noqa: BLE001 - one bad model must not stop the comparison
                failed.add(name)
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "cutoff": test.index.min(),
                        "level": level,
                        "model": name,
                        "date": test.index,
                        "step": np.arange(1, horizon + 1),
                        "actual": test["units_sold"].to_numpy(dtype=float),
                        "forecast": fc,
                    }
                )
            )
    if not rows:
        out = pd.DataFrame(columns=["fold", "cutoff", "level", "model", "date", "step", "actual", "forecast",
                                    "error", "scaled_error"])
    else:
        out = pd.concat(rows, ignore_index=True)
        out["error"] = out["actual"] - out["forecast"]
        out["scaled_error"] = out["error"] / out["level"]
        # A model must have forecasts for every fold to be compared fairly.
        complete = out.groupby("model")["fold"].nunique() == len(cutoffs)
        failed |= set(complete.index[~complete])
        out = out[out["model"].isin(complete.index[complete])].reset_index(drop=True)
    out.attrs["failed"] = sorted(failed)
    return out


def demand_level(history: pd.DataFrame, window: int = 28) -> float:
    """Typical daily demand, used to put forecast errors on a common scale.

    Uses the last 28 days, floored at half the average of the last year so
    that a quiet month for a slow seller does not blow up the scaled errors.
    """
    y = history["units_sold"].astype(float)
    level = max(float(y.tail(window).mean()), 0.5 * float(y.tail(365).mean()))
    return level if level > 0 else 1.0


def fold_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (fold, model), g in bt.groupby(["fold", "model"], sort=True):
        recs.append({"fold": fold, "cutoff": g["cutoff"].iloc[0], "model": model, **score(g["actual"], g["forecast"])})
    return pd.DataFrame(recs)


def summarise(bt: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Aggregate metrics per model across all folds, with improvement versus the baseline."""
    recs = []
    for model, g in bt.groupby("model", sort=False):
        recs.append({"model": model, **score(g["actual"], g["forecast"])})
    summary = pd.DataFrame(recs).set_index("model")
    if baseline in summary.index:
        base = summary.loc[baseline]
        with np.errstate(divide="ignore", invalid="ignore"):
            summary["WAPE_vs_baseline"] = (base["WAPE"] - summary["WAPE"]) / base["WAPE"]
            summary["MAE_vs_baseline"] = (base["MAE"] - summary["MAE"]) / base["MAE"]
    else:
        summary["WAPE_vs_baseline"] = np.nan
        summary["MAE_vs_baseline"] = np.nan
    summary["complexity"] = [MODEL_REGISTRY[m].complexity for m in summary.index]
    return summary


def select_model(summary: pd.DataFrame, tolerance: float = PARSIMONY_TOLERANCE) -> tuple[str, str]:
    """Lowest WAPE wins, unless a simpler model is within `tolerance` (relative) of it.

    If WAPE cannot be computed (no sales at all in the backtest windows),
    MAE is used instead.
    """
    metric = "WAPE" if summary["WAPE"].notna().all() else "MAE"
    ranked = summary.sort_values(metric)
    best = ranked.index[0]
    best_val = ranked[metric].iloc[0]
    close = summary[summary[metric] <= best_val * (1 + tolerance) + 1e-12].sort_values(["complexity", metric])
    chosen = close.index[0]
    fmt = (lambda v: f"{v:.1%}") if metric == "WAPE" else (lambda v: f"{v:.2f} units")
    if chosen == best:
        reason = f"it had the lowest {metric} ({fmt(best_val)})"
    else:
        reason = (
            f"its {metric} ({fmt(summary.loc[chosen, metric])}) is within {tolerance:.0%} of the most accurate "
            f"model ({MODEL_REGISTRY[best].short_label}, {fmt(best_val)}), so the simpler model is preferred"
        )
    return chosen, reason


# ----------------------------------------------------------------------------- intervals

MIN_BUCKET_COUNT = 10


def _bucket(step: np.ndarray) -> np.ndarray:
    return (np.asarray(step) - 1) // 7


def interval_offsets(residuals: pd.DataFrame, steps: np.ndarray, level: float = INTERVAL_LEVEL,
                     min_count: int = MIN_BUCKET_COUNT) -> tuple[np.ndarray, np.ndarray]:
    """Empirical quantiles of scaled errors per forecast week.

    Errors are divided by the demand level at each fold's cutoff, so a
    product entering its busy season gets proportionally wider ranges.
    Multiply the returned offsets by the current demand level. Weeks with
    fewer than `min_count` errors fall back to all errors pooled, and steps
    beyond the backtested horizon reuse the last available week.
    """
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    err_all = residuals["scaled_error"].to_numpy(dtype=float)
    pooled = (np.quantile(err_all, lo_q), np.quantile(err_all, hi_q)) if len(err_all) else (0.0, 0.0)
    res_bucket = _bucket(residuals["step"].to_numpy())
    max_bucket = res_bucket.max() if len(res_bucket) else 0
    lows, highs = np.empty(len(steps)), np.empty(len(steps))
    for i, b in enumerate(np.minimum(_bucket(steps), max_bucket)):
        e = err_all[res_bucket == b]
        if len(e) >= min_count:
            lows[i], highs[i] = np.quantile(e, lo_q), np.quantile(e, hi_q)
        else:
            lows[i], highs[i] = pooled
    return lows, highs


def interval_coverage(bt_model: pd.DataFrame, level: float = INTERVAL_LEVEL) -> float:
    """Share of actuals inside intervals built only from earlier folds' errors.

    Fold k's interval uses residuals from folds 1..k-1, so coverage is
    measured out of sample. Needs at least three folds.
    """
    folds = sorted(bt_model["fold"].unique())
    if len(folds) < 3:
        return float("nan")
    hits, total = 0, 0
    for k in folds[2:]:
        prior = bt_model[bt_model["fold"] < k]
        cur = bt_model[bt_model["fold"] == k]
        lo, hi = interval_offsets(prior, cur["step"].to_numpy(), level)
        scale = cur["level"].to_numpy()
        low = np.clip(cur["forecast"].to_numpy() + lo * scale, 0, None)
        high = cur["forecast"].to_numpy() + hi * scale
        a = cur["actual"].to_numpy()
        hits += int(((a >= low) & (a <= high)).sum())
        total += len(a)
    return hits / total if total else float("nan")


def period_error_sigma(bt_model: pd.DataFrame, days: int, current_level: float) -> tuple[float, bool]:
    """Standard deviation of the total forecast error over `days` consecutive days.

    Measured directly from the backtest: every run of `days` consecutive
    forecast days inside each fold gives one summed (level-scaled) error.
    Summing real errors keeps any day-to-day correlation and bias, which the
    textbook sqrt(days) rule ignores. For periods longer than the backtest
    window, the longest measured period is extended with the sqrt rule and
    the second return value is True.
    """
    horizon = int(bt_model["step"].max())
    d = min(days, horizon)
    sums = []
    for _, g in bt_model.sort_values("step").groupby("fold"):
        e = g["scaled_error"].to_numpy(dtype=float)
        if len(e) >= d:
            c = np.concatenate([[0.0], np.cumsum(e)])
            sums.append(c[d:] - c[:-d])
    if not sums:
        return 0.0, False
    all_sums = np.concatenate(sums)
    sigma = float(np.sqrt(np.mean(all_sums ** 2))) * current_level
    if days > horizon:
        return sigma * float(np.sqrt(days / horizon)), True
    return sigma, False


# ----------------------------------------------------------------------------- end to end

def backtest_horizon(n_days: int, max_horizon: int = BACKTEST_HORIZON) -> int:
    """Longest backtest window (up to `max_horizon`) that still fits two folds."""
    return int(min(max_horizon, (n_days - MIN_TRAIN_DAYS) // 2))


@dataclass
class ForecastResult:
    product_id: str
    horizon: int  # display horizon
    backtest_horizon: int  # length of each backtest window
    history: pd.DataFrame
    backtest: pd.DataFrame  # full backtest, all models, steps 1..backtest_horizon
    fold_metrics: pd.DataFrame  # per fold, scored on the display horizon
    summary: pd.DataFrame  # per model, scored on the display horizon
    selected_model: str
    selection_reason: str
    forecast: pd.DataFrame  # date, forecast, lower, upper for MAX_INTERNAL_HORIZON days
    residual_rmse: float  # daily RMSE of the selected model over the display horizon (units)
    daily_sigma: float  # daily error scale: scaled RMSE x current demand level
    current_level: float
    interval_level: float
    coverage: float  # out-of-sample coverage over the display horizon
    n_folds: int
    filled_days: int = 0
    feature_importance: pd.Series | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def display_forecast(self) -> pd.DataFrame:
        return self.forecast.head(self.horizon)

    @property
    def selected_backtest(self) -> pd.DataFrame:
        return self.backtest[self.backtest["model"] == self.selected_model]

    def error_sigma(self, days: int) -> tuple[float, bool]:
        """Std of total forecast error over the next `days` days (see period_error_sigma)."""
        return period_error_sigma(self.selected_backtest, days, self.current_level)


def run_forecast(daily: pd.DataFrame, product_id: str, horizon: int, n_folds: int = 6,
                 models: tuple[str, ...] = DEFAULT_MODELS, filled_days: int = 0,
                 level: float = INTERVAL_LEVEL, max_backtest_horizon: int = BACKTEST_HORIZON) -> ForecastResult:
    """Backtest all candidates, pick one, refit it on all history and forecast ahead.

    The backtest always uses the longest window that fits (30 days when
    history allows), and the model is selected on that full window. The
    display horizon only changes which part of the backtest is summarised
    and how much of the forecast is charted, so switching between 7, 14 and
    30 days never changes the model or the order recommendation.
    """
    notes = []
    h_bt = backtest_horizon(len(daily), max_backtest_horizon)
    if h_bt < 7:
        raise ValueError(
            f"Not enough history for a time-aware backtest: {len(daily)} days available, "
            f"need at least {MIN_TRAIN_DAYS + 14}."
        )
    if horizon > h_bt:
        notes.append(f"History only supports {h_bt}-day backtests, so the forecast horizon was reduced to {h_bt} days.")
        horizon = h_bt
    folds = feasible_folds(len(daily), h_bt, n_folds)
    if folds < n_folds:
        notes.append(f"Only {folds} backtest windows fit in the available history (wanted {n_folds}).")

    bt = backtest(daily, h_bt, folds, models)
    failed = bt.attrs.get("failed", [])
    if bt.empty:
        raise ValueError("Every model failed on this product's history.")
    if failed:
        notes.append("Left out of the comparison because they could not be fitted: "
                     + ", ".join(MODEL_REGISTRY[m].short_label for m in failed) + ".")
    chosen, reason = select_model(summarise(bt))
    reason += f" over {h_bt}-day backtest windows"

    shown = bt[bt["step"] <= horizon]
    fm = fold_metrics(shown)
    summary = summarise(shown)

    model = make_model(chosen).fit(daily)
    internal_h = max(horizon, MAX_INTERNAL_HORIZON)
    point = model.predict(internal_h)
    sel_bt = bt[bt["model"] == chosen]
    sel_shown = sel_bt[sel_bt["step"] <= horizon]
    current_level = demand_level(daily)
    lo, hi = interval_offsets(sel_bt, np.arange(1, internal_h + 1), level)
    lo, hi = lo * current_level, hi * current_level
    idx = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=internal_h, freq="D", name="date")
    fc = pd.DataFrame(
        {"forecast": point, "lower": np.clip(point + lo, 0, None), "upper": np.clip(point + hi, 0, None)},
        index=idx,
    )
    importance = model.feature_importance() if hasattr(model, "feature_importance") else None

    return ForecastResult(
        product_id=product_id,
        horizon=horizon,
        backtest_horizon=h_bt,
        history=daily,
        backtest=bt,
        fold_metrics=fm,
        summary=summary,
        selected_model=chosen,
        selection_reason=reason,
        forecast=fc,
        residual_rmse=rmse(sel_shown["actual"], sel_shown["forecast"]),
        daily_sigma=float(np.sqrt(np.mean(sel_bt["scaled_error"] ** 2))) * current_level,
        current_level=current_level,
        interval_level=level,
        coverage=interval_coverage(sel_shown, level),
        n_folds=folds,
        filled_days=filled_days,
        feature_importance=importance,
        notes=notes,
    )
