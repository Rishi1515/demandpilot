"""Tests for features, models and backtesting, with a focus on leakage."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    backtest,
    interval_coverage,
    rolling_origin_cutoffs,
    run_forecast,
    select_model,
    smape,
    summarise,
    wape,
)
from src.features import future_frame, make_features
from src.models import MODEL_REGISTRY, SeasonalNaive, make_model


def weekly_series(days=200, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days, freq="D", name="date")
    pattern = np.array([10, 12, 11, 13, 20, 30, 25], dtype=float)
    y = np.tile(pattern, days // 7 + 1)[:days] + rng.normal(0, noise, days)
    return pd.DataFrame({"units_sold": np.clip(y, 0, None), "promotion": 0.0, "unit_price": 2.0}, index=idx)


def test_features_do_not_use_same_day_or_future_values():
    df = weekly_series(120)
    base = make_features(df)
    changed = df.copy()
    changed.iloc[100:, 0] = 9999  # tamper with day 100 onwards
    feats = make_features(changed)
    # every feature for days <= 100 must be unchanged: day 100 may only use days before it
    pd.testing.assert_frame_equal(base.iloc[:101], feats.iloc[:101])


def test_future_frame_assumes_no_promotion_and_last_price():
    df = weekly_series(60)
    df.iloc[-1, df.columns.get_loc("unit_price")] = 3.5
    fut = future_frame(df, 10)
    assert len(fut) == 10 and fut.index.min() == df.index.max() + pd.Timedelta(days=1)
    assert (fut["promotion"] == 0).all() and (fut["unit_price"] == 3.5).all()
    assert fut["units_sold"].isna().all()


def test_seasonal_naive_repeats_last_week():
    df = weekly_series(70)
    fc = SeasonalNaive().fit(df).predict(10)
    expected = np.concatenate([df["units_sold"].to_numpy()[-7:], df["units_sold"].to_numpy()[-7:-4]])
    np.testing.assert_allclose(fc, expected)


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_every_model_returns_non_negative_forecast_of_right_length(name):
    fc = make_model(name).fit(weekly_series(150, noise=2)).predict(14)
    assert fc.shape == (14,)
    assert np.all(np.isfinite(fc)) and np.all(fc >= 0)


def test_rolling_origin_cutoffs_are_ordered_and_non_overlapping():
    cuts = rolling_origin_cutoffs(n=200, horizon=14, n_folds=5)
    assert cuts == [130, 144, 158, 172, 186]
    assert cuts[-1] + 14 == 200


def test_backtest_never_trains_on_test_window():
    df = weekly_series(160, noise=1)
    bt = backtest(df, horizon=7, n_folds=3, models=("seasonal_naive",))
    for _, g in bt.groupby("fold"):
        assert g["date"].min() == g["cutoff"].iloc[0]
        assert len(g) == 7
    # seasonal naive on fold k must equal the 7 days before that fold's cutoff
    first = bt[bt["fold"] == 1]
    cut = df.index.get_loc(first["cutoff"].iloc[0])
    np.testing.assert_allclose(first["forecast"], df["units_sold"].iloc[cut - 7 : cut])


def test_backtest_results_do_not_depend_on_future_data():
    df = weekly_series(160, noise=1, seed=3)
    bt_a = backtest(df, horizon=7, n_folds=2, models=("holt_winters", "lightgbm"))
    tampered = df.copy()
    tampered.iloc[-7:, 0] = 5000  # only the final test window changes
    bt_b = backtest(tampered, horizon=7, n_folds=2, models=("holt_winters", "lightgbm"))
    fc_a = bt_a[bt_a["fold"] == 2]["forecast"].to_numpy()
    fc_b = bt_b[bt_b["fold"] == 2]["forecast"].to_numpy()
    np.testing.assert_allclose(fc_a, fc_b)  # forecasts identical, only actuals differ


def test_metrics():
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)
    assert np.isnan(wape([0, 0], [1, 1]))
    assert smape([0, 10], [0, 10]) == 0.0


def test_selection_prefers_simpler_model_when_tied():
    s = pd.DataFrame({"WAPE": [0.300, 0.250, 0.249], "complexity": [0, 1, 3]},
                     index=["seasonal_naive", "moving_average", "lightgbm"])
    chosen, reason = select_model(s)
    assert chosen == "moving_average" and "simpler" in reason
    s.loc["lightgbm", "WAPE"] = 0.20
    assert select_model(s)[0] == "lightgbm"


def test_run_forecast_end_to_end():
    df = weekly_series(240, noise=1.5)
    res = run_forecast(df, "X", horizon=14, n_folds=4)
    assert res.selected_model in MODEL_REGISTRY
    assert len(res.display_forecast) == 14
    fc = res.forecast
    assert (fc["lower"] <= fc["forecast"] + 1e-9).all() and (fc["upper"] >= fc["forecast"] - 1e-9).all()
    assert 0 <= res.coverage <= 1
    # a regular weekly pattern should beat a flat moving average comfortably
    summ = summarise(res.backtest)
    assert summ.loc[res.selected_model, "WAPE"] < summ.loc["moving_average", "WAPE"]


def test_run_forecast_rejects_too_little_history():
    with pytest.raises(ValueError):
        run_forecast(weekly_series(50), "X", horizon=14)


def test_interval_coverage_needs_three_folds():
    df = weekly_series(200, noise=1)
    bt = backtest(df, 7, 2, models=("seasonal_naive",))
    assert np.isnan(interval_coverage(bt))
