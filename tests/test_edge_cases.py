"""Inputs that used to be risky: zero sales, short histories, stale products, long lead times."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import period_error_sigma, run_forecast
from src.data_validation import validate
from src.explanations import build_facts, template_explanation
from src.pipeline import default_inputs, forecast_product, recommend_for


def frame(pid, units, start="2025-01-01", **extra):
    dates = pd.date_range(start, periods=len(units), freq="D")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "product_id": pid, "units_sold": units})
    for k, v in extra.items():
        df[k] = v
    return df


def test_all_zero_product_does_not_crash():
    res = validate(frame("Z", np.zeros(200)))
    out = forecast_product(res.data, "Z", 14)
    assert np.allclose(out.forecast["forecast"], 0)
    rec = recommend_for(out, default_inputs(res.data, "Z"))
    assert rec.order_quantity >= 0
    assert "nan" not in template_explanation(build_facts(out, rec)).lower()


def test_long_leading_zeros_then_sales():
    units = np.r_[np.zeros(120), np.tile([3, 4, 5, 2, 6, 8, 7], 12)]
    res = validate(frame("L", units))
    out = forecast_product(res.data, "L", 30)
    assert len(out.display_forecast) == out.horizon


def test_short_eligible_history_caps_the_horizon_instead_of_crashing():
    res = validate(frame("S", np.tile([5, 6, 7, 5, 9, 12, 10], 9)[:60]))
    assert res.eligible_products == ["S"]
    out = forecast_product(res.data, "S", 30)
    assert out.horizon < 30 and any("reduced" in n for n in out.notes)


def test_horizon_choice_does_not_change_the_order_recommendation():
    rng = np.random.default_rng(1)
    units = rng.poisson(np.tile([10, 12, 11, 13, 20, 30, 25], 60)[:400])
    res = validate(frame("H", units, current_stock=np.r_[np.full(399, np.nan), 60], lead_time_days=7))
    recs = [recommend_for(forecast_product(res.data, "H", h), default_inputs(res.data, "H")) for h in (7, 14, 30)]
    assert len({r.order_quantity for r in recs}) == 1
    assert len({round(r.safety_stock, 6) for r in recs}) == 1


def test_product_that_stopped_selling_is_extended_to_file_end():
    a = frame("A", np.full(200, 5.0))
    b = frame("B", np.full(150, 5.0))  # last row 50 days before the file ends
    res = validate(pd.concat([a, b]))
    out = forecast_product(res.data, "B", 14)
    assert out.history.index.max() == res.data["date"].max()
    assert out.forecast.index.min() == res.data["date"].max() + pd.Timedelta(days=1)
    assert any("no rows for the last" in n for n in out.notes)


def test_lead_time_above_input_limit_is_clamped():
    res = validate(frame("T", np.full(120, 5.0), lead_time_days=400))
    assert default_inputs(res.data, "T").lead_time_days == 180


def test_period_error_sigma_matches_sqrt_rule_for_independent_errors():
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(1, 41):
        rows.append(pd.DataFrame({"fold": fold, "step": np.arange(1, 31), "scaled_error": rng.normal(0, 1, 30)}))
    bt = pd.concat(rows)
    s1, _ = period_error_sigma(bt, 1, 1.0)
    s9, _ = period_error_sigma(bt, 9, 1.0)
    assert s9 == pytest.approx(3 * s1, rel=0.15)
    s60, extended = period_error_sigma(bt, 60, 1.0)
    assert extended and s60 > s9


def test_period_error_sigma_captures_persistent_bias():
    rows = [pd.DataFrame({"fold": f, "step": np.arange(1, 15), "scaled_error": np.full(14, 0.5)}) for f in range(1, 7)]
    s7, _ = period_error_sigma(pd.concat(rows), 7, 2.0)
    assert s7 == pytest.approx(7 * 0.5 * 2.0)  # grows with L, not with sqrt(L)


def test_run_forecast_with_constant_series():
    idx = pd.date_range("2025-01-01", periods=150, freq="D", name="date")
    out = run_forecast(pd.DataFrame({"units_sold": np.full(150, 7.0)}, index=idx), "C", 14)
    assert np.allclose(out.display_forecast["forecast"], 7, atol=0.5)
