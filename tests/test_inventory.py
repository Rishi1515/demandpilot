import math

import numpy as np
import pytest

from src.inventory import (
    InventoryInputs,
    days_of_cover,
    demand_over,
    expected_shortfall,
    order_quantity,
    recommend,
    reorder_point,
    safety_stock,
    stockout_probability,
    z_value,
)

FLAT = np.full(60, 10.0)  # 10 units a day


def test_z_value_matches_standard_normal_table():
    assert z_value(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert z_value(0.99) == pytest.approx(2.3263, abs=1e-4)
    with pytest.raises(ValueError):
        z_value(1.0)


def test_safety_stock_formula():
    # z(0.95) * sigma * sqrt(L) = 1.6449 * 4 * 3
    assert safety_stock(4.0, 9, 0.95) == pytest.approx(1.6449 * 12, rel=1e-4)
    assert safety_stock(0.0, 9, 0.95) == 0.0


def test_reorder_point_is_lead_time_demand_plus_safety_stock():
    assert reorder_point(70.0, 15.5) == 85.5


def test_demand_over_sums_forecast_and_extends_past_horizon():
    assert demand_over(FLAT, 7) == (70.0, False)
    total, extended = demand_over(np.full(5, 2.0), 8)
    assert extended and total == pytest.approx(16.0)


def test_order_quantity_is_never_negative_and_zero_above_reorder_point():
    assert order_quantity(order_up_to=100, inventory_position=150, reorder_pt=120) == 0
    assert order_quantity(order_up_to=100, inventory_position=110, reorder_pt=120) == 0  # target below position
    assert order_quantity(order_up_to=200, inventory_position=50, reorder_pt=120) == 150
    assert order_quantity(order_up_to=200.2, inventory_position=50, reorder_pt=120) == 151  # rounds up


def test_recommend_orders_up_to_target_when_below_reorder_point():
    inp = InventoryInputs(current_stock=40, lead_time_days=7, service_level=0.95, review_period_days=7)
    rec = recommend(FLAT, daily_sigma=3.0, inputs=inp)
    ss = 1.6449 * 3.0 * math.sqrt(7)
    assert rec.lead_time_demand == pytest.approx(70)
    assert rec.safety_stock == pytest.approx(ss, rel=1e-4)
    assert rec.reorder_point == pytest.approx(70 + ss, rel=1e-4)
    target = 140 + 1.6449 * 3.0 * math.sqrt(14)
    assert rec.order_up_to_level == pytest.approx(target, rel=1e-4)
    assert rec.order_quantity == math.ceil(target - 40)
    assert rec.action == "Reorder now"
    assert rec.stockout_risk == "High"  # 40 < 70 expected lead-time demand


def test_on_order_units_count_towards_position():
    inp = InventoryInputs(current_stock=40, on_order=200, lead_time_days=7, review_period_days=7)
    rec = recommend(FLAT, daily_sigma=3.0, inputs=inp)
    assert rec.inventory_position == 240
    assert rec.order_quantity == 0


def test_changing_inputs_moves_recommendation_in_the_right_direction():
    base = dict(current_stock=60, lead_time_days=7, review_period_days=7)
    low = recommend(FLAT, 3.0, InventoryInputs(service_level=0.90, **base))
    high = recommend(FLAT, 3.0, InventoryInputs(service_level=0.99, **base))
    assert high.safety_stock > low.safety_stock
    assert high.order_quantity >= low.order_quantity
    short = recommend(FLAT, 3.0, InventoryInputs(**{**base, "lead_time_days": 3}))
    long = recommend(FLAT, 3.0, InventoryInputs(**{**base, "lead_time_days": 14}))
    assert long.reorder_point > short.reorder_point
    more_stock = recommend(FLAT, 3.0, InventoryInputs(**{**base, "current_stock": 500}))
    assert more_stock.order_quantity == 0


def test_excess_stock_is_flagged():
    rec = recommend(FLAT, 3.0, InventoryInputs(current_stock=1000, lead_time_days=7, review_period_days=7))
    assert rec.excess_risk == "High"
    assert rec.excess_units == pytest.approx(1000 - rec.order_up_to_level)
    assert rec.action.startswith("Pause ordering")


def test_stockout_probability_and_shortfall_behave():
    assert stockout_probability(100, 100, 10) == pytest.approx(0.5)
    assert stockout_probability(200, 100, 10) < 0.001
    assert stockout_probability(50, 100, 0) == 1.0
    # Normal loss function at k = 0 is 0.3989 * sigma
    assert expected_shortfall(100, 100, 10) == pytest.approx(3.989, abs=1e-3)
    assert expected_shortfall(500, 100, 10) == pytest.approx(0.0, abs=1e-9)


def test_days_of_cover():
    cover, idx = days_of_cover(25, FLAT)
    assert cover == pytest.approx(2.5) and idx == 2
    cover, idx = days_of_cover(1000, FLAT)
    assert idx is None and cover == pytest.approx(100)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        recommend(FLAT, 1.0, InventoryInputs(current_stock=10, lead_time_days=0))


def test_default_flags_produce_warnings():
    rec = recommend(FLAT, 1.0, InventoryInputs(current_stock=0, lead_time_days=7, lead_time_is_default=True,
                                               stock_is_default=True))
    assert any("default" in w for w in rec.warnings)
    assert any("Current stock" in w for w in rec.warnings)
