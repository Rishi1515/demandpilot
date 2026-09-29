import re

import numpy as np
import pandas as pd

from src.backtesting import run_forecast
from src.explanations import build_facts, llm_rephrase, numbers_are_grounded, recent_trend, template_explanation
from src.inventory import InventoryInputs, recommend


def _result():
    idx = pd.date_range("2025-01-01", periods=200, freq="D", name="date")
    y = np.tile([10, 12, 11, 13, 20, 30, 25], 30)[:200].astype(float)
    res = run_forecast(pd.DataFrame({"units_sold": y}, index=idx), "SKU1", horizon=14, n_folds=4)
    rec = recommend(res.forecast["forecast"].to_numpy(), res.daily_sigma,
                    InventoryInputs(current_stock=30, lead_time_days=7))
    return res, rec


def test_recent_trend_labels():
    assert recent_trend(np.r_[np.full(28, 10), np.full(28, 20)])[0] == "rising"
    assert recent_trend(np.r_[np.full(28, 20), np.full(28, 10)])[0] == "falling"
    assert recent_trend(np.full(56, 10))[0] == "stable"
    assert recent_trend(np.full(10, 10))[0] == "unknown"


def test_template_mentions_forecast_recommendation_risk_and_limits():
    res, rec = _result()
    text = template_explanation(build_facts(res, rec))
    for section in ("Forecast.", "Recent trend.", "How reliable this is.", "Recommendation", "Main risk", "Assumptions"):
        assert section in text
    assert f"{rec.order_quantity:,}" in text
    assert "not a guaranteed outcome" in text


def test_numeric_guard():
    source = "Order 120 units. Reorder point 95. Service level 95.0%."
    assert numbers_are_grounded("Please order 120 units; the reorder point is 95 at a 95.0% service level.", source)
    assert not numbers_are_grounded("Please order 150 units. Reorder point 95. Service level 95.0%.", source)
    # swapped numbers, dropped numbers and numbers written as words are all rejected
    assert not numbers_are_grounded("Order 95 units. Reorder point 120. Service level 95.0%.", source)
    assert not numbers_are_grounded("Order 120 units.", source)
    assert not numbers_are_grounded("Order a hundred and twenty units. Reorder point 95. Service level 95.0%.", source)


class _FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.messages = self

    def create(self, **kwargs):
        class Block:
            text = self.reply

        class Msg:
            content = [Block()]

        return Msg()


def test_llm_answer_that_changes_the_action_is_rejected():
    template = "**Recommendation: Reorder now.** Order about 1,200 units."
    text, source = llm_rephrase(template, client=_FakeClient("No need to order yet, maybe 1,200 units later."),
                                action="Reorder now")
    assert text == template and "action" in source


def test_llm_answer_with_invented_number_is_rejected():
    template = "**Forecast.** About 1,200 units over 14 days."
    text, source = llm_rephrase(template, client=_FakeClient("Expect roughly 1,500 units over 14 days."))
    assert text == template and "rejected" in source
    ok_text, ok_source = llm_rephrase(template, client=_FakeClient("You should expect about 1,200 units in the next 14 days."))
    assert ok_text.startswith("You should") and "language model" in ok_source


def test_template_has_no_nan_or_inf():
    res, rec = _result()
    text = template_explanation(build_facts(res, rec))
    assert re.search(r"\d", text)
    assert "nan" not in text.lower() and "inf" not in text.lower().replace("information", "")


def test_new_product_trend_does_not_print_infinity():
    y = np.r_[np.zeros(150), np.full(50, 4.0)]
    idx = pd.date_range("2025-01-01", periods=200, freq="D", name="date")
    res = run_forecast(pd.DataFrame({"units_sold": y}, index=idx), "NEW", horizon=14, n_folds=3)
    rec = recommend(res.forecast["forecast"].to_numpy(), res.daily_sigma, InventoryInputs(current_stock=10, lead_time_days=7))
    text = template_explanation(build_facts(res, rec))
    assert "inf" not in text.lower() and "nan" not in text.lower()
