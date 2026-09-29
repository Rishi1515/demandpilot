"""Glue between the layers, shared by the app and the command-line scripts."""

from __future__ import annotations

import pandas as pd

from .backtesting import ForecastResult, run_forecast
from .data_validation import latest_value, to_daily
from .inventory import InventoryInputs, Recommendation, recommend

DEFAULT_LEAD_TIME = 7
MAX_LEAD_TIME = 180


def forecast_product(clean: pd.DataFrame, product_id: str, horizon: int, n_folds: int = 6) -> ForecastResult:
    end = clean["date"].max()
    daily, filled = to_daily(clean, product_id, end=end)
    res = run_forecast(daily, product_id, horizon, n_folds=n_folds, filled_days=filled)
    last_sale_row = clean.loc[clean["product_id"] == product_id, "date"].max()
    gap = (end - last_sale_row).days
    if gap > 7:
        res.notes.append(
            f"This product has no rows for the last {gap} days of the file (after {last_sale_row:%d %b %Y}). "
            "Those days are treated as zero sales. If it was out of stock, real demand was higher."
        )
    recent = daily["units_sold"].tail(90)
    if (recent == 0).mean() > 0.3:
        res.notes.append(
            f"Sales are intermittent ({(recent == 0).mean():.0%} of the last 90 days had no sales). "
            "Percentage errors and the normal-approximation risk figures are rough for items like this."
        )
    return res


def default_inputs(clean: pd.DataFrame, product_id: str, service_level: float = 0.95,
                   review_period_days: int = 7) -> InventoryInputs:
    """Inventory inputs taken from the data where present, with labelled defaults otherwise."""
    stock = latest_value(clean, product_id, "current_stock")
    lead = latest_value(clean, product_id, "lead_time_days")
    return InventoryInputs(
        current_stock=stock if stock is not None else 0.0,
        lead_time_days=min(int(round(lead)), MAX_LEAD_TIME) if lead and lead >= 1 else DEFAULT_LEAD_TIME,
        service_level=service_level,
        review_period_days=review_period_days,
        unit_cost=latest_value(clean, product_id, "unit_cost"),
        unit_price=latest_value(clean, product_id, "unit_price"),
        lead_time_is_default=not (lead and lead >= 1),
        stock_is_default=stock is None,
    )


def recommend_for(res: ForecastResult, inputs: InventoryInputs) -> Recommendation:
    return recommend(res.forecast["forecast"].to_numpy(), res.daily_sigma, inputs, sigma_fn=res.error_sigma)


def product_name(clean: pd.DataFrame, product_id: str) -> str:
    if "product_name" in clean.columns:
        names = clean.loc[clean["product_id"] == product_id, "product_name"].dropna()
        if not names.empty:
            return f"{names.iloc[-1]} ({product_id})"
    return str(product_id)
