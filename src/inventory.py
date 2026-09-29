"""Inventory decision layer: safety stock, reorder point, order quantity and risk flags.

All formulas are deliberately textbook and visible to the user. They
assume forecast errors on different days are independent and roughly
normal, which is a simplification (see the README limitations section).

    lead-time demand      D_L = sum of the daily forecast over the next L days
    lead-time error std   s_L = std of the total forecast error over L days
    safety stock          SS  = z(service level) x s_L
    reorder point         ROP = D_L + SS
    order-up-to level     S   = D_(L+R) + z x s_(L+R)

In the app, s_L is measured directly from the backtest: the root mean
square of the selected model's summed errors over every run of L
consecutive forecast days, scaled to today's demand level (see
backtesting.period_error_sigma). This keeps day-to-day error correlation.
Without backtest errors, the textbook rule s_L = daily error x sqrt(L) is
used.
    order quantity        Q   = max(0, S - inventory position)   when position <= ROP, else 0
    surplus               E   = max(0, inventory position - S), flagged when position / S >= threshold

where R is the review period (how many days an order should cover
before the next ordering decision) and inventory position is on-hand
stock plus units already on order.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import numpy as np

_N = NormalDist()


@dataclass
class InventoryInputs:
    current_stock: float
    lead_time_days: int
    service_level: float = 0.95
    review_period_days: int = 7
    on_order: float = 0.0
    unit_cost: float | None = None
    unit_price: float | None = None
    holding_cost_rate: float = 0.25  # share of unit cost per year
    excess_ratio_threshold: float = 1.5  # stock above this multiple of S is flagged as excess
    lead_time_is_default: bool = False
    stock_is_default: bool = False


@dataclass
class Recommendation:
    action: str
    order_quantity: int
    reorder_point: float
    safety_stock: float
    order_up_to_level: float
    lead_time_demand: float
    lead_time_sigma: float
    cover_sigma: float
    z: float
    inventory_position: float
    days_of_cover: float
    projected_stockout_date_index: int | None
    stockout_probability: float
    expected_shortfall_units: float
    stockout_risk: str
    excess_units: float
    excess_ratio: float
    excess_risk: str
    revenue_at_risk: float | None
    excess_inventory_value: float | None
    annual_holding_cost_of_excess: float | None
    assumptions: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def z_value(service_level: float) -> float:
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be between 0.5 and 1 (exclusive)")
    return _N.inv_cdf(service_level)


def demand_over(forecast: np.ndarray, days: int) -> tuple[float, bool]:
    """Total forecast demand over the next `days`. Returns (total, extended?).

    If `days` is longer than the forecast, the remaining days use the average
    of the last 28 forecast days and the second value is True.
    """
    f = np.asarray(forecast, dtype=float)
    if days <= 0:
        return 0.0, False
    if days <= len(f):
        return float(f[:days].sum()), False
    tail_mean = float(f[-28:].mean()) if len(f) else 0.0
    return float(f.sum() + tail_mean * (days - len(f))), True


def safety_stock(daily_sigma: float, lead_time_days: int, service_level: float) -> float:
    return z_value(service_level) * daily_sigma * math.sqrt(max(lead_time_days, 0))


def reorder_point(lead_time_demand: float, safety: float) -> float:
    return lead_time_demand + safety


def order_quantity(order_up_to: float, inventory_position: float, reorder_pt: float) -> int:
    """Units to order now. Never negative, and zero while position is above the reorder point."""
    if inventory_position > reorder_pt:
        return 0
    return int(math.ceil(max(0.0, order_up_to - inventory_position)))


def stockout_probability(stock: float, mean_demand: float, sigma: float) -> float:
    """P(demand over the period > stock) under a normal approximation."""
    if mean_demand <= 1e-9:
        return 0.0  # no demand expected; the normal approximation is meaningless here
    if sigma <= 0:
        return 1.0 if mean_demand > stock else 0.0
    return 1 - _N.cdf((stock - mean_demand) / sigma)


def expected_shortfall(stock: float, mean_demand: float, sigma: float) -> float:
    """Expected units of unmet demand over the period (standard normal loss function)."""
    if sigma <= 0:
        return max(0.0, mean_demand - stock)
    k = (stock - mean_demand) / sigma
    loss = _N.pdf(k) - k * (1 - _N.cdf(k))
    return max(0.0, sigma * loss)


def days_of_cover(stock: float, forecast: np.ndarray) -> tuple[float, int | None]:
    """Days until cumulative forecast demand exceeds stock, and the index of that day."""
    f = np.asarray(forecast, dtype=float)
    cum = np.cumsum(f)
    hit = np.nonzero(cum > stock)[0]
    if len(hit):
        i = int(hit[0])
        prev = cum[i - 1] if i > 0 else 0.0
        frac = (stock - prev) / f[i] if f[i] > 0 else 0.0
        return i + frac, i
    avg = f.mean() if len(f) else 0.0
    extra = (stock - cum[-1]) / avg if avg > 0 and len(f) else float("inf")
    return len(f) + extra, None


def classify_stockout(position: float, lead_demand: float, rop: float) -> str:
    if position < lead_demand:
        return "High"
    if position < rop:
        return "Medium"
    return "Low"


def classify_excess(ratio: float, threshold: float) -> str:
    if ratio >= threshold * 4 / 3:
        return "High"
    if ratio >= threshold:
        return "Medium"
    return "Low"


def recommend(forecast: np.ndarray, daily_sigma: float, inputs: InventoryInputs,
              sigma_fn=None) -> Recommendation:
    """Turn a daily forecast and its backtest error into an ordering recommendation.

    `sigma_fn(days)`, when given, returns (std of total forecast error over
    `days`, extrapolated?) measured from the backtest. Without it the
    textbook rule daily_sigma x sqrt(days) is used.
    """
    warn = []
    L, R = int(inputs.lead_time_days), int(inputs.review_period_days)
    if L < 1:
        raise ValueError("lead_time_days must be at least 1")
    if R < 0:
        raise ValueError("review_period_days cannot be negative")
    stock = max(0.0, float(inputs.current_stock))
    on_order = max(0.0, float(inputs.on_order))
    position = stock + on_order
    z = z_value(inputs.service_level)

    d_l, ext1 = demand_over(forecast, L)
    d_lr, ext2 = demand_over(forecast, L + R)
    if ext1 or ext2:
        warn.append(
            f"Lead time plus review period ({L + R} days) is longer than the forecast "
            f"({len(forecast)} days). The extra days use the recent forecast average."
        )
    if sigma_fn is not None:
        sigma_l, ext_l = sigma_fn(L)
        sigma_lr, ext_lr = sigma_fn(L + R)
        method = "measured from summed backtest errors"
        if ext_l or ext_lr:
            method += " (extended with the square-root rule beyond the backtest window)"
    else:
        sigma_l, sigma_lr = daily_sigma * math.sqrt(L), daily_sigma * math.sqrt(L + R)
        method = "daily error x square root of days"
    ss = z * sigma_l
    rop = reorder_point(d_l, ss)
    s_level = d_lr + z * sigma_lr
    qty = order_quantity(s_level, position, rop)

    cover, stockout_idx = days_of_cover(stock, forecast)
    p_out = stockout_probability(position, d_l, sigma_l)
    shortfall = expected_shortfall(position, d_l, sigma_l)
    ratio = position / s_level if s_level > 0 else (float("inf") if position > 0 else 0.0)
    excess_units = max(0.0, position - s_level)  # surplus above the order-up-to level

    if qty > 0:
        action = "Reorder now"
    elif ratio >= inputs.excess_ratio_threshold:
        action = "Pause ordering (overstocked)"
    else:
        action = "No order needed yet"

    if inputs.lead_time_is_default:
        warn.append("Lead time was missing or invalid in the data, so a default value is being used.")
    if inputs.stock_is_default:
        warn.append("Current stock was not in the data. Enter it in the sidebar for a real recommendation.")

    price, cost = inputs.unit_price, inputs.unit_cost
    excess_value = excess_units * cost if cost is not None else None
    return Recommendation(
        action=action,
        order_quantity=qty,
        reorder_point=rop,
        safety_stock=ss,
        order_up_to_level=s_level,
        lead_time_demand=d_l,
        lead_time_sigma=sigma_l,
        cover_sigma=sigma_lr,
        z=z,
        inventory_position=position,
        days_of_cover=cover,
        projected_stockout_date_index=stockout_idx,
        stockout_probability=p_out,
        expected_shortfall_units=shortfall,
        stockout_risk=classify_stockout(position, d_l, rop),
        excess_units=excess_units,
        excess_ratio=ratio,
        excess_risk=classify_excess(ratio, inputs.excess_ratio_threshold),
        revenue_at_risk=shortfall * price if price is not None else None,
        excess_inventory_value=excess_value,
        annual_holding_cost_of_excess=excess_value * inputs.holding_cost_rate if excess_value is not None else None,
        assumptions={
            "Current stock (units)": stock,
            "Units already on order": on_order,
            "Lead time (days)": L,
            "Review period (days)": R,
            "Service level": inputs.service_level,
            "z value": round(z, 3),
            "Forecast error std over lead time (units)": round(sigma_l, 2),
            "Forecast error std over lead time + review (units)": round(sigma_lr, 2),
            "Error std method": method,
            "Excess threshold (x order-up-to level)": inputs.excess_ratio_threshold,
            "Holding cost (% of unit cost per year)": inputs.holding_cost_rate,
            "Unit cost": cost,
            "Unit price": price,
            "Lead time source": "default" if inputs.lead_time_is_default else "data or user input",
            "Stock source": "default" if inputs.stock_is_default else "data or user input",
        },
        warnings=warn,
    )


def projected_inventory(stock: float, forecast: np.ndarray, on_order: float = 0.0,
                        arrival_day: int | None = None) -> np.ndarray:
    """Expected end-of-day stock if nothing new is ordered (on-order units arrive on `arrival_day`)."""
    f = np.asarray(forecast, dtype=float)
    arrivals = np.zeros(len(f))
    if on_order > 0 and arrival_day is not None and 0 <= arrival_day < len(f):
        arrivals[arrival_day] = on_order
    return stock + np.cumsum(arrivals - f)
