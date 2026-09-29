"""Plain-language explanations built from computed values only.

The deterministic template is the source of truth. An optional language
model may rephrase it, but it receives only the template text, and its
answer is rejected unless it states exactly the template's numbers in the
same order (as digits) and keeps the recommended action. That keeps every figure the user
sees traceable to the forecasting and inventory code.
"""

from __future__ import annotations

import os
import re

import numpy as np

from .backtesting import ForecastResult
from .inventory import Recommendation
from .models import BASELINE, model_label

TREND_THRESHOLD = 0.10


def _fmt(x: float, decimals: int = 0) -> str:
    return f"{x:,.{decimals}f}"


def pct_text(p: float) -> str:
    """Probability as a readable percentage that never claims certainty."""
    if p >= 0.995:
        return "over 99%"
    if 0 < p < 0.005:
        return "under 1%"
    return f"{p * 100:.0f}%"


def recent_trend(history_units: np.ndarray, window: int = 28) -> tuple[str, float, float, float]:
    """Compare the last `window` days with the `window` days before them."""
    y = np.asarray(history_units, dtype=float)
    if len(y) < 2 * window:
        return "unknown", float("nan"), float("nan"), float("nan")
    recent, prior = y[-window:].mean(), y[-2 * window : -window].mean()
    change = (recent - prior) / prior if prior > 0 else float("inf") if recent > 0 else 0.0
    if change > TREND_THRESHOLD:
        label = "rising"
    elif change < -TREND_THRESHOLD:
        label = "falling"
    else:
        label = "stable"
    return label, float(recent), float(prior), float(change)


def build_facts(res: ForecastResult, rec: Recommendation, product_name: str | None = None) -> dict:
    fc = res.display_forecast
    s = res.summary
    sel = res.selected_model
    trend, recent, prior, change = recent_trend(res.history["units_sold"].to_numpy())
    return {
        "product": product_name or res.product_id,
        "horizon": res.horizon,
        "total_forecast": float(fc["forecast"].sum()),
        "avg_daily_forecast": float(fc["forecast"].mean()),
        "trend": trend,
        "recent_avg": recent,
        "prior_avg": prior,
        "trend_change": change,
        "model": sel,
        "model_label": model_label(sel, short=True),
        "selection_reason": res.selection_reason,
        "wape": float(s.loc[sel, "WAPE"]),
        "mae": float(s.loc[sel, "MAE"]),
        "improvement_vs_baseline": float(s.loc[sel, "WAPE_vs_baseline"]) if sel != BASELINE else 0.0,
        "n_folds": res.n_folds,
        "interval_level": res.interval_level,
        "coverage": res.coverage,
        "filled_days": res.filled_days,
        "history_days": len(res.history),
        "action": rec.action,
        "order_quantity": rec.order_quantity,
        "reorder_point": rec.reorder_point,
        "safety_stock": rec.safety_stock,
        "lead_time_demand": rec.lead_time_demand,
        "position": rec.inventory_position,
        "lead_time": int(rec.assumptions["Lead time (days)"]),
        "review_period": int(rec.assumptions["Review period (days)"]),
        "service_level": float(rec.assumptions["Service level"]),
        "stockout_risk": rec.stockout_risk,
        "stockout_probability": rec.stockout_probability,
        "excess_risk": rec.excess_risk,
        "excess_units": rec.excess_units,
        "days_of_cover": rec.days_of_cover,
        "warnings": list(rec.warnings),
    }


def template_explanation(f: dict) -> str:
    """Deterministic explanation. Every number comes from `f`."""
    parts = []
    parts.append(
        f"**Forecast.** Over the next {f['horizon']} days, expected demand for {f['product']} is about "
        f"{_fmt(f['total_forecast'])} units, or {_fmt(f['avg_daily_forecast'], 1)} units per day on average."
    )

    if f["trend"] == "unknown":
        parts.append("**Recent trend.** There is not enough history to compare the last two 28-day periods.")
    elif not np.isfinite(f["trend_change"]):
        parts.append(
            f"**Recent trend.** Demand in the last 28 days averaged {_fmt(f['recent_avg'], 1)} units per day, "
            "after no sales in the 28 days before, so this looks like a new or returning product."
        )
    elif f["trend"] == "stable":
        parts.append(
            f"**Recent trend.** Demand in the last 28 days averaged {_fmt(f['recent_avg'], 1)} units per day, "
            f"close to the 28 days before ({_fmt(f['prior_avg'], 1)} per day), so demand is stable."
        )
    else:
        direction = "higher" if f["trend"] == "rising" else "lower"
        parts.append(
            f"**Recent trend.** Demand in the last 28 days averaged {_fmt(f['recent_avg'], 1)} units per day, "
            f"{_fmt(abs(f['trend_change']) * 100)}% {direction} than the 28 days before "
            f"({_fmt(f['prior_avg'], 1)} per day), so demand is {f['trend']}."
        )

    reliab = (
        f"**How reliable this is.** The forecast comes from the {f['model_label']} model, chosen because "
        f"{f['selection_reason']}. "
    )
    if np.isfinite(f["wape"]):
        reliab += (
            f"Over the first {f['horizon']} days of {f['n_folds']} backtest windows, its total error was "
            f"{_fmt(f['wape'] * 100, 1)}% of actual volume (WAPE), or about {_fmt(f['mae'], 1)} units per day."
        )
    else:
        reliab += (
            f"There were no sales in the backtest windows, so percentage error cannot be computed; the average "
            f"error was {_fmt(f['mae'], 1)} units per day."
        )
    if f["model"] != BASELINE and np.isfinite(f["improvement_vs_baseline"]):
        imp = f["improvement_vs_baseline"]
        word = "better" if imp >= 0 else "worse"
        reliab += f" That is {_fmt(abs(imp) * 100)}% {word} than simply repeating last week's sales."
    if not np.isnan(f["coverage"]):
        reliab += (
            f" The shaded band on the chart is an {_fmt(f['interval_level'] * 100)}% range; in backtesting it "
            f"contained the actual value on {_fmt(f['coverage'] * 100)}% of days"
            + (", so treat it as narrower than it looks." if f["coverage"] < f["interval_level"] - 0.05 else ".")
        )
    parts.append(reliab)

    if f["order_quantity"] > 0:
        rec = (
            f"**Recommendation: {f['action']}.** Order about {_fmt(f['order_quantity'])} units. "
            f"Stock on hand plus on order ({_fmt(f['position'])} units) is at or below the reorder point of "
            f"{_fmt(f['reorder_point'])} units, which is the expected demand over the {f['lead_time']}-day lead time "
            f"({_fmt(f['lead_time_demand'])}) plus {_fmt(f['safety_stock'])} units of safety stock for a "
            f"{_fmt(f['service_level'] * 100, 1)}% service level. The order size brings stock up to cover the lead time "
            f"plus a {f['review_period']}-day review period."
        )
    else:
        rec = (
            f"**Recommendation: {f['action']}.** Stock on hand plus on order ({_fmt(f['position'])} units) "
            f"is above the reorder point of {_fmt(f['reorder_point'])} units (lead-time demand {_fmt(f['lead_time_demand'])} "
            f"plus safety stock {_fmt(f['safety_stock'])}), so no order is needed today."
        )
        if np.isfinite(f["days_of_cover"]):
            rec += f" Current stock should last about {_fmt(f['days_of_cover'])} days at the forecast rate."
    parts.append(rec)

    if f["stockout_risk"] in ("High", "Medium"):
        risk = (
            f"**Main risk: stockout ({f['stockout_risk'].lower()}).** There is roughly a "
            f"{pct_text(f['stockout_probability'])} chance that demand during the {f['lead_time']}-day lead time "
            f"exceeds the stock available, so a new order placed today may arrive too late for some sales."
        )
    elif f["excess_risk"] in ("High", "Medium"):
        risk = (
            f"**Main risk: excess stock ({f['excess_risk'].lower()}).** Stock is about {_fmt(f['excess_units'])} units "
            f"above the most this policy would hold, so cash is tied up and the item may age on the shelf."
        )
    elif f["history_days"] < 180 or (not np.isnan(f["coverage"]) and f["coverage"] < f["interval_level"] - 0.15):
        risk = "**Main risk: uncertainty.** The history is short or the error range has been unreliable, so review this recommendation manually."
    else:
        risk = "**Main risk.** No stockout or excess-stock flag at current settings."
    parts.append(risk)

    caveats = [
        "Future promotions are assumed to be zero and price is held at its last value.",
        "Safety stock assumes forecast errors on different days are independent.",
    ]
    if f["filled_days"] > 0:
        caveats.append(f"{f['filled_days']} day(s) had no sales rows and were treated as zero sales.")
    caveats.extend(f["warnings"])
    parts.append("**Assumptions and limits.** " + " ".join(caveats) + " This is decision support, not a guaranteed outcome.")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- optional LLM

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
    r"hundred|thousand|million|dozen|half|double|twice|triple)\b",
    re.IGNORECASE,
)


def _numbers(text: str) -> list[str]:
    return [m.replace(",", "").rstrip(".") for m in _NUM.findall(text)]


def numbers_are_grounded(candidate: str, source: str) -> bool:
    """True if `candidate` states exactly the numbers of `source`, in the same order.

    Also rejects numbers written as words, which would slip past a digit check.
    """
    if _NUMBER_WORDS.search(candidate) and not _NUMBER_WORDS.search(source):
        return False
    return _numbers(candidate) == _numbers(source)


def action_is_preserved(candidate: str, action: str) -> bool:
    return action.lower() in candidate.lower()


def llm_available() -> bool:
    return os.getenv("DEMANDPILOT_LLM_PROVIDER", "").lower() == "anthropic" and bool(os.getenv("ANTHROPIC_API_KEY"))


def llm_rephrase(template_text: str, client=None, action: str | None = None) -> tuple[str, str]:
    """Ask a language model to make the template friendlier. Returns (text, source).

    Falls back to the template if the provider is not configured, the call
    fails, the answer does not contain exactly the template's numbers in the
    same order, writes numbers as words, or drops the recommended action.
    """
    if client is None and not llm_available():
        return template_text, "template"
    try:
        if client is None:
            import anthropic  # optional dependency

            client = anthropic.Anthropic()
        prompt = (
            "Rewrite the inventory briefing below for a busy store manager in at most 170 words. "
            "Keep the bold section labels and the section order. Keep every number exactly as written, in the "
            "same order, written as digits. Do not add, drop, round or change any number, keep the recommended "
            "action wording, and do not add advice that is not in the text.\n\n" + template_text
        )
        msg = client.messages.create(
            model=os.getenv("DEMANDPILOT_LLM_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content).strip()
    except Exception:  # noqa: BLE001 - any provider failure falls back safely
        return template_text, "template (language model unavailable)"
    if not text or not numbers_are_grounded(text, template_text):
        return template_text, "template (language model answer rejected: its numbers did not match)"
    if action and not action_is_preserved(text, action):
        return template_text, "template (language model answer rejected: recommended action changed)"
    return text, "language model rephrasing of the template"
