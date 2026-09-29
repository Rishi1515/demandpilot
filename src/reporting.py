"""Downloadable outputs: forecast CSV, recommendation CSV and a short HTML report."""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from .backtesting import ForecastResult
from .inventory import Recommendation
from .models import model_label


def forecast_table(res: ForecastResult) -> pd.DataFrame:
    fc = res.display_forecast.reset_index()
    fc.insert(0, "product_id", res.product_id)
    fc["model"] = res.selected_model
    fc["interval_level"] = res.interval_level
    fc["date"] = fc["date"].dt.strftime("%Y-%m-%d")
    return fc.round({"forecast": 2, "lower": 2, "upper": 2})


def recommendation_table(res: ForecastResult, rec: Recommendation) -> pd.DataFrame:
    s = res.summary.loc[res.selected_model]
    row = {
        "product_id": res.product_id,
        "forecast_horizon_days": res.horizon,
        "forecast_total_units": round(float(res.display_forecast["forecast"].sum()), 1),
        "selected_model": res.selected_model,
        "backtest_WAPE": round(float(s["WAPE"]), 4),
        "backtest_MAE": round(float(s["MAE"]), 3),
        "backtest_RMSE": round(float(s["RMSE"]), 3),
        "WAPE_improvement_vs_seasonal_naive": round(float(s.get("WAPE_vs_baseline", 0.0)), 4),
        "interval_coverage_backtest": round(float(res.coverage), 3),
        "action": rec.action,
        "order_quantity": rec.order_quantity,
        "reorder_point": round(rec.reorder_point, 1),
        "safety_stock": round(rec.safety_stock, 1),
        "order_up_to_level": round(rec.order_up_to_level, 1),
        "lead_time_demand": round(rec.lead_time_demand, 1),
        "days_of_cover": round(rec.days_of_cover, 1),
        "stockout_risk": rec.stockout_risk,
        "stockout_probability_lead_time": round(rec.stockout_probability, 3),
        "excess_risk": rec.excess_risk,
        "surplus_units_above_target": round(rec.excess_units, 1),
    }
    for k, v in rec.assumptions.items():
        row["assumption: " + k] = v
    return pd.DataFrame([row])


def html_report(res: ForecastResult, rec: Recommendation, explanation_md: str, product_name: str | None = None) -> str:
    """A self-contained, printable report. Plain black-and-white styling."""
    def md_to_html(text: str) -> str:
        out = []
        for para in text.split("\n\n"):
            p = html.escape(para)
            while "**" in p:
                p = p.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
            out.append(f"<p>{p}</p>")
        return "\n".join(out)

    metrics = res.summary[["MAE", "RMSE", "WAPE", "sMAPE", "WAPE_vs_baseline"]].copy()
    metrics.index = [model_label(m) + (" (selected)" if m == res.selected_model else "") for m in metrics.index]
    metrics_html = metrics.to_html(
        float_format=lambda v: f"{v:,.3f}", classes="t", border=0,
        formatters={"WAPE": "{:.1%}".format, "sMAPE": "{:.3f}".format, "WAPE_vs_baseline": "{:+.1%}".format},
    )
    fc = forecast_table(res)[["date", "forecast", "lower", "upper"]].to_html(index=False, classes="t", border=0)
    assumptions = pd.DataFrame(list(rec.assumptions.items()), columns=["Assumption", "Value"]).to_html(
        index=False, classes="t", border=0)
    name = html.escape(product_name or res.product_id)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>DemandPilot report: {name}</title>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; color: #000; background: #fff; max-width: 820px; margin: 32px auto; padding: 0 16px; line-height: 1.5; }}
h1 {{ font-size: 24px; margin-bottom: 4px; }} h2 {{ font-size: 18px; margin-top: 28px; border-bottom: 1px solid #000; }}
.meta {{ color: #333; font-size: 14px; }}
table.t {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
table.t th, table.t td {{ border: 1px solid #999; padding: 4px 8px; text-align: right; }}
table.t th:first-child, table.t td:first-child {{ text-align: left; }}
.box {{ border: 2px solid #000; padding: 12px 16px; margin: 16px 0; }}
.small {{ font-size: 12px; color: #333; }}
</style></head><body>
<h1>DemandPilot report: {name}</h1>
<div class="meta">Product ID {html.escape(res.product_id)} · Horizon {res.horizon} days · Generated {datetime.now():%Y-%m-%d %H:%M}</div>
<div class="box"><strong>{html.escape(rec.action)}</strong>: order {rec.order_quantity:,} units ·
reorder point {rec.reorder_point:,.0f} · safety stock {rec.safety_stock:,.0f} ·
stockout risk {rec.stockout_risk} · excess risk {rec.excess_risk}</div>
<h2>Explanation</h2>
{md_to_html(explanation_md)}
<h2>Model comparison (rolling backtest, {res.n_folds} windows)</h2>
{metrics_html}
<p class="small">Selected model: {html.escape(model_label(res.selected_model))}, because {html.escape(res.selection_reason)}.
WAPE_vs_baseline is the relative WAPE improvement over the seasonal-naive forecast.</p>
<h2>Assumptions</h2>
{assumptions}
<h2>Daily forecast</h2>
{fc}
<p class="small">DemandPilot output is decision support built from historical data and stated assumptions.
It is not a guarantee of future demand or of any business outcome.</p>
</body></html>"""
