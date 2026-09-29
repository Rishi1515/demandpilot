"""DemandPilot: demand forecasting and inventory decision assistant.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.backtesting import INTERVAL_LEVEL, PARSIMONY_TOLERANCE
from src.data_validation import SCHEMA_TABLE, data_hash, read_csv, template_csv, validate
from src.explanations import pct_text, build_facts, llm_available, llm_rephrase, template_explanation
from src.inventory import InventoryInputs
from src.models import MODEL_REGISTRY, model_label
from src.pipeline import default_inputs, forecast_product, product_name, recommend_for
from src.reporting import forecast_table, html_report, recommendation_table
from src.visualization import (
    backtest_chart,
    feature_importance_chart,
    forecast_chart,
    inventory_chart,
    model_comparison_chart,
    risk_badge,
    sales_overview_chart,
)

ROOT = Path(__file__).parent
SAMPLES = {
    "Sample data": ROOT / "data" / "sample_sales.csv",
    "Sample data with errors (validator demo)": ROOT / "data" / "sample_sales_with_issues.csv",
}
UPLOAD = "Upload my own CSV"
HORIZONS = [7, 14, 30]
N_FOLDS = 6

st.set_page_config(page_title="DemandPilot", page_icon="📦", layout="wide")


# ----------------------------------------------------------------------------- cached steps

@st.cache_data(show_spinner=False, max_entries=8)
def load_and_validate(raw: bytes, name: str):
    class _Upload:  # minimal file-like wrapper so read_csv applies the same checks
        def __init__(self, b, n):
            self._b, self.name = b, n

        def getvalue(self):
            return self._b

    frame, err = read_csv(_Upload(raw, name))
    if err:
        return None, err
    return validate(frame), None


@st.cache_data(show_spinner=False, max_entries=64)
def cached_forecast(key: str, product_id: str, horizon: int, _clean: pd.DataFrame):
    return forecast_product(_clean, product_id, horizon, n_folds=N_FOLDS)


def fmt_int(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"


# ----------------------------------------------------------------------------- sidebar: data

st.sidebar.title("📦 DemandPilot")
st.sidebar.caption("Forecast demand, then decide what to reorder.")

st.sidebar.subheader("1. Data")
source = st.sidebar.radio("Data source", list(SAMPLES) + [UPLOAD], label_visibility="collapsed")
with st.sidebar.expander("Expected CSV format"):
    st.dataframe(SCHEMA_TABLE, hide_index=True)
    st.download_button("Download a template CSV", template_csv(), "demandpilot_template.csv", "text/csv")

if source == UPLOAD:
    up = st.sidebar.file_uploader("CSV file", type=["csv"], help="Processed in memory only. Nothing is saved.")
    if up is None:
        st.title("DemandPilot")
        st.info("Upload a CSV in the sidebar, or pick one of the sample datasets to try the app.")
        st.dataframe(SCHEMA_TABLE, hide_index=True)
        st.stop()
    raw, name = up.getvalue(), up.name
else:
    raw, name = SAMPLES[source].read_bytes(), SAMPLES[source].name

with st.spinner("Checking the file..."):
    result, read_error = load_and_validate(raw, name)

st.title("DemandPilot")
st.caption(
    "Demand forecasting and inventory decision support for small retailers. "
    "Outputs are estimates built from your history and the assumptions shown. They are not guaranteed outcomes."
)

if read_error:
    st.error(read_error)
    st.stop()
if not result.ok:
    for e in result.errors:
        st.error(e)
    for w in result.warnings:
        st.warning(w)
    st.stop()

clean = result.data
key = data_hash(clean)
eligible = result.eligible_products
names = {p: product_name(clean, p) for p in eligible}

# ----------------------------------------------------------------------------- sidebar: forecast + inventory

st.sidebar.subheader("2. Forecast")
pid = st.sidebar.selectbox("Product", eligible, format_func=lambda p: names[p])
horizon = st.sidebar.segmented_control("Forecast horizon (days)", HORIZONS, default=14, required=True)

defaults = default_inputs(clean, pid)
st.sidebar.subheader("3. Inventory assumptions")
st.sidebar.caption("Change these and the recommendation updates straight away.")
stock = st.sidebar.number_input(
    "Current stock (units)" + (" · not in data" if defaults.stock_is_default else " · from data"),
    min_value=0, value=int(defaults.current_stock), step=10, key=f"stock_{key}_{pid}")
on_order = st.sidebar.number_input("Units already on order", min_value=0, value=0, step=10, key=f"onorder_{key}_{pid}")
lead = st.sidebar.number_input(
    "Supplier lead time (days)" + (" · default" if defaults.lead_time_is_default else " · from data"),
    min_value=1, max_value=180, value=int(defaults.lead_time_days), key=f"lead_{key}_{pid}")
service = st.sidebar.slider("Target service level", 0.80, 0.995, 0.95, 0.005, format="%.3f",
                            help="Chance of not running out during the lead time. Higher means more safety stock.")
review = st.sidebar.slider("Review period (days)", 1, 30, 7,
                           help="How many days each order should cover before you next decide to order.")
with st.sidebar.expander("Cost assumptions"):
    holding = st.slider("Holding cost (% of unit cost per year)", 5, 50, 25, 1) / 100
    excess_threshold = st.slider("Flag excess when stock exceeds this multiple of the target level", 1.2, 3.0, 1.5, 0.1)

inputs = InventoryInputs(
    current_stock=float(stock), lead_time_days=int(lead), service_level=float(service),
    review_period_days=int(review), on_order=float(on_order), unit_cost=defaults.unit_cost,
    unit_price=defaults.unit_price, holding_cost_rate=holding, excess_ratio_threshold=excess_threshold,
    lead_time_is_default=defaults.lead_time_is_default and lead == defaults.lead_time_days,
    stock_is_default=defaults.stock_is_default and stock == 0,
)

# ----------------------------------------------------------------------------- run

try:
    with st.spinner(f"Backtesting {len(MODEL_REGISTRY)} models on {names[pid]}..."):
        res = cached_forecast(key, pid, horizon, clean)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001 - never show a raw traceback to a business user
    st.error(f"This product could not be forecast ({type(exc).__name__}). Try another product or check its data.")
    st.stop()
horizon = res.horizon  # may be shorter than requested when history is short
rec = recommend_for(res, inputs)
facts = build_facts(res, rec, names[pid])
explanation = template_explanation(facts)

tab_data, tab_fc, tab_rec, tab_all, tab_dl, tab_method = st.tabs(
    ["Data check", "Forecast", "Recommendation", "All products", "Download", "Method and limits"]
)

# ----------------------------------------------------------------------------- tab: data

with tab_data:
    st.subheader("Data quality")
    c = st.columns(4)
    c[0].metric("Valid rows", f"{len(clean):,}", border=True)
    c[1].metric("Products", f"{clean['product_id'].nunique()}", help=f"{len(eligible)} with enough history", border=True)
    span = (clean["date"].max() - clean["date"].min()).days + 1
    c[2].metric("History", f"{span:,} days", help=f"{clean['date'].min():%d %b %Y} to {clean['date'].max():%d %b %Y}",
                border=True)
    c[3].metric("Units sold", f"{clean['units_sold'].sum():,.0f}", border=True)
    if result.warnings:
        st.warning("**Problems found and handled**\n\n" + "\n".join(f"- {w}" for w in result.warnings))
    else:
        st.success("No data problems found.")
    for n in result.notes:
        st.caption("• " + n)

    st.subheader("Products")
    ps = result.product_summary.copy()
    ps["eligible"] = ps["eligible"].map({True: "Yes", False: "No, too short"})
    ps = ps.drop(columns=["days_with_sales_rows"])
    st.dataframe(
        ps, hide_index=True,
        column_config={
            "product_id": "Product ID",
            "product_name": "Name",
            "category": "Category",
            "first_date": st.column_config.DateColumn("First date"),
            "last_date": st.column_config.DateColumn("Last date"),
            "span_days": st.column_config.NumberColumn("Days of history"),
            "avg_daily_units": st.column_config.NumberColumn("Avg units/day", format="%.1f"),
            "total_units": st.column_config.NumberColumn("Total units", format="%d"),
            "missing_days": st.column_config.NumberColumn("Days with no rows", help="Filled with zero sales"),
            "eligible": "Can forecast",
        },
    )
    st.plotly_chart(sales_overview_chart(clean))

# ----------------------------------------------------------------------------- tab: forecast

with tab_fc:
    s = res.summary.loc[res.selected_model]
    st.subheader(f"{names[pid]}: next {horizon} days")
    for n in res.notes:
        st.info(n)
    c = st.columns(4)
    c[0].metric("Forecast demand", f"{res.display_forecast['forecast'].sum():,.0f} units",
                help=f"Sum of the daily forecast over {horizon} days", border=True)
    c[1].metric("Selected model", model_label(res.selected_model, short=True), border=True)
    c[2].metric("Backtest WAPE", f"{s['WAPE']:.1%}" if np.isfinite(s["WAPE"]) else "n/a",
                help="Total absolute error divided by total actual demand, over the first "
                     f"{horizon} days of the backtest windows that were also used to choose the model",
                border=True)
    if res.selected_model != "seasonal_naive":
        imp = s["WAPE_vs_baseline"]
        c[3].metric("vs seasonal naive", f"{imp:+.1%}" if np.isfinite(imp) else "n/a",
                    help="Relative WAPE improvement over repeating last week", border=True)
    else:
        c[3].metric("vs seasonal naive", "baseline selected", border=True)
    st.plotly_chart(forecast_chart(res))
    st.caption(
        f"Shaded band: {INTERVAL_LEVEL:.0%} range built from the model's own backtest errors. "
        f"Out-of-sample coverage in backtesting: {res.coverage:.0%} of days."
        if not np.isnan(res.coverage) else "Shaded band: range built from the model's own backtest errors."
    )

    st.subheader("Model comparison")
    st.write(
        f"Each model was refitted and tested on {res.n_folds} rolling {res.backtest_horizon}-day windows using only "
        f"data available before each window. **{model_label(res.selected_model)}** was selected because "
        f"{res.selection_reason}. The table scores the first {horizon} days of each window. The model is chosen "
        "on the full window, so changing the horizon does not change the model or the order recommendation."
    )
    table = res.summary[["MAE", "RMSE", "WAPE", "sMAPE", "Bias", "WAPE_vs_baseline"]].copy()
    table.index = [model_label(m, short=True) + ("  ✓ selected" if m == res.selected_model else "") for m in table.index]
    st.dataframe(
        table,
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "WAPE": st.column_config.NumberColumn(format="percent"),
            "sMAPE": st.column_config.NumberColumn(format="%.3f"),
            "Bias": st.column_config.NumberColumn(format="%.2f", help="Mean forecast minus actual"),
            "WAPE_vs_baseline": st.column_config.NumberColumn("WAPE vs baseline", format="percent"),
        },
    )
    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(model_comparison_chart(res))
    with right:
        st.plotly_chart(backtest_chart(res))
    with st.expander("Per-window results"):
        fm = res.fold_metrics.copy()
        fm["model"] = fm["model"].map(lambda m: model_label(m, short=True))
        fm["cutoff"] = fm["cutoff"].dt.strftime("%Y-%m-%d")
        st.dataframe(fm, hide_index=True, column_config={"WAPE": st.column_config.NumberColumn(format="percent")})
    if res.feature_importance is not None:
        with st.expander("What drives the LightGBM forecast"):
            st.plotly_chart(feature_importance_chart(res.feature_importance))

# ----------------------------------------------------------------------------- tab: recommendation

with tab_rec:
    st.subheader(f"{rec.action}")
    c = st.columns(4)
    c[0].metric("Order now", f"{rec.order_quantity:,} units", border=True)
    c[1].metric("Reorder point", f"{rec.reorder_point:,.0f}", help="Lead-time demand + safety stock", border=True)
    c[2].metric("Safety stock", f"{rec.safety_stock:,.0f}", border=True)
    cover = "more than a year" if rec.days_of_cover > 365 else f"{rec.days_of_cover:,.1f} days"
    c[3].metric("Current stock lasts", cover, border=True)
    c = st.columns(4)
    c[0].metric("Stockout risk", risk_badge(rec.stockout_risk),
                help="High: position below lead-time demand. Medium: below the reorder point.", border=True)
    c[1].metric("Stockout chance", pct_text(rec.stockout_probability), border=True,
                help="Chance that demand during the lead time exceeds stock on hand plus on order")
    c[2].metric("Excess-stock risk", risk_badge(rec.excess_risk), border=True)
    if rec.excess_inventory_value is not None and rec.excess_units > 0:
        c[3].metric("Surplus value tied up", f"{rec.excess_inventory_value:,.0f}",
                    help=f"Surplus units x unit cost. Annual holding cost about {rec.annual_holding_cost_of_excess:,.0f}.",
                    border=True)
    elif rec.revenue_at_risk is not None:
        c[3].metric("Lost sales at risk", f"{rec.revenue_at_risk:,.0f}",
                    help="Expected units short during the lead time x unit price, in the data's currency", border=True)

    for w in rec.warnings:
        st.warning(w)

    left, right = st.columns([3, 2])
    with left:
        days = max(30, lead + review)
        st.plotly_chart(inventory_chart(inputs.current_stock, res.forecast, rec.reorder_point,
                                        rec.safety_stock, lead, days))
        if inputs.on_order > 0:
            st.caption("The chart shows stock on hand only. Units on order count towards the reorder decision.")
    with right:
        st.markdown("**How the numbers are built**")
        st.markdown(
            f"""
| Step | Value |
|---|---|
| Demand over the {lead}-day lead time | {rec.lead_time_demand:,.0f} |
| Forecast error std over the lead time (from backtest) | {rec.lead_time_sigma:,.1f} |
| z for {service:.1%} service | {rec.z:.2f} |
| Safety stock = z x error std | {rec.safety_stock:,.0f} |
| Reorder point = demand + safety stock | {rec.reorder_point:,.0f} |
| Order-up-to level = demand + z x error std over lead time + {review}-day review | {rec.order_up_to_level:,.0f} |
| Stock on hand + on order | {rec.inventory_position:,.0f} |
| Order = max(0, target - position) if at or below reorder point | {rec.order_quantity:,} |
"""
        )

    st.subheader("Explanation")
    use_llm = False
    if llm_available():
        use_llm = st.toggle("Rephrase with a language model (numbers are checked against the template)")
    if use_llm:
        text, src = llm_rephrase(explanation, action=rec.action)
        st.markdown(text)
        st.caption(f"Source: {src}.")
    else:
        st.markdown(explanation)
        st.caption("Generated from computed values by a fixed template. No language model was used.")

    with st.expander("All assumptions used"):
        st.dataframe(pd.DataFrame({"Assumption": list(rec.assumptions), "Value": [str(v) for v in rec.assumptions.values()]}),
                     hide_index=True)

# ----------------------------------------------------------------------------- tab: all products

with tab_all:
    st.subheader("Risk across all products")
    st.write(
        f"Runs the same backtest and recommendation for every product, using the stock and lead time in the data, "
        f"a {service:.1%} service level and a {review}-day review period."
    )
    run_key = f"portfolio_{key}_{horizon}"
    if st.button("Analyse all products", type="primary") or st.session_state.get(run_key):
        st.session_state[run_key] = True
        rows = []
        bar = st.progress(0.0, text="Forecasting...")
        for i, p in enumerate(eligible):
            try:
                r = cached_forecast(key, p, horizon, clean)
            except Exception:  # noqa: BLE001
                rows.append({"Product": names[p], "Action": "Could not forecast (check history)", "_sort": (1, 1)})
                continue
            base = default_inputs(clean, p, service_level=service, review_period_days=review)
            base.holding_cost_rate, base.excess_ratio_threshold = holding, excess_threshold
            rr = recommend_for(r, base)
            rows.append({
                "Product": names[p],
                "Action": rr.action,
                "Order qty": rr.order_quantity,
                "Stock": base.current_stock if not base.stock_is_default else np.nan,
                "Days of cover": min(rr.days_of_cover, 999),
                "Stockout risk": risk_badge(rr.stockout_risk),
                "Stockout chance": rr.stockout_probability * 100,
                "Excess risk": risk_badge(rr.excess_risk),
                "Surplus value": (rr.excess_inventory_value if rr.excess_risk != "Low" and rr.excess_inventory_value
                                  is not None else np.nan),
                "Model": model_label(r.selected_model, short=True),
                "WAPE": r.summary.loc[r.selected_model, "WAPE"] * 100,
                "Note": "" if r.horizon == horizon else f"{r.horizon}-day horizon (short history)",
                "_sort": (-{"High": 2, "Medium": 1, "Low": 0}[rr.stockout_risk],
                          -{"High": 2, "Medium": 1, "Low": 0}[rr.excess_risk]),
            })
            bar.progress((i + 1) / len(eligible), text=f"Forecasting {names[p]}")
        bar.empty()
        port = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")
        st.dataframe(
            port, hide_index=True,
            column_config={
                "Days of cover": st.column_config.NumberColumn(format="%.1f"),
                "Stockout chance": st.column_config.NumberColumn(format="%.0f%%"),
                "Surplus value": st.column_config.NumberColumn(format="%.0f"),
                "WAPE": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.download_button("Download this table (CSV)", port.to_csv(index=False), f"demandpilot_portfolio_{horizon}d.csv",
                           "text/csv")
    else:
        st.caption(f"Takes a few seconds per product ({len(eligible)} products).")

# ----------------------------------------------------------------------------- tab: download

with tab_dl:
    st.subheader("Download results")
    st.write(f"For **{names[pid]}**, {horizon}-day horizon, with the assumptions currently set in the sidebar.")
    c = st.columns(4)
    c[0].download_button("Forecast (CSV)", forecast_table(res).to_csv(index=False),
                         f"forecast_{pid}_{horizon}d.csv", "text/csv", type="primary")
    c[1].download_button("Recommendation (CSV)", recommendation_table(res, rec).to_csv(index=False),
                         f"recommendation_{pid}.csv", "text/csv")
    c[2].download_button("Report (HTML)", html_report(res, rec, explanation, names[pid]),
                         f"demandpilot_report_{pid}.html", "text/html")
    c[3].download_button("Backtest detail (CSV)", res.backtest.to_csv(index=False),
                         f"backtest_{pid}_{horizon}d.csv", "text/csv")
    st.caption("The HTML report opens in any browser and prints cleanly to PDF.")
    st.dataframe(forecast_table(res), hide_index=True)

# ----------------------------------------------------------------------------- tab: method

with tab_method:
    st.subheader("How DemandPilot works")
    st.markdown(
        f"""
**1. Validation.** Required columns are `date`, `product_id` and `units_sold`. Invalid dates, negative or
non-numeric units and exact duplicate rows are removed and reported. Several rows for the same product and day are
summed. Days with no rows, including days after a product's last row, are treated as zero sales. A product needs at
least 8 weeks of history and 28 days with sales rows. Longer horizons need more history.

**2. Candidate models.** {", ".join(model_label(m) for m in MODEL_REGISTRY)}. The seasonal naive
forecast is the baseline: a more complex model has to beat it to be worth using.

**3. Time-aware backtest.** For this product the history is cut at {res.n_folds} successive points. At each cut
every model is refitted on data before the cut and forecasts the next {res.backtest_horizon} days. Future promotions
are unknown to the model and price is held at its last regular value, exactly as in real use. Random train/test
splits are never used because they let a model learn from the future.

**4. Selection.** The model with the lowest WAPE over the full backtest window wins, unless a simpler model is
within {PARSIMONY_TOLERANCE:.0%} (relative) of it, in which case the simpler one is chosen. The accuracy shown in the
app comes from the same windows used for this choice, so it is slightly optimistic. The README reports results on a
separate holdout period.

**5. Uncertainty.** The shaded range comes from the selected model's backtest errors, scaled by the demand level at
each cut. Coverage is checked out of sample: each window's range is built only from earlier windows' errors.

**6. Inventory decision.** Safety stock = z x the standard deviation of the total forecast error over the lead
time, measured from the model's summed backtest errors. Reorder point = lead-time demand + safety stock. When stock
plus on-order is at or below the reorder point, order enough to reach the order-up-to level (demand over lead time
plus review period, with its own safety buffer). Orders are never negative.

**7. Explanation.** A fixed template writes the explanation from computed values. An optional language model can
rephrase it, but its answer is rejected unless it keeps exactly the template's numbers, in the same order, and the
same recommended action.
"""
    )
    st.subheader("Limitations")
    st.markdown(
        """
- The sample data is synthetic. Results on it say nothing about accuracy on a real business.
- Sales are treated as demand. If a product was out of stock, true demand was higher than recorded sales.
- Lead-time error is measured from a limited number of backtest windows, so it is itself uncertain.
- Risk figures use a normal approximation, which is rough for slow or intermittent sellers.
- Lead time is treated as fixed. Supplier delays are not modelled.
- Future promotions, holidays and price changes are not planned into the forecast.
- One model per product. Products do not share information, so new products with little history are not covered.
"""
    )
