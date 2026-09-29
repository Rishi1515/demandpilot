"""Plotly charts used by the app. Colors come from one small, colorblind-checked palette."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .backtesting import ForecastResult
from .models import model_label

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e6e5e0"
BLUE = "#2a78d6"
BLUE_BAND = "rgba(42,120,214,0.18)"
ORANGE = "#eb6834"
NEUTRAL_BAR = "#c9c8c1"
STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
STATUS_ICON = {"Low": "●", "Medium": "▲", "High": "■"}


def _layout(fig: go.Figure, title: str, y_title: str = "Units per day", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=INK)),
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_2, size=13),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=True, zerolinecolor=GRID, rangemode="tozero")
    return fig


def forecast_chart(res: ForecastResult, history_days: int = 120) -> go.Figure:
    hist = res.history["units_sold"].tail(history_days)
    fc = res.display_forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fc.index) + list(fc.index[::-1]),
        y=list(fc["upper"]) + list(fc["lower"][::-1]),
        fill="toself", fillcolor=BLUE_BAND, line=dict(width=0), hoverinfo="skip",
        name=f"{res.interval_level:.0%} range",
    ))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual sales",
                             line=dict(color=INK_2, width=1.5),
                             hovertemplate="%{y:,.0f} units<extra>Actual</extra>"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="Forecast",
                             line=dict(color=BLUE, width=2.5),
                             customdata=np.stack([fc["lower"], fc["upper"]], axis=1),
                             hovertemplate="%{y:,.1f} units (range %{customdata[0]:,.0f} to %{customdata[1]:,.0f})<extra>Forecast</extra>"))
    start = fc.index.min()
    fig.add_vline(x=start, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=start, y=0.99, yref="paper", text=" Forecast starts", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=MUTED, size=12))
    return _layout(fig, f"Daily demand: last {len(hist)} days and next {res.horizon} days")


def backtest_chart(res: ForecastResult) -> go.Figure:
    """Actual versus the selected model's backtest forecasts, fold by fold."""
    bt = res.backtest[res.backtest["model"] == res.selected_model]
    start = bt["date"].min() - pd.Timedelta(days=28)
    hist = res.history["units_sold"].loc[start: bt["date"].max()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual sales",
                             line=dict(color=INK_2, width=1.5),
                             hovertemplate="%{y:,.0f}<extra>Actual</extra>"))
    for i, (fold, g) in enumerate(bt.groupby("fold")):
        fig.add_trace(go.Scatter(x=g["date"], y=g["forecast"], name="Backtest forecast",
                                 legendgroup="bt", showlegend=i == 0,
                                 line=dict(color=BLUE, width=2),
                                 hovertemplate=f"%{{y:,.1f}}<extra>Fold {fold} forecast</extra>"))
        fig.add_vline(x=g["date"].min(), line=dict(color=GRID, width=1))
    return _layout(fig, f"Backtest: {model_label(res.selected_model, short=True)} vs actual, {res.n_folds} windows", height=360)


def model_comparison_chart(res: ForecastResult) -> go.Figure:
    s = res.summary.sort_values("WAPE")
    colors = [BLUE if m == res.selected_model else NEUTRAL_BAR for m in s.index]
    labels = [model_label(m, short=True) + (" ✓" if m == res.selected_model else "") for m in s.index]
    fig = go.Figure(go.Bar(
        x=s["WAPE"] * 100, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{v:.1f}%" for v in s["WAPE"] * 100],
        textposition="outside", textfont=dict(color=INK_2), cliponaxis=False,
        hovertemplate="%{y}<br>WAPE %{x:.1f}%<extra></extra>",
    ))
    fig = _layout(fig, "Backtest WAPE by model (✓ selected)", y_title="", height=320)
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(title="WAPE (%)", showgrid=True, gridcolor=GRID, range=[0, float(s["WAPE"].max() * 100 * 1.3)])
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def inventory_chart(stock: float, forecast: pd.DataFrame, reorder_point: float, safety_stock: float,
                    lead_time: int, days: int) -> go.Figure:
    """Expected stock over time if no new order is placed, against the reorder point and safety stock."""
    fc = forecast.head(days)
    projected = stock - np.cumsum(fc["forecast"].to_numpy())
    x = [fc.index.min() - pd.Timedelta(days=1)] + list(fc.index)
    y = np.clip([stock] + list(projected), 0, None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name="Stock on hand, nothing new ordered", line=dict(color=BLUE, width=2.5),
                             hovertemplate="%{y:,.0f} units<extra>Expected stock</extra>"))
    out = np.nonzero(projected <= 0)[0]
    if len(out):
        day = fc.index[out[0]]
        fig.add_trace(go.Scatter(x=[day], y=[0], mode="markers", marker=dict(size=10, color=STATUS["High"], symbol="square"),
                                 name="Expected to run out", hovertemplate="Runs out around %{x|%d %b}<extra></extra>"))
        fig.add_annotation(x=day, y=0, text=f"Runs out around {day:%d %b}", showarrow=True, arrowhead=0,
                           ax=40, ay=-30, font=dict(color=INK_2, size=12))
    fig.add_hline(y=reorder_point, line=dict(color=ORANGE, width=2, dash="dash"))
    fig.add_annotation(x=1, xref="paper", y=reorder_point, text=f"Reorder point {reorder_point:,.0f}",
                       showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=INK_2, size=12))
    fig.add_hline(y=safety_stock, line=dict(color=MUTED, width=1.5, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=safety_stock, text=f"Safety stock {safety_stock:,.0f}",
                       showarrow=False, xanchor="right", yanchor="top", font=dict(color=INK_2, size=12))
    fig.add_hline(y=0, line=dict(color=INK_2, width=1))
    if lead_time <= days:
        arrival = fc.index.min() + pd.Timedelta(days=lead_time - 1)
        fig.add_vline(x=arrival, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=arrival, y=1, yref="paper", text="An order placed today arrives",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=MUTED, size=12))
    fig = _layout(fig, "Projected stock if nothing is ordered", y_title="Units on hand", height=380)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0), margin=dict(b=40))
    fig.update_yaxes(range=[0, max(stock, reorder_point) * 1.15])
    return fig


def sales_overview_chart(df: pd.DataFrame, product_id: str | None = None) -> go.Figure:
    """Weekly total units, all products or one product."""
    sub = df if product_id is None else df[df["product_id"] == product_id]
    weekly = sub.set_index("date")["units_sold"].resample("W-MON", label="left", closed="left").sum()
    fig = go.Figure(go.Bar(x=weekly.index, y=weekly.values, marker=dict(color=BLUE, cornerradius=2),
                           hovertemplate="Week of %{x|%d %b %Y}<br>%{y:,.0f} units<extra></extra>"))
    title = "Weekly units sold, all products" if product_id is None else f"Weekly units sold, {product_id}"
    fig = _layout(fig, title, y_title="Units per week", height=320)
    fig.update_layout(hovermode="closest", bargap=0.15)
    return fig


def feature_importance_chart(importance: pd.Series, top: int = 10) -> go.Figure:
    imp = importance.head(top)[::-1]
    fig = go.Figure(go.Bar(x=imp.values * 100, y=imp.index, orientation="h",
                           marker=dict(color=BLUE, cornerradius=4),
                           hovertemplate="%{y}: %{x:.1f}% of total gain<extra></extra>"))
    fig = _layout(fig, "What the LightGBM model relies on (share of gain)", y_title="", height=340)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="% of total gain", showgrid=True, gridcolor=GRID)
    return fig


def risk_badge(level: str) -> str:
    return f"{STATUS_ICON.get(level, '')} {level}"
