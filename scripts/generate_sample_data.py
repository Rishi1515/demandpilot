"""Generate the synthetic sales datasets shipped with DemandPilot.

Everything here is made up. The numbers are shaped to look like a small
retailer's daily sales (weekly rhythm, yearly seasonality, promotions,
trends, one slow mover and one festive product) so the app can be tested
without downloading any external data.

Usage:
    python scripts/generate_sample_data.py

Writes:
    data/sample_sales.csv              clean demo dataset
    data/sample_sales_with_issues.csv  same data with deliberate problems,
                                       used to demo the validator
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = "2024-07-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parents[1]

# id, name, category, base units/day, yearly trend, weekend lift,
# summer amplitude, festive amplitude, promo lift, base price, lead time,
# target days of stock cover at the end of the data (sets the demo scenario)
PRODUCTS = [
    ("P001", "Bottled Water 1L", "Beverages", 120, 0.08, 0.35, 0.40, 0.00, 0.45, 0.60, 5, 3),
    ("P002", "Ground Coffee 250g", "Beverages", 35, 0.12, 0.10, 0.00, 0.05, 0.30, 6.50, 10, 20),
    ("P003", "Instant Noodles 5-pack", "Pantry", 80, 0.00, 0.20, -0.05, 0.05, 0.70, 2.20, 7, 9),
    ("P004", "Basmati Rice 5kg", "Pantry", 18, 0.05, 0.15, 0.00, 0.10, 0.25, 11.00, 14, 35),
    ("P005", "Sunscreen SPF50", "Personal Care", 12, 0.05, 0.25, 0.90, 0.00, 0.35, 9.00, 12, 90),
    ("P006", "Hand Wash 500ml", "Personal Care", 25, -0.25, 0.05, 0.00, 0.00, 0.30, 3.20, 7, 60),
    ("P007", "Premium Olive Oil 1L", "Pantry", 3, 0.00, 0.10, 0.00, 0.10, 0.40, 14.00, 21, 40),
    ("P008", "Festive Gift Hamper", "Seasonal", 4, 0.10, 0.20, 0.00, 2.50, 0.50, 25.00, 10, 10),
]


def _promo_calendar(rng: np.random.Generator, n_days: int, share: float = 0.08) -> np.ndarray:
    """Random promotion blocks of 3 to 7 days covering roughly `share` of days."""
    promo = np.zeros(n_days, dtype=int)
    while promo.mean() < share:
        start = rng.integers(0, n_days - 7)
        promo[start : start + rng.integers(3, 8)] = 1
    return promo


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    t_years = np.arange(n) / 365.25
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    dom = dates.day.to_numpy()

    # Mid-year peak for "summer" items; Oct to Dec peak for festive items.
    summer = np.cos(2 * np.pi * (doy - 172) / 365.25)
    festive = np.exp(-0.5 * ((doy - 305) / 18) ** 2) + 0.6 * np.exp(-0.5 * ((doy - 355) / 7) ** 2)

    frames = []
    for (pid, name, cat, base, trend, wkend, summer_amp, fest_amp,
         promo_lift, price, lead, cover_days) in PRODUCTS:
        promo = _promo_calendar(rng, n)
        weekend = np.where(dow >= 5, 1 + wkend, 1.0)
        payday = np.where(dom <= 3, 1.12, 1.0) if pid == "P004" else 1.0
        level = base * (1 + trend * t_years)
        mu = (
            level
            * weekend
            * payday
            * np.clip(1 + summer_amp * summer, 0.05, None)
            * (1 + fest_amp * festive)
            * (1 + promo_lift * promo)
        )
        # Negative binomial noise: over-dispersed like real retail counts.
        dispersion = 20.0 if base >= 50 else (8.0 if base >= 10 else 1.5)
        p = dispersion / (dispersion + mu)
        units = rng.negative_binomial(dispersion, p)

        unit_price = np.round(price * np.where(promo == 1, 0.85, 1.0) * (1 + 0.03 * t_years), 2)
        unit_cost = np.round(price * 0.62, 2)

        df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "product_id": pid,
                "product_name": name,
                "category": cat,
                "units_sold": units,
                "unit_price": unit_price,
                "unit_cost": unit_cost,
                "promotion": promo,
                "lead_time_days": lead,
                "current_stock": np.nan,
            }
        )
        # current_stock is a snapshot: only the latest row carries it.
        recent_mean = units[-28:].mean()
        df.loc[df.index[-1], "current_stock"] = int(round(recent_mean * cover_days))
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["current_stock"] = out["current_stock"].astype("Int64")
    return out


def add_issues(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Inject problems a real export might have, so the validator can be demoed."""
    rng = np.random.default_rng(seed + 1)
    bad = df.copy()
    bad["date"] = bad["date"].astype(object)
    bad["units_sold"] = bad["units_sold"].astype(object)
    idx = rng.choice(bad.index, size=40, replace=False)
    bad.loc[idx[:10], "date"] = "not-a-date"
    bad.loc[idx[10:18], "units_sold"] = -5
    bad.loc[idx[18:24], "units_sold"] = np.nan
    bad.loc[idx[24:28], "units_sold"] = "twelve"
    dupes = bad.loc[idx[28:40]]
    # A product with only 3 weeks of history, too short to forecast.
    short = df[df["product_id"] == "P002"].tail(21).copy()
    short["product_id"] = "P999"
    short["product_name"] = "New Launch Snack"
    out = pd.concat([bad, dupes, short], ignore_index=True)
    # Drop some whole days for one product to show missing-date handling.
    gap = (out["product_id"] == "P003") & out["date"].astype(str).between("2025-03-01", "2025-03-06")
    return out[~gap].reset_index(drop=True)


def main() -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(data_dir / "sample_sales.csv", index=False)
    add_issues(df).to_csv(data_dir / "sample_sales_with_issues.csv", index=False)
    print(f"Wrote {len(df):,} rows for {df['product_id'].nunique()} products to data/")


if __name__ == "__main__":
    main()
