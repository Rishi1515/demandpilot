"""Schema and data-quality checks for uploaded sales files.

The validator never silently fixes data. Every row it drops or value it
changes is counted and reported back to the user as an error, warning or
note, so the person looking at the forecast knows what went into it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "product_id", "units_sold"]
OPTIONAL_NUMERIC = ["current_stock", "lead_time_days", "unit_price", "unit_cost"]
OPTIONAL_COLUMNS = OPTIONAL_NUMERIC + ["promotion", "category", "product_name", "transaction_id"]

MAX_FILE_MB = 25
MIN_HISTORY_DAYS = 56  # 8 weeks: enough for weekly seasonality and a backtest

SCHEMA_TABLE = pd.DataFrame(
    [
        ("date", "Yes", "Sales date in a parseable format, e.g. 2026-01-31."),
        ("product_id", "Yes", "Stable identifier for a product or SKU."),
        ("units_sold", "Yes", "Non-negative quantity sold on that date."),
        ("current_stock", "Optional", "Latest on-hand inventory. The most recent non-empty value per product is used."),
        ("lead_time_days", "Optional", "Supplier lead time in days. A labelled default is used if missing."),
        ("unit_price", "Optional", "Selling price, used for revenue-at-risk estimates."),
        ("unit_cost", "Optional", "Unit cost, used for inventory value estimates."),
        ("promotion", "Optional", "1/0, true/false or yes/no promotion flag."),
        ("category", "Optional", "Product group for filtering and portfolio views."),
        ("product_name", "Optional", "Readable product name shown in the interface."),
        ("transaction_id", "Optional", "Keeps identical transactions apart. Without it, identical rows are treated as duplicates."),
    ],
    columns=["column", "required", "description"],
)

_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f", ""}


@dataclass
class ValidationResult:
    data: pd.DataFrame | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    product_summary: pd.DataFrame | None = None

    @property
    def ok(self) -> bool:
        return not self.errors and self.data is not None and len(self.data) > 0

    @property
    def eligible_products(self) -> list[str]:
        if self.product_summary is None:
            return []
        s = self.product_summary
        return s.loc[s["eligible"], "product_id"].tolist()


def template_csv() -> str:
    """A tiny example file users can download to see the expected layout."""
    rows = [
        "date,product_id,units_sold,current_stock,lead_time_days,unit_price,unit_cost,promotion,category",
        "2026-01-01,SKU-001,42,,7,2.50,1.60,0,Snacks",
        "2026-01-02,SKU-001,38,,7,2.50,1.60,0,Snacks",
        "2026-01-03,SKU-001,55,310,7,2.20,1.60,1,Snacks",
    ]
    return "\n".join(rows) + "\n"


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def _parse_promotion(series: pd.Series) -> tuple[pd.Series, int]:
    """Accept 1/0, true/false, yes/no or numeric intensities. Returns (values, n_unparsed)."""
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    out = numeric.copy()
    out[numeric.isna() & text.isin(_TRUE)] = 1.0
    out[numeric.isna() & text.isin(_FALSE | {"nan", "none"})] = 0.0
    n_bad = int(out.isna().sum())
    return out.fillna(0.0).clip(lower=0.0), n_bad


def read_csv(source, max_mb: float = MAX_FILE_MB) -> tuple[pd.DataFrame | None, str | None]:
    """Read an uploaded file or path. Returns (frame, error_message)."""
    try:
        if hasattr(source, "getvalue"):
            raw = source.getvalue()
            if len(raw) > max_mb * 1024 * 1024:
                return None, f"File is larger than {max_mb} MB. Please upload a smaller extract."
            name = getattr(source, "name", "upload.csv")
            if not str(name).lower().endswith(".csv"):
                return None, "Only .csv files are accepted."
            if isinstance(raw, str):
                return pd.read_csv(io.StringIO(raw)), None
            try:  # Excel on Windows often saves CSVs as cp1252 rather than UTF-8
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("cp1252", errors="replace")
            return pd.read_csv(io.StringIO(text)), None
        return pd.read_csv(source), None
    except pd.errors.EmptyDataError:
        return None, "The file is empty."
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        return None, f"Could not read the file as CSV: {exc}"


def validate(df: pd.DataFrame, min_history_days: int = MIN_HISTORY_DAYS) -> ValidationResult:
    """Check schema and quality, and return a cleaned row-level frame."""
    res = ValidationResult(data=None)
    if df is None or df.empty:
        res.errors.append("The file contains no rows.")
        return res

    df = _normalise_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        res.errors.append(
            "Missing required column(s): " + ", ".join(missing)
            + ". Expected at least: " + ", ".join(REQUIRED_COLUMNS) + "."
        )
        return res

    n_start = len(df)
    res.notes.append(f"Read {n_start:,} rows and {df.shape[1]} columns.")
    ignored = [c for c in df.columns if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if ignored:
        res.notes.append("Ignored unrecognised column(s): " + ", ".join(ignored) + ".")

    # Exact duplicate rows are almost always export mistakes.
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        df = df.drop_duplicates()
        res.warnings.append(
            f"Removed {n_dupes:,} exact duplicate row(s). If your file lists individual transactions, add a "
            "transaction_id column so identical sales are kept."
        )

    # Required fields
    df["product_id"] = df["product_id"].astype("string").str.strip()
    blank_pid = df["product_id"].isna() | (df["product_id"] == "")
    if blank_pid.any():
        res.warnings.append(f"Dropped {int(blank_pid.sum()):,} row(s) with a blank product_id.")
        df = df[~blank_pid].copy()

    parsed_dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    bad_dates = parsed_dates.isna()
    if bad_dates.any():
        examples = ", ".join(repr(str(v)) for v in df.loc[bad_dates, "date"].astype(str).unique()[:3])
        res.warnings.append(f"Dropped {int(bad_dates.sum()):,} row(s) with an invalid date (e.g. {examples}).")
    df = df.assign(date=parsed_dates.dt.normalize())[~bad_dates].copy()

    units = pd.to_numeric(df["units_sold"], errors="coerce")
    missing_units = df["units_sold"].isna()
    non_numeric = units.isna() & ~missing_units
    if missing_units.any():
        res.warnings.append(f"Dropped {int(missing_units.sum()):,} row(s) with an empty units_sold value.")
    if non_numeric.any():
        res.warnings.append(f"Dropped {int(non_numeric.sum()):,} row(s) where units_sold is not a number.")
    negative = units < 0
    if negative.any():
        res.warnings.append(
            f"Dropped {int(negative.sum()):,} row(s) with negative units_sold. "
            "Returns should be handled separately from demand."
        )
    keep = units.notna() & ~negative
    df = df.assign(units_sold=units)[keep].copy()

    # Optional numeric fields: keep the row, blank out impossible values.
    for col in OPTIONAL_NUMERIC:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            n_bad = int(((vals < 0) | (vals.isna() & df[col].notna())).sum())
            if n_bad:
                res.warnings.append(f"Ignored {n_bad:,} invalid or negative value(s) in {col}.")
            df[col] = vals.where(vals >= 0)
    if "promotion" in df.columns:
        df["promotion"], n_bad = _parse_promotion(df["promotion"])
        if n_bad:
            res.warnings.append(f"Treated {n_bad:,} unreadable promotion value(s) as 0 (no promotion).")

    if df.empty:
        res.errors.append("No valid rows remain after cleaning. Check the date and units_sold columns.")
        return res

    future = df["date"] > pd.Timestamp.today().normalize()
    if future.any():
        res.warnings.append(f"{int(future.sum()):,} row(s) are dated in the future. They were kept; please check them.")

    # Several rows for the same product and day are treated as transactions and summed later.
    multi = int(df.duplicated(subset=["product_id", "date"]).sum())
    if multi:
        res.notes.append(
            f"{multi:,} row(s) share a product and date with another row. They are treated as separate "
            "transactions and summed into daily totals."
        )

    res.product_summary = summarise_products(df, min_history_days)
    short = res.product_summary.loc[~res.product_summary["eligible"], "product_id"].tolist()
    if short:
        res.warnings.append(
            f"{len(short)} product(s) have less than {min_history_days} days of history (or fewer than 28 days "
            f"with sales rows) and cannot be forecast reliably: {', '.join(short[:10])}"
            f"{'...' if len(short) > 10 else ''}."
        )
    if not res.eligible_products:
        res.errors.append(
            f"No product has at least {min_history_days} days of history. More history is needed for a "
            "time-aware backtest."
        )

    res.notes.append(f"{len(df):,} valid rows kept ({n_start - len(df):,} removed).")
    res.data = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return res


def summarise_products(df: pd.DataFrame, min_history_days: int = MIN_HISTORY_DAYS) -> pd.DataFrame:
    g = df.groupby("product_id")
    summary = pd.DataFrame(
        {
            "first_date": g["date"].min(),
            "last_date": g["date"].max(),
            "days_with_sales_rows": g["date"].nunique(),
            "total_units": g["units_sold"].sum(),
        }
    )
    summary["span_days"] = (summary["last_date"] - summary["first_date"]).dt.days + 1
    summary["missing_days"] = summary["span_days"] - summary["days_with_sales_rows"]
    summary["avg_daily_units"] = summary["total_units"] / summary["span_days"]
    summary["eligible"] = (summary["span_days"] >= min_history_days) & (summary["days_with_sales_rows"] >= 28)
    if "product_name" in df.columns:
        summary.insert(0, "product_name", g["product_name"].last())
    if "category" in df.columns:
        summary.insert(1 if "product_name" in df.columns else 0, "category", g["category"].last())
    return summary.reset_index()


def latest_value(df: pd.DataFrame, product_id: str, column: str) -> float | None:
    """Most recent non-empty value of an optional column for one product."""
    if column not in df.columns:
        return None
    s = df.loc[df["product_id"] == product_id].sort_values("date")[column].dropna()
    return None if s.empty else float(s.iloc[-1])


def to_daily(df: pd.DataFrame, product_id: str, end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, int]:
    """Daily series for one product with missing dates filled.

    Missing calendar days are filled with zero units. This assumes a day
    without a row means no sales, which is how most point-of-sale exports
    behave. The series runs to `end` (normally the last date in the whole
    file), so a product that stopped selling is not forecast from a stale
    date. The number of filled days is returned so it can be reported.
    """
    sub = df.loc[df["product_id"] == product_id]
    agg = {"units_sold": "sum"}
    if "promotion" in sub.columns:
        agg["promotion"] = "max"
    if "unit_price" in sub.columns:
        agg["unit_price"] = "mean"
    daily = sub.groupby("date").agg(agg)
    last = max(daily.index.max(), end) if end is not None else daily.index.max()
    full_index = pd.date_range(daily.index.min(), last, freq="D")
    n_filled = len(full_index) - len(daily)
    daily = daily.reindex(full_index)
    daily.index.name = "date"
    daily["units_sold"] = daily["units_sold"].fillna(0.0).astype(float)
    if "promotion" in daily.columns:
        daily["promotion"] = daily["promotion"].fillna(0.0)
    if "unit_price" in daily.columns:
        # Forward fill only: back-filling would copy later prices into earlier days.
        daily["unit_price"] = daily["unit_price"].ffill()
        if daily["unit_price"].isna().all():
            daily = daily.drop(columns="unit_price")
    return daily, n_filled


def data_hash(df: pd.DataFrame) -> str:
    """Stable fingerprint of a cleaned frame, used as a cache key."""
    return str(int(pd.util.hash_pandas_object(df, index=False).sum() % (10**12)))


def history_is_constant(y: np.ndarray) -> bool:
    return bool(np.nanstd(y) == 0)
