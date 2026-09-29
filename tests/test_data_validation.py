import io

import numpy as np
import pandas as pd

from src.data_validation import MIN_HISTORY_DAYS, read_csv, to_daily, validate


def make_frame(days=90, pid="A", start="2025-01-01", **extra):
    dates = pd.date_range(start, periods=days, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "product_id": pid, "units_sold": np.arange(days) % 7})
    for k, v in extra.items():
        df[k] = v
    return df


def test_clean_file_passes():
    res = validate(make_frame())
    assert res.ok and not res.errors and not res.warnings
    assert res.eligible_products == ["A"]


def test_missing_required_column_is_an_error():
    res = validate(make_frame().drop(columns="units_sold"))
    assert not res.ok
    assert "units_sold" in res.errors[0]


def test_column_names_are_normalised():
    df = make_frame().rename(columns={"product_id": "Product ID", "units_sold": " Units Sold "})
    assert validate(df).ok


def test_invalid_dates_negative_and_non_numeric_units_are_dropped_and_reported():
    df = make_frame().astype({"units_sold": object})
    df.loc[0, "date"] = "not a date"
    df.loc[1, "units_sold"] = -3
    df.loc[2, "units_sold"] = "abc"
    df.loc[3, "units_sold"] = None
    res = validate(df)
    assert res.ok
    text = " ".join(res.warnings)
    assert "invalid date" in text
    assert "negative units_sold" in text
    assert "not a number" in text
    assert "empty units_sold" in text
    assert len(res.data) == len(df) - 4


def test_exact_duplicates_removed_and_same_day_rows_summed():
    df = make_frame(days=60)
    dup = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    res = validate(dup)
    assert any("duplicate" in w for w in res.warnings)
    extra = df.iloc[[10]].copy()
    extra["units_sold"] = 100
    res2 = validate(pd.concat([df, extra], ignore_index=True))
    daily, _ = to_daily(res2.data, "A")
    assert daily.iloc[10]["units_sold"] == df.iloc[10]["units_sold"] + 100


def test_insufficient_history_is_flagged():
    df = pd.concat([make_frame(days=90, pid="LONG"), make_frame(days=20, pid="SHORT")])
    res = validate(df)
    assert res.eligible_products == ["LONG"]
    assert any("SHORT" in w for w in res.warnings)
    only_short = validate(make_frame(days=MIN_HISTORY_DAYS - 1))
    assert not only_short.ok


def test_missing_dates_filled_with_zero_and_counted():
    df = make_frame(days=70)
    df = df.drop(index=[20, 21, 22])
    res = validate(df)
    daily, filled = to_daily(res.data, "A")
    assert filled == 3
    assert len(daily) == 70
    assert daily.iloc[20:23]["units_sold"].eq(0).all()


def test_optional_fields_parsed_and_bad_values_ignored():
    df = make_frame(promotion=["yes", "no", "1", "0", "TRUE"] * 18, lead_time_days=7, current_stock=np.nan)
    df.loc[0, "lead_time_days"] = -2
    df.loc[89, "current_stock"] = 55
    res = validate(df)
    assert res.ok
    assert set(res.data["promotion"].unique()) <= {0.0, 1.0}
    assert any("lead_time_days" in w for w in res.warnings)


def test_read_csv_rejects_non_csv_and_empty():
    class Up:
        def __init__(self, b, name):
            self.b, self.name = b, name

        def getvalue(self):
            return self.b

    _, err = read_csv(Up(b"a,b\n1,2\n", "data.xlsx"))
    assert "csv" in err.lower()
    _, err = read_csv(Up(b"", "data.csv"))
    assert err
    frame, err = read_csv(io.StringIO("date,product_id,units_sold\n2025-01-01,A,3\n"))
    assert err is None and len(frame) == 1
