"""Reproducible evaluation with an untouched holdout period.

The app picks a model per product using rolling backtests. Reporting the
winner's score on the same windows used to pick it would be slightly
optimistic, so this script separates the two:

    [ ...... selection backtest (6 x 30 days) ...... ][ holdout (4 x 30 days) ]

1. Model selection uses only data before the holdout.
2. The chosen model, and every other model for comparison, are then
   evaluated on the holdout windows, refitting at each holdout cutoff on
   all data before it.

Usage:
    python scripts/evaluate.py                       # sample data, 30-day windows, scored at 7, 14, 30 days
    python scripts/evaluate.py --data my.csv --save-models

Writes reports/evaluation_summary.md, reports/evaluation_by_product.csv and
reports/evaluation_detail.csv. With --save-models, the selected model for
each product (refitted on all history) is saved to artifacts/ with joblib.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting import (  # noqa: E402
    BACKTEST_HORIZON,
    INTERVAL_LEVEL,
    MIN_TRAIN_DAYS,
    backtest,
    period_error_sigma,
    interval_offsets,
    rmse,
    select_model,
    summarise,
    wape,
)
from src.data_validation import read_csv, to_daily, validate  # noqa: E402
from src.models import BASELINE, MODEL_REGISTRY, make_model, model_label  # noqa: E402


def evaluate_product(daily: pd.DataFrame, window: int, sel_folds: int, hold_folds: int) -> dict:
    """Select on the selection period, then forecast every holdout window with all models."""
    hold_len = window * hold_folds
    selection_data = daily.iloc[:-hold_len]
    sel_bt = backtest(selection_data, window, sel_folds)
    chosen, reason = select_model(summarise(sel_bt))
    hold_bt = backtest(daily, window, hold_folds)  # the last `hold_folds` windows only
    assert hold_bt["date"].min() > selection_data.index.max()
    return {"chosen": chosen, "reason": reason, "selection_bt": sel_bt, "holdout_bt": hold_bt}


def holdout_coverage(sel_model_bt: pd.DataFrame, hold_model_bt: pd.DataFrame) -> float:
    """Coverage of the 80% range on the holdout, with the range built from selection-period errors only."""
    lo, hi = interval_offsets(sel_model_bt, hold_model_bt["step"].to_numpy(), INTERVAL_LEVEL)
    scale = hold_model_bt["level"].to_numpy()
    low = np.clip(hold_model_bt["forecast"].to_numpy() + lo * scale, 0, None)
    high = hold_model_bt["forecast"].to_numpy() + hi * scale
    actual = hold_model_bt["actual"].to_numpy()
    return float(((actual >= low) & (actual <= high)).mean())


def lead_time_checks(sel_model_bt: pd.DataFrame, hold_model_bt: pd.DataFrame, lead: int, z: float,
                     method: str = "measured") -> list[tuple[bool, float]]:
    """For each holdout window: was actual demand over the first `lead` days within forecast + safety stock?

    Safety stock uses errors from the selection period only, rescaled to the
    demand level at the holdout cutoff. `method="measured"` is what the app
    does (summed backtest errors); `method="sqrt"` is the textbook daily
    error x sqrt(lead) rule, kept for comparison. Returns (covered, safety stock).
    """
    daily_scaled = float(np.sqrt(np.mean(sel_model_bt["scaled_error"] ** 2)))
    out = []
    for _, g in hold_model_bt.groupby("fold"):
        g = g.sort_values("step").head(lead)
        lvl = float(g["level"].iloc[0])
        if method == "measured":
            sigma, _ = period_error_sigma(sel_model_bt, lead, lvl)
        else:
            sigma = daily_scaled * lvl * np.sqrt(lead)
        ss = z * sigma
        out.append((bool(g["actual"].sum() <= g["forecast"].sum() + ss), ss))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "sample_sales.csv"))
    ap.add_argument("--horizons", default="7,14,30")
    ap.add_argument("--window", type=int, default=BACKTEST_HORIZON, help="backtest window length in days")
    ap.add_argument("--selection-folds", type=int, default=6)
    ap.add_argument("--holdout-folds", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()

    frame, err = read_csv(args.data)
    if err:
        sys.exit(err)
    val = validate(frame)
    if not val.ok:
        sys.exit("; ".join(val.errors))
    horizons = [int(h) for h in args.horizons.split(",") if int(h) <= args.window]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end = val.data["date"].max()

    per_product, details, chosen_by_product = [], [], {}
    lt_checks = {(lead, m): [] for lead in (7, 14) for m in ("measured", "sqrt")}
    z95 = NormalDist().inv_cdf(0.95)
    t0 = time.time()
    for pid in val.eligible_products:
        daily, _ = to_daily(val.data, pid, end=end)
        needed = MIN_TRAIN_DAYS + args.window * (args.selection_folds + args.holdout_folds)
        if len(daily) < needed:
            print(f"skip {pid}: {len(daily)} days < {needed}")
            continue
        r = evaluate_product(daily, args.window, args.selection_folds, args.holdout_folds)
        chosen = r["chosen"]
        chosen_by_product[pid] = chosen
        sel_c = r["selection_bt"][r["selection_bt"]["model"] == chosen]
        hold = r["holdout_bt"]
        hold_c = hold[hold["model"] == chosen]
        for lead, m in lt_checks:
            lt_checks[(lead, m)] += lead_time_checks(sel_c, hold_c, lead, z95, m)
        for h in horizons:
            hb = hold[hold["step"] <= h].assign(product_id=pid, horizon=h, selected=lambda d: d["model"] == chosen)
            details.append(hb)
            by_model = {m: wape(g["actual"], g["forecast"]) for m, g in hb.groupby("model")}
            cov = holdout_coverage(sel_c, hold_c[hold_c["step"] <= h])
            per_product.append({
                "horizon": h,
                "product_id": pid,
                "selected_model": chosen,
                "selection_reason": r["reason"],
                "holdout_WAPE_selected": by_model[chosen],
                "holdout_WAPE_seasonal_naive": by_model[BASELINE],
                **{f"holdout_WAPE_{m}": v for m, v in by_model.items() if m != BASELINE},
                "holdout_interval_coverage": cov,
            })
            print(f"h={h:>2} {pid}: selected {chosen:<15} holdout WAPE {by_model[chosen]:.3f} "
                  f"(seasonal naive {by_model[BASELINE]:.3f}), coverage {cov:.2f}")

    pp = pd.DataFrame(per_product)
    det = pd.concat(details, ignore_index=True)
    pp.to_csv(out / "evaluation_by_product.csv", index=False, float_format="%.4f")
    det.to_csv(out / "evaluation_detail.csv", index=False, float_format="%.4f")

    # Pooled results: total absolute error over total demand, across all products.
    lines = [
        "# DemandPilot evaluation results",
        "",
        f"Generated by `scripts/evaluate.py` on {datetime.now():%Y-%m-%d} from `{Path(args.data).name}`.",
        f"Products: {det['product_id'].nunique()}. Each backtest window is {args.window} days long. For each product, "
        f"one model was chosen using {args.selection_folds} rolling windows (the same rule the app uses). The results "
        f"below come from the following {args.holdout_folds} windows, which selection never saw. The 7- and 14-day "
        "rows score the first 7 or 14 days of each holdout window.",
        "",
        "WAPE = total absolute error / total actual units (lower is better). "
        "Improvement is relative to the seasonal-naive baseline.",
        "",
        "| Horizon | Approach | Pooled WAPE | Pooled MAE | Pooled RMSE | vs seasonal naive |",
        "|---|---|---|---|---|---|",
    ]
    headline = {}
    for h in horizons:
        dh = det[det["horizon"] == h]
        if dh.empty:
            continue
        base = dh[dh["model"] == BASELINE]
        base_w = wape(base["actual"], base["forecast"])
        rows = [("DemandPilot (selected per product)", dh[dh["selected"]])]
        rows += [(model_label(m), dh[dh["model"] == m]) for m in MODEL_REGISTRY]
        for name, g in rows:
            w = wape(g["actual"], g["forecast"])
            mae = float(np.mean(np.abs(g["actual"] - g["forecast"])))
            imp = (base_w - w) / base_w
            lines.append(f"| {h} days | {name} | {w:.1%} | {mae:.2f} | {rmse(g['actual'], g['forecast']):.2f} | {imp:+.1%} |")
            if name.startswith("DemandPilot"):
                headline[h] = (w, base_w, imp)
    lines += ["", "## Interval coverage on the holdout", "",
              f"Nominal level: {INTERVAL_LEVEL:.0%}. Coverage = share of holdout days where actual demand fell inside the range.", "",
              "| Horizon | Mean coverage across products | Min | Max |", "|---|---|---|---|"]
    for h in horizons:
        c = pp.loc[pp["horizon"] == h, "holdout_interval_coverage"]
        if len(c):
            lines.append(f"| {h} days | {c.mean():.0%} | {c.min():.0%} | {c.max():.0%} |")
    lines += ["", "## Safety stock check on the holdout", "",
              "For each holdout window: was actual demand over the first L days no more than forecast demand plus a "
              "95% safety stock? Errors come from the selection period only. A well-calibrated policy should succeed "
              "in about 95% of windows; much more than that means it holds more stock than needed.", "",
              "The app uses the *measured* method (summed backtest errors over the lead time). The *square-root* "
              "method is the textbook daily error x sqrt(lead time) rule, shown for comparison.", "",
              "| Lead time | Method | Windows | Demand covered | Average safety stock, as % of the measured one |",
              "|---|---|---|---|---|"]
    for lead in (7, 14):
        ref = np.mean([ss for _, ss in lt_checks[(lead, "measured")]]) if lt_checks[(lead, "measured")] else np.nan
        for m, label in (("measured", "Measured (app)"), ("sqrt", "Square-root rule")):
            res_ = lt_checks[(lead, m)]
            if res_:
                hits = [c for c, _ in res_]
                rel = np.mean([ss for _, ss in res_]) / ref
                lines.append(f"| {lead} days | {label} | {len(hits)} | {np.mean(hits):.0%} ({sum(hits)} of {len(hits)}) "
                             f"| {rel:.0%} |")
    lines += ["", "## Model chosen per product", "", "| Product | Selected model | " +
              " | ".join(f"Holdout WAPE {h}d" for h in horizons) + " | Seasonal naive 14d |",
              "|---|---|" + "---|" * len(horizons) + "---|"]
    for pid, g in pp.groupby("product_id"):
        w = {int(r["horizon"]): r["holdout_WAPE_selected"] for _, r in g.iterrows()}
        base14 = g.loc[g["horizon"] == 14, "holdout_WAPE_seasonal_naive"]
        lines.append(f"| {pid} | {model_label(g['selected_model'].iloc[0], short=True)} | "
                     + " | ".join(f"{w[h]:.1%}" for h in horizons)
                     + (f" | {base14.iloc[0]:.1%} |" if len(base14) else " | n/a |"))
    wins = int((pp["holdout_WAPE_selected"] < pp["holdout_WAPE_seasonal_naive"]).sum())
    lines += ["", f"The selected model beat seasonal naive in {wins} of {len(pp)} product and horizon pairs."]
    lines += ["", "## Caveats", "",
              "- The data is synthetic (see `scripts/generate_sample_data.py`). These numbers show the pipeline works and is",
              "  evaluated honestly; they are not evidence of accuracy on any real business.",
              "- Four holdout windows per product is still a small sample, so treat differences of a few points as noise.",
              f"- Runtime: {time.time() - t0:.0f} seconds."]
    (out / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.save_models:
        import joblib

        art = ROOT / "artifacts"
        art.mkdir(exist_ok=True)
        for pid, chosen in chosen_by_product.items():
            daily, _ = to_daily(val.data, pid, end=end)
            joblib.dump(make_model(chosen).fit(daily), art / f"{pid}_{chosen}.joblib")
        print(f"Saved {len(chosen_by_product)} models to {art}")

    print("\nHeadline (pooled holdout WAPE, selected vs seasonal naive):")
    for h, (w, b, imp) in headline.items():
        print(f"  {h:>2}-day horizon: {w:.1%} vs {b:.1%}  ({imp:+.1%})")
    print(f"Wrote {out / 'evaluation_summary.md'}")


if __name__ == "__main__":
    main()
