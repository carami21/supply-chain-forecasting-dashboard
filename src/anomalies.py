"""Stage 5 — flag the weeks where reality broke the forecast.

Two layers:

  demand      Prophet fit on each category's full weekly history; a week is
              an anomaly when actual units fall outside the 95% prediction
              interval. Severity = residual z-score (residual / residual std).
              Caveat: these are *in-sample* interval breaches — the model has
              seen the week it is judging — so they mark unusual weeks, not
              forecast failures. Holdout accuracy lives in forecast.py.
  fulfillment Per region, a week is flagged when its late-delivery rate is
              more than 2 sigma above the trailing 8-week mean (prior weeks
              only, so the spike can't hide itself). Regions with <10 shipped
              orders that week are skipped as too noisy.

Run: python src/anomalies.py
In:  data/processed/demand_weekly.csv, data/processed/fulfillment_weekly.csv
Out: outputs/demand_fit.csv              full-history fit + band per category
     outputs/anomalies.csv               flagged demand weeks
     outputs/fulfillment_anomalies.csv   flagged region-weeks
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from forecast import prophet_forecast
from products import round_numeric

logging.getLogger("cmdstanpy").disabled = True

DEMAND = Path("data/processed/demand_weekly.csv")
FULFILL = Path("data/processed/fulfillment_weekly.csv")
OUT = Path("outputs")

ROLL_WEEKS = 8
SIGMA = 2.0
MIN_ORDERS = 10


def _severity(z: float) -> str:
    z = abs(z)
    return "high" if z >= 3 else "medium" if z >= 2.5 else "low"


def demand_anomalies() -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = pd.read_csv(DEMAND, parse_dates=["week"])
    fits = []
    for cat, g in demand.groupby("category"):
        s = g.set_index("week")["units"].sort_index()
        fc = prophet_forecast(s, s.index)
        fit = fc.assign(category=cat, actual=s.values).rename_axis("week").reset_index()
        fit["residual"] = fit["actual"] - fit["yhat"]
        fit["z"] = fit["residual"] / fit["residual"].std()
        fits.append(fit)

    fit = pd.concat(fits, ignore_index=True)
    fit["is_anomaly"] = (fit["actual"] > fit["yhat_upper"]) | (fit["actual"] < fit["yhat_lower"])
    anomalies = fit[fit["is_anomaly"]].copy()
    anomalies["direction"] = anomalies["residual"].gt(0).map({True: "spike", False: "drop"})
    anomalies["severity"] = anomalies["z"].map(_severity)
    cols = ["week", "category", "actual", "yhat", "yhat_lower", "yhat_upper",
            "residual", "z", "direction", "severity"]
    return round_numeric(fit, 2), round_numeric(anomalies[cols], 2).sort_values("week")


def fulfillment_anomalies() -> pd.DataFrame:
    f = pd.read_csv(FULFILL, parse_dates=["week"]).sort_values(["region", "week"])
    g = f.groupby("region")["late_rate"]
    f["baseline"] = g.transform(lambda s: s.shift(1).rolling(ROLL_WEEKS, min_periods=4).mean())
    f["baseline_std"] = g.transform(lambda s: s.shift(1).rolling(ROLL_WEEKS, min_periods=4).std())
    f["z"] = (f["late_rate"] - f["baseline"]) / f["baseline_std"]
    flagged = f[(f["orders"] >= MIN_ORDERS) & (f["z"] > SIGMA)].copy()
    flagged["severity"] = flagged["z"].map(_severity)
    cols = ["week", "region", "orders", "late_orders", "late_rate", "baseline",
            "z", "avg_ship_delay", "severity"]
    return round_numeric(flagged[cols], 3).sort_values("week")


def run() -> None:
    for p in (DEMAND, FULFILL):
        if not p.exists():
            raise SystemExit(f"missing {p} — run src/aggregate.py first")
    OUT.mkdir(exist_ok=True)

    fit, anomalies = demand_anomalies()
    fit.to_csv(OUT / "demand_fit.csv", index=False)
    anomalies.to_csv(OUT / "anomalies.csv", index=False)
    late = fulfillment_anomalies()
    late.to_csv(OUT / "fulfillment_anomalies.csv", index=False)

    n_weeks = len(fit)
    print(f"wrote {OUT}/demand_fit.csv  ({n_weeks:,} category-weeks)")
    print(f"wrote {OUT}/anomalies.csv  ({len(anomalies)} flagged, "
          f"{len(anomalies) / n_weeks:.1%} of category-weeks; "
          f"{(anomalies['severity'] == 'high').sum()} high severity)")
    print(f"wrote {OUT}/fulfillment_anomalies.csv  ({len(late)} region-weeks "
          f"with late-rate > {SIGMA:.0f} sigma above trailing {ROLL_WEEKS}-wk mean)")


if __name__ == "__main__":
    run()
