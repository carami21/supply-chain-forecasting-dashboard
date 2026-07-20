"""Stage 2 — aggregate order lines into weekly series.

Produces the two tidy tables the modeling stages consume:

  demand_weekly.csv      week x category : units + revenue  (top-N categories)
  fulfillment_weekly.csv week x region   : late-delivery rate + shipping delay

Weeks use the W-MON convention (Monday-start). The raw data starts mid-week
(2015-01-01 is a Thursday) and ends mid-week (2018-01-31), so the first and
last calendar weeks are partial — including them would read as fake demand
collapses. We drop any week whose full 7-day span isn't covered by the data.

Data-quality note — catalog changeover: the dataset's product catalog shifts
hard around October 2017. The primary (top-volume) categories are sporting
goods that sell steadily 2015-2017 then stop dead; a different catalog (video
games, toys, cameras) carries the final ~4 months. We forecast the stable
regime, so the analysis window is trimmed to the contiguous run of weeks where
the modeled categories are active. See DATA_NOTES.md.

Run: python src/aggregate.py
In:  data/processed/orders_clean.parquet
Out: data/processed/demand_weekly.csv, data/processed/fulfillment_weekly.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC = Path("data/processed/orders_clean.parquet")
DEMAND_OUT = Path("data/processed/demand_weekly.csv")
FULFILL_OUT = Path("data/processed/fulfillment_weekly.csv")

TOP_N_CATEGORIES = 8  # top 8 by volume hold the bulk of demand (EDA: top 10 = 89%)
WEEK_FREQ = "W-MON"
ACTIVE_THRESHOLD = 0.5  # a week is "active" if top-N volume >= 50% of median


def _complete_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a `week` column and drop partial boundary weeks."""
    week = df["order_date"].dt.to_period(WEEK_FREQ)
    day_min = df["order_date"].dt.normalize().min()
    day_max = df["order_date"].dt.normalize().max()
    complete = (week.dt.start_time >= day_min) & (week.dt.end_time <= day_max)
    dropped = (~complete).sum()
    print(f"  dropped {dropped:,} rows in partial boundary weeks")
    return df.assign(week=week.dt.start_time)[complete].copy()


def _active_window(df: pd.DataFrame, top) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Contiguous run of weeks where the modeled catalog is active, cutting
    the ramp-up and the post-changeover dead tail. Robust to the exact
    changeover date — driven by volume, not a hard-coded cutoff."""
    weekly = (df[df["category"].isin(top)]
              .groupby("week")["quantity"].sum().sort_index())
    active = weekly >= weekly.median() * ACTIVE_THRESHOLD
    # longest contiguous True run
    best_start = best_len = cur_start = cur_len = 0
    weeks = active.index
    for i, on in enumerate(active.to_numpy()):
        if on:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return weeks[best_start], weeks[best_start + best_len - 1]


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} — run src/etl.py first")

    df = pd.read_parquet(SRC)
    df = _complete_weeks(df)

    top = (df.groupby("category")["quantity"].sum()
             .nlargest(TOP_N_CATEGORIES).index)

    # trim to the stable-catalog analysis window (see DATA_NOTES.md)
    start, end = _active_window(df, top)
    full_weeks = df["week"].nunique()
    df = df[df["week"].between(start, end)]
    print(f"  analysis window: {start:%Y-%m-%d} -> {end:%Y-%m-%d} "
          f"({df['week'].nunique()} of {full_weeks} weeks kept; "
          "post-2017-09 catalog changeover trimmed)")

    # ---- demand: top-N categories, weekly units + revenue ----
    demand = (
        df[df["category"].isin(top)]
        .groupby(["week", "category"], as_index=False)
        .agg(units=("quantity", "sum"), revenue=("sales", "sum"))
        .sort_values(["category", "week"])
        .reset_index(drop=True)
    )
    demand.to_csv(DEMAND_OUT, index=False)

    # ---- fulfillment: weekly late-rate per region (canceled excluded) ----
    shipped = df[~df["is_canceled"]].copy()
    shipped["ship_delay"] = shipped["ship_days_real"] - shipped["ship_days_scheduled"]
    fulfillment = (
        shipped.groupby(["week", "region"], as_index=False)
        .agg(
            orders=("late_delivery_risk", "size"),
            late_orders=("late_delivery_risk", "sum"),
            avg_ship_delay=("ship_delay", "mean"),
        )
        .sort_values(["region", "week"])
        .reset_index(drop=True)
    )
    fulfillment["late_rate"] = fulfillment["late_orders"] / fulfillment["orders"]
    fulfillment.to_csv(FULFILL_OUT, index=False)

    weeks = demand["week"].nunique()
    print(f"wrote {DEMAND_OUT}  ({len(demand):,} rows, "
          f"{demand['category'].nunique()} categories x {weeks} weeks)")
    print(f"wrote {FULFILL_OUT}  ({len(fulfillment):,} rows, "
          f"{fulfillment['region'].nunique()} regions)")
    print(f"  week span : {demand['week'].min():%Y-%m-%d} -> "
          f"{demand['week'].max():%Y-%m-%d}")
    return demand, fulfillment


if __name__ == "__main__":
    run()
