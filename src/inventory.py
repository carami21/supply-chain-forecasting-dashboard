"""Stage 6 — turn demand + fulfillment data into an ordering policy.

The "so what" stage: what should a planner actually do with the forecast?

  ABC-XYZ     ABC = cumulative revenue share (A: top 80%, B: next 15%,
              C: last 5%). XYZ = coefficient of variation of weekly units
              (X <= 0.5 steady, Y <= 1.0 variable, Z > 1.0 erratic).
  Policy      Safety stock SS = z * sigma * sqrt(L), reorder point
              ROP = d * L + SS, EOQ = sqrt(2 * D * S / H). Service level by
              ABC class (A 98%, B 95%, C 90%). d = mean of the 12-week
              Prophet outlook where one exists, else mean weekly demand over
              the analysis window. sigma = Prophet holdout RMSE where one
              exists (forecast error is what safety stock actually buffers),
              else std of weekly demand.
  Value of    For forecasted products, SS is also computed with the seasonal-
  forecasting naive holdout RMSE (what a planner gets without a model); the
              difference x unit cost is the working capital Prophet frees up
              (negative where naive was the better forecast).
  Shipping    Promised vs actual ship days and late rate per shipping mode,
              and per mode x market.

ASSUMPTIONS (not in the data, stated so they can be challenged):
  supplier replenishment lead time L = 2 weeks; ordering cost S = $50/PO;
  annual holding cost H = 25% of unit cost; unit cost = price x (1 - margin).
DataCo's ship days are *customer delivery* times, not supplier lead times,
so they drive the shipping scorecard, not L.

Run: python src/inventory.py
In:  data/processed/orders_clean.parquet, outputs/product_summary.csv,
     outputs/product_metrics.csv, outputs/product_outlook.csv,
     data/processed/demand_weekly.csv (analysis window)
Out: outputs/inventory_policy.csv, outputs/shipping_scorecard.csv,
     outputs/shipping_by_market.csv, outputs/recommendations.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aggregate import _complete_weeks
from products import round_numeric

ORDERS = Path("data/processed/orders_clean.parquet")
DEMAND = Path("data/processed/demand_weekly.csv")
OUT = Path("outputs")

LEAD_TIME_WEEKS = 2
ORDER_COST = 50.0
HOLDING_RATE = 0.25
SERVICE_Z = {"A": 2.054, "B": 1.645, "C": 1.282}  # 98% / 95% / 90%
SERVICE_LEVEL = {"A": "98%", "B": "95%", "C": "90%"}


def abc_class(cum_share: float) -> str:
    return "A" if cum_share <= 0.80 else "B" if cum_share <= 0.95 else "C"


def xyz_class(cv: float) -> str:
    return "X" if cv <= 0.5 else "Y" if cv <= 1.0 else "Z"


ACTIONS = {
    "AX": "High value, steady: forecast-driven weekly replenishment, tight ROP, never stock out.",
    "AY": "High value, variable: forecast weekly, hold extra safety stock, review exceptions.",
    "AZ": "High value, erratic: order to confirmed demand where possible; monitor weekly.",
    "BX": "Steady mid-value: automate reorders at ROP, review monthly.",
    "BY": "Variable mid-value: periodic review every 2 weeks with safety stock.",
    "BZ": "Erratic mid-value: small batches, avoid overstock.",
    "CX": "Low value, steady: bulk-order on EOQ, minimal oversight.",
    "CY": "Low value, variable: periodic review, lean stock.",
    "CZ": "Low value, erratic: make-to-order or candidate for delisting.",
}


def inventory_policy(df: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    products = pd.read_csv(OUT / "product_summary.csv")
    metrics = pd.read_csv(OUT / "product_metrics.csv")
    outlook = pd.read_csv(OUT / "product_outlook.csv")

    grid = pd.MultiIndex.from_product([weeks, products["product_name"]],
                                      names=["week", "product_name"])
    weekly = (df.groupby(["week", "product_name"])["quantity"].sum()
                .reindex(grid, fill_value=0).unstack())

    p = products.set_index("product_name")[["brand", "style", "revenue", "units",
                                            "avg_price", "margin"]].copy()
    p = p.sort_values("revenue", ascending=False)
    p["revenue_share"] = p["revenue"] / p["revenue"].sum()
    p["abc"] = p["revenue_share"].cumsum().map(abc_class)
    p["cv"] = weekly.std() / weekly.mean()
    p["xyz"] = p["cv"].map(xyz_class)
    p["class"] = p["abc"] + p["xyz"]

    rmse = metrics.pivot(index="product_name", columns="model", values="rmse")
    prophet_rmse = rmse["prophet"]
    outlook_mean = outlook.groupby("product_name")["yhat"].mean()
    p["forecasted"] = p.index.isin(prophet_rmse.index)
    p["weekly_demand"] = outlook_mean.reindex(p.index).fillna(weekly.mean())
    p["sigma"] = prophet_rmse.reindex(p.index).fillna(weekly.std())
    p["sigma_naive_forecast"] = rmse["seasonal_naive"].reindex(p.index)

    z = p["abc"].map(SERVICE_Z)
    root_l = np.sqrt(LEAD_TIME_WEEKS)
    p["service_level"] = p["abc"].map(SERVICE_LEVEL)
    p["safety_stock"] = np.ceil(z * p["sigma"] * root_l)
    p["reorder_point"] = np.ceil(p["weekly_demand"] * LEAD_TIME_WEEKS + p["safety_stock"])
    p["unit_cost"] = p["avg_price"] * (1 - p["margin"])
    annual = p["weekly_demand"] * 52
    p["eoq"] = np.ceil(np.sqrt(2 * annual * ORDER_COST / (HOLDING_RATE * p["unit_cost"])))
    p.loc[p["weekly_demand"] <= 0, ["eoq"]] = 0
    p["order_every_weeks"] = (p["eoq"] / p["weekly_demand"]).where(p["weekly_demand"] > 0)

    ss_naive = np.ceil(z * p["sigma_naive_forecast"] * root_l)
    p["ss_saved_units"] = (ss_naive - p["safety_stock"]).where(p["forecasted"], 0)
    p["ss_saved_dollars"] = p["ss_saved_units"] * p["unit_cost"]
    p["action"] = p["class"].map(ACTIONS)
    out = round_numeric(p.reset_index(), 2)
    out["revenue_share"] = p["revenue_share"].round(6).values
    return out


def shipping_scorecard(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = df[~df["is_canceled"]].copy()
    s["delay"] = s["ship_days_real"] - s["ship_days_scheduled"]
    agg = dict(order_lines=("quantity", "size"), late_rate=("late_delivery_risk", "mean"),
               promised_days=("ship_days_scheduled", "mean"),
               actual_days=("ship_days_real", "mean"), avg_delay=("delay", "mean"),
               p95_actual_days=("ship_days_real", lambda x: x.quantile(0.95)))
    by_mode = s.groupby("shipping_mode").agg(**agg).reset_index()
    by_mode["share_of_lines"] = by_mode["order_lines"] / by_mode["order_lines"].sum()
    by_mode = by_mode.sort_values("promised_days")
    by_market = s.groupby(["shipping_mode", "market"]).agg(**agg).reset_index()
    return round_numeric(by_mode, 3), round_numeric(by_market, 3)


def recommendations(policy: pd.DataFrame, modes: pd.DataFrame) -> str:
    m = modes.set_index("shipping_mode")
    nike = policy[policy["brand"] == "Nike"].sort_values("revenue", ascending=False)
    a_share = policy.loc[policy["abc"] == "A", "revenue"].sum() / policy["revenue"].sum()
    n_a = (policy["abc"] == "A").sum()
    saved = policy["ss_saved_dollars"].sum()
    cz = policy[policy["class"].isin(["CZ"])]
    wins = int((policy["ss_saved_units"] > 0).sum())

    lines = [
        "# Order-Logistics Recommendations",
        "",
        "_Generated by `src/inventory.py` from DataCo order history "
        f"({len(policy)} products). Assumptions: {LEAD_TIME_WEEKS}-week supplier lead time, "
        f"${ORDER_COST:.0f}/order, {HOLDING_RATE:.0%}/yr holding cost._",
        "",
        "## 1. Fix the shipping promises (biggest, cheapest win)",
    ]
    for mode in ["First Class", "Second Class", "Standard Class", "Same Day"]:
        if mode in m.index:
            r = m.loc[mode]
            lines.append(f"- **{mode}**: promised {r.promised_days:.0f} d, actual "
                         f"{r.actual_days:.1f} d, **{r.late_rate:.0%} late** "
                         f"({r.share_of_lines:.0%} of order lines).")
    if {"Second Class", "Standard Class"} <= set(m.index):
        lines += [
            f"- Second Class arrives in {m.loc['Second Class', 'actual_days']:.1f} days, "
            f"the same as Standard ({m.loc['Standard Class', 'actual_days']:.1f}). "
            "Customers pay a premium for zero speed. **Action:** re-quote First Class "
            "as 2-day and Second Class as 4-day (or retire Second Class), and audit the "
            "carrier SLA behind First Class: it misses its 1-day promise on every order.",
            "- Late rate is ~flat across all five markets, so this is a promise-setting "
            "problem, not a regional carrier problem. Fixing the quoted days alone "
            "removes most \"late\" deliveries without moving a single package faster.",
        ]
    lines += [
        "",
        "## 2. Concentrate planning effort with ABC-XYZ",
        f"- **{n_a} products drive {a_share:.0%} of revenue** (class A). These get "
        "forecast-driven weekly replenishment and a 98% service level.",
        f"- {len(cz)} products are class CZ (low value, erratic): make-to-order or "
        "review for delisting instead of stocking.",
        "",
        "## 3. Nike line: ordering policy",
        "",
        "| Product | Class | Weekly demand | Safety stock | Reorder point | EOQ | Order every |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in nike.head(6).iterrows():
        every = f"{r.order_every_weeks:.1f} wk" if pd.notna(r.order_every_weeks) else "n/a"
        lines.append(f"| {r.product_name} | {r['class']} | {r.weekly_demand:.0f} | "
                     f"{r.safety_stock:.0f} | {r.reorder_point:.0f} | {r.eoq:.0f} | {every} |")
    lines += [
        "",
        "## 4. What the forecast is worth",
        f"- Sizing safety stock on Prophet's forecast error instead of the "
        f"seasonal-naive forecast's error {'frees' if saved >= 0 else 'adds'} "
        f"**${abs(saved):,.0f}** of inventory across the {int(policy['forecasted'].sum())} "
        "forecasted products at the same service level.",
        f"- Prophet lowers safety stock on {wins} of {int(policy['forecasted'].sum())} "
        "products; where naive wins, keep the naive forecast for that SKU. Pick the "
        "model per product, not one model for the whole catalog.",
        "",
        "## 5. Watch list",
        "- Demand anomalies (`outputs/anomalies.csv`): treat **high/medium** severity "
        "weeks as signal; low-severity flags are within what a 95% band produces by chance.",
        "- Fulfillment anomalies (`outputs/fulfillment_anomalies.csv`): region-weeks "
        "where late rate jumps >2 sigma above its trailing 8-week mean; escalate to the carrier.",
        "",
    ]
    return "\n".join(lines)


def run() -> None:
    needed = [ORDERS, DEMAND, OUT / "product_summary.csv",
              OUT / "product_metrics.csv", OUT / "product_outlook.csv"]
    for p in needed:
        if not p.exists():
            raise SystemExit(f"missing {p} — run the earlier stages first")

    window = pd.read_csv(DEMAND, parse_dates=["week"])["week"]
    df = _complete_weeks(pd.read_parquet(ORDERS))
    df = df[df["week"].between(window.min(), window.max())]
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))

    policy = inventory_policy(df, weeks)
    modes, by_market = shipping_scorecard(df)
    policy.to_csv(OUT / "inventory_policy.csv", index=False)
    modes.to_csv(OUT / "shipping_scorecard.csv", index=False)
    by_market.to_csv(OUT / "shipping_by_market.csv", index=False)
    (OUT / "recommendations.md").write_text(recommendations(policy, modes))

    print(f"wrote {OUT}/inventory_policy.csv  ({len(policy)} products; "
          + ", ".join(f"{k}={v}" for k, v in policy["abc"].value_counts().sort_index().items())
          + ")")
    print(f"wrote {OUT}/shipping_scorecard.csv, shipping_by_market.csv")
    print(f"wrote {OUT}/recommendations.md  (forecast frees "
          f"${policy['ss_saved_dollars'].sum():,.0f} of safety stock)")


if __name__ == "__main__":
    run()
