"""Stage 4 — brand, style, and product-level demand.

Answers "what sells?" before "how much will sell?":

  brand_summary.csv     brand : units, revenue, profit, product count, unit share
  product_summary.csv   product : brand, style, segment, units, revenue, margin, rank
  brand_weekly.csv      week x brand : units  (top brands + "Other")
  style_weekly.csv      week x style x segment : units
  product_forecasts.csv / product_metrics.csv
                        top-N steady-selling products: Prophet vs seasonal-naive
                        on the same 12-week holdout as the category forecast,
                        scored by WAPE (MAPE breaks on zero-sale weeks)
  product_outlook.csv   top-N products: Prophet fit on the full history,
                        forecasting the next 12 weeks (feeds inventory.py)

Uses the same stable-catalog window as aggregate.py (read from
demand_weekly.csv so the two stages can never disagree). See DATA_NOTES.md.

Run: python src/products.py
In:  data/processed/orders_clean.parquet, data/processed/demand_weekly.csv
Out: outputs/*.csv listed above
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate import _complete_weeks
from forecast import HOLDOUT_WEEKS, mape, prophet_forecast, rmse, seasonal_naive

logging.getLogger("cmdstanpy").disabled = True  # one INFO line per fit otherwise

ORDERS = Path("data/processed/orders_clean.parquet")
DEMAND = Path("data/processed/demand_weekly.csv")
OUT = Path("outputs")

TOP_N_PRODUCTS = 10
MIN_ACTIVE_SHARE = 0.95  # forecast only products that sell in >=95% of weeks
TOP_N_BRANDS = 6
OUTLOOK_WEEKS = 12


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute % error — stays defined when a week sells zero,
    which plain MAPE does not. The retail-standard product-level metric."""
    return float(np.abs(actual - pred).sum() / actual.sum() * 100)


def round_numeric(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.assign(**{c: df[c].round(n) for c in num})


def _weekly(df: pd.DataFrame, key: str, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """week x key units, with zero-filled gaps so every series is contiguous."""
    grid = pd.MultiIndex.from_product([weeks, df[key].unique()], names=["week", key])
    return (df.groupby(["week", key])["quantity"].sum()
              .reindex(grid, fill_value=0).rename("units").reset_index())


def run() -> None:
    for p in (ORDERS, DEMAND):
        if not p.exists():
            raise SystemExit(f"missing {p} — run src/etl.py and src/aggregate.py first")

    window = pd.read_csv(DEMAND, parse_dates=["week"])["week"]
    df = _complete_weeks(pd.read_parquet(ORDERS))
    df = df[df["week"].between(window.min(), window.max())]
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    OUT.mkdir(exist_ok=True)

    # ---- brand summary ----
    brands = (df.groupby("brand")
                .agg(units=("quantity", "sum"), revenue=("sales", "sum"),
                     profit=("profit", "sum"), products=("product_name", "nunique"))
                .sort_values("units", ascending=False))
    brands["unit_share"] = brands["units"] / brands["units"].sum()
    brands["margin"] = brands["profit"] / brands["revenue"]
    brands.round(4).reset_index().to_csv(OUT / "brand_summary.csv", index=False)

    # ---- product summary ----
    products = (df.groupby(["product_name", "brand", "style", "segment", "department"])
                  .agg(units=("quantity", "sum"), revenue=("sales", "sum"),
                       profit=("profit", "sum"), avg_price=("unit_price", "mean"),
                       orders=("order_id", "nunique"))
                  .sort_values("units", ascending=False).reset_index())
    products["margin"] = products["profit"] / products["revenue"]
    products["rank"] = range(1, len(products) + 1)
    products.round(4).to_csv(OUT / "product_summary.csv", index=False)

    # ---- brand + style weekly ----
    top_brands = brands.index[:TOP_N_BRANDS]
    bdf = df.assign(brand=df["brand"].where(df["brand"].isin(top_brands), "Other"))
    _weekly(bdf, "brand", weeks).to_csv(OUT / "brand_weekly.csv", index=False)
    (df.groupby(["week", "style", "segment"], as_index=False)["quantity"].sum()
       .rename(columns={"quantity": "units"})
       .to_csv(OUT / "style_weekly.csv", index=False))

    # ---- top-N product forecasts (steady sellers only) ----
    active = (df.groupby("product_name")["week"].nunique() / len(weeks))
    steady = products[products["product_name"].map(active) >= MIN_ACTIVE_SHARE]
    top = steady["product_name"].head(TOP_N_PRODUCTS)
    pw = _weekly(df[df["product_name"].isin(top)], "product_name", weeks)
    forecasts, metrics, outlook = [], [], []
    future = pd.date_range(weeks.max() + pd.Timedelta(weeks=1),
                           periods=OUTLOOK_WEEKS, freq="7D")

    for prod, g in pw.groupby("product_name"):
        s = g.set_index("week")["units"].sort_index()
        train, test = s.iloc[:-HOLDOUT_WEEKS], s.iloc[-HOLDOUT_WEEKS:]
        for name, fc in [("seasonal_naive", seasonal_naive(train, test.index)),
                         ("prophet", prophet_forecast(train, test.index))]:
            fc = fc.reindex(test.index)
            forecasts.append(pd.DataFrame({
                "week": test.index, "product_name": prod, "actual": test.values,
                "yhat": fc["yhat"].values, "yhat_lower": fc["yhat_lower"].values,
                "yhat_upper": fc["yhat_upper"].values, "model": name}))
            metrics.append({"product_name": prod, "model": name,
                            "wape": round(wape(test.values, fc["yhat"].values), 2),
                            "mape": round(mape(test.values, fc["yhat"].values), 2),
                            "rmse": round(rmse(test.values, fc["yhat"].values), 1),
                            "n_test": len(test)})
        fut = prophet_forecast(s, future)
        outlook.append(pd.DataFrame({"week": future, "product_name": prod,
                                     "yhat": fut["yhat"].values,
                                     "yhat_lower": fut["yhat_lower"].values,
                                     "yhat_upper": fut["yhat_upper"].values}))

    pw.to_csv(OUT / "product_weekly.csv", index=False)
    round_numeric(pd.concat(forecasts)).to_csv(OUT / "product_forecasts.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / "product_metrics.csv", index=False)
    round_numeric(pd.concat(outlook)).to_csv(OUT / "product_outlook.csv", index=False)

    summary = metrics.groupby("model")["wape"].mean()
    nike = brands.loc["Nike"] if "Nike" in brands.index else None
    print(f"wrote {OUT}/brand_summary.csv  ({len(brands)} brands)")
    print(f"wrote {OUT}/product_summary.csv  ({len(products)} products)")
    if nike is not None:
        print(f"  Nike: #{list(brands.index).index('Nike') + 1} brand, "
              f"{nike['unit_share']:.1%} of units, {int(nike['products'])} products")
    print(f"  top product: {products.iloc[0]['product_name']} "
          f"({int(products.iloc[0]['units']):,} units)")
    print(f"wrote product forecasts for top {len(top)} steady sellers: mean WAPE "
          f"seasonal_naive={summary['seasonal_naive']:.1f}%  "
          f"prophet={summary['prophet']:.1f}%")


if __name__ == "__main__":
    run()
