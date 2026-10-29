"""Stage 8 — tidy, Tableau-ready extracts (one CSV per dashboard view).

Tableau Public can't run Python, so every number it shows comes from these
files. Long (tidy) format throughout so each view is a simple drag-and-drop;
see docs/tableau_build_guide.md for the click-by-click build.

Run: python src/build_extract.py
In:  outputs/*.csv
Out: tableau/*.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("outputs")
TABLEAU = Path("tableau")


def _read(name: str) -> pd.DataFrame | None:
    path = OUT / name
    return pd.read_csv(path) if path.exists() else None


def run() -> None:
    TABLEAU.mkdir(exist_ok=True)
    written = []

    def save(df: pd.DataFrame, name: str) -> None:
        df.to_csv(TABLEAU / name, index=False)
        written.append(f"{name} ({len(df):,} rows)")

    # 1. demand: full-history fit + anomaly flag, one row per category-week
    fit, anomalies = _read("demand_fit.csv"), _read("anomalies.csv")
    sev = anomalies.set_index(["week", "category"])["severity"]
    fit["severity"] = [sev.get((w, c), "") for w, c in zip(fit["week"], fit["category"])]
    save(fit[["week", "category", "actual", "yhat", "yhat_lower", "yhat_upper",
              "is_anomaly", "severity"]], "demand_fit.csv")

    # 2. holdout forecasts: categories + products in one long table
    cat = _read("forecasts.csv").rename(columns={"category": "item"}).assign(level="Category")
    prod = _read("product_forecasts.csv").rename(columns={"product_name": "item"}) \
        .assign(level="Product")
    save(pd.concat([cat, prod], ignore_index=True), "forecast_holdout.csv")

    # 3. forecast accuracy: one row per item x model
    cm = _read("metrics.csv").rename(columns={"category": "item", "mape": "error_pct"})
    pm = _read("product_metrics.csv").rename(columns={"product_name": "item",
                                                      "wape": "error_pct"})
    save(pd.concat([cm.assign(level="Category", metric="MAPE"),
                    pm.drop(columns="mape").assign(level="Product", metric="WAPE")],
                   ignore_index=True), "forecast_accuracy.csv")

    # 4. brands + products
    save(_read("brand_summary.csv"), "brands.csv")
    save(_read("brand_weekly.csv"), "brand_weekly.csv")
    save(_read("product_summary.csv"), "products.csv")
    save(_read("style_weekly.csv"), "style_weekly.csv")
    save(_read("product_outlook.csv"), "product_outlook.csv")

    # 5. inventory + logistics
    save(_read("inventory_policy.csv"), "inventory_policy.csv")
    save(_read("shipping_by_market.csv"), "shipping.csv")
    save(_read("fulfillment_anomalies.csv"), "fulfillment_alerts.csv")

    # 6. H&M style module (optional)
    for name in ["hm_style_summary.csv", "hm_style_weekly.csv", "hm_trends.csv",
                 "hm_style_forecasts.csv", "hm_colour_month.csv"]:
        df = _read(name)
        if df is not None:
            save(df, name)

    print(f"wrote {len(written)} extracts to {TABLEAU}/")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    run()
