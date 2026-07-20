"""Stage 1 — ingest + clean the raw DataCo order lines.

Reads the raw Mendeley CSV (latin-1), keeps only the columns the forecast
and anomaly stages need, parses the order date, and writes a compact
parquet. EDA (notebooks/01_eda.ipynb) confirmed zero duplicate rows,
zero null/negative quantities, and zero null dates, so cleaning is light
— the main jobs are column selection, date parsing, and flagging the
`Shipping canceled` rows so downstream stages can treat demand and
fulfillment consistently.

Run: python src/etl.py
In:  data/raw/DataCoSupplyChainDataset.csv
Out: data/processed/orders_clean.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/DataCoSupplyChainDataset.csv")
OUT = Path("data/processed/orders_clean.parquet")

# Source columns → tidy names. Only these leave the ETL stage.
COLUMNS = {
    "order date (DateOrders)": "order_date",
    "Category Name": "category",
    "Order Item Quantity": "quantity",
    "Sales": "sales",
    "Market": "market",
    "Order Region": "region",
    "Days for shipping (real)": "ship_days_real",
    "Days for shipment (scheduled)": "ship_days_scheduled",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
}

DATE_FORMAT = "%m/%d/%Y %H:%M"  # e.g. "1/31/2018 22:56"


def run() -> pd.DataFrame:
    if not RAW.exists():
        raise SystemExit(
            f"missing {RAW} — download DataCoSupplyChainDataset.csv from "
            "https://data.mendeley.com/datasets/8gx2fvg2k6/5 into data/raw/"
        )

    df = pd.read_csv(RAW, encoding="latin-1", usecols=list(COLUMNS))
    df = df.rename(columns=COLUMNS)

    df["order_date"] = pd.to_datetime(df["order_date"], format=DATE_FORMAT)

    # Canceled orders still represent demand a planner faced, but must be
    # excluded from delivery-timeliness metrics. Flag once, here.
    df["is_canceled"] = df["delivery_status"].eq("Shipping canceled")

    df = df.sort_values("order_date").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"wrote {OUT}  ({len(df):,} rows x {df.shape[1]} cols)")
    print(f"  date range : {df['order_date'].min():%Y-%m-%d} -> "
          f"{df['order_date'].max():%Y-%m-%d}")
    print(f"  categories : {df['category'].nunique()}")
    print(f"  canceled   : {df['is_canceled'].sum():,} rows "
          f"({df['is_canceled'].mean():.1%})")
    return df


if __name__ == "__main__":
    run()
