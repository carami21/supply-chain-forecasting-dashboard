"""Stage 1 — ingest + clean the raw DataCo order lines.

Reads the raw Mendeley CSV (latin-1), keeps only the columns the forecast,
anomaly, product, and inventory stages need (no customer PII), derives
brand / style / segment from the product name, parses the order date, and
writes a compact parquet. EDA (notebooks/01_eda.ipynb) confirmed zero duplicate rows,
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
    # product + logistics fields for the brand, inventory, and shipping-mode
    # stages. Customer PII (name/email/password/street) is never selected.
    "Order Id": "order_id",
    "Product Name": "product_name",
    "Department Name": "department",
    "Order Item Product Price": "unit_price",
    "Order Profit Per Order": "profit",
    "Shipping Mode": "shipping_mode",
}

# Multi-word brands; anything else takes the first word of the product name.
MULTI_WORD_BRANDS = [
    "Under Armour", "Field & Stream", "Perfect Fitness", "The North Face",
    "Glove It", "Team Golf", "Top Flite", "Total Gym", "Bag Boy",
    "Cleveland Golf", "Elevation Training", "Fitbit", "GoPro",
]

# Product-type ("style") keywords, checked in order — first match wins.
STYLE_KEYWORDS = [
    # source names are truncated at 45 chars, so some keys are fragments
    ("Comfort Bi", "Bikes"), ("Rip Deck", "Fitness Equipment"),
    ("Dumbbells", "Fitness Equipment"), ("Elliptical", "Fitness Equipment"),
    ("Total Gym", "Fitness Equipment"), ("Training Mask", "Fitness Equipment"),
    ("Gun Fire Safe", "Outdoor Gear"), ("Hammock", "Outdoor Gear"),
    ("Air Mattress", "Outdoor Gear"), ("Bridgestone", "Golf Balls"),
    ("Pro V1", "Golf Balls"), ("Away Match Soc", "Jerseys"),
    ("Winners Bl", "T-Shirts"), ("Hik", "Hiking Shoes"),
    ("Match Ball", "Balls"), ("Travel Cover", "Bags"),
    ("Shoe Brush", "Accessories"), ("Camera", "Electronics"),
    ("Competition Indoor", "Games"),
    ("Cleat", "Cleats"), ("Slide", "Slides"), ("Running Sho", "Running Shoes"),
    ("Runni", "Running Shoes"), ("Training S", "Training Shoes"),
    ("Basketball Shoe", "Basketball Shoes"), ("Hiking", "Hiking Shoes"),
    ("Golf Shoe", "Golf Shoes"), ("Polo", "Polos"), ("T-Shirt", "T-Shirts"),
    ("Tee", "T-Shirts"), ("Shorts", "Shorts"), ("Dress", "Dresses"),
    ("Sock", "Socks"), ("Jammer", "Swimwear"), ("Glove", "Gloves"),
    ("Life Vest", "Life Vests"), ("Kayak", "Kayaks"), ("Bike", "Bikes"),
    ("Hybr", "Bikes"), ("Watch", "Wearables"), ("Tracker", "Wearables"),
    ("Monitor", "Wearables"), ("Trac", "Wearables"), ("Putter", "Golf Clubs"),
    ("Wedge", "Golf Clubs"), ("Rescue", "Golf Clubs"), ("Hybrid", "Golf Clubs"),
    ("Golf Bal", "Golf Balls"), ("Gol", "Golf Balls"), ("Bag", "Bags"),
    ("Backpack", "Bags"), ("Duffle", "Bags"),
]


def brand_of(name: str) -> str:
    for b in MULTI_WORD_BRANDS:
        if name.startswith(b):
            return b
    return name.split()[0]


def style_of(name: str) -> str:
    for key, style in STYLE_KEYWORDS:
        if key in name:
            return style
    return "Other"


def segment_of(name: str) -> str:
    for key, seg in [("Women's", "Women"), ("Men's", "Men"), ("Girls'", "Kids"),
                     ("Boys'", "Kids"), ("Kids'", "Kids"), ("Youth", "Kids")]:
        if key in name:
            return seg
    return "Unisex"

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

    names = df["product_name"].drop_duplicates()
    df["brand"] = df["product_name"].map(dict(zip(names, names.map(brand_of))))
    df["style"] = df["product_name"].map(dict(zip(names, names.map(style_of))))
    df["segment"] = df["product_name"].map(dict(zip(names, names.map(segment_of))))

    df = df.sort_values("order_date").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"wrote {OUT}  ({len(df):,} rows x {df.shape[1]} cols)")
    print(f"  date range : {df['order_date'].min():%Y-%m-%d} -> "
          f"{df['order_date'].max():%Y-%m-%d}")
    print(f"  categories : {df['category'].nunique()}")
    print(f"  products   : {df['product_name'].nunique()}  "
          f"brands: {df['brand'].nunique()}")
    print(f"  canceled   : {df['is_canceled'].sum():,} rows "
          f"({df['is_canceled'].mean():.1%})")
    return df


if __name__ == "__main__":
    run()
