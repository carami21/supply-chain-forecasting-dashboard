# Supply Chain Demand Forecasting Dashboard

Python + Tableau pipeline that forecasts weekly demand across 180K+
supply-chain order records and flags the weeks where reality broke the
forecast.

**Status:** Week 1 build in progress (target: live dashboard by 2026-07-31).

## Pipeline

```
data/raw/DataCoSupplyChainDataset.csv
  → src/etl.py          clean + type-fix           → data/processed/orders_clean.parquet
  → src/aggregate.py    order lines → weekly demand → data/processed/demand_weekly.csv
  → src/forecast.py     Prophet vs seasonal-naive   → outputs/forecasts.csv, metrics.csv
  → src/anomalies.py    interval-breach flags       → outputs/anomalies.csv
  → src/build_extract.py                            → tableau/dashboard_data.csv
  → Tableau Public dashboard
```

Run everything: `python run_pipeline.py`

## Dataset

DataCo Smart Supply Chain — 180,519 order lines × 53 columns (2015–2018),
licensed **CC BY 4.0**. Not committed to this repo; download from Mendeley:
https://data.mendeley.com/datasets/8gx2fvg2k6/5 → place
`DataCoSupplyChainDataset.csv` and `DescriptionDataCoSupplyChain.csv`
in `data/raw/`.

> Constante, F., Silva, F., & Pereira, A. (2019). DataCo Smart Supply
> Chain for Big Data Analysis. Mendeley Data, V5. doi:10.17632/8gx2fvg2k6.5

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
