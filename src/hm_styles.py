"""Stage 7 (optional) — fashion style trends from H&M transactions.

DataCo answers the logistics questions; it has no real fashion attributes.
H&M's Personalized Fashion Recommendations data (Kaggle, ~31M purchases,
Sep 2018 - Sep 2020) does: product type, colour, garment group, and
Ladieswear/Menswear/Kids index. This stage answers "which styles are popular,
which are rising, and how much will each sell next?"

  hm_style_summary.csv    every product type: purchases, customers, avg price
  hm_style_weekly.csv     week x top styles : purchases
  hm_trends.csv           8-week momentum per style (last 8 wks vs prior 8)
  hm_style_forecasts.csv / hm_style_metrics.csv
                          top styles: Prophet vs seasonal-naive, 8-week
                          holdout, scored by WAPE
  hm_colour_month.csv     month x colour group : purchases + share of month
  hm_group_weekly.csv     week x index group (Ladieswear, Menswear, ...)

The 3.5 GB transactions file never loads into pandas: DuckDB aggregates it
out-of-core straight from CSV. Only aggregates are written; raw H&M data is
never committed or redistributed (Kaggle competition terms).

Note: H&M's `price` column is scaled/anonymized, so revenue is reported as a
relative index, not currency.

Get the data (needs a Kaggle API token and the competition rules accepted):
  kaggle competitions download -c h-and-m-personalized-fashion-recommendations \
      -f articles.csv -p data/raw/hm
  kaggle competitions download -c h-and-m-personalized-fashion-recommendations \
      -f transactions_train.csv -p data/raw/hm
  (unzip if the files arrive zipped)

Run: python src/hm_styles.py
In:  data/raw/hm/articles.csv, data/raw/hm/transactions_train.csv
Out: outputs/hm_*.csv listed above
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from forecast import prophet_forecast, rmse, seasonal_naive
from products import round_numeric, wape

logging.getLogger("cmdstanpy").disabled = True

RAW = Path("data/raw/hm")
ARTICLES = RAW / "articles.csv"
TRANSACTIONS = RAW / "transactions_train.csv"
OUT = Path("outputs")

TOP_N_WEEKLY = 20
TOP_N_FORECAST = 12
HOLDOUT_WEEKS = 8
MOMENTUM_WEEKS = 8
MIN_TREND_UNITS = 2000  # ignore tiny styles in the momentum ranking


def _query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def run(raw: Path = RAW, out: Path = OUT) -> bool:
    articles, transactions = raw / ARTICLES.name, raw / TRANSACTIONS.name
    if not (articles.exists() and transactions.exists()):
        print(f"skip H&M stage — {raw}/articles.csv and transactions_train.csv not found "
              "(see module docstring to download)")
        return False
    out.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW art AS SELECT * FROM read_csv_auto('{articles}',
            types={{'article_id': 'VARCHAR'}});
        CREATE VIEW tx AS SELECT CAST(t_dat AS DATE) AS t_dat, customer_id,
                                 CAST(article_id AS VARCHAR) AS article_id, price
            FROM read_csv_auto('{transactions}', types={{'article_id': 'VARCHAR'}});
        -- complete Monday-start weeks only (data starts/ends mid-week)
        CREATE TABLE bounds AS SELECT
            date_trunc('week', min(t_dat) + INTERVAL 6 DAY) AS first_week,
            date_trunc('week', max(t_dat) + INTERVAL 1 DAY) - INTERVAL 7 DAY AS last_week
            FROM tx;
        CREATE TABLE t AS
            SELECT date_trunc('week', tx.t_dat) AS week, tx.t_dat, tx.customer_id,
                   tx.price, a.product_type_name, a.product_group_name,
                   a.colour_group_name, a.index_group_name, a.garment_group_name
            FROM tx JOIN art a USING (article_id), bounds
            WHERE date_trunc('week', tx.t_dat) BETWEEN bounds.first_week AND bounds.last_week;
    """)

    summary = _query(con, """
        SELECT product_type_name, any_value(product_group_name) AS product_group_name,
               count(*) AS units, count(DISTINCT customer_id) AS customers,
               avg(price) AS avg_price_index, sum(price) AS revenue_index
        FROM t GROUP BY 1 ORDER BY units DESC""")
    summary["unit_share"] = summary["units"] / summary["units"].sum()
    round_numeric(summary, 6).to_csv(out / "hm_style_summary.csv", index=False)

    top = list(summary["product_type_name"].head(TOP_N_WEEKLY))
    weekly_all = _query(con, """
        SELECT week, product_type_name, count(*) AS units FROM t
        GROUP BY 1, 2 ORDER BY 1""")
    weekly_all["week"] = pd.to_datetime(weekly_all["week"])
    weeks = pd.DatetimeIndex(sorted(weekly_all["week"].unique()))
    grid = pd.MultiIndex.from_product([weeks, weekly_all["product_type_name"].unique()],
                                      names=["week", "product_type_name"])
    wide = (weekly_all.set_index(["week", "product_type_name"])["units"]
                      .reindex(grid, fill_value=0).unstack())
    (wide[top].stack().rename("units").reset_index()
        .to_csv(out / "hm_style_weekly.csv", index=False))

    recent = wide.tail(MOMENTUM_WEEKS).sum()
    prior = wide.iloc[-2 * MOMENTUM_WEEKS:-MOMENTUM_WEEKS].sum()
    trend = pd.DataFrame({"recent_units": recent, "prior_units": prior})
    trend = trend[(trend["recent_units"] + trend["prior_units"]) >= MIN_TREND_UNITS]
    trend["growth"] = trend["recent_units"] / trend["prior_units"] - 1
    trend = trend.replace([float("inf")], pd.NA).dropna().sort_values("growth", ascending=False)
    round_numeric(trend.rename_axis("product_type_name").reset_index(), 4) \
        .to_csv(out / "hm_trends.csv", index=False)

    forecasts, metrics = [], []
    for style in top[:TOP_N_FORECAST]:
        s = wide[style].astype(float)
        train, test = s.iloc[:-HOLDOUT_WEEKS], s.iloc[-HOLDOUT_WEEKS:]
        for name, fc in [("seasonal_naive", seasonal_naive(train, test.index)),
                         ("prophet", prophet_forecast(train, test.index))]:
            fc = fc.reindex(test.index)
            forecasts.append(pd.DataFrame({
                "week": test.index, "product_type_name": style, "actual": test.values,
                "yhat": fc["yhat"].values, "yhat_lower": fc["yhat_lower"].values,
                "yhat_upper": fc["yhat_upper"].values, "model": name}))
            metrics.append({"product_type_name": style, "model": name,
                            "wape": round(wape(test.values, fc["yhat"].values), 2),
                            "rmse": round(rmse(test.values, fc["yhat"].values), 1),
                            "n_test": len(test)})
    round_numeric(pd.concat(forecasts)).to_csv(out / "hm_style_forecasts.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "hm_style_metrics.csv", index=False)

    colour = _query(con, """
        SELECT strftime(t_dat, '%Y-%m') AS month, colour_group_name, count(*) AS units
        FROM t GROUP BY 1, 2 ORDER BY 1""")
    colour["share"] = colour["units"] / colour.groupby("month")["units"].transform("sum")
    round_numeric(colour, 5).to_csv(out / "hm_colour_month.csv", index=False)

    _query(con, """SELECT week, index_group_name, count(*) AS units FROM t
                   GROUP BY 1, 2 ORDER BY 1""").to_csv(out / "hm_group_weekly.csv", index=False)

    m = metrics.groupby("model")["wape"].mean()
    print(f"wrote {out}/hm_style_summary.csv  ({len(summary)} styles, "
          f"{int(summary['units'].sum()):,} purchases, {len(weeks)} weeks)")
    print(f"  top style: {summary.iloc[0]['product_type_name']} "
          f"({summary.iloc[0]['unit_share']:.1%} of purchases)")
    if len(trend):
        print(f"  fastest riser: {trend.index[0]} ({trend['growth'].iloc[0]:+.0%}); "
              f"fastest faller: {trend.index[-1]} ({trend['growth'].iloc[-1]:+.0%})")
    print(f"  style forecasts (top {TOP_N_FORECAST}): mean WAPE "
          f"seasonal_naive={m['seasonal_naive']:.1f}%  prophet={m['prophet']:.1f}%")
    return True


if __name__ == "__main__":
    ok = run(*(Path(a) for a in sys.argv[1:3]))
    sys.exit(0)
