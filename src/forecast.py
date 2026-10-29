"""Stage 3 — forecast weekly demand per category.

Two models, one honest comparison:

  seasonal_naive : yhat(t) = actual(t - 52 weeks). The benchmark every real
                   forecast must beat; ~1 year of seasonality, no fitting.
  prophet        : trend + yearly seasonality with native prediction
                   intervals (yhat_lower/upper) that the anomaly stage reuses.

Each category is scored on a 12-week holdout (the last 12 weeks), reporting
MAPE and RMSE per model. The whole point of the baseline is the headline
metric: how much Prophet beats seasonal-naive.

Run: python src/forecast.py
In:  data/processed/demand_weekly.csv
Out: outputs/forecasts.csv  (week, category, actual, yhat, yhat_lower, yhat_upper, model)
     outputs/metrics.csv     (category, model, mape, rmse, n_test)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Prophet/cmdstanpy are noisy; quiet them before import.
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.simplefilter("ignore", FutureWarning)
from prophet import Prophet  # noqa: E402

SRC = Path("data/processed/demand_weekly.csv")
FORECAST_OUT = Path("outputs/forecasts.csv")
METRICS_OUT = Path("outputs/metrics.csv")

HOLDOUT_WEEKS = 12
SEASON_WEEKS = 52  # yearly seasonality for the naive baseline
SEED = 42


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def seasonal_naive(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forecast each test week with the value 52 weeks earlier."""
    history = train.copy()
    preds = []
    for wk in test_index:
        ref = wk - pd.Timedelta(weeks=SEASON_WEEKS)
        preds.append(history.get(ref, history.iloc[-1]))  # fall back to last value
    return pd.DataFrame({"yhat": preds, "yhat_lower": np.nan, "yhat_upper": np.nan},
                        index=test_index)


def prophet_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.DataFrame:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, interval_width=0.95)
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    future = pd.DataFrame({"ds": test_index})
    np.random.seed(SEED)  # interval bounds are sampled; seed for reproducible runs
    fc = m.predict(future).set_index("ds")
    return fc[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} — run src/aggregate.py first")

    demand = pd.read_csv(SRC, parse_dates=["week"])
    forecasts, metrics = [], []

    for cat, g in demand.groupby("category"):
        s = g.set_index("week")["units"].sort_index()
        train, test = s.iloc[:-HOLDOUT_WEEKS], s.iloc[-HOLDOUT_WEEKS:]

        for name, fc in [
            ("seasonal_naive", seasonal_naive(train, test.index)),
            ("prophet", prophet_forecast(train, test.index)),
        ]:
            block = fc.reindex(test.index)
            forecasts.append(pd.DataFrame({
                "week": test.index, "category": cat, "actual": test.values,
                "yhat": block["yhat"].values,
                "yhat_lower": block["yhat_lower"].values,
                "yhat_upper": block["yhat_upper"].values,
                "model": name,
            }))
            metrics.append({
                "category": cat, "model": name,
                "mape": round(mape(test.values, block["yhat"].values), 2),
                "rmse": round(rmse(test.values, block["yhat"].values), 1),
                "n_test": len(test),
            })

    forecasts = pd.concat(forecasts, ignore_index=True)
    metrics = pd.DataFrame(metrics)

    FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(FORECAST_OUT, index=False)
    metrics.to_csv(METRICS_OUT, index=False)

    # headline: mean MAPE per model and Prophet's improvement
    summary = metrics.groupby("model")["mape"].mean()
    naive, prop = summary["seasonal_naive"], summary["prophet"]
    lift = (naive - prop) / naive * 100
    print(f"wrote {FORECAST_OUT}  ({len(forecasts):,} rows)")
    print(f"wrote {METRICS_OUT}")
    print(f"  mean MAPE  seasonal_naive={naive:.1f}%  prophet={prop:.1f}%")
    print(f"  Prophet beats naive by {lift:.0f}% MAPE "
          f"({'better' if lift > 0 else 'WORSE — investigate'})")
    return forecasts, metrics


if __name__ == "__main__":
    run()
