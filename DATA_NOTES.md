# Data Notes — DataCo Smart Supply Chain

Findings from EDA (`notebooks/01_eda.ipynb`) and aggregation that shape how
the pipeline treats the data. Kept separate from the README so the modeling
decisions are auditable.

## Dataset

- **180,519 order lines × 53 columns**, order dates **2015-01-01 → 2018-01-31**.
- Exceptionally clean: **0 duplicate rows, 0 null/negative quantities, 0 null
  dates**. ETL cleaning is therefore light (column select + date parse).
- 50 product categories; the **top 10 hold 89%** of all units.
- 5 markets, 23 order regions. **54.8% late-delivery rate** overall — strong
  signal for the fulfillment-anomaly layer.
- ~4.3% of orders are `Shipping canceled`. These are **kept as demand** (a
  planner still faced the order) but **excluded from fulfillment metrics**
  (they never shipped, so timeliness is undefined). Flagged in ETL as
  `is_canceled`.

## Catalog changeover (the load-bearing finding)

The product catalog **shifts hard around October 2017**. The highest-volume
categories are sporting goods — Cleats, Women's Apparel, Indoor/Outdoor Games,
Cardio Equipment, Shop By Sport, Men's Footwear, Fishing, Water Sports — which
sell **steadily at ~2,200 units/week through 2017-09-26, then stop dead**.
The final ~4 months (Oct 2017 – Jan 2018) are carried by an entirely different
catalog: Video Games, Children's Clothing, Cameras, Toys, Pet Supplies.

Weekly top-8 volume across the transition:

```
2017-09-05  2220
2017-09-12  2106
2017-09-19  2261
2017-09-26  2035   <- last active week
(after)        0   <- catalog replaced
```

This is a hard changeover, not a decline. **Decision:** forecast the *stable
regime*. `aggregate.py` trims the analysis to the longest contiguous run of
weeks where the modeled categories are active (`ACTIVE_THRESHOLD = 0.5 ×
median`), yielding **143 of 160 complete weeks: 2015-01-06 → 2017-09-26**.
The post-changeover catalog is only ~16 weeks — too short to establish yearly
seasonality — so it is out of scope rather than modeled separately.

The trim is volume-driven, not a hard-coded date, so it stays correct if the
data is revised. If a future version wants the later catalog, model it as a
second regime; do not merge the two — they are different product mixes.

## Weekly aggregation

- Weeks are **Monday-start (`W-MON`)**.
- The first and last calendar weeks of the raw data are partial (data begins
  Thursday 2015-01-01, ends Wednesday 2018-01-31); partial boundary weeks are
  dropped so they don't read as demand cliffs.
- Demand series: **top 8 categories × 143 weeks** (`demand_weekly.csv`).
- Fulfillment series: **week × region** late-rate and avg shipping delay,
  canceled orders excluded (`fulfillment_weekly.csv`).

## Brand / product layer (added 2026-10-28)

- DataCo product names carry real brands. Brand = first word, except a
  whitelist of multi-word brands (Under Armour, Field & Stream, ...) in
  `etl.py`. **Nike is #1: 126.5K units, 33.8% of the stable window, 13 products.**
- Source product names are **truncated at 45 characters** ("... Comfort Bi",
  "... Spine Surge Runni"), so style keywords in `etl.py` include fragments.
- Volume is extremely concentrated: each top category is effectively one
  product (e.g. Women's Apparel ≈ Nike Dri-FIT polo, Fishing ≈ Field & Stream
  gun safe — DataCo's category labels are loose). Product- and category-level
  forecasts for those pairs therefore match closely; that is expected.
- 9 products sell in ≥95% of weeks and are forecast; the long tail is sparse
  (zero-sale weeks), which is why product accuracy is reported as **WAPE**.

## Shipping (added 2026-10-28)

- Delivery promise vs actual is deterministic by mode: First Class always
  takes 2 days against a 1-day promise (100% late); Second Class takes ~4
  days against 2 (80% late), identical to Standard's ~4 days. Late rate is
  ~57% in every market, so lateness is driven by the quoted days, not geography.
