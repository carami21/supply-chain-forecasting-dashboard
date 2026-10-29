# Tableau Public Build Guide

Rebuild the Streamlit dashboard in Tableau Public (free) for the portfolio link.
Every number comes from `tableau/*.csv`, so run `python run_pipeline.py` first.
Budget: ~2 hours for all four dashboards.

**Colors** (match the Streamlit app): primary `#2a78d6`, second series `#eb6834`,
third `#1baf7a`, muted/other `#b9b8b2`, anomaly `#d03b3b`, background `#fcfcfb`.

## 0. Connect the data
1. Tableau Public → **Connect → Text file** → `tableau/demand_fit.csv`.
2. Add the other CSVs as **separate data sources** (Data → New Data Source).
   Don't join them; each sheet uses one source.
3. For every `week` field: right-click → **Change Data Type → Date**.

## 1. Overview dashboard
| Sheet | Source | Build |
|---|---|---|
| KPI: Units | `brands.csv` | `SUM(units)` on Text. Format as `#,##0,K`. |
| KPI: Nike share | `brands.csv` | Filter brand = Nike; `SUM(unit_share)` on Text, % format. |
| KPI: Forecast MAPE | `forecast_accuracy.csv` | Filter level = Category; `AVG(error_pct)` on Text; `model` on Columns. |
| KPI: Late rate | `shipping.csv` | Calculated field `SUM([late_rate]*[order_lines])/SUM([order_lines])`. |
| Weekly demand | `demand_fit.csv` | `week` (exact week, continuous) → Columns; `SUM(actual)` → Rows; line. Axis: include zero. |

Dashboard: KPIs in a horizontal container across the top, line chart below.

## 2. Brands & Products
| Sheet | Source | Build |
|---|---|---|
| Top brands | `brands.csv` | `brand` → Rows, `SUM(units)` → Columns, sort desc, Top 10 filter. Color: calc `IF [brand]="Nike" THEN "Nike" ELSE "Other" END` → Nike `#2a78d6`, Other `#b9b8b2`. Label `unit_share`. |
| Brand trend | `brand_weekly.csv` | `week` → Columns, `SUM(units)` → Rows, `brand` → Color; filter to Nike / Perfect Fitness / O'Brien. Quick table calc: moving average, 4 weeks. |
| Top products | `products.csv` | `product_name` → Rows, `SUM(units)` → Columns, Top 15, brand-highlight color as above; tooltip: brand, style, revenue, margin. |
| Styles | `style_weekly.csv` | `style` → Rows, `SUM(units)` → Columns; `segment` as a **filter** (show as single-value list). |

Add `brand` from `products.csv` as a dashboard filter applied to Top products.

## 3. Demand Forecast
| Sheet | Source | Build |
|---|---|---|
| Forecast vs actual | `forecast_holdout.csv` | `week` → Columns; Measure Values (`actual`, `yhat`) → Rows; `model` → Color. Filters: `level`, `item` (single value dropdown). |
| Band | same sheet | Dual-axis is **not** used — instead add `yhat_lower`/`yhat_upper` as a **reference band** (Analytics → Reference Band, per cell). |
| Anomalies | `demand_fit.csv` | `week` → Columns, `SUM(actual)` → Rows (line) + second mark: filter `severity` in (high, medium) as red `#d03b3b` diamonds. Filter `category`. |
| Accuracy table | `forecast_accuracy.csv` | `item` → Rows, `model` → Columns, `AVG(error_pct)` → Text; highlight the lower value per row. |

## 4. Inventory & Logistics
| Sheet | Source | Build |
|---|---|---|
| Promised vs actual | `shipping.csv` | `shipping_mode` → Columns; Measure Names/Values with `AVG(promised_days)`, `AVG(actual_days)` side-by-side bars (promised `#b9b8b2`, actual `#2a78d6`). |
| Late rate | `shipping.csv` | `shipping_mode` → Columns, weighted late rate (calc from §1) → Rows, label %. Add `market` as a filter. |
| ABC-XYZ matrix | `inventory_policy.csv` | `abc` → Rows, `xyz` → Columns, `SUM(revenue_share)` → Color (blue sequential), `CNT(product_name)` + `SUM(revenue_share)` → Label. Square marks. |
| Ordering policy | `inventory_policy.csv` | Text table: product_name, class, service_level, weekly_demand, safety_stock, reorder_point, eoq, order_every_weeks, action. Filter `brand` (default Nike). |
| Fulfillment alerts | `fulfillment_alerts.csv` | Text table sorted by `z` desc. |

## 5. Style Trends (only if the H&M stage ran)
| Sheet | Source | Build |
|---|---|---|
| Top styles | `hm_style_summary.csv` | `product_type_name` → Rows, `SUM(units)` → Columns, Top 15. |
| Momentum | `hm_trends.csv` | `product_type_name` → Rows, `growth` → Columns; color calc `[growth] > 0` (green `#1baf7a` / orange `#eb6834`). Top 8 + bottom 8. |
| Style forecast | `hm_style_forecasts.csv` | Same as §3 forecast sheet with `product_type_name` filter. |
| Colour seasonality | `hm_colour_month.csv` | `month` → Columns, `colour_group_name` → Rows (top 8), `SUM(share)` → Color, square marks. |

## Publish
File → **Save to Tableau Public** → name it "Supply Chain Demand & Logistics Dashboard".
Copy the public URL into the README (`Live dashboard` line) and onto the resume.
