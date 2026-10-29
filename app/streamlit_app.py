"""Apparel & Sporting Goods Supply Chain Dashboard.

Reads only pipeline outputs (outputs/*.csv) — no models run in the app, so it
loads instantly. Run the pipeline first: python run_pipeline.py

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Reference data-viz palette (validated; categorical order is fixed).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#b9b8b2"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e8e7e3"
STATUS = {"high": "#d03b3b"}  # reserved critical red; size encodes severity
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

st.set_page_config(page_title="Supply Chain Dashboard", page_icon="📦", layout="wide")


@st.cache_data
def load(name: str, dates: tuple[str, ...] = ("week",)) -> pd.DataFrame | None:
    path = OUT / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def style(fig: go.Figure, title: str = "", height: int = 380, ytitle: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)), height=height,
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        font=dict(color=INK_2, size=12), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        hovermode="x unified", bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_text=ytitle)
    return fig


def short(name: str, n: int = 34) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


# ---------------------------------------------------------------- data
brands = load("brand_summary.csv", ())
products = load("product_summary.csv", ())
brand_weekly = load("brand_weekly.csv")
style_weekly = load("style_weekly.csv")
demand_fit = load("demand_fit.csv")
cat_fc = load("forecasts.csv")
cat_metrics = load("metrics.csv", ())
prod_weekly = load("product_weekly.csv")
prod_fc = load("product_forecasts.csv")
prod_metrics = load("product_metrics.csv", ())
outlook = load("product_outlook.csv")
anomalies = load("anomalies.csv")
late_anoms = load("fulfillment_anomalies.csv")
policy = load("inventory_policy.csv", ())
ship = load("shipping_scorecard.csv", ())
ship_mkt = load("shipping_by_market.csv", ())
hm_top = load("hm_style_summary.csv", ())

if brands is None or policy is None:
    st.error("Pipeline outputs missing — run `python run_pipeline.py` first.")
    st.stop()

st.title("Supply Chain Demand & Logistics Dashboard")
st.caption("DataCo Smart Supply Chain (CC BY 4.0) · 180K order lines · stable catalog "
           "Jan 2015 – Sep 2017 · Prophet forecasting · ABC-XYZ inventory policy"
           + (" · H&M style trends" if hm_top is not None else ""))

tabs = st.tabs(["Overview", "Brands & Products", "Demand Forecast",
                "Inventory & Logistics"] + (["Style Trends (H&M)"] if hm_top is not None else []))

# ---------------------------------------------------------------- overview
with tabs[0]:
    nike = brands.set_index("brand").loc["Nike"]
    cm = cat_metrics.groupby("model")["mape"].mean()
    pm = prod_metrics.groupby("model")["wape"].mean()
    late = (ship["late_rate"] * ship["order_lines"]).sum() / ship["order_lines"].sum()
    saved = policy["ss_saved_dollars"].sum()

    c = st.columns(5)
    c[0].metric("Units sold", f"{brands['units'].sum() / 1e3:,.0f}K")
    c[1].metric("Revenue", f"${brands['revenue'].sum() / 1e6:,.1f}M")
    c[2].metric("Nike share of units", f"{nike['unit_share']:.0%}", "#1 brand", delta_color="off")
    c[3].metric("Category forecast MAPE", f"{cm['prophet']:.1f}%",
                f"{cm['prophet'] - cm['seasonal_naive']:+.1f} pts vs naive", delta_color="inverse")
    c[4].metric("Late deliveries", f"{late:.0%}", "promise problem", delta_color="off")

    st.markdown("#### What the data says")
    fc_mode = ship.set_index("shipping_mode")
    st.markdown(
        f"- **Nike is the #1 brand** — {nike['unit_share']:.0%} of all units across "
        f"{int(nike['products'])} products; the Dri-FIT Victory Golf Polo alone is the "
        "#2 product in the catalog.\n"
        f"- **First Class ships late on {fc_mode.loc['First Class', 'late_rate']:.0%} of orders** "
        f"and Second Class takes {fc_mode.loc['Second Class', 'actual_days']:.0f} days — "
        "the same as Standard. Re-quoting delivery promises fixes most lateness for free.\n"
        f"- **Prophet beats a seasonal-naive forecast** ({cm['prophet']:.1f}% vs "
        f"{cm['seasonal_naive']:.1f}% MAPE by category; {pm['prophet']:.1f}% vs "
        f"{pm['seasonal_naive']:.1f}% WAPE by product), which frees **${saved:,.0f}** "
        "of safety stock at the same service level.\n"
        f"- **{(policy['abc'] == 'A').sum()} products drive "
        f"{policy.loc[policy['abc'] == 'A', 'revenue_share'].sum():.0%} of revenue** — "
        "that's where planning effort belongs.")

    total = demand_fit.groupby("week")["actual"].sum()
    fig = go.Figure(go.Scatter(x=total.index, y=total.values, mode="lines",
                               line=dict(color=SERIES[0], width=2), name="Units",
                               hovertemplate="%{y:,.0f} units<extra></extra>"))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(style(fig, "Weekly units, top 8 categories", 320, "Units / week"),
                    use_container_width=True)

# ---------------------------------------------------------------- brands
with tabs[1]:
    left, right = st.columns(2)
    top = brands.head(10).iloc[::-1]
    colors = [SERIES[0] if b == "Nike" else MUTED for b in top["brand"]]
    fig = go.Figure(go.Bar(x=top["units"], y=top["brand"], orientation="h",
                           marker=dict(color=colors, cornerradius=4),
                           text=[f"{s:.0%}" for s in top["unit_share"]],
                           textposition="outside",
                           hovertemplate="%{y}: %{x:,.0f} units<extra></extra>"))
    left.plotly_chart(style(fig, "Top 10 brands by units (Nike highlighted)", 420),
                      use_container_width=True)

    bw = brand_weekly[brand_weekly["brand"].isin(brands["brand"].head(3))]
    fig = go.Figure()
    for i, b in enumerate(brands["brand"].head(3)):
        s = bw[bw["brand"] == b].set_index("week")["units"].rolling(4).mean()
        fig.add_scatter(x=s.index, y=s.values, mode="lines", name=b,
                        line=dict(color=SERIES[i], width=2))
    right.plotly_chart(style(fig, "Top 3 brands, weekly units (4-wk avg)", 420, "Units / week"),
                       use_container_width=True)

    st.markdown("#### Most purchased products")
    brand_pick = st.selectbox("Brand", ["All"] + list(brands["brand"].head(10)), index=0)
    pv = products if brand_pick == "All" else products[products["brand"] == brand_pick]
    pv = pv.head(15)
    fig = go.Figure(go.Bar(
        x=pv["units"][::-1], y=[short(n) for n in pv["product_name"][::-1]], orientation="h",
        marker=dict(color=[SERIES[0] if b == "Nike" else MUTED for b in pv["brand"][::-1]],
                    cornerradius=4),
        customdata=pv[["brand", "style", "revenue", "margin"]][::-1].values,
        hovertemplate="%{y}<br>%{customdata[0]} · %{customdata[1]}<br>%{x:,.0f} units · "
                      "$%{customdata[2]:,.0f} · %{customdata[3]:.1%} margin<extra></extra>"))
    st.plotly_chart(style(fig, "", max(260, 32 * len(pv))), use_container_width=True)

    st.markdown("#### Popular styles")
    seg = st.radio("Segment", ["All", "Men", "Women", "Kids", "Unisex"], horizontal=True)
    sw = style_weekly if seg == "All" else style_weekly[style_weekly["segment"] == seg]
    by_style = sw.groupby("style")["units"].sum().nlargest(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=by_style.values, y=by_style.index, orientation="h",
                           marker=dict(color=SERIES[0], cornerradius=4),
                           hovertemplate="%{y}: %{x:,.0f} units<extra></extra>"))
    st.plotly_chart(style(fig, f"Units by style — {seg}", 380), use_container_width=True)
    with st.expander("Product table"):
        st.dataframe(products, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- forecast
with tabs[2]:
    level = st.radio("Forecast level", ["Category", "Product"], horizontal=True)
    if level == "Category":
        item = st.selectbox("Category", sorted(demand_fit["category"].unique()))
        hist = demand_fit[demand_fit["category"] == item].set_index("week")["actual"]
        fc = cat_fc[cat_fc["category"] == item]
        met = cat_metrics[cat_metrics["category"] == item].set_index("model")["mape"]
        metric_name, fut = "MAPE", None
        an = anomalies[(anomalies["category"] == item) & anomalies["severity"].isin(["high", "medium"])]
    else:
        item = st.selectbox("Product", sorted(prod_weekly["product_name"].unique()))
        hist = prod_weekly[prod_weekly["product_name"] == item].set_index("week")["units"]
        fc = prod_fc[prod_fc["product_name"] == item]
        met = prod_metrics[prod_metrics["product_name"] == item].set_index("model")["wape"]
        metric_name, an = "WAPE", None
        fut = outlook[outlook["product_name"] == item]

    p, n = fc[fc["model"] == "prophet"], fc[fc["model"] == "seasonal_naive"]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="Actual",
                    line=dict(color=INK_2, width=1.5))
    fig.add_scatter(x=list(p["week"]) + list(p["week"][::-1]),
                    y=list(p["yhat_upper"]) + list(p["yhat_lower"][::-1]),
                    fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                    name="Prophet 95% band", hoverinfo="skip")
    fig.add_scatter(x=p["week"], y=p["yhat"], mode="lines", name="Prophet (holdout)",
                    line=dict(color=SERIES[0], width=2))
    fig.add_scatter(x=n["week"], y=n["yhat"], mode="lines", name="Seasonal naive",
                    line=dict(color=SERIES[1], width=2, dash="dot"))
    if fut is not None and len(fut):
        fig.add_scatter(x=list(fut["week"]) + list(fut["week"][::-1]),
                        y=list(fut["yhat_upper"]) + list(fut["yhat_lower"][::-1]),
                        fill="toself", fillcolor="rgba(27,175,122,0.15)", line=dict(width=0),
                        name="Outlook band", hoverinfo="skip")
        fig.add_scatter(x=fut["week"], y=fut["yhat"], mode="lines", name="12-wk outlook",
                        line=dict(color=SERIES[2], width=2))
    if an is not None and len(an):
        fig.add_scatter(x=an["week"], y=an["actual"], mode="markers", name="Anomaly (med/high)",
                        marker=dict(size=[12 if s == "high" else 9 for s in an["severity"]],
                                    color=STATUS["high"],
                                    symbol="diamond", line=dict(color="#fcfcfb", width=2)),
                        customdata=an[["direction", "severity", "z"]].values,
                        hovertemplate="%{customdata[0]} · %{customdata[1]} · z=%{customdata[2]}"
                                      "<extra></extra>")
    end = (fut["week"].max() if fut is not None and len(fut) else hist.index.max())
    fig.update_xaxes(range=[hist.index.max() - pd.Timedelta(weeks=52), end + pd.Timedelta(days=4)])
    st.plotly_chart(style(fig, f"{item}: weekly units (last year; drag to zoom out)", 460,
                          "Units / week"),
                    use_container_width=True)

    c = st.columns(3)
    c[0].metric(f"Prophet {metric_name}", f"{met['prophet']:.1f}%")
    c[1].metric(f"Seasonal-naive {metric_name}", f"{met['seasonal_naive']:.1f}%")
    winner = "Prophet" if met["prophet"] < met["seasonal_naive"] else "Seasonal naive"
    c[2].metric("Use for this item", winner)
    st.caption("Holdout = last 12 weeks, unseen by the model. Anomaly diamonds are in-sample "
               "95% interval breaches (medium/high severity only — low flags are within what a "
               "95% band produces by chance).")

    with st.expander("All forecast metrics"):
        a, b = st.columns(2)
        a.dataframe(cat_metrics.pivot(index="category", columns="model", values="mape"),
                    use_container_width=True)
        b.dataframe(prod_metrics.pivot(index="product_name", columns="model", values="wape"),
                    use_container_width=True)

# ---------------------------------------------------------------- inventory
with tabs[3]:
    st.markdown("#### Shipping promises vs reality")
    left, right = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=ship["shipping_mode"], y=ship["promised_days"], name="Promised days",
                marker=dict(color=MUTED, cornerradius=4))
    fig.add_bar(x=ship["shipping_mode"], y=ship["actual_days"], name="Actual days",
                marker=dict(color=SERIES[0], cornerradius=4))
    left.plotly_chart(style(fig, "Promised vs actual delivery days", 360, "Days"),
                      use_container_width=True)
    fig = go.Figure(go.Bar(x=ship["shipping_mode"], y=ship["late_rate"],
                           marker=dict(color=SERIES[1], cornerradius=4),
                           text=[f"{v:.0%}" for v in ship["late_rate"]], textposition="outside",
                           hovertemplate="%{x}: %{y:.0%} late<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.12])
    right.plotly_chart(style(fig, "Late-delivery rate by mode", 360), use_container_width=True)
    with st.expander("Late rate by mode × market"):
        st.dataframe(ship_mkt.pivot(index="shipping_mode", columns="market", values="late_rate")
                     .style.format("{:.0%}"), use_container_width=True)

    st.markdown("#### ABC-XYZ classification")
    left, right = st.columns([1, 2])
    grid = (policy.pivot_table(index="abc", columns="xyz", values="product_name",
                               aggfunc="count", fill_value=0)
                  .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0))
    rev = (policy.pivot_table(index="abc", columns="xyz", values="revenue_share",
                              aggfunc="sum", fill_value=0)
                 .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0))
    fig = go.Figure(go.Heatmap(
        z=rev.values, x=["X steady", "Y variable", "Z erratic"],
        y=["A (80% rev)", "B (15%)", "C (5%)"], colorscale=SEQ, showscale=False, xgap=2, ygap=2,
        text=[[f"{int(grid.values[i][j])} SKUs<br>{rev.values[i][j]:.0%} rev"
               for j in range(3)] for i in range(3)],
        texttemplate="%{text}", hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    left.plotly_chart(style(fig, "Products and revenue by class", 340), use_container_width=True)
    right.markdown(
        "- **AX** — high value, steady: forecast-driven weekly replenishment, 98% service.\n"
        "- **BX/BY** — mid value: automate reorders at ROP, review every 2 weeks.\n"
        "- **CY/CZ** — low value, erratic: lean stock, make-to-order or review for delisting.\n\n"
        "Safety stock is sized on the **forecast error** (Prophet holdout RMSE) for forecasted "
        "SKUs. Assumptions: 2-week supplier lead time, $50/order, 25%/yr holding cost.")

    st.markdown("#### Ordering policy")
    b = st.selectbox("Filter by brand", ["Nike", "All"] + sorted(policy["brand"].unique()))
    view = policy if b == "All" else policy[policy["brand"] == b]
    cols = ["product_name", "class", "service_level", "weekly_demand", "safety_stock",
            "reorder_point", "eoq", "order_every_weeks", "ss_saved_dollars", "action"]
    st.dataframe(view[cols].sort_values("weekly_demand", ascending=False), hide_index=True,
                 use_container_width=True,
                 column_config={"ss_saved_dollars": st.column_config.NumberColumn(
                     "SS $ freed by forecast", format="$%.0f"),
                     "order_every_weeks": st.column_config.NumberColumn(format="%.1f")})

    st.markdown("#### Fulfillment alerts")
    st.caption("Region-weeks where late-delivery rate jumped >2σ above its trailing 8-week mean.")
    st.dataframe(late_anoms.sort_values("z", ascending=False).assign(
                     week=late_anoms["week"].dt.date), hide_index=True,
                 use_container_width=True,
                 column_config={"late_rate": st.column_config.NumberColumn(format="percent"),
                                "baseline": st.column_config.NumberColumn(format="percent")})

    rec = OUT / "recommendations.md"
    if rec.exists():
        with st.expander("Full recommendations memo", expanded=False):
            st.markdown(rec.read_text())

# ---------------------------------------------------------------- H&M styles
if hm_top is not None:
    with tabs[4]:
        hm_weekly = load("hm_style_weekly.csv")
        hm_trend = load("hm_trends.csv", ())
        hm_fc = load("hm_style_forecasts.csv")
        hm_metrics = load("hm_style_metrics.csv", ())
        hm_colour = load("hm_colour_month.csv", ())

        st.caption("H&M Personalized Fashion Recommendations (Kaggle) — real fashion "
                   "transactions, aggregated to style level. Raw data is not redistributed.")
        left, right = st.columns(2)
        t = hm_top.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=t["units"], y=t["product_type_name"], orientation="h",
                               marker=dict(color=SERIES[0], cornerradius=4),
                               hovertemplate="%{y}: %{x:,.0f} purchases<extra></extra>"))
        left.plotly_chart(style(fig, "Most purchased styles", 460), use_container_width=True)
        if hm_trend is not None:
            tr = pd.concat([hm_trend.nlargest(8, "growth"), hm_trend.nsmallest(8, "growth")])
            tr = tr.sort_values("growth")
            fig = go.Figure(go.Bar(
                x=tr["growth"], y=tr["product_type_name"], orientation="h",
                marker=dict(color=[SERIES[2] if g > 0 else SERIES[1] for g in tr["growth"]],
                            cornerradius=4),
                hovertemplate="%{y}: %{x:+.0%} (last 8 wks vs prior 8)<extra></extra>"))
            fig.update_xaxes(tickformat="+.0%")
            right.plotly_chart(style(fig, "Rising and falling styles (8-wk momentum)", 460),
                               use_container_width=True)

        if hm_fc is not None:
            pick = st.selectbox("Style forecast", sorted(hm_fc["product_type_name"].unique()))
            h = hm_weekly[hm_weekly["product_type_name"] == pick].set_index("week")["units"]
            f = hm_fc[(hm_fc["product_type_name"] == pick)]
            fp, fn = f[f["model"] == "prophet"], f[f["model"] == "seasonal_naive"]
            fig = go.Figure()
            fig.add_scatter(x=h.index, y=h.values, mode="lines", name="Actual",
                            line=dict(color=INK_2, width=1.5))
            fig.add_scatter(x=list(fp["week"]) + list(fp["week"][::-1]),
                            y=list(fp["yhat_upper"]) + list(fp["yhat_lower"][::-1]),
                            fill="toself", fillcolor="rgba(42,120,214,0.15)",
                            line=dict(width=0), name="Prophet 95% band", hoverinfo="skip")
            fig.add_scatter(x=fp["week"], y=fp["yhat"], mode="lines", name="Prophet",
                            line=dict(color=SERIES[0], width=2))
            fig.add_scatter(x=fn["week"], y=fn["yhat"], mode="lines", name="Naive",
                            line=dict(color=SERIES[1], width=2, dash="dot"))
            st.plotly_chart(style(fig, f"{pick}: weekly purchases", 420, "Purchases / week"),
                            use_container_width=True)
            if hm_metrics is not None:
                with st.expander("Style forecast accuracy (WAPE, 8-week holdout)"):
                    st.dataframe(hm_metrics.pivot(index="product_type_name", columns="model",
                                                  values="wape"), use_container_width=True)

        if hm_colour is not None:
            top_c = hm_colour.groupby("colour_group_name")["units"].sum().nlargest(8).index
            hc = hm_colour[hm_colour["colour_group_name"].isin(top_c)]
            share = hc.pivot_table(index="colour_group_name", columns="month",
                                   values="share", aggfunc="sum").fillna(0)
            fig = go.Figure(go.Heatmap(z=share.values, x=share.columns.astype(str),
                                       y=share.index, colorscale=SEQ, xgap=2, ygap=2,
                                       colorbar=dict(tickformat=".0%"),
                                       hovertemplate="%{y} · %{x}: %{z:.1%}<extra></extra>"))
            st.plotly_chart(style(fig, "Colour share of purchases by month", 380),
                            use_container_width=True)
