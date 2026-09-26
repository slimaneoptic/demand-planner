"""Demand forecast & reorder planner: forecasts every product, then says what to order this week."""
from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data import DataError, default_settings, future_promos, load_upload, sample_history, sample_settings, template_csv
from export import workbook
from forecast import HORIZON, MODELS, accuracy_table, forecast_all
from inventory import SETTINGS_COLUMNS, plan, plan_table, projection

# ---- edit these two lines -------------------------------------------------
AUTHOR = "Slimane"
HIRE_URL = "https://www.upwork.com/freelancers/~01d6ae0f1a06a43ac0"
# ---------------------------------------------------------------------------

INK, MUTED = "#14202B", "#4B5866"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"  # validated palette, slots 1-3
STATUS_ICON = {"Order now": "🔴 Order now", "Cut by budget": "⚠️ Cut by budget", "Order soon": "🟠 Order soon",
               "OK": "🟢 OK", "Overstocked": "🔵 Overstocked", "No demand": "⚪ No demand"}

st.set_page_config(page_title="Demand forecast & reorder planner", page_icon="📦", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1280px;}
    [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
    .note {background:#ECEFF3; border-radius:6px; padding:.7rem .9rem; margin:.3rem 0 .8rem; color:#14202B;}
    .footer {margin-top:2.5rem; padding-top:1rem; border-top:1px solid #D5DBE1; color:#4B5866; font-size:.92rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def run_forecast(history: pd.DataFrame, with_promos: bool):
    start = history["week"].max() + pd.Timedelta(weeks=1)
    promos = future_promos(start, HORIZON) if with_promos else None
    return forecast_all(history, promos)


@st.cache_data(show_spinner=False)
def run_plan(history: pd.DataFrame, with_promos: bool, settings: pd.DataFrame, service: float, budget: float | None):
    return plan(run_forecast(history, with_promos), settings, service, budget)


def money(x: float) -> str:
    return f"${x:,.0f}"


# ---------------------------------------------------------------- data source
if "source" not in st.session_state:
    st.session_state.source = "sample"
    st.session_state.history = sample_history()
    st.session_state.settings = sample_settings(st.session_state.history)

st.title("Demand forecast & reorder planner")
st.markdown(
    f"Forecasts the next {HORIZON} weeks for every product, tests four models on your own sales history and keeps "
    "the most accurate one per product, then tells you **what to order this week**: how much, and within your budget."
)

c1, c2, c3 = st.columns([1.3, 1, 1.2])
source_label = "Sample: coffee equipment shop (8 products, 3 years)" if st.session_state.source == "sample" \
    else f"Your file: {st.session_state.get('file_name', 'uploaded data')}"
c1.markdown(f"**Data**  \n{html.escape(source_label)}  \n<span style='color:{MUTED}'>Load your own in the "
            f"*Your data* tab.</span>", unsafe_allow_html=True)
service = c2.select_slider("Chance of no stock-out before the next delivery", options=[0.90, 0.95, 0.98, 0.99],
                           value=0.95, format_func=lambda v: f"{v:.0%}",
                           help="How often stock should last until the next delivery. Higher means more safety stock.")
budget_in = c3.number_input("Budget for this week's orders ($, 0 = no limit)", min_value=0, value=0, step=500)
budget = float(budget_in) if budget_in > 0 else None

def editor_key() -> str:
    return f"ed_{st.session_state.get('data_version', 0)}"


def effective_settings() -> pd.DataFrame:
    """The stored settings plus any edits made in the table on the Your data tab."""
    base = st.session_state.settings[SETTINGS_COLUMNS].copy().reset_index(drop=True)
    delta = st.session_state.get(editor_key()) or {}
    for i, changes in delta.get("edited_rows", {}).items():
        for col, val in changes.items():
            base.at[int(i), col] = val
    return base


def arrival(product: str) -> int:
    v = settings.set_index("product").loc[product, "arrives_in_weeks"]
    return 1 if pd.isna(v) else int(v)


history: pd.DataFrame = st.session_state.history
settings: pd.DataFrame = effective_settings()
with_promos = st.session_state.source == "sample"
try:
    with st.spinner("Testing 4 models on every product and forecasting…"):
        forecasts = run_forecast(history, with_promos)
        lines, summary = run_plan(history, with_promos, settings, service, budget)
except Exception as exc:  # never leave a visitor on a traceback
    st.error(f"Couldn't forecast this data ({exc}). Back to the sample shop.")
    for k in ("source", "history", "settings"):
        st.session_state.pop(k, None)
    st.button("Reload")
    st.stop()
if st.session_state.get("upload_notes"):
    st.info(" ".join(st.session_state.upload_notes))
if st.session_state.source != "sample" and not st.session_state.get(editor_key()):
    st.warning("Stock, open orders, lead times, costs and prices below are placeholders. "
               "Enter yours in the *Your data* tab to get a real order plan.")
table = plan_table(lines)
acc = accuracy_table(forecasts)

# ---------------------------------------------------------------- headline numbers
to_order = [l for l in lines if l.order_units > 0]
due = [l for l in lines if l.status in ("Order now", "Cut by budget")]
tested = [f for f in forecasts.values() if f.tests and f.wape.get(f.model) == f.wape.get(f.model) and f.naive_wape == f.naive_wape]
kept_err = float(np.average([f.wape[f.model] for f in tested], weights=[max(f.mean.sum(), 1) for f in tested])) if tested else 0.0
naive_err = float(np.average([f.naive_wape for f in tested], weights=[max(f.mean.sum(), 1) for f in tested])) if tested else 0.0
weights = [l.demand_lead + l.demand_cycle for l in lines]
fill_without = float(np.average([l.fill_without for l in lines], weights=weights)) if lines else 1.0
fill_with = float(np.average([l.fill_with for l in lines], weights=weights)) if lines else 1.0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Products to order now", f"{len(due)} of {len(lines)}")
k2.metric("This week's order", money(summary["order_cost"]),
          f"{sum(l.order_units for l in to_order):,} units", delta_color="off", delta_arrow="off")
k3.metric("Forecast error", f"{kept_err:.0%}", f"{(kept_err - naive_err) * 100:+.0f} pts vs copying last year",
          delta_color="inverse", help="Average absolute error of the kept models on weeks we already know "
                                      "(1 to 16 weeks ahead), compared with copying last year's sales.")
k4.metric("Demand served after delivery", f"{fill_with:.0%}", f"{(fill_with - fill_without) * 100:+.0f} pts vs no order",
          help="Expected share of demand served from each delivery until the next regular order arrives, "
               "with this week's orders, compared with ordering nothing.")

if summary["budget_binding"]:
    st.markdown(
        f"<div class='note'><b>Budget limit:</b> the full plan costs {money(summary['recommended_cost'])}, your budget is "
        f"{money(budget)}. The optimizer spent it where each dollar protects the most margin, over 300 demand "
        f"scenarios per product (whole cases and minimum orders respected). Products it left out are the ones whose "
        f"extra sales are worth least per dollar.</div>", unsafe_allow_html=True)

tab_plan, tab_fc, tab_acc, tab_data, tab_how = st.tabs(
    ["Order plan", "Forecast by product", "Model accuracy", "Your data", "How it works"])

# ---------------------------------------------------------------- order plan
with tab_plan:
    show = table.copy()
    show["Status"] = show["Status"].map(STATUS_ICON)
    if not summary["budget_binding"]:
        show = show.drop(columns=["Recommended (units)"])
    show["Runs out"] = [f"{pd.Timestamp(d):%d %b}" if d is not None and not pd.isna(d) else "—" for d in show["Runs out"]]
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "Order cost": st.column_config.NumberColumn(format="dollar"),
            "Order (units)": st.column_config.NumberColumn(format="%d"),
            "Weeks of cover": st.column_config.NumberColumn(format="%.1f"),
            "Served after delivery, no order": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Served after delivery, with order": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Safety stock": st.column_config.NumberColumn(help="Buffer inside the reorder point, above average demand until an order arrives"),
            "Runs out": st.column_config.TextColumn(help="Week the stock runs out if you order nothing new"),
            "Reorder point": st.column_config.NumberColumn(help="Order when stock + open orders fall below this"),
        },
    )
    soon = [l for l in lines if l.status == "Order soon"]
    tips = []
    if due:
        urgent = [l for l in due if l.runs_out is not None and (l.runs_out - forecasts[l.product].weeks[0]).days // 7 < l.lead_weeks]
        if urgent:
            tips.append("**Gap before the order can arrive:** " + "; ".join(
                f"{l.product} runs out the week of {l.runs_out:%d %b}, and an order placed today takes "
                f"{l.lead_weeks} weeks" for l in urgent) + ". Ask the supplier for express delivery or a partial shipment.")
    if soon:
        tips.append("**Within 2 weeks:** " + ", ".join(f"{l.product}" for l in soon) + " will reach the reorder point.")
    over = [l for l in lines if l.status == "Overstocked"]
    if over:
        tips.append("**Overstocked:** " + ", ".join(f"{l.product} ({l.cover_weeks:.0f} weeks of cover)" for l in over)
                    + ". Skip orders and consider a promotion.")
    for tip in tips:
        st.markdown(tip)
    notes = [f"Order plan, week of {forecasts[lines[0].product].weeks[0]:%d %b %Y}" if lines else "Order plan",
             f"Service level {service:.0%}; budget {money(budget) if budget else 'no limit'}; "
             f"order cost {money(summary['order_cost'])}."]
    st.download_button("Download the plan (Excel)", workbook(table, forecasts, acc, settings, notes),
                       "order_plan.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------------------------------------- forecast chart
with tab_fc:
    names = [l.product for l in lines] or list(forecasts)
    pick = st.selectbox("Product", names, index=0)
    f = forecasts[pick]
    line = next(l for l in lines if l.product == pick)
    hist = history[history["product"] == pick].sort_values("week").tail(78)
    hist_df = pd.DataFrame({"week": hist["week"], "units": hist["units"], "series": "Actual sales",
                            "promo": np.where(hist["promo"] > 0, "Promotion week", "")})
    fc_df = pd.DataFrame({"week": f.weeks, "units": f.mean.round(0), "low": f.low.round(0), "high": f.high.round(0),
                          "series": "Forecast"})
    bridge = pd.DataFrame({"week": [hist_df["week"].iloc[-1]], "units": [hist_df["units"].iloc[-1]],
                           "series": ["Forecast"]})
    colors = alt.Scale(domain=["Actual sales", "Forecast"], range=[MUTED, BLUE])
    x = alt.X("week:T", title=None, axis=alt.Axis(format="%b %Y", grid=False))
    band = alt.Chart(fc_df).mark_area(opacity=0.18, color=BLUE).encode(
        x=x, y="low:Q", y2="high:Q",
        tooltip=[alt.Tooltip("week:T", title="Week of", format="%d %b %Y"), alt.Tooltip("units:Q", title="Forecast"),
                 alt.Tooltip("low:Q", title="Low (80 %)"), alt.Tooltip("high:Q", title="High (80 %)")])
    lines_chart = alt.Chart(pd.concat([hist_df, bridge, fc_df], ignore_index=True)).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("units:Q", title="Units per week", axis=alt.Axis(gridColor="#E6EAEE")),
        color=alt.Color("series:N", scale=colors, legend=alt.Legend(title=None, orient="top", labelLimit=0)),
        tooltip=[alt.Tooltip("week:T", title="Week of", format="%d %b %Y"), alt.Tooltip("series:N", title=""),
                 alt.Tooltip("units:Q", title="Units")])
    promo_pts = alt.Chart(hist_df[hist_df["promo"] != ""]).mark_point(filled=True, size=70, color=ORANGE).encode(
        x=x, y="units:Q", tooltip=[alt.Tooltip("week:T", title="Promotion week of", format="%d %b %Y"),
                                   alt.Tooltip("units:Q", title="Units")])
    today = alt.Chart(pd.DataFrame({"week": [f.weeks[0] - pd.Timedelta(days=3)]})).mark_rule(
        strokeDash=[4, 4], color=MUTED).encode(x="week:T")
    st.altair_chart((band + lines_chart + promo_pts + today).properties(height=340), width="stretch")
    err = f.wape.get(f.model, float("nan"))
    if not f.tests:
        verdict = f"**Model: {f.model}.** Too little history to test the models; the range is a default."
    elif f.model == "Seasonal naive":
        verdict = f"**Model kept: same week last year.** It was the most accurate here: {err:.0%} error on past weeks."
    else:
        verdict = (f"**Model kept: {f.model}.** Error on past weeks {err:.0%}, versus {f.naive_wape:.0%} for copying "
                   "last year.")
    st.markdown(verdict + " Shaded area: 80 % range, from how far this model missed in its tests. "
                "Orange dots: promotion weeks.")

    st.subheader("Expected stock")
    base = projection(f, line.on_hand, line.on_order, arrival(pick), 0, line.lead_weeks).assign(plan="If you order nothing")
    with_order = projection(f, line.on_hand, line.on_order, arrival(pick), line.order_units, line.lead_weeks).assign(plan=f"With this week's order ({line.order_units:,} units)")
    proj = pd.concat([base, with_order], ignore_index=True)
    proj["stock"] = proj["stock"].round(0)
    pc = alt.Scale(domain=[base["plan"].iloc[0], with_order["plan"].iloc[0]], range=[ORANGE, GREEN])
    stock_lines = alt.Chart(proj).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30)).encode(
        x=alt.X("week:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
        y=alt.Y("stock:Q", title="Units in stock (end of week)", axis=alt.Axis(gridColor="#E6EAEE")),
        color=alt.Color("plan:N", scale=pc, legend=alt.Legend(title=None, orient="top", labelLimit=0)),
        tooltip=[alt.Tooltip("week:T", title="Week of", format="%d %b %Y"), alt.Tooltip("plan:N", title=""),
                 alt.Tooltip("stock:Q", title="Stock")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
    st.altair_chart((stock_lines + zero).properties(height=260), width="stretch")
    lost = base["lost"].sum()
    st.caption(f"At zero, sales are lost{f' (about {lost:,.0f} units if you order nothing)' if lost >= 1 else ''}. "
               f"Lead time {line.lead_weeks} weeks: an order placed today arrives the week of "
               f"{f.weeks[0] + pd.Timedelta(weeks=line.lead_weeks):%d %b}. Uses the average forecast.")

# ---------------------------------------------------------------- accuracy
with tab_acc:
    n_tests = max((f.tests for f in forecasts.values()), default=0)
    season = all(f.season_tested for f in forecasts.values())
    st.markdown(
        f"Each model forecast weeks we already know ({n_tests} test windows per product"
        + (", including the same season one year earlier, so a model must also handle what's coming: Christmas, summer…"
           if season else "; with less than about 2 years of history the coming season can't be tested, so for seasonal "
                          "products only models that use last year's pattern are allowed")
        + "). The error is the total absolute miss divided by total sales (lower is better). "
          "The best model per product is kept.")
    a = acc.copy()
    for m in MODELS:
        a[m] = (a[m] * 100).round(1)
    st.dataframe(a, hide_index=True, width="stretch",
                 column_config={m: st.column_config.NumberColumn(f"{m} (error %)", format="%.1f") for m in MODELS})
    wins = acc["Model kept"].value_counts()
    st.caption("Kept: " + " · ".join(f"{m} for {n} product{'s' if n > 1 else ''}" for m, n in wins.items()))

# ---------------------------------------------------------------- your data
with tab_data:
    st.markdown(
        "**1. Sales history:** a CSV or Excel file with one row per sale, day or week: **date, product, units** "
        "(names like *sku* or *quantity* also work). An optional **promo** column (0/1) marks promotion weeks. "
        "Each product needs at least 26 weeks; two years or more lets the models learn the seasons.")
    up = st.file_uploader("Sales file", type=["csv", "xlsx"])
    cc1, cc2 = st.columns(2)
    cc1.download_button("Download a template (sample data)", template_csv(), "sales_template.csv", "text/csv")
    if st.session_state.source != "sample" and cc2.button("Back to the sample shop"):
        st.session_state.source = "sample"
        st.session_state.history = sample_history()
        st.session_state.settings = sample_settings(st.session_state.history)
        st.session_state.upload_notes = []
        st.session_state.data_version = st.session_state.get("data_version", 0) + 1
        st.rerun()
    if up is not None and st.session_state.get("file_key") != (up.name, up.size):
        try:
            hist_new, notes = load_upload(up.name, up.getvalue())
        except DataError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Couldn't read this file: {exc}")
        else:
            st.session_state.update(source="upload", history=hist_new, settings=default_settings(hist_new),
                                    file_name=up.name, file_key=(up.name, up.size), upload_notes=notes,
                                    data_version=st.session_state.get("data_version", 0) + 1)
            st.rerun()
    st.markdown(
        "**2. Stock and supplier settings** per product: edit the table, the plan updates. "
        "*Arrives in* is when the open order lands; *order every* is how often you usually order from that supplier.")
    st.data_editor(
        st.session_state.settings[SETTINGS_COLUMNS].reset_index(drop=True), hide_index=True, width="stretch",
        disabled=["product"], key=editor_key(),
        column_config={
            "product": "Product", "on_hand": st.column_config.NumberColumn("On hand", min_value=0, step=1),
            "on_order": st.column_config.NumberColumn("On order", min_value=0, step=1),
            "arrives_in_weeks": st.column_config.NumberColumn("Arrives in (weeks)", min_value=1, max_value=26, step=1),
            "lead_weeks": st.column_config.NumberColumn("Lead time (weeks)", min_value=1, max_value=14, step=1),
            "order_every_weeks": st.column_config.NumberColumn("Order every (weeks)", min_value=1, max_value=12, step=1),
            "unit_cost": st.column_config.NumberColumn("Unit cost ($)", min_value=0.0, format="%.2f"),
            "price": st.column_config.NumberColumn("Price ($)", min_value=0.0, format="%.2f"),
            "case_size": st.column_config.NumberColumn("Case size", min_value=1, step=1),
            "min_cases": st.column_config.NumberColumn("Min. cases", min_value=0, step=1),
        })

# ---------------------------------------------------------------- method
with tab_how:
    st.markdown(f"""
**Forecasts.** Four models per product: *same week last year* (scaled to this year's level), an 8-week *moving average*,
*Holt-Winters* (trend + yearly season) and one *gradient boosting* model trained on all products together
(season, last year's pattern, recent trend, promotions). Each is tested on weeks we already know, and the most accurate
is kept per product. The 80 % range comes from the kept model's own past misses.

**When to order.** You can order any week; an order arrives after the lead time (up to 14 weeks in this demo). A product is due when stock + open
orders won't cover demand until an order placed *next* week could arrive, at your service level.

**How much.** Enough for the demand after the order arrives until the next regular order, minus the stock expected to be
left, at your service level. Sales you can't serve are counted as lost (not delayed), so a late order doesn't
pretend to recover them. Rounded up to whole cases and the supplier's minimum.

**Budget.** When the plan costs more than your budget, an optimizer (Google OR-Tools CP-SAT) picks quantities over
300 demand scenarios per product to protect the most expected margin per dollar.

Client versions connect to Shopify, WooCommerce, an ERP or Google Sheets, handle several warehouses and suppliers,
container and pallet limits, shelf life, and send the weekly plan by email.
""")

st.markdown(
    f'<div class="footer">Built by {AUTHOR}. I build forecasting, planning and optimization tools on your own data: '
    f'your products, your suppliers, running in days. <a href="{HIRE_URL}" target="_blank">Work with me</a></div>',
    unsafe_allow_html=True,
)
