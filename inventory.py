"""Reorder decisions from the forecasts: when to order, how much, and a budget-aware optimizer.

Model: you can place an order any week; it arrives after the lead time and has to last until the next
regular order arrives (the order cycle). Sales you can't serve are lost (not back-ordered), so an order that
arrives after a stock-out can't recover the sales missed before it lands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from forecast import ProductForecast

SCENARIOS = 300          # demand scenarios per product
ERROR_CORRELATION = 0.3  # forecast errors of nearby weeks move together
URGENCY_WEIGHT = 4        # a unit needed right after delivery counts 5x: it can't wait for next week's order
PERIOD_CORRELATION = 0.5  # demand before and after the delivery move together (a busy month stays busy)
SETTINGS_COLUMNS = ["product", "on_hand", "on_order", "arrives_in_weeks", "lead_weeks", "order_every_weeks",
                    "unit_cost", "price", "case_size", "min_cases"]


@dataclass
class Line:
    product: str
    status: str               # "Order now", "Cut by budget", "Order soon", "OK", "Overstocked", "No demand"
    on_hand: int
    on_order: int
    lead_weeks: int
    cycle_weeks: int
    cover_weeks: float | None  # weeks stock + open orders last at forecast demand (None: no demand)
    reorder_point: float      # order when stock + open orders fall below this
    demand_lead: float        # forecast units until an order placed today arrives
    demand_cycle: float       # forecast units the order has to cover after it arrives
    safety_stock: float       # buffer inside the reorder point, above the average demand until an order arrives
    recommended_units: int    # at the service level, rounded to whole cases and the minimum order
    order_units: int          # what is ordered after the budget (same as recommended if no budget limit)
    order_cost: float
    fill_without: float       # expected share of the demand after delivery (until the next order) served without this order
    fill_with: float
    lost_margin_without: float
    lost_margin_with: float
    runs_out: pd.Timestamp | None  # week the stock runs out if nothing new is ordered
    order_in_weeks: int | None     # weeks until an order is due (0 = now)


def relative_sd(fc: ProductForecast, n: int, offset: int = 0) -> float:
    """Spread of the total of n forecast weeks starting `offset` weeks out, relative to that total.
    Measured on the kept model's own backtests, shrunk toward a prior built from the per-week spread."""
    w = fc.week_error if fc.week_error is not None else np.full(offset + n, 0.25)
    part = w[offset:offset + n] if offset + n <= len(w) else w[-n:]
    wk = float(np.mean(part)) if len(part) else float(np.mean(w))
    prior = wk * np.sqrt((1 + (n - 1) * ERROR_CORRELATION) / n)
    own = fc.window_errors.get((offset, n))
    if own is not None and len(own) >= 3:
        k = len(own)
        est = (k * float(np.sqrt(np.mean(np.square(own)))) + 3 * prior) / (k + 3)
    else:
        est = prior
    return float(np.clip(est, 0.05, 1.5))


def window_mean(fc: ProductForecast, start: int, n: int) -> float:
    """Forecast units in weeks start+1 .. start+n (weeks beyond the horizon repeat the last month's average)."""
    m = fc.mean
    total = 0.0
    for i in range(start, start + n):
        total += float(m[i]) if i < len(m) else float(m[-4:].mean())
    return total


def _lognormal(mu: float, sd_rel: float, z: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.log(1 + sd_rel**2))
    return mu * np.exp(sigma * z - sigma**2 / 2)


def _round_up(units: float, case: int, min_cases: int) -> int:
    if units <= 0:
        return 0
    return max(int(np.ceil(units / case - 1e-9)), min_cases) * case


def _get(s, name, default):
    v = getattr(s, name, default)
    try:
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else v
    except TypeError:
        return default


def plan(forecasts: dict[str, ProductForecast], settings: pd.DataFrame, service_level: float,
         budget: float | None, seed: int = 11) -> tuple[list[Line], dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for s in settings.itertuples(index=False):
        fc = forecasts.get(s.product)
        if fc is None:
            continue
        lead = int(max(1, _get(s, "lead_weeks", 4)))
        cycle = int(max(1, _get(s, "order_every_weeks", 4)))
        case = int(max(1, _get(s, "case_size", 1)))
        min_cases = int(max(0, _get(s, "min_cases", 0)))
        on_hand, on_order = float(max(0, _get(s, "on_hand", 0))), float(max(0, _get(s, "on_order", 0)))
        arrives = int(max(1, _get(s, "arrives_in_weeks", 1)))
        mu_lead, mu_cycle = window_mean(fc, 0, lead), window_mean(fc, lead, cycle)
        sd_lead, sd_cycle = relative_sd(fc, lead), relative_sd(fc, cycle, offset=lead)
        z1 = rng.standard_normal(SCENARIOS)
        z2 = PERIOD_CORRELATION * z1 + np.sqrt(1 - PERIOD_CORRELATION**2) * rng.standard_normal(SCENARIOS)
        d_lead = np.round(_lognormal(mu_lead, sd_lead, z1))
        d_cycle = np.round(_lognormal(mu_cycle, sd_cycle, z2))
        # the open order lands at the start of week arrives+1 (same convention as a new order: lead+1)
        if arrives < lead and on_order > 0:
            share = window_mean(fc, 0, arrives) / max(mu_lead, 1e-9)
            before = np.round(d_lead * share)
            left_before = np.maximum(on_hand - before, 0)
            served_lead = np.minimum(before, on_hand) + np.minimum(d_lead - before, left_before + on_order)
            left = np.maximum(left_before + on_order - (d_lead - before), 0)
            position = on_hand + on_order
        else:  # nothing arrives before a new order could: the open order only helps the cycle after
            served_lead = np.minimum(d_lead, on_hand)
            left = np.maximum(on_hand - d_lead, 0) + (on_order if arrives <= lead + cycle else 0)
            position = on_hand
        need = d_cycle - left
        rec_raw = float(np.quantile(need, service_level))
        # order now only if waiting one more week would miss the service level
        demand_wait = window_mean(fc, 0, lead + 1)
        d_wait = np.round(_lognormal(demand_wait, relative_sd(fc, lead + 1), z1))
        reorder_point = float(np.quantile(d_wait, service_level))
        due = position < reorder_point and mu_cycle > 0.5
        rec = _round_up(rec_raw, case, min_cases) if due else 0
        # first week after the delivery: what a one-week delay would cost (used to rank orders under a budget)
        first_share = float(fc.mean[lead]) / max(mu_cycle, 1e-9) if lead < len(fc.mean) else 1.0 / cycle
        rows.append(dict(s=s, fc=fc, lead=lead, cycle=cycle, case=case, min_cases=min_cases, on_hand=on_hand,
                         on_order=on_order, arrives=arrives, position=position, mu_lead=mu_lead, mu_cycle=mu_cycle,
                         ss=max(reorder_point - demand_wait, 0.0), rop=reorder_point, due=due, rec=rec,
                         d_lead=d_lead, d_cycle=d_cycle, left=left, served_lead=served_lead,
                         d_first=np.round(d_cycle * min(first_share, 1.0))))

    orders = _optimize(rows, budget)
    lines, total_cost = [], 0.0
    for r, q in zip(rows, orders):
        s, fc = r["s"], r["fc"]
        margin = max(float(_get(s, "price", 0)) - float(_get(s, "unit_cost", 0)), 0.0)
        d_cycle, left = r["d_cycle"], r["left"]
        cyc = max(float(d_cycle.sum()), 1.0)
        served0, served1 = np.minimum(d_cycle, left), np.minimum(d_cycle, left + q)
        avg_week = float(np.mean(fc.mean[:8]))
        cover = (r["on_hand"] + r["on_order"]) / avg_week if avg_week > 0.05 else None
        proj = projection(fc, int(r["on_hand"]), int(r["on_order"]), r["arrives"], 0, r["lead"])
        short = proj[proj["lost"] > 0.5]
        order_in = 0 if r["due"] else _weeks_until(proj, r["rop"], fc)
        if r["mu_cycle"] <= 0.5 and avg_week <= 0.05:
            status = "No demand"
        elif r["due"] and q < r["rec"]:
            status = "Cut by budget"
        elif r["due"]:
            status = "Order now"
        elif cover is not None and cover > r["lead"] + r["cycle"] + 12:
            status = "Overstocked"
        elif order_in is not None and order_in <= 2:
            status = "Order soon"
        else:
            status = "OK"
        cost = q * float(_get(s, "unit_cost", 0))
        total_cost += cost
        lines.append(Line(s.product, status, int(r["on_hand"]), int(r["on_order"]), r["lead"], r["cycle"],
                          round(min(cover, 99.0), 1) if cover is not None else None, r["rop"], r["mu_lead"],
                          r["mu_cycle"], r["ss"], r["rec"], int(q), cost,
                          float(served0.sum()) / cyc if r["mu_cycle"] > 0.5 else 1.0,
                          float(served1.sum()) / cyc if r["mu_cycle"] > 0.5 else 1.0,
                          margin * float((d_cycle - served0).mean()), margin * float((d_cycle - served1).mean()),
                          short["week"].iloc[0] if len(short) else None, order_in))
    rec_cost = sum(r["rec"] * float(_get(r["s"], "unit_cost", 0)) for r in rows)
    summary = {"recommended_cost": rec_cost, "order_cost": total_cost,
               "budget_binding": budget is not None and rec_cost > budget + 1e-6}
    return lines, summary


def _weeks_until(proj: pd.DataFrame, reorder_point: float, fc: ProductForecast) -> int | None:
    """Rough weeks until stock + open orders drop to the reorder point (mean forecast, no new orders)."""
    below = proj.index[proj["position"] < reorder_point]
    return int(below[0]) + 1 if len(below) else None


def _optimize(rows: list[dict], budget: float | None) -> list[int]:
    """Choose order quantities (whole cases, minimum order respected) that keep the most margin within the budget.

    Each product's recommended quantity is the upper bound. When the budget can't pay for all of them,
    the model spends it where the expected extra sales are worth the most, over the same demand scenarios.
    """
    recs = [r["rec"] for r in rows]
    if budget is None or sum(r["rec"] * float(_get(r["s"], "unit_cost", 0)) for r in rows) <= budget + 1e-6:
        return recs
    m = cp_model.CpModel()
    cases, spend, objective = [], [], []
    for i, r in enumerate(rows):
        max_cases = r["rec"] // r["case"]
        c = m.NewIntVar(0, max_cases, f"c{i}")
        if max_cases > 0 and r["min_cases"] > 0:
            use = m.NewBoolVar(f"u{i}")
            m.Add(c >= r["min_cases"]).OnlyEnforceIf(use)
            m.Add(c == 0).OnlyEnforceIf(use.Not())
        cases.append(c)
        spend.append(int(round(float(_get(r["s"], "unit_cost", 0)) * r["case"] * 100)) * c)
        if max_cases == 0:
            continue
        margin_cents = int(round(max(float(_get(r["s"], "price", 0)) - float(_get(r["s"], "unit_cost", 0)), 0.0) * 100))
        for k, (dc, df, lk) in enumerate(zip(r["d_cycle"], r["d_first"], r["left"])):
            dc, df, lk = int(dc), int(df), int(lk)
            if dc <= lk:
                continue  # already covered by stock: the order changes nothing in this scenario
            extra = m.NewIntVar(0, dc - lk, f"x{i}_{k}")  # sales the order adds over the cycle
            m.Add(extra <= r["case"] * c)
            objective.append(margin_cents * extra)
            if df > lk:  # sales in the first week after delivery: lost for good if this order waits a week
                urgent = m.NewIntVar(0, df - lk, f"f{i}_{k}")
                m.Add(urgent <= r["case"] * c)
                objective.append(URGENCY_WEIGHT * margin_cents * urgent)
        objective.append(c)  # tie-breaker: prefer ordering when it costs nothing extra
    m.Add(sum(spend) <= int(budget * 100))
    m.Maximize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.max_deterministic_time = 2.0  # same inputs, same plan
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [0] * len(rows)
    return [int(solver.Value(c)) * r["case"] for c, r in zip(cases, rows)]


def projection(fc: ProductForecast, on_hand: int, on_order: int, arrives_in: int, new_order: int,
               lead: int) -> pd.DataFrame:
    """Expected stock at the end of each forecast week (mean forecast). Demand you can't serve is lost."""
    stock, position, rows = float(on_hand), float(on_hand + on_order), []
    for i, (w, d) in enumerate(zip(fc.weeks, fc.mean), start=1):
        if i == arrives_in + 1:
            stock += on_order
        if i == lead + 1:  # ordered today, on the shelf after the lead time
            stock += new_order
        lost = max(d - stock, 0.0)
        stock = max(stock - d, 0.0)
        position -= d
        rows.append({"week": w, "stock": stock, "lost": lost, "position": position + (new_order if i <= lead else 0)})
    return pd.DataFrame(rows)


def plan_table(lines: list[Line]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Product": l.product, "Status": l.status, "Order (units)": l.order_units,
        "Recommended (units)": l.recommended_units, "Order cost": round(l.order_cost, 2),
        "On hand": l.on_hand, "On order": l.on_order, "Weeks of cover": l.cover_weeks,
        "Runs out": l.runs_out.date() if l.runs_out is not None else None,
        "Reorder point": round(l.reorder_point), "Safety stock": round(l.safety_stock),
        "Lead time (weeks)": l.lead_weeks, "Order every (weeks)": l.cycle_weeks,
        "Served after delivery, no order": l.fill_without, "Served after delivery, with order": l.fill_with,
    } for l in lines])
