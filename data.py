"""Sample data and loading of the user's own sales history."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEK = "W-MON"  # weeks are labelled by their Monday
LAST_WEEK = pd.Timestamp("2026-09-14")  # most recent full week in the sample
SAMPLE_WEEKS = 156


@dataclass(frozen=True)
class SampleProduct:
    name: str
    base: float          # units per week at the start
    growth: float        # yearly growth, 0.10 = +10 %/year
    season: str          # "flat", "winter", "summer", "gifts"
    promo_lift: float    # extra sales in a promo week, 0.6 = +60 %
    noise: float         # week-to-week randomness
    lead_weeks: int
    cycle_weeks: int     # how often you normally order from this supplier
    unit_cost: float
    price: float
    case: int
    moq_cases: int


SAMPLE = [
    SampleProduct("Espresso beans 1 kg", 180, 0.12, "winter", 0.45, 0.10, 3, 2, 11.0, 24.0, 12, 2),
    SampleProduct("House blend 500 g", 240, 0.05, "winter", 0.35, 0.10, 3, 2, 5.5, 13.0, 24, 2),
    SampleProduct("Cold brew bottle", 90, 0.20, "summer", 0.50, 0.14, 2, 2, 2.1, 5.5, 24, 1),
    SampleProduct("Paper filters x100", 150, 0.02, "flat", 0.20, 0.12, 5, 4, 1.2, 4.0, 50, 1),
    SampleProduct("Ceramic mug", 60, 0.00, "gifts", 0.55, 0.18, 6, 6, 3.8, 12.0, 12, 1),
    SampleProduct("French press", 35, 0.06, "gifts", 0.70, 0.22, 8, 8, 14.0, 39.0, 6, 1),
    SampleProduct("Milk frother", 28, 0.15, "gifts", 0.65, 0.22, 8, 8, 9.0, 29.0, 8, 1),
    SampleProduct("Gift box", 12, 0.10, "gifts", 0.40, 0.25, 6, 8, 8.0, 32.0, 10, 1),
]


def _season(kind: str, week_of_year: np.ndarray) -> np.ndarray:
    angle = 2 * np.pi * week_of_year / 52.0
    if kind == "winter":
        return 1 + 0.18 * np.cos(angle - 2 * np.pi * 2 / 52)
    if kind == "summer":
        return np.maximum(0.35, 1 + 0.65 * np.cos(angle - 2 * np.pi * 29 / 52))
    if kind == "gifts":  # quiet most of the year, strong November-December peak
        peak = np.exp(-0.5 * ((week_of_year - 50) / 3.0) ** 2) + np.exp(-0.5 * ((week_of_year + 2) / 3.0) ** 2)
        return 0.75 + 2.6 * peak + 0.25 * np.exp(-0.5 * ((week_of_year - 7) / 1.2) ** 2)  # small Valentine's bump
    return np.ones_like(week_of_year, dtype=float)


def promo_calendar(weeks: pd.DatetimeIndex) -> np.ndarray:
    """Store-wide promotions: Black Friday week, a spring sale and a September sale."""
    flag = np.zeros(len(weeks), dtype=int)
    for i, w in enumerate(weeks):
        if w.month == 11 and 19 <= w.day <= 25:  # the Monday of Black Friday week
            flag[i] = 1
        if w.isocalendar().week in (14, 38):
            flag[i] = 1
    return flag


def sample_history(seed: int = 7) -> pd.DataFrame:
    """Three years of weekly unit sales for 8 products of a small coffee equipment shop."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(end=LAST_WEEK, periods=SAMPLE_WEEKS, freq=WEEK)
    woy = weeks.isocalendar().week.to_numpy().astype(float)
    promo = promo_calendar(weeks)
    years = np.arange(SAMPLE_WEEKS) / 52.0
    rows = []
    for p in SAMPLE:
        mean = p.base * (1 + p.growth) ** years * _season(p.season, woy) * (1 + p.promo_lift * promo)
        shape = 1 / p.noise**2  # gamma-poisson: coefficient of variation close to p.noise
        units = rng.poisson(rng.gamma(shape, mean / shape))
        rows.append(pd.DataFrame({"week": weeks, "product": p.name, "units": units, "promo": promo}))
    return pd.concat(rows, ignore_index=True)


def future_promos(start: pd.Timestamp, horizon: int) -> pd.Series:
    weeks = pd.date_range(start=start, periods=horizon, freq=WEEK)
    return pd.Series(promo_calendar(weeks), index=weeks)


def sample_settings(history: pd.DataFrame) -> pd.DataFrame:
    by_name = {p.name: p for p in SAMPLE}
    recent = history[history["week"] > history["week"].max() - pd.Timedelta(weeks=8)].groupby("product")["units"].mean()
    rows = []
    for name, p in by_name.items():
        weekly = recent.get(name, p.base)
        # a realistic mix: some products are low, one is overstocked, one has an order on the way
        # a realistic week: one urgent product, gift items to buy ahead of Christmas, summer stock left over
        cover = {"Espresso beans 1 kg": 2.2, "House blend 500 g": 4.5, "Cold brew bottle": 16.0, "Paper filters x100": 6.5,
                 "Ceramic mug": 8.0, "French press": 10.0, "Milk frother": 9.5, "Gift box": 11.0}[name]
        on_order = int(round(weekly * 2 / p.case)) * p.case if name == "House blend 500 g" else 0
        rows.append({"product": name, "on_hand": int(weekly * cover), "on_order": on_order,
                     "arrives_in_weeks": 1.0 if on_order else np.nan, "lead_weeks": p.lead_weeks,
                     "order_every_weeks": p.cycle_weeks, "unit_cost": p.unit_cost, "price": p.price,
                     "case_size": p.case, "min_cases": p.moq_cases})
    return pd.DataFrame(rows)


def default_settings(history: pd.DataFrame) -> pd.DataFrame:
    """Starting values for uploaded data; the user edits them in the app."""
    recent = history[history["week"] > history["week"].max() - pd.Timedelta(weeks=8)].groupby("product")["units"].mean()
    return pd.DataFrame([{"product": name, "on_hand": int(round(avg * 3)), "on_order": 0, "arrives_in_weeks": np.nan,
                          "lead_weeks": 4, "order_every_weeks": 4, "unit_cost": 10.0, "price": 25.0,
                          "case_size": 1, "min_cases": 0}
                         for name, avg in recent.items()])


class DataError(ValueError):
    pass


def _pick(columns: list[str], options: tuple[str, ...]) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for o in options:
        if o in lower:
            return lower[o]
    return None


def _dates(col: pd.Series) -> pd.Series:
    """Parse dates whatever the export: ISO, dd/mm/yyyy, mm/dd/yyyy, with or without time zones."""
    s = col.astype(str).str.strip()
    best, best_ok = None, -1
    for kw in ({"format": "ISO8601"}, {"dayfirst": True, "format": "mixed"}, {"dayfirst": False, "format": "mixed"}):
        try:
            d = pd.to_datetime(s, errors="coerce", utc=True, **kw)
        except (ValueError, TypeError):
            continue
        ok = int(d.notna().sum())
        if d.notna().any() and d.max() > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=7):
            ok -= len(s)  # future dates: day and month are swapped
        if ok > best_ok:
            best, best_ok = d, ok
    if best is None:
        raise DataError("Couldn't read the dates.")
    return best.dt.tz_convert(None)


def _numbers(col: pd.Series) -> pd.Series:
    """Numbers as exported by Excel: '1,234', '1 234', '1.234,5' (European) all become 1234(.5)."""
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce")
    s = col.astype(str).str.strip().str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    # decide once per column: is the comma a decimal mark (European) or a thousands separator?
    comma_decimal = s.str.match(r"^-?\d+,\d{1,2}$").any() or s.str.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$").any()
    if comma_decimal:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_upload(name: str, raw: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Read a CSV or Excel file of sales (daily or weekly) into weekly units per product, plus notes for the user."""
    try:
        df = pd.read_excel(io.BytesIO(raw)) if name.lower().endswith(".xlsx") else \
            pd.read_csv(io.BytesIO(raw), sep=None, engine="python", dtype=str)
    except Exception as exc:
        raise DataError(f"Couldn't read the file: {exc}") from exc
    cols = list(df.columns)
    date_col = _pick(cols, ("date", "week", "day", "order date", "order_date", "created at", "week start", "period"))
    prod_col = _pick(cols, ("product", "sku", "item", "product name", "product title", "lineitem name",
                            "lineitem sku", "variant sku", "product id", "item name", "title", "name"))
    qty_col = _pick(cols, ("units", "quantity", "qty", "sales", "sold", "demand", "units sold", "quantity sold",
                           "lineitem quantity", "net quantity", "units ordered", "net items sold"))
    if not (date_col and prod_col and qty_col):
        raise DataError("The file needs three columns: date, product and units (names like sku, quantity or "
                        f"Shopify's Lineitem quantity also work). Columns found: {', '.join(map(str, cols[:12]))}.")
    promo_col = _pick(cols, ("promo", "promotion", "on_promo"))
    product = df[prod_col]
    if pd.api.types.is_float_dtype(product):
        product = product.map(lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer() else v)
    promo = 0
    if promo_col:
        promo = df[promo_col].astype(str).str.strip().str.lower().isin(["1", "1.0", "yes", "y", "true", "x", "promo"]).astype(int)
    out = pd.DataFrame({"date": _dates(df[date_col]), "product": product.astype(str).str.strip(),
                        "units": _numbers(df[qty_col]), "promo": promo})
    total = len(out)
    out = out.dropna(subset=["date", "units"])
    out = out[(out["product"] != "") & (out["product"].str.lower() != "nan")]
    notes = []
    dropped = total - len(out)
    if total and dropped / total > 0.02:
        raise DataError(f"{dropped} of {total} rows have a date or quantity that couldn't be read. Check those columns.")
    if dropped:
        notes.append(f"{dropped} rows without a readable date or quantity were left out.")
    if out.empty:
        raise DataError("No usable rows: check the date and units columns.")
    out["units"] = out["units"].clip(lower=0)  # returns are not demand
    out["week"] = out["date"].dt.to_period("W-SUN").dt.start_time.dt.normalize()  # Monday of each Mon-Sun week
    daily = out["date"].dt.normalize().nunique() > out["week"].nunique() * 1.5
    last_day = out["date"].max().normalize()
    if daily and last_day.weekday() != 6:  # the last week isn't finished: leave it out
        cut = last_day - pd.Timedelta(days=last_day.weekday())
        out = out[out["week"] < cut]
        notes.append(f"The week of {cut:%d %b %Y} isn't complete yet and was left out.")
    weekly = out.groupby(["product", "week"], as_index=False).agg(units=("units", "sum"), promo=("promo", "max"))
    last = weekly["week"].max()
    parts = []
    for prod, g in weekly.groupby("product"):
        idx = pd.date_range(g["week"].min(), last, freq=WEEK)
        g = g.set_index("week").reindex(idx).rename_axis("week").reset_index()
        g["product"] = prod
        g["units"] = g["units"].fillna(0)
        g["promo"] = g["promo"].fillna(0).astype(int)
        parts.append(g)
    weekly = pd.concat(parts, ignore_index=True)
    counts = weekly.groupby("product")["week"].count()
    short = sorted(counts[counts < 26].index)
    weekly = weekly[~weekly["product"].isin(short)]
    if weekly.empty:
        raise DataError("Every product has less than 26 weeks of history. The planner needs at least 6 months per product.")
    if short:
        notes.append(f"{len(short)} product(s) with less than 26 weeks of sales were left out: {', '.join(short[:8])}"
                     + ("…" if len(short) > 8 else "") + ".")
    totals = weekly.groupby("product")["units"].sum()
    if len(totals) > 30:
        notes.append(f"The demo keeps your 30 best-selling products; {len(totals) - 30} smaller ones were left out.")
    top = totals.nlargest(30).index
    weekly = weekly[weekly["product"].isin(top)][["week", "product", "units", "promo"]].reset_index(drop=True)
    return weekly, notes


def template_csv() -> bytes:
    h = sample_history()
    h = h[h["week"] >= h["week"].max() - pd.Timedelta(weeks=60)]
    return h.rename(columns={"week": "date"}).to_csv(index=False).encode()
