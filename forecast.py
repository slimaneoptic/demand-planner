"""Weekly demand forecasts: four models per product, tested on the past, best one kept."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from data import WEEK

HORIZON = 16          # weeks forecast ahead
BACKTEST_ORIGINS = 6  # recent past forecast dates used to test every model (plus 3 from the same season last year)
ORIGIN_STEP = 2       # weeks between those dates
MODELS = ("Seasonal naive", "Moving average", "Holt-Winters", "Gradient boosting")
Z80 = 1.2816          # 80 % interval


@dataclass
class ProductForecast:
    product: str
    model: str                       # the model kept for this product
    weeks: pd.DatetimeIndex
    mean: np.ndarray                 # weekly forecast
    low: np.ndarray                  # 80 % interval
    high: np.ndarray
    wape: dict[str, float]           # model -> weighted absolute % error in backtests
    naive_wape: float
    window_errors: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)  # (offset, n) -> relative errors of n-week totals
    week_error: np.ndarray | None = None  # relative error spread per horizon week (smoothed)
    tests: int = 0                        # backtest windows used
    season_tested: bool = False           # did a test cover the same season last year?


# ---------------------------------------------------------------- single-series models

def seasonal_naive(y: np.ndarray, h: int) -> np.ndarray:
    """Same week last year, scaled by how this year compares with last year (last 8 weeks)."""
    if len(y) < 52 + 8:
        return moving_average(y, h)
    ratio = (y[-8:].sum() + 1) / (y[-60:-52].sum() + 1)
    ratio = float(np.clip(ratio, 0.5, 2.0))
    base = np.array([y[len(y) - 52 + i] if i < 52 else y[len(y) - 104 + i] for i in range(h)], dtype=float)
    return base * ratio


def moving_average(y: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(np.mean(y[-8:])))


def holt_winters(y: np.ndarray, h: int, seasonal: str | None = None) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    if len(y) < 104:
        seasonal = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(y.astype(float), trend="add", damped_trend=True, seasonal=seasonal,
                                       seasonal_periods=52 if seasonal else None,
                                       initialization_method="heuristic").fit(use_brute=False)
            f = np.asarray(fit.forecast(h), dtype=float)
        except Exception:
            f = moving_average(y, h)
    if not np.all(np.isfinite(f)):
        f = moving_average(y, h)
    return np.maximum(f, 0)


# ---------------------------------------------------------------- global gradient boosting

def _woy(weeks: pd.DatetimeIndex) -> np.ndarray:
    return weeks.isocalendar().week.to_numpy().astype(int)


def _rolling_mean(y: np.ndarray, k: int) -> np.ndarray:
    """out[t] = mean(y[t-k+1 .. t]); NaN where the window is incomplete."""
    c = np.concatenate([[0.0], np.cumsum(y)])
    out = np.full(len(y), np.nan)
    out[k - 1:] = (c[k:] - c[:-k]) / k
    return out


def _feature_block(y: np.ndarray, promo: np.ndarray, woy: np.ndarray, ts: np.ndarray, h: int, pid: int,
                   target_promo: np.ndarray) -> np.ndarray:
    """Features known at forecast dates `ts` (last observed index) for the week ts+h, one row per date."""
    lvl8 = _rolling_mean(y, 8) + 1.0
    lvl4 = _rolling_mean(y, 4) + 1.0
    lvl3c = _rolling_mean(y, 3)  # centred on index-1 when read at i+1
    level = lvl8[ts]
    ly_t = ts - 52
    ly_target = ts + h - 52
    ok = (ly_t >= 7) & (ly_target + 1 < len(y)) & (ly_target >= 1)
    ly_level = np.where(ok, lvl8[np.clip(ly_t, 0, len(y) - 1)], 1.0)
    ly_week = np.where(ok, lvl3c[np.clip(ly_target + 1, 0, len(y) - 1)] + 1.0, 1.0)
    ly_level = np.where(np.isfinite(ly_level), ly_level, 1.0)
    ly_week = np.where(np.isfinite(ly_week), ly_week, 1.0)
    target_w = ((woy[ts] - 1 + h) % 52) + 1
    ang = 2 * np.pi * target_w / 52
    return np.column_stack([np.full(len(ts), h), np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang),
                            ly_week / ly_level, np.where(ok, level / ly_level, 1.0), lvl4[ts] / level, target_promo,
                            promo[ts], ok.astype(float), np.full(len(ts), pid)])


def _gb_train(series: dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]], end: int | None, h_max: int):
    """Train one model on every product, using data up to index `end` (exclusive, counted from the end)."""
    X, Y = [], []
    for pid, (y, promo, weeks) in enumerate(series.values()):
        n = len(y) if end is None else len(y) - end
        if n < 20:
            continue
        yy, pp = y[:n], promo[:n]
        woy = _woy(weeks[:n])
        for h in range(1, h_max + 1):
            ts = np.arange(11, n - h)
            if len(ts) == 0:
                continue
            X.append(_feature_block(yy, pp, woy, ts, h, pid, pp[ts + h]))
            Y.append(yy[ts + h] / (_rolling_mean(yy, 8)[ts] + 1.0))
    if not X or sum(len(x) for x in X) < 200:
        return None
    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=25,
                                          l2_regularization=1.0, categorical_features=[11], random_state=0)
    model.fit(np.vstack(X), np.concatenate(Y))
    return model


def _gb_predict_many(model, jobs: list[tuple]) -> list[np.ndarray]:
    """One predict call for many series (much faster than one call each). job = (y, promo, weeks, pid, h, future_promos)."""
    blocks, sizes, levels = [], [], []
    for y, promo, weeks, pid, h, fp in jobs:
        t = len(y) - 1
        woy = _woy(weeks)
        fp = np.zeros(h) if fp is None else np.asarray(fp, dtype=float)
        blocks.append(np.vstack([_feature_block(y, promo, woy, np.array([t]), k, pid, np.array([fp[k - 1]]))
                                 for k in range(1, h + 1)]))
        sizes.append(h)
        levels.append(_rolling_mean(y, 8)[t] + 1.0)
    pred = model.predict(np.vstack(blocks))
    out, i = [], 0
    for h, lvl in zip(sizes, levels):
        out.append(np.maximum(pred[i:i + h] * lvl, 0))
        i += h
    return out


# ---------------------------------------------------------------- backtest + final forecast

def _smooth(sd: np.ndarray) -> np.ndarray:
    """Rolling median over neighbouring horizons, then never decreasing: a few noisy tests shouldn't zig-zag the band."""
    s = pd.Series(sd).rolling(5, center=True, min_periods=1).median().to_numpy()
    return np.maximum.accumulate(np.clip(s, 0.05, 1.5))


def _season_matters(y: np.ndarray, horizon: int) -> bool:
    """Last year, did the coming weeks differ from the weeks before them by more than 30 %?"""
    if len(y) < 52 + 8:
        return False
    ahead = y[len(y) - 52:len(y) - 52 + horizon].mean() + 1
    before = y[len(y) - 60:len(y) - 52].mean() + 1
    return abs(ahead / before - 1) > 0.3


def forecast_all(history: pd.DataFrame, future_promo: pd.Series | None = None,
                 horizon: int = HORIZON) -> dict[str, ProductForecast]:
    """Backtest the four models on each product, keep the most accurate, forecast `horizon` weeks."""
    series: dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]] = {}
    for prod, g in history.sort_values("week").groupby("product", sort=True):
        promo = g["promo"].to_numpy(dtype=float) if future_promo is not None else np.zeros(len(g))
        series[prod] = (g["units"].to_numpy(dtype=float), promo, pd.DatetimeIndex(g["week"]))

    recent = [horizon - 4 + ORIGIN_STEP * k for k in range(BACKTEST_ORIGINS)]  # 12..22 weeks held out
    seasonal_cuts = (50, 52, 54)  # the same calendar weeks last year

    def cuts_for(n: int) -> list[int]:
        c = [x for x in recent if n - x >= 20] or ([8] if n >= 28 else [])  # short products: one 8-week test
        return c + [x for x in seasonal_cuts if n >= x + 60]

    all_cuts = sorted({c for y, _, _ in series.values() for c in cuts_for(len(y))})
    gb_models = {c: _gb_train(series, c, horizon) for c in all_cuts}
    gb_final = _gb_train(series, None, horizon)
    names = list(series)
    gb_bt: dict[int, dict[str, np.ndarray]] = {}
    for c in all_cuts:
        if gb_models[c] is None:
            continue
        jobs = [(pid, y, promo, weeks) for pid, (y, promo, weeks) in enumerate(series.values()) if c in cuts_for(len(y))]
        preds = _gb_predict_many(gb_models[c], [(y[:-c], promo[:-c], weeks[:-c], pid, min(horizon, c),
                                                 promo[len(y) - c:len(y) - c + min(horizon, c)]) for pid, y, promo, weeks in jobs])
        gb_bt[c] = {names[pid]: p for (pid, *_), p in zip(jobs, preds)}
    fut = {}
    for prod, (y, promo, weeks) in series.items():
        fw = pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=horizon, freq=WEEK)
        fut[prod] = (fw, future_promo.reindex(fw).fillna(0).to_numpy() if future_promo is not None else np.zeros(horizon))
    gb_fut = dict(zip(names, _gb_predict_many(gb_final, [(y, promo, weeks, pid, horizon, fut[prod][1])
                                                         for pid, (prod, (y, promo, weeks)) in enumerate(series.items())]))) \
        if gb_final is not None else {}

    out: dict[str, ProductForecast] = {}
    for prod, (y, promo, weeks) in series.items():
        cuts = cuts_for(len(y))
        # Holt-Winters is tested and used the same way: seasonal only if every test still had 2 years of training data
        hw_season = "add" if cuts and len(y) - max(cuts) >= 104 else None
        abs_err = {m: 0.0 for m in MODELS}
        actual_total = 0.0
        preds_by_model: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in MODELS}
        for c in cuts:
            train_y, h = y[:-c], min(horizon, c)
            test_y = y[-c:][:h]
            preds = {"Seasonal naive": seasonal_naive(train_y, h), "Moving average": moving_average(train_y, h),
                     "Holt-Winters": holt_winters(train_y, h, hw_season)}
            if c in gb_bt and prod in gb_bt[c]:
                preds["Gradient boosting"] = gb_bt[c][prod]
            for m, p in preds.items():
                abs_err[m] += float(np.abs(test_y - p).sum())
                preds_by_model[m].append((p, test_y))
            actual_total += float(test_y.sum())
        allowed = [m for m in MODELS if len(preds_by_model[m]) == len(cuts) and (m != "Gradient boosting" or prod in gb_fut)]
        season_tested = any(c in seasonal_cuts for c in cuts)
        if not season_tested and _season_matters(y, horizon):
            # the season ahead was never tested: keep only models that know last year
            allowed = [m for m in allowed if m in ("Seasonal naive", "Gradient boosting")] or allowed
        wape = {m: (abs_err[m] / max(actual_total, 1.0) if m in allowed and cuts else float("nan")) for m in MODELS}
        best = min(allowed, key=lambda m: wape[m]) if cuts else "Moving average"

        rel_by_h = [[] for _ in range(horizon)]
        window_errors: dict[tuple[int, int], list[float]] = {}
        for p, a in preds_by_model[best]:
            for i in range(len(p)):
                rel_by_h[i].append((a[i] - p[i]) / max(p[i], 1.0))
            for off in range(len(p)):
                for n in range(1, len(p) - off + 1):
                    ps = p[off:off + n].sum()
                    window_errors.setdefault((off, n), []).append((a[off:off + n].sum() - ps) / max(ps, 1.0))
        pooled = [e for r in rel_by_h for e in r]
        base = float(np.sqrt(np.mean(np.square(pooled)))) if pooled else 0.3
        week_sd = _smooth(np.array([np.sqrt(np.mean(np.square(r))) if len(r) >= 3 else base for r in rel_by_h]))

        mean = gb_fut[prod] if best == "Gradient boosting" else (
            holt_winters(y, horizon, hw_season) if best == "Holt-Winters" else
            seasonal_naive(y, horizon) if best == "Seasonal naive" else moving_average(y, horizon))
        sig = np.sqrt(np.log(1 + week_sd**2))  # band in log space: asymmetric, never collapses to 0
        low, high = mean * np.exp(-Z80 * sig), mean * np.exp(Z80 * sig)
        out[prod] = ProductForecast(prod, best, fut[prod][0], mean, low, high, wape, wape["Seasonal naive"],
                                    {k: np.array(v) for k, v in window_errors.items()}, week_sd, len(cuts), season_tested)
    return out


def accuracy_table(forecasts: dict[str, ProductForecast]) -> pd.DataFrame:
    rows = []
    for f in forecasts.values():
        row = {"Product": f.product, "Model kept": f.model}
        row.update({m: f.wape[m] for m in MODELS})
        rows.append(row)
    return pd.DataFrame(rows)
