import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import DataError, load_upload, sample_history, sample_settings, future_promos  # noqa: E402
from forecast import HORIZON, MODELS, forecast_all  # noqa: E402
from inventory import SETTINGS_COLUMNS, plan, projection  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    h = sample_history()
    s = sample_settings(h)
    fc = forecast_all(h, future_promos(h["week"].max() + pd.Timedelta(weeks=1), HORIZON))
    return h, s, fc


def test_sample_is_three_years_of_weeks(sample):
    h, s, _ = sample
    assert h["product"].nunique() == 8
    assert h.groupby("product")["week"].count().eq(156).all()
    assert (h["units"] >= 0).all()
    assert list(s.columns) == SETTINGS_COLUMNS


def test_every_product_gets_a_full_forecast(sample):
    _, _, fc = sample
    for f in fc.values():
        assert len(f.mean) == HORIZON and f.model in MODELS
        assert np.all(f.low <= f.mean + 1e-9) and np.all(f.mean <= f.high + 1e-9)
        assert np.all(f.mean >= 0)


def test_kept_model_is_the_most_accurate(sample):
    _, _, fc = sample
    for f in fc.values():
        assert f.wape[f.model] == min(v for v in f.wape.values() if v == v)


def test_gift_products_forecast_the_christmas_peak(sample):
    _, _, fc = sample
    for name in ("Gift box", "Milk frother", "French press"):
        f = fc[name]
        dec = f.mean[[i for i, w in enumerate(f.weeks) if w.month == 12]]
        sep_oct = f.mean[:4]
        assert dec.mean() > 1.8 * sep_oct.mean(), name


def test_plan_orders_what_is_due_in_whole_cases(sample):
    _, s, fc = sample
    lines, summary = plan(fc, s, 0.95, None)
    by = {l.product: l for l in lines}
    assert by["Espresso beans 1 kg"].status == "Order now"
    assert all(l.safety_stock <= l.reorder_point for l in lines)
    assert by["Cold brew bottle"].status == "Overstocked" and by["Cold brew bottle"].order_units == 0
    cases = s.set_index("product")
    for l in lines:
        c = cases.loc[l.product]
        assert l.order_units % int(c.case_size) == 0
        if l.order_units:
            assert l.order_units >= int(c.min_cases) * int(c.case_size)
        assert l.fill_with >= l.fill_without - 1e-9
    assert summary["order_cost"] == pytest.approx(sum(l.order_cost for l in lines))


def test_higher_service_level_orders_more(sample):
    _, s, fc = sample
    low = sum(l.order_units for l in plan(fc, s, 0.90, None)[0])
    high = sum(l.order_units for l in plan(fc, s, 0.99, None)[0])
    assert high >= low


def test_budget_is_respected_and_spent_where_it_protects_most(sample):
    _, s, fc = sample
    lines, summary = plan(fc, s, 0.95, 3000)
    assert summary["budget_binding"]
    assert summary["order_cost"] <= 3000 + 1e-6
    assert summary["order_cost"] > 2500  # the optimizer uses most of the budget
    full = {l.product: l.order_units for l in plan(fc, s, 0.95, None)[0]}
    for l in lines:
        assert l.order_units <= full[l.product]


def test_budget_goes_first_to_the_product_running_out(sample):
    _, s, fc = sample
    lines, _ = plan(fc, s, 0.95, 3000)
    esp = next(l for l in lines if l.product == "Espresso beans 1 kg")
    assert esp.order_units > 0 and esp.runs_out is not None


def test_open_order_arriving_late_does_not_hide_a_stockout(sample):
    _, s, fc = sample
    s = s.copy()
    i = s.index[s["product"] == "Paper filters x100"][0]
    s.loc[i, ["on_hand", "on_order", "arrives_in_weeks"]] = [50, 5000, 10]
    l = next(l for l in plan(fc, s, 0.95, None)[0] if l.product == "Paper filters x100")
    assert l.status in ("Order now", "Order soon") and l.runs_out is not None


def test_projection_adds_the_order_after_the_lead_time(sample):
    _, _, fc = sample
    f = fc["Paper filters x100"]
    a = projection(f, 5000, 0, 1, 0, 5)
    b = projection(f, 5000, 0, 1, 1000, 5)
    assert (b["stock"] - a["stock"]).iloc[:5].abs().max() < 1e-9
    assert (b["stock"] - a["stock"]).iloc[5] == pytest.approx(1000, abs=1)
    assert (a["stock"] >= 0).all()


def test_upload_daily_csv_is_turned_into_weeks():
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", "2025-12-28", freq="D")
    rows = [{"Order Date": d.strftime("%Y-%m-%d"), "SKU": sku, "Qty": int(rng.poisson(5 if sku == "A" else 2))}
            for d in days for sku in ("A", "B")]
    raw = pd.DataFrame(rows).to_csv(index=False).encode()
    weekly, _ = load_upload("sales.csv", raw)
    assert set(weekly["product"]) == {"A", "B"}
    assert weekly.groupby("product")["week"].count().min() >= 100
    assert weekly["units"].sum() == sum(r["Qty"] for r in rows)
    fc = forecast_all(weekly, None)
    assert set(fc) == {"A", "B"}


def test_upload_rejects_missing_columns_and_short_history():
    with pytest.raises(DataError):
        load_upload("x.csv", b"a,b\n1,2\n")
    short = pd.DataFrame({"date": pd.date_range("2025-01-06", periods=10, freq="W-MON"), "product": "A", "units": 3})
    with pytest.raises(DataError):
        load_upload("x.csv", short.to_csv(index=False).encode())


def test_excel_upload():
    df = pd.DataFrame({"week": pd.date_range("2024-01-01", periods=60, freq="W-MON"), "product": "X", "units": 10})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    weekly, _ = load_upload("s.xlsx", buf.getvalue())
    assert len(weekly) == 60


@pytest.mark.parametrize("weeks", [26, 40, 52, 60, 82, 104, 110])
def test_short_histories_do_not_crash(weeks):
    h = sample_history()
    h = h[h["week"] > h["week"].max() - pd.Timedelta(weeks=weeks)]
    fc = forecast_all(h, None)
    assert len(fc) == 8 and all(np.isfinite(f.mean).all() for f in fc.values())


def test_template_round_trip():
    from data import template_csv
    weekly, _ = load_upload("sales_template.csv", template_csv())
    assert len(forecast_all(weekly, None)) == 8


def test_one_new_product_does_not_shrink_the_others_tests():
    h = sample_history()
    new = h[(h["product"] == "Gift box") & (h["week"] > h["week"].max() - pd.Timedelta(weeks=30))].assign(product="New item")
    fc = forecast_all(pd.concat([h, new]), None)
    assert fc["Cold brew bottle"].season_tested and fc["Cold brew bottle"].tests == 9
    assert fc["New item"].tests >= 1


def test_two_years_keeps_a_seasonal_model_for_gifts():
    h = sample_history()
    h = h[h["week"] > h["week"].max() - pd.Timedelta(weeks=104)]
    fc = forecast_all(h, None)
    for name in ("Ceramic mug", "Milk frother", "Gift box"):
        assert fc[name].model in ("Seasonal naive", "Gradient boosting"), name


def test_band_never_collapses_to_zero(sample):
    _, _, fc = sample
    for f in fc.values():
        assert np.all(f.low[f.mean > 1] > 0)


def test_european_dates_and_thousands_separators():
    days = pd.date_range("2024-01-01", "2025-12-28", freq="D")
    df = pd.DataFrame({"Date": days.strftime("%d/%m/%Y"), "Product": "A", "Qty": "1.234"})
    weekly, notes = load_upload("eu.csv", df.to_csv(index=False, sep=";").encode())
    assert weekly["units"].sum() == 1234 * len(days)
    assert weekly["week"].max() <= pd.Timestamp("2025-12-22")
    df2 = pd.DataFrame({"Created at": days.strftime("%Y-%m-%dT10:00:00+01:00"), "Lineitem name": "B",
                        "Lineitem quantity": "1,500"})
    weekly2, _ = load_upload("shopify.csv", df2.to_csv(index=False).encode())
    assert weekly2["units"].sum() == 1500 * len(days)
