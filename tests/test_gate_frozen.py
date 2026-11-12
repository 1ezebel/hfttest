import numpy as np
import pandas as pd
import pytest

from hftlab.frozen import frozen_trades, funding_for_trades, summary, verdict
from hftlab.gate import DAY_S, START_S, contract_of, fetch_daily, fetch_funding, inst_of, universe

DAY = 86_400_000


def test_names_roundtrip():
    assert contract_of("ZEC-USDT-SWAP") == "ZEC_USDT" and inst_of("ZEC_USDT") == "ZEC-USDT-SWAP"


def test_fetch_daily_windows_and_drops_forming_candle():
    first = START_S + 400 * DAY_S
    now = first + 2500 * DAY_S + 3600                      # сейчас идёт день номер 2500
    calls = []

    def get(path):
        calls.append(path)
        lo, hi = (int(path.split(k + "=")[1].split("&")[0]) for k in ("from", "to"))
        ts = range(max(lo, first), min(hi, now) + 1, DAY_S)
        return [{"t": t, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": 10, "sum": "15"} for t in ts]

    df = fetch_daily("X-USDT-SWAP", get, now_s=now, sleep_s=0)
    assert len(calls) == 2                                  # 2900 дней не влезают в один запрос
    assert len(df) == 2500 and not df["ts_ms"].duplicated().any()
    assert df["ts_ms"].iloc[-1] == (first + 2499 * DAY_S) * 1000   # сегодняшняя незакрытая свеча отброшена
    assert (df["ts_ms"].diff().dropna() == DAY).all() and df["quote_vol"].iloc[0] == 15


def test_fetch_funding_pages_backwards_until_empty():
    all_t = [i * 28_800 for i in range(1, 251)]

    def get(path):
        to = int(path.split("to=")[1]) if "to=" in path else 10**12
        page = [t for t in reversed(all_t) if t <= to][:90]
        return [{"t": t, "r": "0.0001"} for t in page]

    df = fetch_funding("X-USDT-SWAP", get, sleep_s=0)
    assert len(df) == 250 and df["ts_ms"].is_monotonic_increasing and df["rate"].iloc[0] == 0.0001


def test_universe_applies_all_three_rules():
    contracts = [
        {"name": "OLD_USDT", "create_time": 1_600_000_000, "in_delisting": False},
        {"name": "NEW_USDT", "create_time": 1_700_000_000, "in_delisting": False},     # слишком новый
        {"name": "DEAD_USDT", "create_time": 1_600_000_000, "in_delisting": True},     # снимается с торгов
        {"name": "THIN_USDT", "create_time": 1_600_000_000, "in_delisting": False},    # маленький оборот
        {"name": "USED_USDT", "create_time": 1_600_000_000, "in_delisting": False},    # уже был в серии
    ]
    tickers = [{"contract": c["name"], "volume_24h_quote": "5000000"} for c in contracts]
    tickers[3]["volume_24h_quote"] = "1000"
    get = lambda path: contracts if path == "/contracts" else tickers  # noqa: E731
    assert [r["inst"] for r in universe(get, exclude=("USED-USDT-SWAP",))] == ["OLD-USDT-SWAP"]


def bars(n=60):
    o = 100.0 + np.arange(n)
    return pd.DataFrame({"ts_ms": np.arange(n) * DAY, "open": o, "close": o})


def test_funding_is_charged_for_holding_period_only():
    b = bars()
    side = np.zeros(60, dtype=np.int8)
    side[10] = 1                                             # сигнал в день 10: вход день 11, выход день 18
    # одна выплата в сутки ровно в полночь, ставка 1 б.п.
    f = pd.DataFrame({"ts_ms": np.arange(5, 40) * DAY, "rate": np.full(35, 1e-4)})
    tr = frozen_trades(b, side, 7, DAY, f)
    assert len(tr) == 1
    assert tr["price_bps"][0] == pytest.approx((118 - 111) / 111 * 1e4)
    assert tr["funding_bps"][0] == pytest.approx(-7.0)       # длинная позиция заплатила 7 выплат (дни 12..18)
    side[10] = -1
    assert frozen_trades(b, side, 7, DAY, f)["funding_bps"][0] == pytest.approx(7.0)   # короткая получила


def test_trades_outside_funding_history_are_excluded_from_summary():
    b = bars()
    side = np.zeros(60, dtype=np.int8)
    side[2], side[30] = 1, 1
    f = pd.DataFrame({"ts_ms": np.arange(20, 50) * DAY, "rate": np.full(30, 1e-4)})
    tr = frozen_trades(b, side, 7, DAY, f)
    assert np.isnan(tr["funding_bps"][0]) and np.isfinite(tr["funding_bps"][1])
    s = summary(tr, 7 * DAY, cost_bps=20)
    assert (s["n"], s["n_no_funding"]) == (1, 1)
    none = funding_for_trades({"gross_bps": np.zeros(2), "t_entry": np.zeros(2), "t_exit": np.ones(2),
                               "side": np.ones(2)}, None)
    assert np.isnan(none).all()


def test_summary_breakdowns_and_verdict():
    rng = np.random.default_rng(0)
    n = 400
    tr = pd.DataFrame({"t_signal": np.arange(n) * 7 * DAY + 1_600_000_000_000,
                       "side": np.where(np.arange(n) % 2, 1, -1),
                       "price_bps": rng.normal(60, 200, n), "funding_bps": np.full(n, -5.0)})
    tr["total_bps"] = tr["price_bps"] + tr["funding_bps"]
    s = summary(tr, 7 * DAY, cost_bps=20)
    assert s["n"] == n and s["n_long"] + s["n_short"] == n
    assert s["mean_total_bps"] == pytest.approx(s["mean_price_bps"] - 5)
    assert s["mean_without_top10_bps"] < s["mean_total_bps"]
    assert sum(v["n"] for v in s["by_year"].values()) == n
    ok, text = verdict(s)
    assert ok and "не доказательство" in text
    weak = dict(s, t=1.2)
    assert verdict(weak)[0] is False and "1.20" in verdict(weak)[1]
    costly = dict(s, mean_net_bps=-3.0)
    assert verdict(costly)[0] is False and "после издержек" in verdict(costly)[1]
    assert verdict(dict(s, n=0, t=None))[0] is False
