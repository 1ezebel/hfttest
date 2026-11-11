import numpy as np
import pandas as pd
import pytest

from hftlab.funding import basis_at, carry_backtest, fade_trades, fetch_all, funding_z, load_funding, update

H = 3_600_000
P8 = 8 * H


def fake_get(total):
    """Фальшивая биржа: выплаты с номерами 0..total-1 каждые 8 часов, страницы по 100 от новых к старым."""
    def get(inst, after):
        hi = total if after is None else after // P8
        return [{"fundingTime": str(i * P8), "realizedRate": "0.0001", "fundingRate": "0.0002"}
                for i in range(hi - 1, max(hi - 101, -1), -1)]
    return get


def test_fetch_paginates_and_update_appends_history(tmp_path):
    df = fetch_all("X", fake_get(278), sleep_s=0)
    assert len(df) == 278 and df["ts_ms"].is_monotonic_increasing and df["rate"].iloc[0] == 0.0001
    update(tmp_path, "X", fake_get(278), sleep_s=0)
    # месяц спустя биржа отдаёт уже другое окно: старые записи обязаны сохраниться

    def later(inst, after):
        hi = 370 if after is None else after // P8
        return [{"fundingTime": str(i * P8), "realizedRate": "0.0003", "fundingRate": "0.0003"}
                for i in range(hi - 1, max(hi - 101, 91), -1)]
    df2 = update(tmp_path, "X", later, sleep_s=0)
    assert len(df2) == 370 and df2["ts_ms"].iloc[0] == 0
    assert len(load_funding(tmp_path, "X")) == 370
    with pytest.raises(FileNotFoundError):
        load_funding(tmp_path, "Y")


def test_funding_z_excludes_current_value_and_future():
    r = np.random.default_rng(0).normal(1e-4, 2e-5, 200)
    z = funding_z(r, 30)
    assert np.isnan(z[:30]).all() and np.isfinite(z[30:]).all()
    assert z[50] == pytest.approx((r[50] - r[20:50].mean()) / r[20:50].std(ddof=1))
    r2 = r.copy()
    r2[120:] += 1.0
    np.testing.assert_allclose(funding_z(r2, 30)[:120], z[:120], equal_nan=True)
    assert np.isnan(funding_z(np.full(100, 1e-4), 30)).all()      # ставка не меняется: нормы нет, сигнала нет


def bars_1h(n, price=lambda i: 100.0):
    o = np.array([price(i) for i in range(n)], dtype=float)
    return pd.DataFrame({"ts_ms": np.arange(n) * H, "open": o, "close": o})


def test_fade_trade_timing_and_funding_sign():
    n_f = 60
    rate = np.full(n_f, 1e-4) + np.tile([1e-6, -1e-6], n_f // 2)   # почти ровная ставка
    rate[40] = 5e-4                                                # всплеск: толпа в длинных
    f = pd.DataFrame({"ts_ms": np.arange(n_f) * P8, "rate": rate})
    t_spike = 40 * P8
    # цена падает на 1% за сутки после входа (вход через час после выплаты)
    b = bars_1h(n_f * 8, lambda i: 100.0 if i * H <= t_spike + H else 99.0)
    tr = fade_trades(b, f)
    assert list(tr["t_signal"]) == [t_spike] and tr["side"][0] == -1
    assert tr["price_bps"][0] == pytest.approx(100.0)              # продали по 100, выкупили по 99
    # за 24 часа в короткой позиции получили три обычные выплаты по ~1 б.п.
    assert tr["funding_bps"][0] == pytest.approx(3.0, abs=0.05)
    assert tr["gross_bps"][0] == pytest.approx(103.0, abs=0.05)
    # отрицательный всплеск: покупаем и ПЛАТИМ положительную ставку
    rate[40] = -3e-4
    tr = fade_trades(b, pd.DataFrame({"ts_ms": f["ts_ms"], "rate": rate}))
    assert tr["side"][0] == 1 and tr["funding_bps"][0] == pytest.approx(-3.0, abs=0.05)


def test_fade_skips_when_candles_are_missing():
    n_f = 60
    rate = np.full(n_f, 1e-4) + np.tile([1e-6, -1e-6], n_f // 2)
    rate[40] = 5e-4
    f = pd.DataFrame({"ts_ms": np.arange(n_f) * P8, "rate": rate})
    b = bars_1h(n_f * 8)
    hole = b[(b["ts_ms"] < 40 * P8 + 5 * H) | (b["ts_ms"] > 40 * P8 + 9 * H)].reset_index(drop=True)
    assert len(fade_trades(hole, f)["gross_bps"]) == 0


def test_basis_at_uses_last_closed_candle():
    m5 = 300_000
    ts = np.arange(100) * m5
    perp = pd.DataFrame({"ts_ms": ts, "close": 100.0 + np.arange(100) * 0.01})
    spot = pd.DataFrame({"ts_ms": ts, "close": np.full(100, 100.0)})
    b = basis_at(perp, spot, np.array([10 * m5, 10 * m5 + 1, 0, 200 * m5]))
    assert b[0] == pytest.approx(0.09 * 100, rel=1e-6)     # свеча 9 закрылась ровно в момент 10*5м
    assert b[1] == pytest.approx(b[0])
    assert np.isnan(b[2]) and np.isnan(b[3])                # до данных и слишком давно после


def test_carry_arithmetic():
    f = pd.DataFrame({"ts_ms": np.arange(10) * P8, "rate": np.full(10, 1e-4)})
    flat = np.full(10, 5.0)
    r = carry_backtest(f, flat, cost_bps=30, rule="always")
    # вошли после выплаты 0, получили выплаты 1..9: 9 б.п.; один круг стоит 30
    assert (r["periods_in"], r["round_trips"]) == (9, 1)
    assert r["funding_bps"] == pytest.approx(9.0) and r["net_bps"] == pytest.approx(-21.0)
    # базис сжался с 5 до 1 б.п.: короткому фьючерсу это плюс 4
    shrink = np.r_[5.0, np.full(8, 3.0), 1.0]
    assert carry_backtest(f, shrink, 30, "always")["basis_bps"] == pytest.approx(4.0)


def test_carry_threshold_enters_late_exits_on_negative_and_pays_per_trip():
    rate = np.r_[np.full(5, 0.2e-4), np.full(10, 3e-4), np.full(6, -2e-4), np.full(4, 0.1e-4)]
    f = pd.DataFrame({"ts_ms": np.arange(len(rate)) * P8, "rate": rate})
    r = carry_backtest(f, np.zeros(len(rate)), cost_bps=30, rule="threshold")
    # средняя за 3 выплаты переходит 1 б.п. на выплате 5 (0.2, 0.2, 3.0): вход, доход с выплаты 6;
    # отрицательной становится на выплате 16 (3, -2, -2): выход
    assert r["round_trips"] == 1 and r["periods_in"] == 11
    assert r["funding_bps"] == pytest.approx(9 * 3.0 - 2 * 2.0)
    assert r["net_bps"] == pytest.approx(23.0 - 30.0)
    # ставка всегда низкая: ни одного входа, ни копейки издержек
    low = carry_backtest(pd.DataFrame({"ts_ms": f["ts_ms"], "rate": np.full(len(rate), 0.5e-4)}),
                         np.zeros(len(rate)), 30, "threshold")
    assert low["round_trips"] == 0 and low["net_bps"] == 0
