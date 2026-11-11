import numpy as np
import pandas as pd
import pytest

from hftlab.slow import (clustered_t, evaluate, load_bars, make_side, placebo_p, pooled_summary, run_trades,
                         sig_basis, sig_price, sig_price_vol, vol_rank, walk_forward_bars, zscore)

MR = lambda b, **p: sig_price(b, flip=False, **p)  # noqa: E731

BAR = 300_000


def frame(close, seg_gap_at=None):
    """Свечи, где open = предыдущий close (как на реальном рынке без гэпов)."""
    c = np.asarray(close, dtype=float)
    o = np.r_[c[0], c[:-1]]
    ts = np.arange(len(c)) * BAR
    if seg_gap_at is not None:
        ts[seg_gap_at:] += 5 * BAR
    return pd.DataFrame({"ts_ms": ts, "open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c})


def ou(n, theta, seed, sigma=2e-4):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] * (1 - theta) + rng.normal(0, sigma)
    return 100 * np.exp(x)


def walk(n, seed, sigma=2e-4):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, sigma, n)))


def test_entry_next_open_exit_after_hold_and_one_position():
    b = frame(np.arange(100.0, 130.0))                 # цена растёт на 1 за свечу
    side = np.zeros(30, dtype=np.int8)
    side[3] = 1                                        # сигнал на закрытии свечи 3
    side[5] = -1                                       # приходит, пока позиция открыта: пропускаем
    tr = run_trades(b, side, hold=4, bar_ms=BAR)
    assert len(tr["gross_bps"]) == 1
    # вход по open свечи 4 (= close 3 = 103), выход по open свечи 8 (= close 7 = 107)
    assert tr["gross_bps"][0] == pytest.approx((107 - 103) / 103 * 1e4)


def test_trade_must_fit_inside_range_and_skip_gaps():
    b = frame(np.arange(100.0, 130.0))
    side = np.zeros(30, dtype=np.int8)
    side[3] = 1
    assert len(run_trades(b, side, hold=4, lo=0, hi=8, bar_ms=BAR)["gross_bps"]) == 0   # выход в 8 не влезает
    assert len(run_trades(b, side, hold=4, lo=0, hi=9, bar_ms=BAR)["gross_bps"]) == 1
    gapped = frame(np.arange(100.0, 130.0), seg_gap_at=6)                                # дыра внутри сделки
    assert len(run_trades(gapped, side, hold=4, bar_ms=BAR)["gross_bps"]) == 0


def test_signal_uses_only_the_past():
    c = walk(3000, 1)
    a = zscore(frame(c), 12)
    c2 = c.copy()
    c2[2000:] *= 1.5                                   # меняем будущее
    b = zscore(frame(c2), 12)
    np.testing.assert_allclose(a[:2000], b[:2000], equal_nan=True)


def test_mean_reversion_wins_on_reverting_series_and_momentum_loses():
    b = frame(ou(30000, theta=0.05, seed=2))
    z = zscore(b, 12)
    mr = evaluate(b, make_side(z, 2.0, flip=False), 6, 0, len(b), BAR, cost_bps=0)
    mom = evaluate(b, make_side(z, 2.0, flip=True), 6, 0, len(b), BAR, cost_bps=0)
    assert mr["n"] > 100 and mr["mean_gross_bps"] > 0 and mr["t"] > 3
    assert mom["mean_gross_bps"] < 0 and mom["t"] < -3
    assert evaluate(b, make_side(z, 2.0, False), 6, 0, len(b), BAR, cost_bps=5)["mean_net_bps"] == \
        pytest.approx(mr["mean_gross_bps"] - 5)


def test_random_walk_has_no_edge_and_placebo_agrees():
    b = frame(walk(30000, 3))
    z = zscore(b, 12)
    side = make_side(z, 2.0, False)
    m = evaluate(b, side, 6, 0, len(b), BAR, 0)
    assert abs(m["t"]) < 3.5
    p = placebo_p(b, side, 6, 0, len(b), BAR, m["mean_gross_bps"])
    assert p is not None and p > 0.01


def test_placebo_is_small_when_the_edge_is_real():
    b = frame(ou(30000, theta=0.05, seed=4))
    side = make_side(zscore(b, 12), 2.0, False)
    m = evaluate(b, side, 6, 0, len(b), BAR, 0)
    assert placebo_p(b, side, 6, 0, len(b), BAR, m["mean_gross_bps"]) < 0.05


def test_walk_forward_logs_every_variant_and_finds_nothing_on_noise():
    logged = []
    grid = {"span": [12, 48], "z_entry": [1.5, 2.5], "hold": [6, 12]}
    res = walk_forward_bars(frame(walk(40000, 5)), BAR, MR, grid, cost_bps=10, n_chunks=4, min_n=20,
                            on_result=lambda s, p, m, r: logged.append(s))
    assert len(res) == 3
    assert logged.count("train") == 3 * 8                              # каждый вариант на каждом куске
    assert [s for s in logged if s.startswith("wf-")] == ["wf-1", "wf-2", "wf-3"]
    for r in res:
        assert r["test"]["mean_net_bps"] < 0                           # на шуме после издержек только минус


def test_walk_forward_finds_real_reversion_before_costs():
    grid = {"span": [12], "z_entry": [2.0], "hold": [6]}
    res = walk_forward_bars(frame(ou(40000, 0.05, seed=6)), BAR, MR, grid, cost_bps=0, n_chunks=3, min_n=20)
    assert all(r["test"]["mean_gross_bps"] > 0 and r["test"]["placebo_p"] < 0.05 for r in res)


def test_pooled_summary_thresholds_and_empty_instrument():
    rng = np.random.default_rng(0)
    mk = lambda n: {"gross": rng.normal(0, 10, n), "t": np.arange(n) * 86_400_000}  # noqa: E731
    one = pooled_summary({"A": mk(100)})
    assert one["t_threshold_inst"] == pytest.approx(1.96, abs=0.01)    # одна проверка: обычные 5%
    eight = pooled_summary({f"I{i}": mk(100) for i in range(8)}, family=6)
    assert eight["t_threshold_inst"] > 2.6 and eight["n"] == 800       # восемь инструментов: порог строже
    assert eight["t_threshold_pooled"] == pytest.approx(2.64, abs=0.01)  # шесть стратегий в серии
    empty = pooled_summary({"A": {"gross": np.array([]), "t": np.array([], dtype=np.int64)}})
    assert empty["n"] == 0 and empty["t"] is None


def test_clustered_t_does_not_count_copies_as_new_evidence():
    """Десять монет с одинаковым результатом в один день: это одно наблюдение, а не десять."""
    rng = np.random.default_rng(1)
    daily = rng.normal(2.0, 10.0, 60)                 # 60 дней, слабый эффект
    days = np.arange(60)
    t_one = clustered_t(daily, days)
    copies, copy_days = np.tile(daily, 10), np.tile(days, 10)
    assert clustered_t(copies, copy_days) == pytest.approx(t_one, rel=1e-9)
    naive = copies.mean() / (copies.std(ddof=1) / np.sqrt(len(copies)))
    assert naive > 3 * t_one                          # обычный t раздут втрое
    # независимые наблюдения: группировка ничего не меняет
    x = rng.normal(1.0, 5.0, 500)
    assert clustered_t(x, np.arange(500)) == pytest.approx(x.mean() / (x.std(ddof=1) / np.sqrt(500)), rel=0.01)
    assert clustered_t(np.array([1.0]), np.array([0])) is None


def test_vol_rank_uses_only_the_past_and_filter_masks_regime():
    rng = np.random.default_rng(2)
    # первая половина тихая, вторая бурная
    ret = np.r_[rng.normal(0, 1e-4, 1500), rng.normal(0, 1e-3, 1500)]
    b = frame(100 * np.exp(np.cumsum(ret)))
    r = vol_rank(b)
    assert np.isnan(r[:700]).all() and r[1600] > 0.85                    # после скачка волатильность у верхней границы
    b2 = b.copy()
    b2.loc[2000:, "close"] *= 2.0                                           # меняем будущее
    np.testing.assert_allclose(vol_rank(b2)[:2000], r[:2000], equal_nan=True)
    low = sig_price_vol(b, 12, 1.5, flip=False, regime="low")
    high = sig_price_vol(b, 12, 1.5, flip=True, regime="high")
    plain = sig_price(b, 12, 1.5, flip=False)
    assert (low[:743] == 0).all() and (high[:743] == 0).all()               # режим ещё неизвестен: не торгуем
    assert (low[1600:1900] == 0).all() and (high[1600:1900] != 0).any()     # в бурной части low молчит
    assert ((low != 0) <= (plain != 0)).all()                               # фильтр только убирает сигналы


def test_basis_signal_fades_premium():
    n = 2000
    b = frame(np.full(n, 100.0))
    basis = np.random.default_rng(3).normal(2.0, 0.5, n)
    basis[1500], basis[1700] = 12.0, -9.0          # фьючерс резко подорожал / подешевел к споту
    b["basis_bps"] = basis
    side = sig_basis(b, 12, 3.0)
    assert side[1500] == -1 and side[1700] == 1


def test_load_bars_adds_basis_from_spot(tmp_path):
    from hftlab.candles import path_for
    ts = np.arange(600) * BAR
    for inst, px in (("X-USDT-SWAP", 100.1), ("X-USDT", 100.0)):
        df = pd.DataFrame({"ts_ms": ts, "open": px, "high": px, "low": px, "close": px, "vol": 1.0, "quote_vol": 1.0})
        if inst == "X-USDT":
            df = df.drop(index=range(300, 303)).reset_index(drop=True)   # дыра в споте: одна 15-минутка выпадет
        p = path_for(tmp_path, inst, "5m")
        p.parent.mkdir(parents=True)
        df.to_parquet(p, index=False)
    bars, _ = load_bars(str(tmp_path), "X-USDT-SWAP", "15m", needs="basis")
    assert len(bars) == 199 and bars["basis_bps"].iloc[0] == pytest.approx(10.0, rel=1e-3)
    plain, n_gaps = load_bars(str(tmp_path), "X-USDT-SWAP", "15m")
    assert len(plain) == 200 and n_gaps == 0 and "basis_bps" not in plain
