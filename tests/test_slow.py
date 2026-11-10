import numpy as np
import pandas as pd
import pytest

from hftlab.slow import evaluate, make_side, placebo_p, run_trades, walk_forward_bars, zscore

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
    res = walk_forward_bars(frame(walk(40000, 5)), BAR, flip=False, cost_bps=10, n_chunks=4, min_n=20,
                            grid=grid, on_result=lambda s, p, m, r: logged.append(s))
    assert len(res) == 3
    assert logged.count("train") == 3 * 8                              # каждый вариант на каждом куске
    assert [s for s in logged if s.startswith("wf-")] == ["wf-1", "wf-2", "wf-3"]
    for r in res:
        assert r["test"]["mean_net_bps"] < 0                           # на шуме после издержек только минус


def test_walk_forward_finds_real_reversion_before_costs():
    grid = {"span": [12], "z_entry": [2.0], "hold": [6]}
    res = walk_forward_bars(frame(ou(40000, 0.05, seed=6)), BAR, flip=False, cost_bps=0, n_chunks=3,
                            min_n=20, grid=grid)
    assert all(r["test"]["mean_gross_bps"] > 0 and r["test"]["placebo_p"] < 0.05 for r in res)
