import numpy as np
import pandas as pd
import pytest

from hftlab.research import evaluate, forward_mid, markout_scorer, onsets, sig_mr_ema, walk_forward


def frame(mid, seg=None, step_ms=100):
    mid = np.asarray(mid, dtype=float)
    n = len(mid)
    return pd.DataFrame({
        "ts_ms": np.arange(n) * step_ms, "seg": np.ones(n, dtype=int) if seg is None else seg,
        "mid": mid, "spread": np.full(n, 0.1), "imb1": np.zeros(n), "imb5": np.zeros(n),
    })


def test_forward_mid_stops_at_segment_border():
    b = frame([1, 2, 3, 4, 5, 6], seg=[1, 1, 1, 2, 2, 2])
    f = forward_mid(b, 200)
    # строка 0 видит строку 2; строка 1 уже упирается в границу куска
    assert f[0] == 3 and np.isnan(f[1]) and np.isnan(f[2])
    assert f[3] == 6 and np.isnan(f[4])


def test_onsets_dedupe_and_cooldown():
    side = np.array([0, 1, 1, 1, 0, 1, -1, -1, 0, 1])
    ts = np.arange(10) * 100
    seg = np.ones(10, dtype=int)
    assert list(onsets(side, ts, seg, cooldown_ms=0)) == [1, 5, 6, 9]
    assert list(onsets(side, ts, seg, cooldown_ms=500)) == [1, 6]


def test_evaluate_sign_and_costs():
    # цена растёт на 1 б.п. каждые 100 мс; сигнал "купить" в строке 1
    mid = 100 * (1 + 1e-4) ** np.arange(50)
    b = frame(mid)
    side = np.zeros(50, dtype=np.int8)
    side[1] = 1
    h = evaluate(b, side, (1000,))["horizons"]["1000ms"]
    assert h["n"] == 1 and h["gross_bps"] == pytest.approx(10.0, rel=1e-3)
    assert h["net_taker_bps"] < h["gross_bps"] - 10  # спред и две taker-комиссии
    side[1] = -1
    assert evaluate(b, side, (1000,))["horizons"]["1000ms"]["gross_bps"] == pytest.approx(-10.0, rel=1e-3)


def test_signal_has_no_lookahead():
    """Меняем будущее: сигнал в прошлом обязан остаться тем же."""
    rng = np.random.default_rng(1)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, 3000))
    a = sig_mr_ema(frame(mid), halflife_s=1.0, z_entry=1.5)
    mid2 = mid.copy()
    mid2[2000:] += 5.0
    b = sig_mr_ema(frame(mid2), halflife_s=1.0, z_entry=1.5)
    assert (a[:2000] == b[:2000]).all()
    assert (a != 0).any()


def test_walk_forward_on_noise_finds_nothing_and_logs_everything():
    """На случайном блуждании edge нет: test-результат должен быть около нуля."""
    rng = np.random.default_rng(7)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 2e-5, 40000)))
    logged = []
    grid = {"halflife_s": [1.0, 5.0], "z_entry": [1.5, 2.5]}
    res = walk_forward(frame(mid), sig_mr_ema, grid, markout_scorer(1000), n_chunks=4,
                       on_result=lambda split, p, m, c: logged.append(split))
    assert len(res) == 3
    assert logged.count("train") == 3 * 4          # каждый вариант на каждом fold
    assert [s for s in logged if s.startswith("wf-")] == ["wf-1", "wf-2", "wf-3"]
    for r in res:
        assert abs(r["test"]["t"]) < 4             # никакого "значимого" edge на шуме


def test_decay_curve_finds_peak_and_half_life(tmp_path):
    from hftlab.decay import decay_curve, plot
    # после каждого сигнала цена 1 с растёт на 2 б.п., потом к 5 с возвращается обратно
    n = 20000
    mid = np.full(n, 100.0)
    side = np.zeros(n, dtype=np.int8)
    bump = np.r_[np.linspace(0, 2e-4, 11), np.linspace(2e-4, 0, 41)[1:]]
    for i in range(100, n - 800, 800):
        side[i] = 1
        mid[i:i + len(bump)] *= 1 + bump
    c = decay_curve(frame(mid), side)
    assert c["peak_ms"] == 1000 and c["peak_bps"] == pytest.approx(2.0, rel=1e-2)
    assert c["half_life_ms"] == 5000
    assert decay_curve(frame(np.full(500, 100.0)), np.zeros(500, dtype=np.int8))["peak_ms"] is None
    plot(c, "тест", tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 5000
