import json

import numpy as np
import pandas as pd
import pytest

from hftlab.oms import OState
from hftlab.paper import PaperTrader, replay
from hftlab.research import sig_micro_drift, sig_mr_ema
from hftlab.sim import SimConfig, simulate

CFG = SimConfig(latency_ms=50, ttl_ms=700, hold_ms=1500)


def market(n=6000, seed=3):
    """Синтетический рынок: цена гуляет по тикам, объёмы и сделки случайные."""
    rng = np.random.default_rng(seed)
    tick = 0.1
    bid = np.round(1000 + np.cumsum(rng.choice([-1, 0, 0, 1], n)) * tick, 1)
    ask = np.round(bid + tick, 1)
    bid_sz, ask_sz = rng.uniform(0.5, 6, n), rng.uniform(0.5, 6, n)
    b = pd.DataFrame({"ts_ms": np.arange(n) * 100, "seg": 1, "bid": bid, "ask": ask,
                      "bid_sz": bid_sz, "ask_sz": ask_sz, "mid": (bid + ask) / 2,
                      "imb1": (bid_sz - ask_sz) / (bid_sz + ask_sz)})
    b["imb5"] = b["imb1"]
    m = n * 3
    t_ts = np.sort(rng.integers(0, n * 100, m))
    row = np.clip(np.searchsorted(b["ts_ms"].to_numpy(), t_ts, side="right") - 1, 0, n - 1)
    side = rng.choice([-1, 1], m)
    px = np.where(side > 0, ask[row], bid[row])   # покупатель бьёт в аск, продавец в бид
    t = pd.DataFrame({"ts_ms": t_ts, "px": px, "sz": rng.uniform(0.1, 3, m), "side": side})
    return b, t


KEYS = ("t_signal", "t_fill", "side", "entry", "exit", "gross_bps", "net_bps")


@pytest.mark.parametrize("name,fn,params", [
    ("micro_drift", sig_micro_drift, {"imb_entry": 0.6, "depth": 1}),
    ("mr_ema", sig_mr_ema, {"halflife_s": 1.0, "z_entry": 1.5}),
])
def test_paper_replay_matches_simulator(name, fn, params):
    """Два независимых кода (пакетный и событийный) обязаны дать одни и те же сделки."""
    b, t = market()
    sim = simulate(b, t, fn(b, **params), CFG)
    trader = PaperTrader("X", name, params, CFG)
    replay(trader, b, t)
    assert len(sim["trades"]) >= 10                       # тест не пустой
    got = [{k: tr[k] for k in KEYS} for tr in trader.trades]
    want = [{k: tr[k] for k in KEYS} for tr in sim["trades"]]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g == pytest.approx(w)
    s = trader.summary()
    assert s["rejected"] == sim["rejected"] and s["anomalies"] == 0
    assert all(o.state in (OState.FILLED, OState.CANCELED, OState.REJECTED)
               for o in list(trader.oms.orders.values())[:-2])


def test_loss_limit_stops_trading_and_log_is_written(tmp_path):
    b, t = market(3000)
    log = tmp_path / "paper.jsonl"
    # лимит убытка заведомо мал: первая же сделка (комиссии 7 б.п.) его пробьёт
    trader = PaperTrader("X", "micro_drift", {"imb_entry": 0.6, "depth": 1}, CFG,
                         qty=1.0, max_daily_loss=0.01, log_path=log)
    replay(trader, b, t)
    trader.close()
    s = trader.summary()
    assert s["killed"] is True and s["fills"] <= 2 and s["risk_rejected"] > 0
    assert abs(trader.oms.position.get("X", 0.0)) < 1e-12      # позиция закрыта
    types = [json.loads(line)["type"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert "submit" in types and "kill" in types and types.count("kill") == 1


def test_paper_compare_on_replay_matches_fully(tmp_path):
    from hftlab.paper_compare import compare, read_log
    b, t = market(4000)
    params = {"imb_entry": 0.6, "depth": 1}
    log = tmp_path / "p.jsonl"
    trader = PaperTrader("X", "micro_drift", params, CFG, log_path=log)
    replay(trader, b, t)
    trader.close()
    r = compare(read_log(log), b, t, "micro_drift", params, CFG)
    assert r["paper_trades"] >= 5
    assert r["matched"] == r["paper_trades"] == r["same_fill_time"]
    assert r["matched_mean_abs_diff_bps"] == pytest.approx(0.0)
    # симулятор может знать ещё одну сделку после последней заявки paper, не больше
    assert r["only_paper"] == [] and len(r["only_sim"]) <= 1
