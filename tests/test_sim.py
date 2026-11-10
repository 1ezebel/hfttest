import numpy as np
import pandas as pd
import pytest

from hftlab.sim import SimConfig, simulate

CFG = SimConfig(latency_ms=100, ttl_ms=1000, hold_ms=1000, maker_fee_bps=2.0, taker_fee_bps=5.0)


def book(n=60, bid=100.0, ask=100.1, bid_sz=10.0, ask_sz=10.0, seg=None):
    return pd.DataFrame({
        "ts_ms": np.arange(n) * 100, "seg": np.ones(n, dtype=int) if seg is None else seg,
        "bid": np.full(n, bid), "ask": np.full(n, ask),
        "bid_sz": np.full(n, bid_sz), "ask_sz": np.full(n, ask_sz),
        "mid": np.full(n, (bid + ask) / 2),
    })


def trades(rows):
    return pd.DataFrame(rows, columns=["ts_ms", "px", "sz", "side"])


def buy_signal(n=60, at=5):
    side = np.zeros(n, dtype=np.int8)
    side[at] = 1
    return side


def test_no_fill_until_queue_is_eaten():
    b = book()
    # очередь перед нами 10; продали в бид только 9: нас не исполнили
    r = simulate(b, trades([(700, 100.0, 9.0, -1)]), buy_signal(), CFG)
    assert (r["orders"], r["fills"]) == (1, 0)
    # продали ещё 2: очередь съедена, остаток достался нам
    r = simulate(b, trades([(700, 100.0, 9.0, -1), (800, 100.0, 2.0, -1)]), buy_signal(), CFG)
    assert r["fills"] == 1 and r["trades"][0]["t_fill"] == 800


def test_buy_trades_and_other_prices_do_not_fill_us():
    b = book()
    r = simulate(b, trades([(700, 100.0, 50.0, 1), (750, 100.1, 50.0, -1)]), buy_signal(), CFG)
    assert r["fills"] == 0


def test_trade_through_our_price_fills_regardless_of_queue():
    r = simulate(book(), trades([(700, 99.9, 0.1, -1)]), buy_signal(), CFG)
    assert r["fills"] == 1


def test_latency_and_ttl_window():
    b = book()
    # сигнал в 500 мс: заявка активна с 600, отмена действует в 1600
    assert simulate(b, trades([(600, 99.9, 1.0, -1)]), buy_signal(), CFG)["fills"] == 0   # ещё не дошла
    assert simulate(b, trades([(1601, 99.9, 1.0, -1)]), buy_signal(), CFG)["fills"] == 0  # уже отменена
    assert simulate(b, trades([(1600, 99.9, 1.0, -1)]), buy_signal(), CFG)["fills"] == 1  # отмена в пути


def test_pnl_pays_spread_and_fees():
    # цена стоит на месте: купили по 100.0 (бид), вышли по рынку в тот же бид 100.0
    r = simulate(book(), trades([(700, 99.9, 1.0, -1)]), buy_signal(), CFG)
    tr = r["trades"][0]
    assert tr["gross_bps"] == pytest.approx(0.0)
    assert tr["net_bps"] == pytest.approx(-7.0)          # 2 maker + 5 taker
    assert tr["mid_markout_bps"] == pytest.approx(5.0, rel=1e-2)  # mid выше бида на полспреда


def test_post_only_reject_when_price_crosses():
    b = book()
    b.loc[6:, ["bid", "ask"]] = [99.8, 99.9]   # к приходу заявки аск уже ниже нашей цены 100.0
    b["mid"] = (b["bid"] + b["ask"]) / 2
    r = simulate(b, trades([(700, 99.9, 1.0, -1)]), buy_signal(), CFG)
    assert (r["rejected"], r["orders"], r["fills"]) == (1, 0, 0)


def test_sell_side_mirror_and_one_position_at_a_time():
    side = np.zeros(60, dtype=np.int8)
    side[5], side[8] = -1, -1          # второй сигнал приходит, пока первый ещё в работе
    side[6] = 0
    r = simulate(book(), trades([(700, 100.2, 0.1, 1)]), side, CFG)
    assert r["fills"] == 1 and r["skipped_busy"] == 1
    assert r["trades"][0]["entry"] == 100.1 and r["trades"][0]["net_bps"] == pytest.approx(-7.0)


def test_gap_in_data_skips_order():
    seg = np.r_[np.ones(10, dtype=int), np.full(50, 2)]   # дыра внутри окна заявки
    r = simulate(book(seg=seg), trades([(700, 99.9, 1.0, -1)]), buy_signal(), CFG)
    assert (r["orders"], r["fills"], r["skipped_gap"]) == (0, 0, 1)
