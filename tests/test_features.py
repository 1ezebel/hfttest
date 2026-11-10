import json

import pytest

from hftlab.features import add_features, load, parse, write_by_day

T0 = 1_794_318_000_000  # мс


def book(action, seq, prev, bids, asks, ts):
    return json.dumps({"arg": {"channel": "books", "instId": "X"}, "action": action,
                       "data": [{"bids": bids, "asks": asks, "ts": str(ts), "prevSeqId": prev, "seqId": seq}]})


def trade(tid, px, sz, side, ts):
    return json.dumps({"arg": {"channel": "trades", "instId": "X"},
                       "data": [{"tradeId": str(tid), "px": str(px), "sz": str(sz), "side": side, "ts": str(ts)}]})


def L(p, s):
    return [str(p), str(s), "0", "1"]


def lines():
    return [
        (1, "#CONNECT"),
        (2, book("snapshot", 1, -1, [L(100, 3)], [L(101, 1)], T0)),
        (3, trade(1, 101, 0.5, "buy", T0 + 50)),
        (4, trade(2, 100, 0.2, "sell", T0 + 60)),
        (5, book("update", 2, 1, [L(100, 5)], [], T0 + 100)),          # бид вырос на 2
        (6, trade(2, 100, 0.2, "sell", T0 + 60)),                      # дубль сделки
        (7, book("update", 3, 2, [], [L(101, 0), L(102, 4)], T0 + 200)),  # аск ушёл выше
        (8, book("update", 9, 7, [L(100, 1)], [], T0 + 300)),          # gap: стакан недействителен
        (9, book("update", 10, 9, [L(100, 1)], [], T0 + 400)),         # без snapshot: игнор
        (10, book("snapshot", 20, -1, [L(100, 2)], [L(102, 2)], T0 + 500)),
    ]


def test_parse_and_features():
    b, t = parse(lines(), "X")
    assert len(t) == 2                      # дубль сделки убран
    assert list(b["ts_ms"] - T0) == [0, 100, 200, 500]
    f = add_features(b, t)

    assert list(f["seg"]) == [1, 1, 1, 2]   # после gap начался новый кусок
    r0, r1, r2, r3 = (f.iloc[i] for i in range(4))
    assert r0["mid"] == 100.5 and r0["spread"] == 1
    assert r0["microprice"] == pytest.approx((100 * 1 + 101 * 3) / 4)   # тянется к аску: бид толще
    assert r0["imb1"] == pytest.approx(0.5)
    assert r1["ofi"] == pytest.approx(2.0)          # на биде прибавилось 2
    assert r2["ofi"] == pytest.approx(1.0)          # аск 101 (размер 1) исчез: давление вверх
    assert r0["ofi"] == 0 and r3["ofi"] == 0        # первая строка куска
    # сделки из интервала (T0, T0+100] попали в строку T0+100, а не в прошлую
    assert (r0["buy_vol"], r0["n_trades"]) == (0, 0)
    assert (r1["buy_vol"], r1["sell_vol"], r1["n_trades"]) == (0.5, 0.2, 2)


def test_quiet_gap_starts_new_segment():
    ls = [
        (1, "#CONNECT"),
        (2, book("snapshot", 1, -1, [L(100, 1)], [L(101, 1)], T0)),
        (3, book("update", 2, 1, [L(100, 2)], [], T0 + 5000)),   # 5 секунд тишины
    ]
    f = add_features(*parse(ls, "X"))
    assert list(f["seg"]) == [1, 2]


def test_write_and_load_roundtrip(tmp_path):
    b, t = parse(lines(), "X")
    f = add_features(b, t)
    files = write_by_day(f, t, tmp_path / "X")
    assert len(files) == 2
    assert len(load(tmp_path, "X")) == 4
    assert len(load(tmp_path, "X", "trades")) == 2
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "X", date_from="2099-01-01")
