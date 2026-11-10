import pytest

from hftlab.universe import build_table


def test_build_table_math_and_filters():
    inst = [
        {"instId": "AAA-USDT-SWAP", "state": "live", "settleCcy": "USDT", "tickSz": "0.001", "ctVal": "10"},
        {"instId": "BBB-USD-SWAP", "state": "live", "settleCcy": "BBB", "tickSz": "0.1", "ctVal": "1"},
        {"instId": "CCC-USDT-SWAP", "state": "suspend", "settleCcy": "USDT", "tickSz": "0.1", "ctVal": "1"},
    ]
    tick = [
        {"instId": "AAA-USDT-SWAP", "bidPx": "0.999", "askPx": "1.001", "last": "1.0", "vol24h": "500000"},
        {"instId": "BBB-USD-SWAP", "bidPx": "10", "askPx": "10.1", "last": "10", "vol24h": "1"},
        {"instId": "CCC-USDT-SWAP", "bidPx": "10", "askPx": "10.1", "last": "10", "vol24h": "1"},
        {"instId": "DDD-USDT-SWAP", "bidPx": "", "askPx": "", "last": "", "vol24h": ""},
    ]
    [r] = build_table(inst, tick, maker_fee_bps=2.0)       # не-USDT, неактивный и пустой отброшены
    assert r["inst"] == "AAA-USDT-SWAP"
    assert r["tick_bps"] == pytest.approx(10.0) and r["spread_bps"] == pytest.approx(20.0)
    assert r["spread_ticks"] == 2
    assert r["vol_musd"] == pytest.approx(5.0)             # 500000 контрактов x 10 x 1.0
    assert r["edge_bps"] == pytest.approx(16.0)
