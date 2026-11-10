from hftlab.netcheck import TS_RE, verdict

OK = {"seconds": 30, "cpu_share": 0.05, "lag_first_ms": 110.0, "lag_last_ms": 120.0}


def test_ts_regex_finds_exchange_time():
    raw = '{"arg":{"channel":"trades"},"data":[{"px":"1","ts":"1794318113689","seqId":1}]}'
    assert TS_RE.search(raw).group(1) == "1794318113689"


def test_verdicts():
    assert verdict(20, 150, OK)[0].startswith("ВСЁ В ПОРЯДКЕ")
    assert verdict(1400, 150, OK)[0].startswith("ЧАСЫ") and "отстают" in verdict(1400, 150, OK)[0]
    assert "спешат" in verdict(-900, 150, OK)[0]
    growing = {**OK, "lag_first_ms": 2000.0, "lag_last_ms": 20000.0}
    assert verdict(0, 150, growing)[0].startswith("КАНАЛ")
    assert verdict(0, 150, {**growing, "cpu_share": 0.95})[0].startswith("ПРОЦЕССОР")
    assert "постоянное" in verdict(0, 150, {**OK, "lag_first_ms": 3000.0, "lag_last_ms": 3100.0})[0]
    assert any(v.startswith("СЕТЬ") for v in verdict(0, 1500, OK))
