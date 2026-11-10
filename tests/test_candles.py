import pandas as pd
import pytest

from hftlab.candles import BAR_MS, fetch_range, gaps, load, parse_rows, resample, update

B = BAR_MS["5m"]


def row(i, close=100.0, confirm="1"):
    """Строка в формате OKX: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]."""
    return [str(i * B), "100", "101", "99", str(close), "10", "1", "1000", confirm]


def fake_get(total, newest_unconfirmed=False):
    """Фальшивая биржа: свечи с номерами 0..total-1, страницы по 100 от новых к старым."""
    calls = []

    def get(inst, bar, after):
        calls.append(after)
        hi = total if after is None else after // B       # номера строго меньше `after`
        idx = list(range(hi - 1, max(hi - 101, -1), -1))
        return [row(i, confirm="0" if (newest_unconfirmed and i == total - 1) else "1") for i in idx]

    get.calls = calls
    return get


def test_parse_drops_unconfirmed_and_sorts():
    df = parse_rows([row(3, confirm="0"), row(2), row(1), row(1)])
    assert list(df["ts_ms"]) == [B, 2 * B]
    assert df["close"].iloc[0] == 100.0 and df["quote_vol"].iloc[0] == 1000.0


def test_fetch_range_paginates_back_to_start():
    get = fake_get(1000)
    df = fetch_range("X", "5m", start_ms=250 * B, get=get, sleep_s=0)
    assert df["ts_ms"].iloc[0] == 250 * B and df["ts_ms"].iloc[-1] == 999 * B
    assert len(df) == 750 and gaps(df, "5m") == []
    assert len(get.calls) == 8                      # 750 свечей = 8 страниц по 100


def test_fetch_range_stops_when_exchange_runs_out():
    df = fetch_range("X", "5m", start_ms=-10**9, get=fake_get(250), sleep_s=0)
    assert len(df) == 250                           # не зациклились, когда история кончилась


def test_update_is_incremental_and_skips_forming_candle(tmp_path):
    now = 600 * B
    df = update(tmp_path, "X", "5m", days=10, get=fake_get(500, newest_unconfirmed=True), now_ms=now, sleep_s=0)
    assert df["ts_ms"].iloc[-1] == 498 * B          # незакрытая свеча 499 в данные не попала
    get2 = fake_get(520)
    df2 = update(tmp_path, "X", "5m", days=10, get=get2, now_ms=now, sleep_s=0)
    assert df2["ts_ms"].iloc[-1] == 519 * B and len(df2) == 520 and not df2["ts_ms"].duplicated().any()
    assert len(get2.calls) == 1                     # докачали одной страницей, а не всё заново
    assert len(load(tmp_path, "X", "5m")) == 520
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "Y", "5m")


def test_resample_aggregates_and_drops_incomplete_groups():
    rows = []
    for i in range(1, 12):                          # 5-минутки с номера 1: первая группа неполная
        r = row(i, close=100 + i)
        r[2], r[3] = str(200 + i), str(50 + i)      # high, low
        rows.append(r)
    df = parse_rows(rows)
    out = resample(df, "5m", "15m")
    # полные 15-минутки: свечи 3,4,5 и 6,7,8 и 9,10,11 (свечи 1,2 неполная группа)
    assert list(out["ts_ms"]) == [3 * B, 6 * B, 9 * B]
    first = out.iloc[0]
    assert first["open"] == 100 and first["close"] == 105
    assert first["high"] == 205 and first["low"] == 53 and first["vol"] == 30
    with pytest.raises(ValueError):
        resample(df, "15m", "5m")


def test_gaps_found():
    df = pd.DataFrame({"ts_ms": [0, B, 4 * B, 5 * B]})
    assert gaps(df, "5m") == [(B, 2)]
