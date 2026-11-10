from datetime import datetime, timezone

from hftlab.storage import HourlyWriter, format_event, hour_path, iter_lines, parse_line


def ns(y, mo, d, h, mi=0):
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() * 1e9)


def test_hour_path_is_utc(tmp_path):
    p = hour_path(tmp_path, "okx", "BTC-USDT-SWAP", ns(2026, 11, 10, 14, 59))
    assert p == tmp_path / "okx" / "BTC-USDT-SWAP" / "2026-11-10" / "14.jsonl.gz"


def test_parse_line_roundtrip_and_garbage():
    assert parse_line('123\t{"a":1}\n') == (123, '{"a":1}')
    assert parse_line(format_event(5, "GAP", "gap").decode()) == (5, "#GAP gap")
    assert parse_line("no tab here") is None
    assert parse_line("abc\tpayload") is None


def test_write_read_and_hourly_rotation(tmp_path):
    w = HourlyWriter(tmp_path)
    t1, t2 = ns(2026, 11, 10, 14, 59), ns(2026, 11, 10, 15, 0)
    w.write_raw("BTC", t1, '{"x":1}')
    w.write_event("BTC", t2, "CONNECT")
    w.write_raw("BTC", t2 + 1, '{"x":2}')
    w.close()

    f14 = tmp_path / "okx" / "BTC" / "2026-11-10" / "14.jsonl.gz"
    f15 = tmp_path / "okx" / "BTC" / "2026-11-10" / "15.jsonl.gz"
    assert list(iter_lines(f14)) == [(t1, '{"x":1}')]
    assert list(iter_lines(f15)) == [(t2, "#CONNECT"), (t2 + 1, '{"x":2}')]


def test_truncated_tail_does_not_crash(tmp_path):
    w = HourlyWriter(tmp_path)
    t = ns(2026, 11, 10, 9)
    for i in range(2000):
        w.write_raw("BTC", t + i, '{"payload":"%s"}' % ("z" * 50))
    w.flush()
    w.close()
    f = tmp_path / "okx" / "BTC" / "2026-11-10" / "09.jsonl.gz"
    f.write_bytes(f.read_bytes()[:-6])  # имитация аварийного обрыва
    list(iter_lines(f))  # главное: без исключения
