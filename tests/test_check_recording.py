import json
from datetime import datetime, timezone
from pathlib import Path

from hftlab.check_recording import collect_by_instrument, replay, _chain
from hftlab.storage import HourlyWriter

T0 = int(datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc).timestamp() * 1e9)
MS = 1_000_000


def book_msg(action, seq, prev, bids, asks, ts_ms):
    return json.dumps(
        {
            "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
            "action": action,
            "data": [
                {
                    "bids": bids,
                    "asks": asks,
                    "ts": str(ts_ms),
                    "checksum": 0,
                    "prevSeqId": prev,
                    "seqId": seq,
                }
            ],
        },
        separators=(",", ":"),
    )


def trade_msg(ts_ms):
    return json.dumps(
        {
            "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
            "data": [{"px": "100", "sz": "1", "side": "buy", "ts": str(ts_ms)}],
        },
        separators=(",", ":"),
    )


def make_recording(root: Path):
    w = HourlyWriter(root)
    k = "BTC-USDT-SWAP"
    t = T0
    w.write_event(k, t, "CONNECT")
    # snapshot, обычный update, сделка, потом пропуск (prev=99 вместо 12), recovery
    w.write_raw(k, t + 10 * MS, book_msg("snapshot", 10, -1, [["99", "5", "0", "1"]], [["100", "2", "0", "1"]], 10_000))
    w.write_raw(k, t + 20 * MS, book_msg("update", 12, 10, [["99", "6", "0", "1"]], [], 10_005))
    w.write_raw(k, t + 25 * MS, trade_msg(10_006))
    w.write_raw(k, t + 30 * MS, book_msg("update", 100, 99, [], [["100", "1", "0", "1"]], 10_020))
    w.write_event(k, t + 30 * MS, "GAP", "gap")
    w.write_raw(k, t + 40 * MS, book_msg("snapshot", 200, -1, [["99", "5", "0", "1"]], [["100", "2", "0", "1"]], 10_035))
    w.write_raw(k, t + 50 * MS, book_msg("update", 201, 200, [["98", "1", "0", "1"]], [], 10_045))
    w.close()


def test_replay_counts_and_detects_gap(tmp_path):
    make_recording(tmp_path)
    groups = collect_by_instrument(tmp_path)
    assert len(groups) == 1
    (inst_dir, files), = groups.items()
    assert inst_dir.name == "BTC-USDT-SWAP"

    rep = replay(_chain(files), inst_dir.name)
    assert rep.channels["books"] == 5 and rep.channels["trades"] == 1
    assert rep.snapshots == 2
    assert rep.connects == 1
    assert rep.gaps_detected == 1
    assert rep.gap_events == rep.gaps_detected  # рекордер и повторный проход согласны
    assert rep.crossed == 0
    assert len(rep.latencies_ms) == 5
    assert all(abs(x - 0.0) < 1e-6 or x > 0 for x in rep.latencies_ms)  # recv не раньше ts


def _snap(seq, bids, asks, ts=1000):
    return json.dumps({"arg": {"channel": "books", "instId": "X"}, "action": "snapshot",
                       "data": [{"bids": bids, "asks": asks, "ts": str(ts), "prevSeqId": -1, "seqId": seq}]})


def _upd(seq, prev, bids, asks, ts=1000):
    return json.dumps({"arg": {"channel": "books", "instId": "X"}, "action": "update",
                       "data": [{"bids": bids, "asks": asks, "ts": str(ts), "prevSeqId": prev, "seqId": seq}]})


def test_audit_ok_bad_pending_and_unmatched():
    from hftlab.check_recording import replay
    b, a = [["100", "1", "0", "1"]], [["101", "2", "0", "1"]]
    lines = [
        (1, "#CONNECT"),
        (2, _snap(10, b, a)),
        (3, "#AUDIT " + _snap(10, [["100.0", "1.00", "0", "1"]], a)),   # тот же стакан, иная запись чисел
        (4, "#AUDIT " + _snap(12, [["100", "7", "0", "1"]], a)),        # из будущего: ждёт seq 12
        (5, _upd(12, 10, [["100", "7", "0", "1"]], [])),                 # дошли: совпало
        (6, _upd(13, 12, [["100", "9", "0", "1"]], [])),
        (7, "#AUDIT " + _snap(12, [["100", "8", "0", "1"]], a)),        # прошлое состояние, размер не тот
        (8, "#AUDIT " + _snap(5, b, a)),                                 # такого seq у нас не было
        (9, "#RESUB"),
    ]
    rep = replay(lines, "X")
    assert rep.audits == 4 and rep.resub_events == 1
    assert (rep.audit_ok, rep.audit_bad, rep.audit_unmatched) == (2, 1, 1)
