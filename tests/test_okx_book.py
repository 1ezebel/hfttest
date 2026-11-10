import pytest

from hftlab.okx_book import OrderBook, Status


def snap(seq, bids, asks, ts="1000"):
    return {"bids": bids, "asks": asks, "ts": ts, "prevSeqId": -1, "seqId": seq}


def upd(prev, seq, bids=(), asks=(), ts="1100"):
    return {"bids": list(bids), "asks": list(asks), "ts": ts, "prevSeqId": prev, "seqId": seq}


def lvl(p, s):
    return [p, s, "0", "1"]


@pytest.fixture
def book():
    b = OrderBook("BTC-USDT-SWAP")
    st = b.apply(
        "snapshot",
        snap(10, [lvl("99.0", "5"), lvl("98.5", "3")], [lvl("100.0", "2"), lvl("100.5", "4")]),
    )
    assert st is Status.OK
    return b


def test_snapshot_builds_book(book):
    assert book.best_bid() == (99.0, 5.0)
    assert book.best_ask() == (100.0, 2.0)
    assert book.mid() == 99.5
    assert book.spread() == 1.0
    assert not book.is_crossed()
    assert book.last_seq == 10


def test_update_adds_changes_and_deletes_levels(book):
    st = book.apply("update", upd(10, 11, bids=[lvl("99.0", "0"), lvl("98.0", "7")], asks=[lvl("100.0", "9")]))
    assert st is Status.OK
    bids, asks = book.top(5)
    assert bids == [(98.5, 3.0), (98.0, 7.0)]  # 99.0 удалён
    assert asks[0] == (100.0, 9.0)  # размер заменён, не сложен


def test_gap_invalidates_book_until_new_snapshot(book):
    st = book.apply("update", upd(12, 13, bids=[lvl("99.0", "1")]))  # ждали prev=10
    assert st is Status.GAP
    assert not book.ready and book.best_bid() is None
    assert book.apply("update", upd(13, 14)) is Status.NO_SNAPSHOT
    assert book.apply("snapshot", snap(50, [lvl("1", "1")], [lvl("2", "1")])) is Status.OK
    assert book.apply("update", upd(50, 51)) is Status.OK


def test_seq_reset_is_reported(book):
    st = book.apply("update", upd(500, 3))  # seqId < prevSeqId: нумерацию сбросили
    assert st is Status.RESET
    assert not book.ready


def test_heartbeat_update_keeps_book(book):
    assert book.apply("update", upd(10, 10)) is Status.OK
    assert book.best_bid() == (99.0, 5.0)
    assert book.last_seq == 10


def test_update_before_snapshot_is_rejected():
    b = OrderBook("X")
    assert b.apply("update", upd(1, 2)) is Status.NO_SNAPSHOT


def test_microprice_leans_toward_heavier_side():
    b = OrderBook("X")
    b.apply("snapshot", snap(1, [lvl("99", "9")], [lvl("101", "1")]))
    # на биде в 9 раз больше объёма -> цена вероятнее вверх -> ближе к аску
    assert b.microprice() == pytest.approx((99 * 1 + 101 * 9) / 10)
    assert b.microprice() > b.mid()


def test_imbalance_range():
    b = OrderBook("X")
    b.apply("snapshot", snap(1, [lvl("99", "3")], [lvl("101", "1")]))
    assert b.imbalance(1) == pytest.approx(0.5)


def test_crossed_book_detected():
    b = OrderBook("X")
    b.apply("snapshot", snap(1, [lvl("101", "1")], [lvl("100", "1")]))
    assert b.is_crossed()
