"""Локальный стакан OKX (канал `books`) с проверкой непрерывности по seqId/prevSeqId.

Формат сообщений канала `books`:
    {"arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
     "action": "snapshot" | "update",
     "data": [{"asks": [[price, size, "0", numOrders], ...],
               "bids": [...], "ts": "<ms>", "checksum": 0,
               "prevSeqId": -1, "seqId": 123}]}

Правила:
  * snapshot полностью заменяет стакан (prevSeqId у него -1).
  * update применим, только если его prevSeqId == seqId предыдущего сообщения.
    Иначе мы потеряли данные: стакан считаем недействительным до нового snapshot.
  * size == 0 означает "удалить уровень".
  * Поле checksum НЕ используем: OKX отключил его в каналах books
    (в продакшене с 23.06.2026 оно всегда 0). Целостность = цепочка seqId.
    Это слабее, чем checksum (ловит пропуски, но не подмену содержимого),
    поэтому полезно периодически пересоздавать подписку (см. docs/ROADMAP.md).

Цены и размеры хранятся строками, как пришли от биржи (без ошибок float
при сравнении ключей), во float переводим только при расчётах.
"""
from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    OK = "ok"
    NO_SNAPSHOT = "no_snapshot"  # update пришёл, а snapshot ещё не было
    GAP = "gap"  # prevSeqId не совпал с последним seqId: пропуск
    RESET = "reset"  # seqId < prevSeqId: биржа сбросила нумерацию (обслуживание)


def _merge(side: dict[str, str], levels) -> None:
    for level in levels:
        price, size = level[0], level[1]
        if float(size) == 0.0:
            side.pop(price, None)
        else:
            side[price] = size


class OrderBook:
    def __init__(self, inst_id: str) -> None:
        self.inst_id = inst_id
        self.bids: dict[str, str] = {}
        self.asks: dict[str, str] = {}
        self.last_seq: int | None = None
        self.ts_ms: int | None = None

    # ---- состояние -------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.last_seq is not None

    def invalidate(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.last_seq = None
        self.ts_ms = None

    def apply(self, action: str, data: dict) -> Status:
        """Применить одно сообщение (один элемент `data`). Возвращает статус."""
        seq_id = int(data["seqId"])
        prev_seq_id = int(data.get("prevSeqId", -1))

        if action == "snapshot":
            self.bids.clear()
            self.asks.clear()
            _merge(self.bids, data["bids"])
            _merge(self.asks, data["asks"])
            self.last_seq = seq_id
            self.ts_ms = int(data["ts"])
            return Status.OK

        if not self.ready:
            return Status.NO_SNAPSHOT

        if prev_seq_id != self.last_seq:
            status = Status.RESET if seq_id < prev_seq_id else Status.GAP
            self.invalidate()
            return status

        # prevSeqId == seqId (heartbeat без изменений) тоже сюда: списки пусты.
        _merge(self.bids, data["bids"])
        _merge(self.asks, data["asks"])
        self.last_seq = seq_id
        self.ts_ms = int(data["ts"])
        return Status.OK

    # ---- чтение стакана --------------------------------------------------
    def top(self, n: int = 5):
        """(bids, asks): списки (price, size) во float. Бид по убыванию, аск по возрастанию."""
        bids = sorted(((float(p), float(s)) for p, s in self.bids.items()), reverse=True)[:n]
        asks = sorted((float(p), float(s)) for p, s in self.asks.items())[:n]
        return bids, asks

    def best_bid(self):
        if not self.bids:
            return None
        p = max(self.bids, key=float)
        return float(p), float(self.bids[p])

    def best_ask(self):
        if not self.asks:
            return None
        p = min(self.asks, key=float)
        return float(p), float(self.asks[p])

    def is_crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        return bb is not None and ba is not None and bb[0] >= ba[0]

    def mid(self):
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb[0] + ba[0]) / 2

    def spread(self):
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba[0] - bb[0]

    def microprice(self):
        """Mid, взвешенный объёмами на лучших уровнях.

        Если на биде объёма больше, чем на аске, цена вероятнее пойдёт вверх,
        и microprice смещается к аску.
        """
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        (bp, bs), (ap, as_) = bb, ba
        return (bp * as_ + ap * bs) / (bs + as_)

    def imbalance(self, depth: int = 5):
        """Дисбаланс объёма по top-`depth` уровням: от -1 (давит продавец) до +1 (давит покупатель)."""
        bids, asks = self.top(depth)
        b = sum(s for _, s in bids)
        a = sum(s for _, s in asks)
        if b + a == 0:
            return None
        return (b - a) / (b + a)
