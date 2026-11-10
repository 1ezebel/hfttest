"""Этап 4: учёт ордеров (OMS), лимиты риска и kill switch. Реальных денег здесь нет.

Этот модуль не знает про биржу. Он общается с "шлюзом" (gateway): объектом с методами
send(order), cancel(client_id). Сейчас единственный шлюз это бумажный (hftlab.paper).
Настоящий шлюз появится только после этапов 4-5 и отдельного решения пользователя.

Жизнь ордера (автомат состояний):

    NEW ──ack──> ACKED ──fill──> PARTIAL ──fill──> FILLED
     │             │                │
     │             └──cancel ack────┴──> CANCELED
     ├──reject──> REJECTED
     └──нет ответа дольше таймаута──> UNKNOWN  (судьба неизвестна, нужна сверка с биржей)

Неприятные случаи из реальной жизни, которые здесь обработаны и покрыты тестами:
  * повторная отправка с тем же client_id не создаёт второй ордер (идемпотентность);
  * один и тот же fill пришёл дважды: второй игнорируется (по fill_id);
  * fill пришёл ПОСЛЕ подтверждения отмены (гонка fill/cancel): позиция всё равно
    меняется, потому что биржа права, а случай записывается в anomalies;
  * подтверждение отмены пришло после полного исполнения: ордер остаётся FILLED;
  * ответ на ордер потерялся: состояние UNKNOWN, и риск считает его худшим образом
    (как будто он ещё может исполниться);
  * сверка (reconcile): позиция и список открытых ордеров сравниваются с биржей,
    расхождения возвращаются списком, истиной считается биржа.

Kill switch: один вызов kill() отменяет всё, закрывает позицию и запрещает новые ордера.
Срабатывает сам при превышении дневного убытка.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class OState(str, Enum):
    NEW = "new"            # отправлен, биржа ещё не ответила
    ACKED = "acked"        # биржа приняла, стоит в стакане
    PARTIAL = "partial"    # исполнен частично
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"    # ответа нет слишком долго


TERMINAL = {OState.FILLED, OState.CANCELED, OState.REJECTED}


@dataclass
class Order:
    client_id: str
    inst: str
    side: int               # +1 покупка, -1 продажа
    qty: float
    price: float | None     # None = рыночный (используется только для выхода и kill switch)
    post_only: bool = True
    reduce_only: bool = False
    state: OState = OState.NEW
    filled: float = 0.0
    avg_px: float = 0.0
    sent_ms: int = 0
    cancel_requested: bool = False
    reject_reason: str = ""
    fill_ids: set[str] = field(default_factory=set)

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled)

    @property
    def open(self) -> bool:
        return self.state not in TERMINAL


class Gateway(Protocol):
    def send(self, order: Order, now_ms: int) -> None: ...
    def cancel(self, client_id: str, now_ms: int) -> None: ...


@dataclass(frozen=True)
class RiskLimits:
    max_order_qty: float            # размер одной заявки
    max_position: float             # модуль позиции по инструменту (с учётом открытых заявок)
    max_daily_loss: float           # в валюте котировки (USDT); при достижении срабатывает kill switch
    max_orders_per_sec: int = 5
    max_open_orders: int = 4


class OrderManager:
    def __init__(self, gateway: Gateway, limits: RiskLimits) -> None:
        self.gw = gateway
        self.limits = limits
        self.orders: dict[str, Order] = {}
        self.position: dict[str, float] = {}
        self.cash: dict[str, float] = {}      # деньги от сделок и комиссии, по инструментам
        self.anomalies: list[str] = []         # всё странное: поздние fills, перелив, чужие fills
        self.killed = False
        self.kill_reason = ""
        self._sent_times: deque[int] = deque()
        self._seen_fills: set[tuple[str, str]] = set()
        self._n_flatten = 0

    # ---------- отправка ----------

    def risk_check(self, o: Order, now_ms: int) -> str | None:
        """Причина отказа или None. Проверяется ДО отправки на биржу (pre-trade risk check)."""
        lim = self.limits
        if self.killed and not o.reduce_only:
            return f"kill switch: {self.kill_reason}"
        if o.qty <= 0:
            return "qty <= 0"
        if o.reduce_only:
            pos = self.position.get(o.inst, 0.0)
            if pos * o.side >= 0 or o.qty > abs(pos) + 1e-12:
                return "reduce_only увеличил бы позицию"
            return None  # закрытие позиции лимитами не блокируем
        if o.qty > lim.max_order_qty:
            return f"qty {o.qty} > max_order_qty {lim.max_order_qty}"
        if len(self.open_orders()) >= lim.max_open_orders:
            return f"открытых заявок уже {lim.max_open_orders}"
        while self._sent_times and now_ms - self._sent_times[0] >= 1000:
            self._sent_times.popleft()
        if len(self._sent_times) >= lim.max_orders_per_sec:
            return f"больше {lim.max_orders_per_sec} заявок в секунду"
        lo, hi = self.worst_case_position(o.inst)
        worst = hi + o.qty if o.side > 0 else lo - o.qty
        if abs(worst) > lim.max_position + 1e-12:
            return f"позиция могла бы стать {worst:+g} при лимите {lim.max_position}"
        return None

    def submit(self, client_id: str, inst: str, side: int, qty: float, price: float | None,
               now_ms: int, *, post_only: bool = True, reduce_only: bool = False) -> Order:
        """Отправить ордер. Повтор с тем же client_id вернёт уже существующий, ничего не отправляя."""
        if client_id in self.orders:
            return self.orders[client_id]
        o = Order(client_id, inst, side, qty, price, post_only=post_only and price is not None,
                  reduce_only=reduce_only, sent_ms=now_ms)
        reason = self.risk_check(o, now_ms)  # до регистрации: заявка не должна считать саму себя
        self.orders[client_id] = o
        if reason is not None:
            o.state, o.reject_reason = OState.REJECTED, f"risk: {reason}"
            return o
        self._sent_times.append(now_ms)
        self.gw.send(o, now_ms)
        return o

    def cancel(self, client_id: str, now_ms: int) -> None:
        o = self.orders.get(client_id)
        if o is None or not o.open or o.cancel_requested:
            return
        o.cancel_requested = True
        self.gw.cancel(client_id, now_ms)

    # ---------- ответы биржи ----------

    def on_ack(self, client_id: str) -> None:
        o = self.orders.get(client_id)
        if o is not None and o.state in (OState.NEW, OState.UNKNOWN):
            o.state = OState.ACKED

    def on_reject(self, client_id: str, reason: str) -> None:
        o = self.orders.get(client_id)
        if o is not None and o.state in (OState.NEW, OState.UNKNOWN):
            o.state, o.reject_reason = OState.REJECTED, reason

    def on_cancel_ack(self, client_id: str) -> None:
        o = self.orders.get(client_id)
        if o is not None and o.state is not OState.FILLED and o.state is not OState.REJECTED:
            o.state = OState.CANCELED

    def on_fill(self, client_id: str, fill_id: str, inst: str, side: int, qty: float, px: float,
                fee: float = 0.0) -> bool:
        """Исполнение. Возвращает False для дубля. Позиция меняется всегда: биржа права."""
        if (client_id, fill_id) in self._seen_fills:
            return False
        self._seen_fills.add((client_id, fill_id))
        self.position[inst] = self.position.get(inst, 0.0) + side * qty
        self.cash[inst] = self.cash.get(inst, 0.0) - side * qty * px - fee

        o = self.orders.get(client_id)
        if o is None:
            self.anomalies.append(f"fill {fill_id} по неизвестному ордеру {client_id}")
            return True
        if o.state in (OState.CANCELED, OState.REJECTED):
            self.anomalies.append(f"поздний fill {fill_id}: ордер {client_id} уже {o.state.value}")
        o.avg_px = (o.avg_px * o.filled + px * qty) / (o.filled + qty)
        o.filled += qty
        if o.filled > o.qty + 1e-12:
            self.anomalies.append(f"перелив по {client_id}: исполнено {o.filled} из {o.qty}")
        if o.filled >= o.qty - 1e-12:
            o.state = OState.FILLED
        elif o.state not in (OState.CANCELED, OState.REJECTED):
            o.state = OState.PARTIAL
        return True

    # ---------- состояние ----------

    def open_orders(self, inst: str | None = None) -> list[Order]:
        return [o for o in self.orders.values() if o.open and (inst is None or o.inst == inst)]

    def worst_case_position(self, inst: str) -> tuple[float, float]:
        """(минимум, максимум) позиции, если исполнятся все открытые заявки на продажу / покупку."""
        pos = self.position.get(inst, 0.0)
        buys = sum(o.remaining for o in self.open_orders(inst) if o.side > 0)
        sells = sum(o.remaining for o in self.open_orders(inst) if o.side < 0)
        return pos - sells, pos + buys

    def pnl(self, mids: dict[str, float]) -> float:
        """Результат с начала дня: деньги от сделок минус комиссии плюс открытая позиция по mid."""
        return sum(self.cash.get(i, 0.0) + self.position.get(i, 0.0) * mids.get(i, 0.0)
                   for i in set(self.cash) | set(self.position))

    def check_timeouts(self, now_ms: int, ack_timeout_ms: int = 5000) -> list[str]:
        """Ордера без ответа дольше таймаута переводятся в UNKNOWN. Вернуть их id."""
        lost = []
        for o in self.orders.values():
            if o.state is OState.NEW and now_ms - o.sent_ms > ack_timeout_ms:
                o.state = OState.UNKNOWN
                lost.append(o.client_id)
        return lost

    def check_loss_limit(self, mids: dict[str, float], now_ms: int) -> bool:
        """Вызывать регулярно. При убытке больше лимита срабатывает kill switch."""
        if not self.killed and self.pnl(mids) <= -self.limits.max_daily_loss:
            self.kill(f"дневной убыток {self.pnl(mids):.4f} достиг лимита", now_ms)
        return self.killed

    # ---------- аварийные действия ----------

    def kill(self, reason: str, now_ms: int) -> None:
        """Kill switch: запретить новое, отменить всё, закрыть позицию по рынку."""
        if not self.killed:
            self.killed, self.kill_reason = True, reason
        for o in self.open_orders():
            if not o.reduce_only:
                self.cancel(o.client_id, now_ms)
        for inst, pos in self.position.items():
            closing = sum(o.remaining for o in self.open_orders(inst) if o.reduce_only)
            qty = abs(pos) - closing
            if qty > 1e-12:
                self._n_flatten += 1
                self.submit(f"kill-{inst}-{self._n_flatten}", inst, -1 if pos > 0 else 1, qty, None,
                            now_ms, reduce_only=True)

    def reconcile(self, exch_position: dict[str, float], exch_open_ids: set[str]) -> list[str]:
        """Сверка с биржей. Истина на стороне биржи; возвращает список расхождений."""
        issues = []
        for inst in set(self.position) | set(exch_position):
            ours, theirs = self.position.get(inst, 0.0), exch_position.get(inst, 0.0)
            if abs(ours - theirs) > 1e-9:
                issues.append(f"позиция {inst}: у нас {ours:+g}, на бирже {theirs:+g}")
                self.position[inst] = theirs
        for o in self.orders.values():
            if o.open and o.client_id not in exch_open_ids:
                issues.append(f"ордер {o.client_id} ({o.state.value}) на бирже не открыт: закрываем у себя")
                o.state = OState.FILLED if o.remaining <= 1e-12 else OState.CANCELED
            elif o.state is OState.UNKNOWN and o.client_id in exch_open_ids:
                o.state = OState.PARTIAL if o.filled > 0 else OState.ACKED
        for cid in exch_open_ids - set(self.orders):
            issues.append(f"на бирже открыт неизвестный нам ордер {cid}")
        return issues
