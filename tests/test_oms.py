import pytest

from hftlab.oms import OrderManager, OState, RiskLimits

LIMITS = RiskLimits(max_order_qty=1.0, max_position=1.0, max_daily_loss=10.0,
                    max_orders_per_sec=3, max_open_orders=2)


class FakeGateway:
    def __init__(self):
        self.sent, self.canceled = [], []

    def send(self, order, now_ms):
        self.sent.append(order.client_id)

    def cancel(self, client_id, now_ms):
        self.canceled.append(client_id)


def make():
    gw = FakeGateway()
    return OrderManager(gw, LIMITS), gw


def test_happy_path_and_pnl():
    oms, gw = make()
    oms.submit("a", "X", 1, 1.0, 100.0, 0)
    assert oms.orders["a"].state is OState.NEW
    oms.on_ack("a")
    oms.on_fill("a", "f1", "X", 1, 0.4, 100.0)
    assert oms.orders["a"].state is OState.PARTIAL
    oms.on_fill("a", "f2", "X", 1, 0.6, 100.0, fee=0.5)
    o = oms.orders["a"]
    assert o.state is OState.FILLED and o.avg_px == pytest.approx(100.0)
    assert oms.position["X"] == pytest.approx(1.0)
    assert oms.pnl({"X": 103.0}) == pytest.approx(3.0 - 0.5)


def test_idempotent_submit():
    oms, gw = make()
    a = oms.submit("a", "X", 1, 1.0, 100.0, 0)
    b = oms.submit("a", "X", 1, 1.0, 100.0, 5)      # повтор (например, после таймаута сети)
    assert a is b and gw.sent == ["a"]


def test_duplicate_fill_ignored():
    oms, _ = make()
    oms.submit("a", "X", 1, 1.0, 100.0, 0)
    assert oms.on_fill("a", "f1", "X", 1, 0.5, 100.0) is True
    assert oms.on_fill("a", "f1", "X", 1, 0.5, 100.0) is False
    assert oms.position["X"] == 0.5


def test_fill_cancel_race_both_orders():
    oms, gw = make()
    # 1) отмена подтверждена, потом приходит опоздавший fill: позиция меняется, случай записан
    oms.submit("a", "X", 1, 1.0, 100.0, 0)
    oms.on_ack("a")
    oms.cancel("a", 10)
    oms.cancel("a", 11)                              # повторная отмена не шлётся
    assert gw.canceled == ["a"]
    oms.on_cancel_ack("a")
    oms.on_fill("a", "f1", "X", 1, 1.0, 100.0)
    assert oms.position["X"] == 1.0 and len(oms.anomalies) == 1
    assert oms.orders["a"].state is OState.FILLED
    # 2) исполнился раньше, чем дошла отмена: остаётся FILLED
    oms.on_fill("z", "f9", "X", -1, 1.0, 100.0)      # чужой fill: позиция в ноль, аномалия
    oms.submit("b", "X", 1, 1.0, 100.0, 2000)
    oms.cancel("b", 2001)
    oms.on_fill("b", "f2", "X", 1, 1.0, 100.0)
    oms.on_cancel_ack("b")
    assert oms.orders["b"].state is OState.FILLED
    assert any("неизвестному" in a for a in oms.anomalies)


def test_risk_limits():
    oms, gw = make()
    assert oms.submit("big", "X", 1, 2.0, 100.0, 0).state is OState.REJECTED
    oms.submit("a", "X", 1, 0.6, 100.0, 0)
    # вторая покупка дала бы 1.2 > max_position, даже если первая ещё не исполнена
    r = oms.submit("b", "X", 1, 0.6, 100.0, 1)
    assert r.state is OState.REJECTED and "позиция" in r.reject_reason
    oms.submit("c", "X", -1, 0.6, 101.0, 2)
    assert oms.submit("d", "X", -1, 0.1, 101.0, 3).state is OState.REJECTED   # max_open_orders
    assert gw.sent == ["a", "c"]


def test_rate_limit():
    oms, gw = make()
    for i in range(5):
        oms.submit(f"o{i}", "X", 1, 0.1, 100.0, i)
        oms.on_reject(f"o{i}", "test")               # освобождаем слот открытых заявок
    assert len(gw.sent) == 3                         # 4-я и 5-я в ту же секунду отбиты
    assert oms.submit("later", "X", 1, 0.1, 100.0, 1500).state is OState.NEW


def test_lost_ack_becomes_unknown_and_reconcile():
    oms, _ = make()
    oms.submit("a", "X", 1, 1.0, 100.0, 0)
    assert oms.check_timeouts(6000) == ["a"]
    assert oms.orders["a"].state is OState.UNKNOWN
    # неизвестный ордер всё ещё считается риском: вторая покупка запрещена
    assert oms.submit("b", "X", 1, 1.0, 100.0, 7000).state is OState.REJECTED
    # биржа говорит: позиция 1.0 (ордер исполнился, fill мы не получили), открытых нет
    issues = oms.reconcile({"X": 1.0}, set())
    assert len(issues) == 2 and oms.position["X"] == 1.0
    assert not oms.orders["a"].open
    assert oms.reconcile({"X": 1.0}, set()) == []    # после сверки расхождений нет


def test_kill_switch_cancels_flattens_and_blocks():
    oms, gw = make()
    oms.submit("a", "X", 1, 1.0, 100.0, 0)
    oms.on_fill("a", "f1", "X", 1, 0.5, 100.0)       # позиция +0.5, остаток заявки ещё стоит
    oms.kill("ручная остановка", 100)
    assert gw.canceled == ["a"]
    flat = [o for o in oms.orders.values() if o.reduce_only]
    assert len(flat) == 1 and (flat[0].side, flat[0].qty, flat[0].price) == (-1, 0.5, None)
    oms.kill("ещё раз", 101)                         # повторный вызов не шлёт второй выход
    assert len([o for o in oms.orders.values() if o.reduce_only]) == 1
    assert oms.submit("new", "X", 1, 0.1, 100.0, 5000).state is OState.REJECTED
    assert "ручная остановка" in oms.orders["new"].reject_reason


def test_daily_loss_trips_kill_switch():
    oms, _ = make()
    oms.submit("a", "X", 1, 1.0, 100.0, 0)
    oms.on_fill("a", "f1", "X", 1, 1.0, 100.0)
    assert oms.check_loss_limit({"X": 95.0}, 10) is False
    assert oms.check_loss_limit({"X": 89.0}, 20) is True
    assert oms.killed and "убыток" in oms.kill_reason


def test_reduce_only_cannot_increase_position():
    oms, _ = make()
    assert oms.submit("x", "X", -1, 0.5, None, 0, reduce_only=True).state is OState.REJECTED
