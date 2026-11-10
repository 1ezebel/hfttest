"""Этап 5: бумажная торговля (paper trading). Ордера НИКУДА не отправляются.

Вживую (публичные данные, ключи не нужны):
    python -m hftlab.paper --inst BTC-USDT-SWAP --strategy mr_ema \\
        --params '{"halflife_s": 5, "z_entry": 2}' --minutes 60

На записанных данных (для сверки с симулятором):
    python -m hftlab.paper --replay processed --inst BTC-USDT-SWAP --strategy mr_ema --params '{...}'

Как устроено:
    рынок (стакан, сделки) -> PaperTrader (сигнал, таймеры) -> OrderManager (риск, учёт)
                                                             -> PaperGateway (бумажная "биржа")

PaperGateway исполняет заявки по тем же правилам, что симулятор (hftlab.sim): задержка,
конец очереди, post-only, выход по рынку. Разница в том, что здесь всё происходит
событие за событием, как в реальной торговле: код не может заглянуть вперёд даже по ошибке.
На одних и тех же записанных данных paper и симулятор обязаны дать одинаковые сделки
(это проверяет тест). Вживую добавляется то, чего в симуляторе нет: реальная задержка
доставки данных до нас (feed delay). Она прибавляется к задержке заявки.

Каждое событие пишется в results/paper/<время>.jsonl, итог идёт в журнал бэктестов
как стратегия "<имя>_paper" со split="oos" (это настоящие данные из будущего).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from hftlab.oms import Order, OrderManager, OState, RiskLimits
from hftlab.research import ROW_MS, STRATEGIES, summarize
from hftlab.sim import SimConfig

# Сколько последних строк нужно сигналу, чтобы совпасть с расчётом по всей истории
WINDOW_ROWS: dict[str, Callable[[dict], int]] = {
    "mr_ema": lambda p: int(50 * max(1.0, p["halflife_s"] * 1000 / ROW_MS)) + 1,
    "micro_drift": lambda p: 1,
}


class PaperGateway:
    """Бумажная биржа для одного инструмента: модель очереди из hftlab.sim, но по событиям."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.oms: OrderManager | None = None
        self.events: list[dict[str, Any]] = []   # что произошло; забирает PaperTrader
        self._pending: list[tuple[float, str, Any]] = []      # (когда дойдёт, "send"/"cancel", что)
        self._market: list[tuple[float, Order]] = []          # рыночные заявки, ждут ближайший стакан
        self._working: dict[str, list] = {}                   # client_id -> [order, очередь перед нами]
        self._book: tuple[float, float, float, float] | None = None  # bid, bid_sz, ask, ask_sz
        self._n_fills = 0

    # --- интерфейс шлюза (вызывает OrderManager) ---
    def send(self, order: Order, now_ms: float) -> None:
        eff = now_ms + self.cfg.latency_ms
        if order.price is None:
            self._market.append((eff, order))
        else:
            self._pending.append((eff, "send", order))

    def cancel(self, client_id: str, now_ms: float) -> None:
        self._pending.append((now_ms + self.cfg.latency_ms, "cancel", client_id))

    # --- рынок ---
    def _advance(self, ts: float) -> None:
        """Выполнить всё, что дошло до биржи строго раньше момента ts."""
        if not self._pending:
            return
        self._pending.sort(key=lambda x: x[0])
        while self._pending and self._pending[0][0] < ts:
            eff, kind, obj = self._pending.pop(0)
            if kind == "cancel":
                if obj in self._working:
                    del self._working[obj]
                    self.oms.on_cancel_ack(obj)
                    self.events.append({"type": "canceled", "id": obj, "ts": eff})
                continue
            o: Order = obj
            if self._book is None:
                self._reject(o, eff, "нет стакана")
                continue
            bid, bid_sz, ask, ask_sz = self._book
            if (o.side > 0 and o.price >= ask) or (o.side < 0 and o.price <= bid):
                self._reject(o, eff, "post-only: заявка пересекла бы спред")
                continue
            if o.side > 0:
                queue = 0.0 if o.price > bid else bid_sz
            else:
                queue = 0.0 if o.price < ask else ask_sz
            self._working[o.client_id] = [o, queue]
            self.oms.on_ack(o.client_id)
            self.events.append({"type": "ack", "id": o.client_id, "ts": eff, "queue": queue})

    def _reject(self, o: Order, eff: float, reason: str) -> None:
        self.oms.on_reject(o.client_id, reason)
        self.events.append({"type": "rejected", "id": o.client_id, "ts": eff, "reason": reason})

    def _fill(self, o: Order, px: float, ts: float, fee_bps: float, kind: str) -> None:
        self._n_fills += 1
        qty = o.remaining
        self.oms.on_fill(o.client_id, f"paper-{self._n_fills}", o.inst, o.side, qty, px,
                         fee=qty * px * fee_bps / 1e4)
        self.events.append({"type": kind, "id": o.client_id, "ts": ts, "px": px, "side": o.side})

    def on_trade(self, ts: float, px: float, sz: float, side: int) -> None:
        self._advance(ts)
        for cid, slot in list(self._working.items()):
            o, queue = slot
            if side != -o.side:
                continue  # нас исполняет только встречный агрессор
            through = px < o.price if o.side > 0 else px > o.price
            if not through:
                if px != o.price:
                    continue
                slot[1] = queue = queue - sz
                if queue >= 0:
                    continue  # очередь перед нами ещё не съедена
            del self._working[cid]
            self._fill(o, o.price, ts, self.cfg.maker_fee_bps, "fill")

    def on_book(self, ts: float, bid: float, bid_sz: float, ask: float, ask_sz: float) -> None:
        self._advance(ts)
        self._book = (bid, bid_sz, ask, ask_sz)
        ready = [(e, o) for e, o in self._market if e <= ts]
        self._market = [(e, o) for e, o in self._market if e > ts]
        for _, o in ready:  # рыночная заявка: продаём в бид, покупаем по аску
            self._fill(o, bid if o.side < 0 else ask, ts, self.cfg.taker_fee_bps, "exit")


class PaperTrader:
    def __init__(self, inst: str, strategy: str, params: dict, cfg: SimConfig = SimConfig(),
                 qty: float = 0.01, max_daily_loss: float = 50.0,
                 signal_fn: Callable[..., np.ndarray] | None = None, window_rows: int | None = None,
                 log_path: str | Path | None = None) -> None:
        self.inst, self.strategy, self.params, self.cfg, self.qty = inst, strategy, params, cfg, qty
        self.fn = signal_fn or STRATEGIES[strategy]["fn"]
        self.window = window_rows or WINDOW_ROWS[strategy](params)
        self.gw = PaperGateway(cfg)
        self.oms = OrderManager(self.gw, RiskLimits(
            max_order_qty=qty, max_position=qty, max_daily_loss=max_daily_loss,
            max_orders_per_sec=5, max_open_orders=2))
        self.gw.oms = self.oms
        self.rows: deque[tuple] = deque(maxlen=self.window)   # (ts, seg, mid, imb1, imb5)
        self.prev_side = 0
        self.prev_seg: int | None = None
        self.free_at = -math.inf
        self.entry: dict[str, Any] | None = None      # текущая заявка/позиция
        self.timers: list[tuple[float, str, str]] = []  # (когда, "cancel"/"exit", client_id)
        self.trades: list[dict[str, Any]] = []
        self.stats = {"signals": 0, "orders": 0, "rejected": 0, "risk_rejected": 0, "skipped_busy": 0}
        self._n = 0
        self._kill_logged = False
        self._mid = float("nan")
        self._log = open(log_path, "a", encoding="utf-8") if log_path else None

    # ---------- события рынка ----------

    def on_trade(self, ts: float, px: float, sz: float, side: int) -> None:
        self._fire_timers(ts)
        self.gw.on_trade(ts, px, sz, side)
        self._drain()

    def on_book_row(self, ts: float, seg: int, bid: float, bid_sz: float, ask: float, ask_sz: float,
                    imb1: float, imb5: float, feed_delay_ms: float = 0.0) -> None:
        self._fire_timers(ts)
        self.gw.on_book(ts, bid, bid_sz, ask, ask_sz)
        self._drain()
        self._mid = (bid + ask) / 2
        if self.oms.check_loss_limit({self.inst: self._mid}, int(ts)) and not self._kill_logged:
            self._kill_logged = True
            self._write({"type": "kill", "ts": ts, "reason": self.oms.kill_reason})

        new_seg = seg != self.prev_seg
        if new_seg:
            self.rows.clear()  # через дыру в данных окно сигнала не тянем
            if self.entry and self.entry["state"] == "working":
                self.oms.cancel(self.entry["id"], int(ts))
        self.rows.append((ts, seg, self._mid, imb1, imb5))
        side = self._signal()
        onset = side != 0 and (side != self.prev_side or new_seg)
        self.prev_side, self.prev_seg = side, seg
        if not onset:
            return
        self.stats["signals"] += 1
        if self.entry is not None or ts < self.free_at:
            self.stats["skipped_busy"] += 1
            return
        self._n += 1
        cid = f"{self.strategy}-{int(ts)}-{self._n}"
        price = bid if side > 0 else ask
        now = ts + feed_delay_ms  # вживую мы узнаём о строке с опозданием
        o = self.oms.submit(cid, self.inst, side, self.qty, price, int(now))
        if o.state is OState.REJECTED:
            self.stats["risk_rejected"] += 1
            self._write({"type": "risk_rejected", "ts": ts, "reason": o.reject_reason})
            return
        self.entry = {"id": cid, "state": "working", "side": side, "price": price, "t_signal": ts}
        self.timers.append((now + self.cfg.ttl_ms, "cancel", cid))
        self._write({"type": "submit", "ts": ts, "id": cid, "side": side, "px": price,
                     "feed_delay_ms": feed_delay_ms})

    # ---------- внутреннее ----------

    def _signal(self) -> int:
        a = np.array(self.rows, dtype=float)
        df = pd.DataFrame({"ts_ms": a[:, 0], "seg": a[:, 1].astype(np.int64), "mid": a[:, 2],
                           "imb1": a[:, 3], "imb5": a[:, 4]})
        return int(self.fn(df, **self.params)[-1])

    def _fire_timers(self, ts: float) -> None:
        if not self.timers:
            return
        self.timers.sort(key=lambda x: x[0])
        while self.timers and self.timers[0][0] <= ts:
            when, kind, cid = self.timers.pop(0)
            if kind == "cancel":
                self.oms.cancel(cid, int(when))  # если уже исполнен, OMS просто ничего не сделает
            elif kind == "exit" and self.entry and self.entry["id"] == cid:
                self.oms.submit(cid + "-exit", self.inst, -self.entry["side"], self.qty, None,
                                int(when), reduce_only=True)

    def _drain(self) -> None:
        for ev in self.gw.events:
            self._write(ev)
            e = self.entry
            if e is None:
                continue
            if ev["type"] == "ack" and ev["id"] == e["id"]:
                self.stats["orders"] += 1
            elif ev["type"] == "rejected" and ev["id"] == e["id"]:
                self.stats["rejected"] += 1
                self.entry, self.free_at = None, ev["ts"] + 1
            elif ev["type"] == "canceled" and ev["id"] == e["id"]:
                self.entry, self.free_at = None, ev["ts"] + 1
            elif ev["type"] == "fill" and ev["id"] == e["id"]:
                e["state"], e["t_fill"] = "position", ev["ts"]
                self.timers.append((ev["ts"] + self.cfg.hold_ms, "exit", e["id"]))
            elif ev["type"] == "exit" and e["state"] == "position":
                # обычный выход по таймеру или аварийный выход от kill switch
                gross = e["side"] * (ev["px"] - e["price"]) / e["price"] * 1e4
                fees = self.cfg.maker_fee_bps + self.cfg.taker_fee_bps
                tr = {"t_signal": int(e["t_signal"]), "t_fill": int(e["t_fill"]), "side": e["side"],
                      "by_kill": ev["id"].startswith("kill-"),
                      "entry": float(e["price"]), "exit": float(ev["px"]),
                      "gross_bps": float(gross), "net_bps": float(gross - fees)}
                self.trades.append(tr)
                self._write({"type": "trade", **tr})
                self.entry, self.free_at = None, ev["ts"]
        self.gw.events.clear()

    def _write(self, ev: dict) -> None:
        if self._log:
            self._log.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        net = np.array([t["net_bps"] for t in self.trades])
        gross = np.array([t["gross_bps"] for t in self.trades])
        s = summarize(net)
        n = len(self.trades)
        return {
            **self.stats, "fills": n,
            "fill_rate": n / self.stats["orders"] if self.stats["orders"] else None,
            "mean_gross_bps": summarize(gross)["mean"], "mean_net_bps": s["mean"], "t_net": s["t"],
            "sum_gross_bps": float(gross.sum()), "sum_net_bps": float(net.sum()),
            "fees_bps": (self.cfg.maker_fee_bps + self.cfg.taker_fee_bps) * n,
            "pnl_quote": self.oms.pnl({self.inst: self._mid}) if n else 0.0,
            "killed": self.oms.killed, "anomalies": len(self.oms.anomalies),
            "open_at_end": self.entry is not None,
        }

    def close(self) -> None:
        if self._log:
            self._log.close()
            self._log = None


# ---------- источники данных ----------

def replay(trader: PaperTrader, b: pd.DataFrame, t: pd.DataFrame) -> None:
    """Прогнать записанные данные через трейдера в порядке биржевого времени."""
    bt, tt = b["ts_ms"].to_numpy(), t["ts_ms"].to_numpy()
    cols = [b[c].to_numpy() for c in ("seg", "bid", "bid_sz", "ask", "ask_sz", "imb1", "imb5")]
    tpx, tsz, tside = t["px"].to_numpy(), t["sz"].to_numpy(), t["side"].to_numpy()
    j, nt = 0, len(t)
    for i in range(len(b)):
        while j < nt and tt[j] < bt[i]:
            trader.on_trade(float(tt[j]), float(tpx[j]), float(tsz[j]), int(tside[j]))
            j += 1
        trader.on_book_row(float(bt[i]), int(cols[0][i]), *(float(c[i]) for c in cols[1:]))
    while j < nt:
        trader.on_trade(float(tt[j]), float(tpx[j]), float(tsz[j]), int(tside[j]))
        j += 1


async def run_live(trader: PaperTrader, url: str, minutes: float) -> None:
    import websockets

    from hftlab.okx_book import OrderBook, Status
    from hftlab.recorder import sub_msg

    inst = trader.inst
    deadline = time.monotonic() + minutes * 60
    seg = 0
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(url, ping_interval=None, max_size=2**23, open_timeout=15) as ws:
                book = OrderBook(inst)
                seg += 1
                await ws.send(sub_msg("subscribe", "books", inst))
                await ws.send(sub_msg("subscribe", "trades", inst))
                last_ping = time.monotonic()
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    recv_ms = time.time_ns() / 1e6
                    if time.monotonic() - last_ping > 20:
                        await ws.send("ping")
                        last_ping = time.monotonic()
                    if raw == "pong":
                        continue
                    msg = json.loads(raw)
                    channel = msg.get("arg", {}).get("channel")
                    if "event" in msg or channel is None:
                        continue
                    if channel == "trades":
                        for d in msg["data"]:
                            trader.on_trade(float(d["ts"]), float(d["px"]), float(d["sz"]),
                                            1 if d["side"] == "buy" else -1)
                        continue
                    for d in msg["data"]:
                        status = book.apply(msg["action"], d)
                        if status in (Status.GAP, Status.RESET):
                            seg += 1
                            await ws.send(sub_msg("unsubscribe", "books", inst))
                            await ws.send(sub_msg("subscribe", "books", inst))
                            break
                        if status is not Status.OK:
                            continue
                        bids, asks = book.top(5)
                        if not bids or not asks or bids[0][0] >= asks[0][0]:
                            seg += 1
                            continue
                        b5, a5 = sum(s for _, s in bids), sum(s for _, s in asks)
                        ts = float(d["ts"])
                        trader.on_book_row(
                            ts, seg, bids[0][0], bids[0][1], asks[0][0], asks[0][1],
                            (bids[0][1] - asks[0][1]) / (bids[0][1] + asks[0][1]), (b5 - a5) / (b5 + a5),
                            feed_delay_ms=max(0.0, recv_ms - ts))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - любая причина: переподключаемся
            print(f"соединение потеряно: {e!r}, переподключение через 3 с")
            await asyncio.sleep(3)


def main() -> int:
    from hftlab.recorder import URL
    from hftlab.runs import RunDB

    ap = argparse.ArgumentParser(description="Бумажная торговля: ордера никуда не отправляются")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="mr_ema")
    ap.add_argument("--params", required=True, help='JSON, например \'{"halflife_s": 5, "z_entry": 2}\'')
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--replay", help="папка processed: прогнать записанные данные вместо живых")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--latency-ms", type=int, default=50)
    ap.add_argument("--ttl-ms", type=int, default=2000)
    ap.add_argument("--hold-ms", type=int, default=5000)
    ap.add_argument("--qty", type=float, default=0.01, help="размер заявки в базовой монете")
    ap.add_argument("--db", default="results/runs.sqlite")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    params = json.loads(a.params)
    cfg = SimConfig(latency_ms=a.latency_ms, ttl_ms=a.ttl_ms, hold_ms=a.hold_ms)
    started = pd.Timestamp.now(tz="UTC")
    log_path = None
    if not a.no_log:
        Path("results/paper").mkdir(parents=True, exist_ok=True)
        log_path = f"results/paper/{started:%Y%m%d-%H%M%S}-{a.inst}-{a.strategy}.jsonl"
    trader = PaperTrader(a.inst, a.strategy, params, cfg, qty=a.qty, log_path=log_path)

    mode = "replay" if a.replay else "live"
    print(f"paper ({mode}): {a.inst} {a.strategy} {params} {json.dumps(asdict(cfg))}")
    if a.replay:
        from hftlab.features import load
        replay(trader, load(a.replay, a.inst), load(a.replay, a.inst, "trades"))
    else:
        try:
            asyncio.run(run_live(trader, a.url, a.minutes))
        except KeyboardInterrupt:
            print("остановлено пользователем")
    s = trader.summary()
    trader.close()
    for k, v in s.items():
        print(f"   {k:<16} {v}")
    if trader.oms.anomalies:
        print("   аномалии:", trader.oms.anomalies[:5])

    if not a.no_log:
        name = f"{a.strategy}_paper"
        ended = pd.Timestamp.now(tz="UTC")
        with RunDB(a.db) as db:
            db.ensure_strategy(name, hypothesis=STRATEGIES[a.strategy]["hypothesis"]
                               + "; бумажная торговля на живом потоке")
            metrics = {**s, "unit": "bps", "n_trades": s["fills"], "pnl_gross": s["sum_gross_bps"],
                       "pnl_net": s["sum_net_bps"], "fees": s["fees_bps"], "mode": mode,
                       "sim": asdict(cfg), "minutes": (ended - started).total_seconds() / 60}
            rid = db.log_run(name, split="oos" if mode == "live" else "test", params=params,
                             metrics=metrics, data_from=f"{started:%Y-%m-%d %H:%M}",
                             data_to=f"{ended:%Y-%m-%d %H:%M}", instruments=[a.inst],
                             artifacts=[log_path], note=f"paper {mode}")
        print(f"записано в журнал: {rid}, лог событий: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
