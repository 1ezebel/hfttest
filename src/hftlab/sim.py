"""Этап 3: симулятор пассивного исполнения. Здесь умирает большинство идей.

    python -m hftlab.sim processed --inst BTC-USDT-SWAP --strategy mr_ema

Что моделируем (одна позиция за раз, размер минимальный):
  1. По сигналу ставим лимитную заявку на лучший бид (покупка) или аск (продажа).
     Заявка post-only: если к моменту прихода на биржу она пересекла бы спред, биржа её отклонит.
  2. Заявка доходит до биржи через latency_ms. Мы встаём В КОНЕЦ очереди: перед нами
     весь объём, который стоял на этой цене.
  3. Нас исполняют, только когда сделки по нашей цене съели всю очередь перед нами,
     либо когда цена прошла сквозь наш уровень.
  4. Не исполнили за ttl_ms: отменяем (отмена тоже идёт latency_ms, и за это время нас
     ещё могут исполнить).
  5. Исполнили: через hold_ms выходим по рынку (пересекаем спред, платим taker).

Консервативные допущения (все в худшую для нас сторону):
  * отмены чужих заявок очередь перед нами НЕ сокращают (считаем, что отменяют те, кто за нами);
  * сделки между последним известным стаканом и приходом заявки очередь не сокращают;
  * выход всегда по рынку, по цене стакана ПОСЛЕ задержки.

Чего симулятор не знает (граница точности публичных L2-данных):
  * настоящей позиции в очереди (видим только суммарный объём уровня раз в ~100 мс);
  * влияния нашей заявки на рынок (для минимального размера им пренебрегаем);
  * скрытых заявок и реальной задержки именно вашего канала.
Поэтому хороший результат здесь означает "стоит проверить на paper trading", а не "заработает".

Adverse selection: пассивную заявку чаще исполняют, когда цена идёт против нас.
Симулятор показывает это прямо: markout по всем заявкам против markout только по исполненным.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from hftlab.features import load
from hftlab.research import (MAKER_FEE_BPS, STRATEGIES, TAKER_FEE_BPS, _day, _fmt, summarize,
                             walk_forward)
from hftlab.runs import RunDB


@dataclass(frozen=True)
class SimConfig:
    latency_ms: int = 50      # от решения до появления заявки на бирже (и до отмены)
    ttl_ms: int = 2000        # сколько ждём исполнения
    hold_ms: int = 5000       # сколько держим позицию
    maker_fee_bps: float = MAKER_FEE_BPS
    taker_fee_bps: float = TAKER_FEE_BPS


def simulate(b: pd.DataFrame, t: pd.DataFrame, side: np.ndarray, cfg: SimConfig = SimConfig()) -> dict[str, Any]:
    """Прогнать сигнал через модель очереди. Возвращает метрики и список сделок."""
    ts, seg = b["ts_ms"].to_numpy(), b["seg"].to_numpy()
    bid, ask = b["bid"].to_numpy(), b["ask"].to_numpy()
    bid_sz, ask_sz = b["bid_sz"].to_numpy(), b["ask_sz"].to_numpy()
    mid = b["mid"].to_numpy()
    t_ts, t_px, t_sz, t_side = (t[c].to_numpy() for c in ("ts_ms", "px", "sz", "side"))
    n = len(b)

    def row_at(time_ms: float) -> int:
        """Последняя строка стакана с ts <= time_ms: то, что было известно в этот момент."""
        return int(np.searchsorted(ts, time_ms, side="right")) - 1

    def row_after(time_ms: float) -> int:
        return int(np.searchsorted(ts, time_ms, side="left"))

    prev = np.roll(side, 1)
    prev[0] = 0
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    candidates = np.flatnonzero((side != 0) & ((side != prev) | new_seg))

    stats = {"signals": len(candidates), "orders": 0, "rejected": 0, "skipped_busy": 0,
             "skipped_gap": 0, "fills": 0}
    trades: list[dict[str, Any]] = []
    all_orders_markout: list[float] = []
    free_at = -math.inf  # момент, когда прошлая заявка/позиция закрыта

    for i in candidates:
        t0, s = ts[i], int(side[i])
        if t0 < free_at:
            stats["skipped_busy"] += 1
            continue
        price = bid[i] if s > 0 else ask[i]
        t_active = t0 + cfg.latency_ms
        t_cancel = t0 + cfg.ttl_ms + cfg.latency_ms
        j, j_end = row_at(t_active), row_after(t_cancel)
        if j_end >= n or seg[j] != seg[i] or seg[j_end] != seg[i]:
            stats["skipped_gap"] += 1  # дыра в данных или конец записи: не знаем, что было
            continue

        # Состояние стакана, когда заявка дошла до биржи
        if (s > 0 and price >= ask[j]) or (s < 0 and price <= bid[j]):
            stats["rejected"] += 1  # post-only: пересекла бы спред
            free_at = t_active + 1  # об отказе узнаём после момента прихода заявки
            continue
        stats["orders"] += 1
        if s > 0:
            queue = 0.0 if price > bid[j] else bid_sz[j]
        else:
            queue = 0.0 if price < ask[j] else ask_sz[j]

        # Куда ушла цена за время удержания, если бы нас исполнили сразу (для сравнения)
        k_all = row_after(t_active + cfg.hold_ms)
        if k_all < n and seg[k_all] == seg[i]:
            all_orders_markout.append(s * (mid[k_all] - price) / price * 1e4)

        # Сделки, пока заявка стоит: нас исполняют встречные агрессоры
        lo = int(np.searchsorted(t_ts, t_active, side="right"))
        hi = int(np.searchsorted(t_ts, t_cancel, side="right"))
        fill_ts = None
        for m in range(lo, hi):
            if t_side[m] != -s:
                continue
            px = t_px[m]
            through = px < price if s > 0 else px > price
            if through:
                fill_ts = t_ts[m]
                break
            if px == price:
                queue -= t_sz[m]
                if queue < 0:  # очередь перед нами съедена, остаток сделки достался нам
                    fill_ts = t_ts[m]
                    break
        if fill_ts is None:
            free_at = t_cancel + 1  # заявка снята только после момента отмены
            continue

        # Выход по рынку через hold_ms (+ задержка): продаём в бид / покупаем по аску
        t_exit = fill_ts + cfg.hold_ms + cfg.latency_ms
        k = row_after(t_exit)
        if k >= n or seg[k] != seg[i]:
            stats["skipped_gap"] += 1
            stats["orders"] -= 1
            free_at = t_cancel
            continue
        exit_px = bid[k] if s > 0 else ask[k]
        gross = s * (exit_px - price) / price * 1e4
        fees = cfg.maker_fee_bps + cfg.taker_fee_bps
        stats["fills"] += 1
        trades.append({
            "t_signal": int(t0), "t_fill": int(fill_ts), "side": s, "entry": float(price),
            "exit": float(exit_px), "gross_bps": float(gross), "net_bps": float(gross - fees),
            "mid_markout_bps": float(s * (mid[k] - price) / price * 1e4),
        })
        free_at = ts[k]

    gross = np.array([x["gross_bps"] for x in trades])
    net = np.array([x["net_bps"] for x in trades])
    sg, sn = summarize(gross), summarize(net)
    filled_mo = summarize(np.array([x["mid_markout_bps"] for x in trades]))
    all_mo = summarize(np.array(all_orders_markout))
    return {
        **stats,
        "fill_rate": stats["fills"] / stats["orders"] if stats["orders"] else None,
        "mean_gross_bps": sg["mean"], "mean_net_bps": sn["mean"], "t_net": sn["t"],
        "sum_gross_bps": float(gross.sum()), "sum_net_bps": float(net.sum()),
        "fees_bps": (cfg.maker_fee_bps + cfg.taker_fee_bps) * stats["fills"],
        # adverse selection: второе число обычно заметно хуже первого
        "markout_all_orders_bps": all_mo["mean"], "markout_filled_bps": filled_mo["mean"],
        "trades": trades,
    }


def sim_scorer(t: pd.DataFrame, cfg: SimConfig):
    """Оценщик для walk_forward: средний результат сделки после комиссий."""
    def score(chunk: pd.DataFrame, side: np.ndarray) -> dict[str, Any]:
        lo, hi = chunk["ts_ms"].iloc[0], chunk["ts_ms"].iloc[-1]
        tc = t[(t["ts_ms"] >= lo) & (t["ts_ms"] <= hi)]
        r = simulate(chunk, tc, side, cfg)
        r.pop("trades")
        return {"n": r["fills"], "score": r["mean_net_bps"], **r}
    return score


def main() -> int:
    ap = argparse.ArgumentParser(description="Симулятор пассивного исполнения + walk-forward")
    ap.add_argument("processed", nargs="?", default="processed")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="mr_ema")
    ap.add_argument("--latency-ms", type=int, default=50)
    ap.add_argument("--ttl-ms", type=int, default=2000)
    ap.add_argument("--hold-ms", type=int, default=5000)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--min-fills", type=int, default=20, help="минимум сделок на train для выбора параметров")
    ap.add_argument("--db", default="results/runs.sqlite")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    b, t = load(a.processed, a.inst), load(a.processed, a.inst, "trades")
    cfg = SimConfig(latency_ms=a.latency_ms, ttl_ms=a.ttl_ms, hold_ms=a.hold_ms)
    spec = STRATEGIES[a.strategy]
    name = f"{a.strategy}_maker"
    hours = (b["ts_ms"].iloc[-1] - b["ts_ms"].iloc[0]) / 3.6e6
    print(f"{a.inst}: {hours:.2f} ч, стратегия {name}, {json.dumps(asdict(cfg))}")
    if hours < 24:
        print("! данных меньше суток: это проверка кода, а не вывод о стратегии")

    db = None if a.no_log else RunDB(a.db)
    if db:
        db.ensure_strategy(name, hypothesis=spec["hypothesis"] + "; вход пассивный, выход по рынку")

    def log(split: str, params: dict, m: dict, chunk: pd.DataFrame) -> None:
        if db is None or chunk.empty:
            return
        metrics = {k: v for k, v in m.items() if k not in ("n", "score")}
        metrics.update(unit="bps", n_trades=m["fills"], pnl_gross=m["sum_gross_bps"],
                       pnl_net=m["sum_net_bps"], fees=m["fees_bps"],
                       execution="maker entry (queue tail), taker exit", sim=asdict(cfg),
                       hours=(chunk["ts_ms"].iloc[-1] - chunk["ts_ms"].iloc[0]) / 3.6e6)
        db.log_run(name, split=split, params=params, metrics=metrics,
                   data_from=_day(chunk["ts_ms"].iloc[0]), data_to=_day(chunk["ts_ms"].iloc[-1]),
                   instruments=[a.inst])

    print("\n  fold  params                               orders  fills  fill%  net_bps  t_net  "
          "markout_all  markout_filled")
    for r in walk_forward(b, spec["fn"], spec["grid"], sim_scorer(t, cfg), a.chunks,
                          min_n=a.min_fills, on_result=log):
        if r["params"] is None:
            print(f"  {r['fold']:>4}  (на train не набралось {a.min_fills} сделок ни у одного варианта)")
            continue
        m = r["test"]
        fr = "-" if m["fill_rate"] is None else f"{m['fill_rate'] * 100:.0f}"
        print(f"  {r['fold']:>4}  {str(r['params']):<36} {m['orders']:>6}  {m['fills']:>5}  {fr:>5}  "
              f"{_fmt(m['mean_net_bps'], 2):>7}  {_fmt(m['t_net'], 1):>5}  "
              f"{_fmt(m['markout_all_orders_bps'], 2):>11}  {_fmt(m['markout_filled_bps'], 2):>14}")
    print(f"\nnet_bps: средний результат сделки после комиссий ({MAKER_FEE_BPS} maker + {TAKER_FEE_BPS} taker).")
    print("markout_all против markout_filled: если второе хуже, нас исполняют в плохие моменты (adverse selection).")
    if db:
        print(f"Всего вариантов в журнале по {name}: {db.count_runs(name)}")
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
