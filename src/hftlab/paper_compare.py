"""Этап 5: сравнить бумажную торговлю с тем, что ожидал симулятор на тех же данных.

    python -m hftlab.paper_compare results/paper/<файл>.jsonl --processed processed

Нужно, чтобы рекордер писал тот же инструмент в то же время, и чтобы после
paper-сессии был запущен `python -m hftlab.features`.

Как читать результат:
  * сделки сопоставляются по времени сигнала (t_signal);
  * "только в paper" / "только в sim" в самом начале сессии это норма: бумажный трейдер
    стартует с пустым окном сигнала, а симулятор видит историю до старта;
  * расхождения в середине сессии надо объяснять. Обычные причины: задержка доставки
    данных вживую менялась (в симуляторе берём её медиану), разрыв соединения у одного из двух.
  Необъяснимые расхождения означают, что симулятору нельзя верить, и это важнее любой доходности.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from hftlab.features import load
from hftlab.research import STRATEGIES
from hftlab.sim import SimConfig, simulate


def read_log(path: str | Path) -> dict[str, Any]:
    start, submits, trades = None, [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev["type"] == "start":
            start = ev
        elif ev["type"] == "submit":
            submits.append(ev)
        elif ev["type"] == "trade":
            trades.append(ev)
    return {"start": start, "submits": submits, "trades": trades}


def compare(log: dict[str, Any], b, t, strategy: str, params: dict, cfg: SimConfig) -> dict[str, Any]:
    if not log["submits"]:
        return {"error": "в логе нет ни одной заявки"}
    delay = statistics.median(s.get("feed_delay_ms", 0.0) for s in log["submits"])
    t_lo, t_hi = log["submits"][0]["ts"], log["submits"][-1]["ts"]
    sim_cfg = SimConfig(latency_ms=cfg.latency_ms + round(delay), ttl_ms=cfg.ttl_ms, hold_ms=cfg.hold_ms,
                        maker_fee_bps=cfg.maker_fee_bps, taker_fee_bps=cfg.taker_fee_bps)
    sim = simulate(b, t, STRATEGIES[strategy]["fn"](b, **params), sim_cfg)
    sim_trades = {x["t_signal"]: x for x in sim["trades"] if t_lo <= x["t_signal"] <= t_hi}
    paper_trades = {x["t_signal"]: x for x in log["trades"]}
    both = sorted(set(sim_trades) & set(paper_trades))

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "feed_delay_ms": delay, "sim_latency_ms": sim_cfg.latency_ms,
        "paper_trades": len(paper_trades), "sim_trades": len(sim_trades), "matched": len(both),
        "only_paper": sorted(set(paper_trades) - set(sim_trades)),
        "only_sim": sorted(set(sim_trades) - set(paper_trades)),
        "same_fill_time": sum(1 for k in both if paper_trades[k]["t_fill"] == sim_trades[k]["t_fill"]),
        "paper_mean_net_bps": mean([x["net_bps"] for x in paper_trades.values()]),
        "sim_mean_net_bps": mean([x["net_bps"] for x in sim_trades.values()]),
        "matched_mean_abs_diff_bps": mean([abs(paper_trades[k]["net_bps"] - sim_trades[k]["net_bps"]) for k in both]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Сравнение paper trading с симулятором")
    ap.add_argument("log")
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--inst")
    ap.add_argument("--strategy")
    ap.add_argument("--params", help="JSON; нужен, только если в логе нет заголовка start")
    a = ap.parse_args()
    log = read_log(a.log)
    st = log["start"] or {}
    inst, strategy = a.inst or st.get("inst"), a.strategy or st.get("strategy")
    params = json.loads(a.params) if a.params else st.get("params")
    if not (inst and strategy and params):
        print("в логе нет заголовка: укажите --inst, --strategy и --params")
        return 2
    cfg = SimConfig(**st["cfg"]) if "cfg" in st else SimConfig()
    r = compare(log, load(a.processed, inst), load(a.processed, inst, "trades"), strategy, params, cfg)
    for k, v in r.items():
        if isinstance(v, list):
            v = f"{len(v)} шт." + (f" (первые: {v[:3]})" if v else "")
        elif isinstance(v, float):
            v = f"{v:.3f}"
        print(f"   {k:<26} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
