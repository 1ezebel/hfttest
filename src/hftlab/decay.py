"""Этап 2: затухание сигнала (signal decay). На каком горизонте сигнал живёт и когда умирает.

    python -m hftlab.decay processed --strategy micro_drift --params '{"imb_entry": 0.9, "depth": 5}'

Зачем: от этого зависит, сколько держать позицию и насколько критична задержка.
Сигнал, который живёт 200 мс, с задержкой 150 мс почти бесполезен; сигнал, который
набирает силу 10 секунд, можно исполнять спокойно.

Таблица и график показывают средний markout (до издержек) на горизонтах от 100 мс до 60 с.
  peak        горизонт, на котором эффект максимален
  half-life   первый горизонт после пика, где эффект упал вдвое (или "не затух")

Важно: кривая строится по ВСЕМ данным с заранее заданными параметрами. Это описание
сигнала, а не честная оценка доходности. Прогон пишется в журнал со split="train".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hftlab.features import load
from hftlab.research import STRATEGIES, TAKER_FEE_BPS, _day, _fmt, evaluate
from hftlab.runs import RunDB

HORIZONS_MS = (100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000)


def decay_curve(b, side: np.ndarray, horizons_ms=HORIZONS_MS) -> dict[str, Any]:
    ev = evaluate(b, side, horizons_ms)["horizons"]
    rows = [{"horizon_ms": h, **ev[f"{h}ms"]} for h in horizons_ms]
    valid = [r for r in rows if r["gross_bps"] is not None]
    peak = max(valid, key=lambda r: r["gross_bps"]) if valid else None
    half = None
    if peak is not None and peak["gross_bps"] > 0:
        after = [r for r in valid if r["horizon_ms"] > peak["horizon_ms"]]
        half = next((r["horizon_ms"] for r in after if r["gross_bps"] <= peak["gross_bps"] / 2), None)
    return {"rows": rows,
            "peak_ms": peak["horizon_ms"] if peak else None,
            "peak_bps": peak["gross_bps"] if peak else None,
            "half_life_ms": half}


def _label(ms: int) -> str:
    return f"{ms} мс" if ms < 1000 else f"{ms // 1000} с"


def plot(curve: dict[str, Any], title: str, path: str | Path, subtitle: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in curve["rows"] if r["gross_bps"] is not None]
    x = [r["horizon_ms"] for r in rows]
    y = [r["gross_bps"] for r in rows]
    # 95% интервал: среднее +- 1.96 стандартной ошибки (ошибка = |среднее / t|)
    err = [1.96 * abs(r["gross_bps"] / r["t"]) if r["t"] else 0.0 for r in rows]
    ink, muted, blue, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.axhline(0, color=muted, linewidth=1)
    ax.fill_between(x, [a - e for a, e in zip(y, err)], [a + e for a, e in zip(y, err)],
                    color=blue, alpha=0.15, linewidth=0)
    ax.plot(x, y, color=blue, linewidth=2, marker="o", markersize=6,
            markeredgecolor=surface, markeredgewidth=1.5)
    ax.set_xscale("log")
    ax.set_xticks(x, [_label(h) for h in x])
    ax.minorticks_off()
    ax.set_xlabel("время после сигнала", color=muted)
    ax.set_ylabel("средний markout, б.п.", color=muted)
    ax.set_title(title, color=ink, loc="left", fontsize=12, fontweight="bold", pad=30)
    ax.text(0, 1.07, subtitle, transform=ax.transAxes, color=muted, fontsize=9)
    ax.margins(y=0.18)  # место над линией для подписи пика
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.tick_params(colors=muted, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if curve["peak_ms"] is not None:
        i = x.index(curve["peak_ms"])
        ax.annotate(f"пик {y[i]:.2f} б.п.", (x[i], y[i]), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=ink, fontsize=9)
    fig.text(0.01, 0.01, f"До издержек. Круг по рынку стоит около {2 * TAKER_FEE_BPS:.0f} б.п. "
             "Полоса: 95% интервал.", color=muted, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Кривая затухания сигнала")
    ap.add_argument("processed", nargs="?", default="processed")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    ap.add_argument("--params", required=True, help="JSON с параметрами сигнала")
    ap.add_argument("--db", default="results/runs.sqlite")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    params = json.loads(a.params)
    b = load(a.processed, a.inst)
    hours = (b["ts_ms"].iloc[-1] - b["ts_ms"].iloc[0]) / 3.6e6
    c = decay_curve(b, STRATEGIES[a.strategy]["fn"](b, **params))
    print(f"{a.inst} {a.strategy} {params}, {hours:.2f} ч")
    print("  горизонт       n   gross_bps      t")
    for r in c["rows"]:
        print(f"  {_label(r['horizon_ms']):>8}  {r['n']:>6}  {_fmt(r['gross_bps']):>10}  {_fmt(r['t'], 1):>5}")
    hl = "не затух за 60 с" if c["half_life_ms"] is None else _label(c["half_life_ms"])
    if c["peak_ms"] is None or c["peak_bps"] <= 0:
        print("  положительного эффекта нет ни на одном горизонте")
    else:
        print(f"  peak: {_label(c['peak_ms'])} ({c['peak_bps']:.3f} б.п.), half-life: {hl}")
    if a.no_log:
        return 0

    tag = "-".join(f"{k}{v}" for k, v in sorted(params.items()))
    chart = f"results/charts/decay-{a.inst}-{a.strategy}-{tag}.png"
    plot(c, f"Затухание сигнала {a.strategy}", chart,
         subtitle=f"{a.inst}, {hours:.1f} ч данных, параметры: {json.dumps(params)}")
    best = next((r for r in c["rows"] if r["horizon_ms"] == c["peak_ms"]), None)
    with RunDB(a.db) as db:
        db.ensure_strategy(a.strategy, hypothesis=STRATEGIES[a.strategy]["hypothesis"])
        rid = db.log_run(
            a.strategy, split="train", params=params,
            metrics={"unit": "bps", "kind": "decay", "curve": c["rows"], "peak_ms": c["peak_ms"],
                     "peak_bps": c["peak_bps"], "half_life_ms": c["half_life_ms"], "hours": hours,
                     "n_trades": best["n"] if best else 0},
            data_from=_day(b["ts_ms"].iloc[0]), data_to=_day(b["ts_ms"].iloc[-1]),
            instruments=[a.inst], artifacts=[chart],
            note="кривая затухания по всем данным (описание сигнала, не оценка доходности)")
        db.pin_run(rid)
    print(f"график: {chart}; записано в журнал: {rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
