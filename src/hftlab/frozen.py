"""Проверка стратегии с ЗАМОРОЖЕННЫМИ параметрами на новых данных.

    python -m hftlab.frozen --strategy mom_daily --params '{"span": 5, "z_entry": 1.5, "hold": 7}' \\
        --bars bars_gate --name mom_daily_gate_new --inst ZEC-USDT-SWAP --inst LTC-USDT-SWAP

Чем это отличается от hftlab.slow. Там параметры подбираются на одном куске и проверяются
на следующем. Здесь подбора нет вообще: параметры выбраны раньше и на других данных, поэтому
вся история новых инструментов служит тестом. Это самый честный вид проверки, доступный без
ожидания будущих данных: одна гипотеза, один набор параметров, один ответ.

Плата за финансирование вычитается из каждой сделки, если в папке инструмента лежит
funding.parquet. Сделки, для которых истории funding нет, в итог не входят.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hftlab.candles import BAR_MS
from hftlab.research import summarize
from hftlab.runs import RunDB
from hftlab.slow import DAY_MS, STRATEGIES, _day, _fmt, clustered_t, load_bars, run_trades


def funding_for_trades(tr: dict[str, np.ndarray], f: pd.DataFrame | None) -> np.ndarray:
    """Funding в б.п. для каждой сделки (плюс = получили). NaN, если история не покрывает сделку."""
    n = len(tr["gross_bps"])
    out = np.full(n, np.nan)
    if f is None or f.empty:
        return out
    ft, fr = f["ts_ms"].to_numpy(), f["rate"].to_numpy()
    csum = np.r_[0.0, np.cumsum(fr)]
    lo = np.searchsorted(ft, tr["t_entry"], side="right")   # выплаты строго после входа
    hi = np.searchsorted(ft, tr["t_exit"], side="right")    # и не позже выхода
    paid = csum[hi] - csum[lo]
    covered = (tr["t_entry"] >= ft[0]) & (tr["t_exit"] <= ft[-1] + DAY_MS)
    out[covered] = -tr["side"][covered] * paid[covered] * 1e4   # длинная позиция платит положительную ставку
    return out


def frozen_trades(bars: pd.DataFrame, side: np.ndarray, hold: int, bar_ms: int,
                  funding: pd.DataFrame | None) -> pd.DataFrame:
    tr = run_trades(bars, side, hold, bar_ms=bar_ms)
    fund = funding_for_trades(tr, funding)
    return pd.DataFrame({"t_signal": tr["t_signal"], "side": tr["side"], "price_bps": tr["gross_bps"],
                         "funding_bps": fund, "total_bps": tr["gross_bps"] + fund})


def summary(trades: pd.DataFrame, cluster_ms: int, cost_bps: float) -> dict[str, Any]:
    """Итог по сделкам с известным funding. Все числа в б.п. на сделку."""
    d = trades.dropna(subset=["total_bps"])
    g = d["total_bps"].to_numpy()
    s = summarize(g)
    out: dict[str, Any] = {
        "n": s["n"], "n_no_funding": int(len(trades) - len(d)),
        "mean_price_bps": float(d["price_bps"].mean()) if len(d) else None,
        "mean_funding_bps": float(d["funding_bps"].mean()) if len(d) else None,
        "mean_total_bps": s["mean"], "t_naive": s["t"],
        "t": clustered_t(g, (d["t_signal"] // cluster_ms).to_numpy()) if len(d) else None,
        "mean_net_bps": None if s["mean"] is None else s["mean"] - cost_bps,
        "win_rate": float((g > 0).mean()) if len(d) else None,
        "median_bps": float(np.median(g)) if len(d) else None,
    }
    for name, sel in (("long", d["side"] > 0), ("short", d["side"] < 0)):
        out[f"n_{name}"] = int(sel.sum())
        out[f"mean_{name}_bps"] = float(d.loc[sel, "total_bps"].mean()) if sel.any() else None
    top = np.sort(g)[::-1]
    out["mean_without_top10_bps"] = float(top[10:].mean()) if len(top) > 10 else None
    years = pd.to_datetime(d["t_signal"], unit="ms").dt.year
    out["by_year"] = {int(y): {"n": int(len(x)), "mean": float(x.mean())}
                      for y, x in d.groupby(years)["total_bps"]}
    return out


def verdict(s: dict[str, Any], t_threshold: float = 1.96) -> tuple[bool, str]:
    """Правило из плана проверки: t выше порога с верным знаком И плюс после издержек."""
    if not s["n"] or s["t"] is None:
        return False, "сделок нет, проверять нечего"
    stat = s["t"] > t_threshold
    econ = s["mean_net_bps"] > 0
    if stat and econ:
        return True, "оба условия выполнены: кандидат получает право на бумажную торговлю, это ещё не доказательство"
    why = []
    if not stat:
        why.append(f"t = {s['t']:.2f} не выше {t_threshold}")
    if not econ:
        why.append(f"после издержек {s['mean_net_bps']:+.1f} б.п.")
    return False, "не пройдено: " + "; ".join(why)


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка с замороженными параметрами")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    ap.add_argument("--params", required=True, help="JSON, включая hold")
    ap.add_argument("--inst", action="append", required=True)
    ap.add_argument("--bars", default="bars_gate")
    ap.add_argument("--name", required=True, help="имя стратегии в журнале, например mom_daily_gate_new")
    ap.add_argument("--split", default="oos", choices=["oos", "test"],
                    help="oos = новые данные (главный тест), test = описательная проверка")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--db", default="results/runs.sqlite")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    spec = STRATEGIES[a.strategy]
    params = json.loads(a.params)
    hold = params["hold"]
    sig_params = {k: v for k, v in params.items() if k != "hold"}
    bar = spec["bar"]
    bar_ms = BAR_MS[bar]
    cluster_ms = 7 * DAY_MS if bar == "1D" else DAY_MS

    print(f"{a.name}: стратегия {a.strategy}, параметры {params} (заморожены), свечи {bar}, "
          f"стоимость круга {a.cost_bps:g} б.п.")
    print("  инструмент            сделок    цена  funding   всего      t   без funding")
    parts = []
    for inst in a.inst:
        bars, _ = load_bars(a.bars, inst, bar, spec.get("needs"))
        fp = Path(a.bars) / inst / "funding.parquet"
        tr = frozen_trades(bars, spec["signal"](bars, **sig_params), hold, bar_ms,
                           pd.read_parquet(fp) if fp.exists() else None)
        tr["inst"] = inst
        parts.append(tr)
        s = summary(tr, cluster_ms, a.cost_bps)
        print(f"  {inst:<20} {s['n']:>6} {_fmt(s['mean_price_bps'], 1):>7} {_fmt(s['mean_funding_bps'], 1):>8} "
              f"{_fmt(s['mean_total_bps'], 1):>7} {_fmt(s['t'], 1):>6} {s['n_no_funding']:>8}")
    allt = pd.concat(parts, ignore_index=True)
    s = summary(allt, cluster_ms, a.cost_bps)
    print(f"  {'ВМЕСТЕ':<20} {s['n']:>6} {_fmt(s['mean_price_bps'], 1):>7} {_fmt(s['mean_funding_bps'], 1):>8} "
          f"{_fmt(s['mean_total_bps'], 1):>7} {_fmt(s['t'], 1):>6} {s['n_no_funding']:>8}"
          f"   (t без учёта связи между монетами: {_fmt(s['t_naive'], 1)})")
    print(f"\n  После стоимости круга: {_fmt(s['mean_net_bps'], 1)} б.п. на сделку. "
          f"Доля прибыльных: {_fmt(None if s['win_rate'] is None else s['win_rate'] * 100, 0)}%, "
          f"медиана {_fmt(s['median_bps'], 1)} б.п.")
    print(f"  Длинные: {s['n_long']} сделок, {_fmt(s['mean_long_bps'], 1)} б.п. "
          f"Короткие: {s['n_short']} сделок, {_fmt(s['mean_short_bps'], 1)} б.п.")
    print(f"  Без 10 лучших сделок: {_fmt(s['mean_without_top10_bps'], 1)} б.п.")
    print("  По годам: " + ", ".join(f"{y}: {v['mean']:+.0f} ({v['n']})" for y, v in s["by_year"].items()))
    ok, text = verdict(s)
    if a.split == "oos":
        print(f"\n  ВЕРДИКТ: {text}")
    if not a.no_log:
        d = allt.dropna(subset=["total_bps"])
        with RunDB(a.db) as db:
            db.ensure_strategy(a.name, hypothesis=spec["hypothesis"] + "; параметры заморожены, новые данные")
            metrics = {k: v for k, v in s.items()}
            metrics.update(unit="bps", kind="frozen", n_trades=s["n"], cost_bps=a.cost_bps, passed=ok,
                           pnl_gross=float(d["total_bps"].sum()),
                           pnl_net=float((d["total_bps"] - a.cost_bps).sum()), fees=a.cost_bps * s["n"],
                           by_year={str(k): v for k, v in s["by_year"].items()})
            rid = db.log_run(a.name, split=a.split, params=params, metrics=metrics,
                             data_from=_day(allt["t_signal"].min()), data_to=_day(allt["t_signal"].max()),
                             instruments=a.inst, note=f"замороженные параметры, источник {a.bars}; {text}")
            db.pin_run(rid)
        print(f"  записано в журнал: {rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
