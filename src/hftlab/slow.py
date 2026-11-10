"""Медленные стратегии на свечах: возврат к среднему (mr_bars) и импульс (mom_bars).

    python -m hftlab.slow --inst BTC-USDT-SWAP --bar 15m --strategy mr_bars
    python -m hftlab.slow --inst BTC-USDT-SWAP --bar 15m --strategy mom_bars --cost-bps 30

Зачем: на горизонтах в секунды движение цены (около 1 б.п.) меньше любой комиссии.
На горизонтах в десятки минут и часы движение измеряется десятками б.п., и комиссия
перестаёт быть приговором. Здесь проверяем, остаётся ли после неё что-нибудь.

Как устроена сделка (одна позиция за раз):
  1. По ЗАКРЫТИЮ свечи i считаем z-оценку: насколько цена ушла от своей скользящей средней
     (в стандартных отклонениях за недавнее время). Сигнал смотрит только в прошлое.
  2. Сигнал: mr_bars ставит ПРОТИВ отклонения (ждёт возврата), mom_bars ПО отклонению.
  3. Вход по цене ОТКРЫТИЯ следующей свечи (i+1): раньше узнать о сигнале и исполниться нельзя.
  4. Выход по цене открытия свечи через `hold` свечей после входа.
  5. Из результата вычитается стоимость круга (вход + выход): комиссии и проскальзывание.

Честность:
  * walk-forward: параметры выбираются на одном куске времени, проверяются на следующем;
    сделка целиком (вход и выход) обязана лежать внутри своего куска;
  * сделки, которые пересекают дыру в данных, пропускаются;
  * плацебо-тест: тот же сигнал, сдвинутый во времени на случайный срок, показывает,
    какие результаты дают пустые совпадения. p-значение = доля плацебо не хуже настоящего;
  * каждый перебранный вариант пишется в журнал (hftlab.runs): число проб видно всегда;
  * стоимость круга это параметр. "Окупается" имеет смысл только вместе с ней, поэтому
    всегда печатается безубыточная стоимость: сколько б.п. за круг стратегия выдерживает.

Не учтено: плата за финансирование (funding) фьючерсов, влияние нашей заявки на цену,
стоимость плеча. Для размера сделки в сотни долларов влияние на цену несущественно.
"""
from __future__ import annotations

import argparse
import math
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd

from hftlab.candles import BAR_MS, gaps, load, resample
from hftlab.research import grid_points, summarize
from hftlab.runs import RunDB

COSTS_BPS = (10.0, 20.0, 40.0)  # стоимость круга для сводной таблицы: дёшево / средне / дорого


# ---------- сигналы ----------

def zscore(bars: pd.DataFrame, span: int) -> np.ndarray:
    """На сколько стандартных отклонений цена выше своей EMA. Использует только прошлое."""
    c = np.log(bars["close"])
    dev = c - c.ewm(span=span, adjust=False).mean()
    win = max(5 * span, 100)
    return (dev / dev.rolling(win, min_periods=win).std()).to_numpy()


def make_side(z: np.ndarray, z_entry: float, flip: bool) -> np.ndarray:
    """+1 купить, -1 продать, 0 ничего. По умолчанию против отклонения (mean reversion)."""
    side = np.zeros(len(z), dtype=np.int8)
    with np.errstate(invalid="ignore"):
        side[z >= z_entry] = -1
        side[z <= -z_entry] = 1
    return -side if flip else side


STRATEGIES: dict[str, dict[str, Any]] = {
    "mr_bars": {
        "flip": False,
        "hypothesis": "цена, ушедшая от скользящей средней на z сигм, возвращается за часы (mean reversion на свечах)",
    },
    "mom_bars": {
        "flip": True,
        "hypothesis": "цена, ушедшая от скользящей средней на z сигм, продолжает движение за часы (импульс на свечах)",
    },
}
GRID = {"span": [12, 48, 192], "z_entry": [1.5, 2.0, 3.0], "hold": [6, 12, 36]}


# ---------- сделки ----------

def run_trades(bars: pd.DataFrame, side: np.ndarray, hold: int, lo: int = 0, hi: int | None = None,
               bar_ms: int = 300_000) -> dict[str, np.ndarray]:
    """Сделки с входами в строках [lo, hi). Вход и выход обязаны лежать внутри [lo, hi)."""
    n = len(bars)
    hi = n if hi is None else hi
    ts, op = bars["ts_ms"].to_numpy(), bars["open"].to_numpy()
    prev = np.roll(side, 1)
    prev[0] = 0
    onsets = np.flatnonzero((side != 0) & (side != prev))   # начало сигнала, а не каждая строка
    t_sig, sides, gross = [], [], []
    free = 0  # строка, начиная с которой можно открывать новую позицию (прошлая закрыта)
    for i in onsets:
        if i < lo or i >= hi or i < free:
            continue
        e, x = i + 1, i + 1 + hold
        if x >= hi:
            continue
        if ts[e] - ts[i] != bar_ms or ts[x] - ts[e] != hold * bar_ms:
            continue  # дыра в данных между сигналом, входом и выходом
        t_sig.append(ts[i])
        sides.append(side[i])
        gross.append(side[i] * (op[x] - op[e]) / op[e] * 1e4)
        free = x
    return {"t_signal": np.array(t_sig), "side": np.array(sides), "gross_bps": np.array(gross, dtype=float)}


def evaluate(bars: pd.DataFrame, side: np.ndarray, hold: int, lo: int, hi: int, bar_ms: int,
             cost_bps: float) -> dict[str, Any]:
    tr = run_trades(bars, side, hold, lo, hi, bar_ms)
    g = tr["gross_bps"]
    s = summarize(g)
    n = int(s["n"])
    mean = s["mean"]
    return {
        "n": n, "mean_gross_bps": mean, "t": s["t"],
        "mean_net_bps": None if mean is None else mean - cost_bps,
        "breakeven_cost_bps": mean,                       # при какой стоимости круга результат нулевой
        "sum_gross_bps": float(g.sum()), "sum_net_bps": float((g - cost_bps).sum()),
        "fees_bps": cost_bps * n,
        "win_rate": float((g > 0).mean()) if n else None,
        "gross": g,
    }


def placebo_p(bars: pd.DataFrame, side: np.ndarray, hold: int, lo: int, hi: int, bar_ms: int,
              observed: float, n_shifts: int = 200, seed: int = 0) -> float | None:
    """Доля сдвигов сигнала во времени, давших средний результат не хуже настоящего (односторонний p)."""
    span = hi - lo
    min_shift = min(288, span // 4)       # сдвиг не меньше суток на 5-минутках, чтобы совпадения были случайны
    if span < 4 * max(min_shift, 1) or observed is None:
        return None
    rng = np.random.default_rng(seed)
    window = side[lo:hi]
    better = used = 0
    for _ in range(n_shifts):
        sh = int(rng.integers(min_shift, span - min_shift))
        shifted = np.zeros(len(side), dtype=side.dtype)
        shifted[lo:hi] = np.roll(window, sh)
        g = run_trades(bars, shifted, hold, lo, hi, bar_ms)["gross_bps"]
        if len(g) == 0:
            continue
        used += 1
        better += g.mean() >= observed
    return (better + 1) / (used + 1) if used else None


# ---------- walk-forward ----------

def walk_forward_bars(
    bars: pd.DataFrame, bar_ms: int, flip: bool, cost_bps: float, n_chunks: int = 4, min_n: int = 30,
    grid: dict[str, list] = GRID,
    on_result: Callable[[str, dict, dict, tuple[int, int]], None] | None = None,
) -> list[dict[str, Any]]:
    """Кусок k: выбираем лучшие параметры по результату после издержек. Кусок k+1: проверяем только их."""
    edges = np.linspace(0, len(bars), n_chunks + 1).astype(int)
    signals: dict[tuple, np.ndarray] = {}   # сигнал не зависит от hold: считаем один раз на (span, z)
    zs = {sp: zscore(bars, sp) for sp in grid["span"]}
    for p in grid_points(grid):
        key = (p["span"], p["z_entry"])
        if key not in signals:
            signals[key] = make_side(zs[p["span"]], p["z_entry"], flip)

    out = []
    for k in range(n_chunks - 1):
        tr_rng, te_rng = (edges[k], edges[k + 1]), (edges[k + 1], edges[k + 2])
        best, best_score = None, -math.inf
        for p in grid_points(grid):
            side = signals[(p["span"], p["z_entry"])]
            m = evaluate(bars, side, p["hold"], *tr_rng, bar_ms, cost_bps)
            if on_result:
                on_result("train", p, m, tr_rng)
            if m["n"] >= min_n and m["mean_net_bps"] is not None and m["mean_net_bps"] > best_score:
                best, best_score = p, m["mean_net_bps"]
        if best is None:
            out.append({"fold": k + 1, "params": None})
            continue
        side = signals[(best["span"], best["z_entry"])]
        m = evaluate(bars, side, best["hold"], *te_rng, bar_ms, cost_bps)
        m["placebo_p"] = placebo_p(bars, side, best["hold"], *te_rng, bar_ms, m["mean_gross_bps"])
        if on_result:
            on_result(f"wf-{k + 1}", best, m, te_rng)
        out.append({"fold": k + 1, "params": best, "train_net": best_score, "test": m})
    return out


# ---------- запуск ----------

def _fmt(v, nd=2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _day(ms: float) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def pooled_summary(per_inst: dict[str, np.ndarray]) -> dict[str, Any]:
    """Сводка тестовых сделок по инструментам и вместе, с порогом t на множественные проверки.

    Порог Бонферрони: при k независимых проверках "значимым" считаем только |t| выше порога
    для уровня 5% / k. Инструменты крипторынка сильно двигаются вместе, поэтому общий t
    завышает независимость выборки: относиться к нему как к верхней оценке.
    """
    rows = []
    for inst, g in per_inst.items():
        s = summarize(g)
        rows.append({"inst": inst, "n": s["n"], "mean": s["mean"], "t": s["t"]})
    parts = [g for g in per_inst.values() if len(g)]
    allg = np.concatenate(parts) if parts else np.array([])
    tot = summarize(allg)
    k = max(len(per_inst), 1)
    return {"rows": rows, "n": tot["n"], "mean": tot["mean"], "t": tot["t"],
            "t_threshold": NormalDist().inv_cdf(1 - 0.05 / (2 * k)), "k": k}


def main() -> int:
    ap = argparse.ArgumentParser(description="Возврат к среднему / импульс на свечах, walk-forward")
    ap.add_argument("--bars", default="bars", help="папка со свечами (hftlab.candles)")
    ap.add_argument("--inst", action="append", help="инструмент, можно несколько раз")
    ap.add_argument("--bar", default="15m", choices=sorted(BAR_MS), help="размер свечи для сигнала")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="mr_bars")
    ap.add_argument("--cost-bps", type=float, default=20.0,
                    help="стоимость круга (вход+выход) в б.п.: 2 x (комиссия + проскальзывание)")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--min-n", type=int, default=30, help="минимум сделок на train, чтобы вариант мог победить")
    ap.add_argument("--db", default="results/runs.sqlite")
    ap.add_argument("--no-log", action="store_true", help="не писать в журнал (только отладка кода)")
    ap.add_argument("--quiet", action="store_true", help="печатать только сводку по инструментам")
    a = ap.parse_args()
    insts = a.inst or ["BTC-USDT-SWAP"]
    bar_ms, spec = BAR_MS[a.bar], STRATEGIES[a.strategy]

    db = None if a.no_log else RunDB(a.db)
    if db:
        db.ensure_strategy(a.strategy, hypothesis=spec["hypothesis"])

    per_inst: dict[str, np.ndarray] = {}
    for inst in insts:
        base = load(a.bars, inst, "5m")
        bars = resample(base, "5m", a.bar)
        days = (bars["ts_ms"].iloc[-1] - bars["ts_ms"].iloc[0]) / 86_400_000
        if not a.quiet:
            print(f"{inst} {a.bar}: {len(bars)} свечей, {days:.0f} дней, дыр в 5m: {len(gaps(base, '5m'))}, "
                  f"стратегия {a.strategy}, стоимость круга {a.cost_bps:g} б.п.")

        def log(split: str, params: dict, m: dict, rng: tuple[int, int], bars=bars, inst=inst) -> None:
            if db is None:
                return
            metrics = {k: v for k, v in m.items() if k != "gross"}
            metrics.update(unit="bps", kind="bars", bar=a.bar, cost_bps=a.cost_bps, n_trades=m["n"],
                           pnl_gross=m["sum_gross_bps"], pnl_net=m["sum_net_bps"], fees=m["fees_bps"],
                           execution="signal at bar close, fill at next bar open, exit hold bars later")
            db.log_run(a.strategy, split=split, params={**params, "bar": a.bar}, metrics=metrics,
                       data_from=_day(bars["ts_ms"].iloc[rng[0]]), data_to=_day(bars["ts_ms"].iloc[rng[1] - 1]),
                       instruments=[inst])

        res = walk_forward_bars(bars, bar_ms, spec["flip"], a.cost_bps, a.chunks, a.min_n, on_result=log)
        tests = []
        if not a.quiet:
            print("\n  fold  params                                         train_net  n   gross   t   plac_p   net")
        for r in res:
            if r["params"] is None:
                if not a.quiet:
                    print(f"  {r['fold']:>4}  (на train не набралось {a.min_n} сделок ни у одного варианта)")
                continue
            t = r["test"]
            tests.append(t["gross"])
            if not a.quiet:
                print(f"  {r['fold']:>4}  {str(r['params']):<46} {_fmt(r['train_net']):>8} {t['n']:>4} "
                      f"{_fmt(t['mean_gross_bps']):>7} {_fmt(t['t'], 1):>5} {_fmt(t['placebo_p']):>7} "
                      f"{_fmt(t['mean_net_bps']):>6}")
        per_inst[inst] = np.concatenate(tests) if tests else np.array([])

        if len(insts) == 1 and tests:
            g = per_inst[inst]
            s = summarize(g)
            print(f"\nВсе тестовые куски вместе: сделок {s['n']}, средний результат до издержек "
                  f"{_fmt(s['mean'])} б.п., t = {_fmt(s['t'], 1)}")
            print("Результат на сделку после издержек при разной стоимости круга:")
            for c in COSTS_BPS:
                print(f"   {c:>4.0f} б.п.: {_fmt(s['mean'] - c) if s['mean'] is not None else '-':>7}")
            be = s["mean"]
            if be is not None and be > 0:
                print(f"Безубыточная стоимость круга: {be:.1f} б.п. Если ваша выше, стратегия убыточна.")
            else:
                print("До издержек результат не положительный: на этих данных сигнал не работает.")

    if len(insts) > 1:
        ps = pooled_summary(per_inst)
        print(f"\n{a.strategy}, свечи {a.bar}, стоимость круга {a.cost_bps:g} б.п.: только тестовые куски walk-forward")
        print("  инструмент              сделок   до издержек, б.п.     t")
        for r in ps["rows"]:
            print(f"  {r['inst']:<22} {r['n']:>7} {_fmt(r['mean']):>13} {_fmt(r['t'], 1):>12}")
        print(f"  {'ВМЕСТЕ':<22} {ps['n']:>7} {_fmt(ps['mean']):>13} {_fmt(ps['t'], 1):>12}")
        print(f"  Порог значимости для {ps['k']} проверок: |t| > {ps['t_threshold']:.1f} "
              "(инструменты двигаются вместе, общий t завышен)")
    if db:
        n = db.count_runs(a.strategy)
        print(f"\nВсего вариантов в журнале по {a.strategy}: {n}. Лучший из многих выглядит лучше, чем есть: "
              "верьте только результатам тестовых кусков и p плацебо-теста.")
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
