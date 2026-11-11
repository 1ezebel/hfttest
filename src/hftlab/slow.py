"""Медленные стратегии на свечах: от 15 минут до дней.

    python -m hftlab.slow --strategy mr_bars --inst BTC-USDT-SWAP --bar 15m
    python -m hftlab.slow --strategy mr_daily   --inst BTC-USDT-SWAP --inst ETH-USDT-SWAP --quiet
    python -m hftlab.slow --strategy mr_lowvol  --inst BTC-USDT-SWAP
    python -m hftlab.slow --strategy basis_fade --inst BTC-USDT-SWAP

Зачем: на горизонтах в секунды движение цены (около 1 б.п.) меньше любой комиссии.
На горизонтах в часы и дни движение измеряется десятками и сотнями б.п., и комиссия
перестаёт быть приговором. Здесь проверяем, остаётся ли после неё что-нибудь.

Стратегии (каждая это отдельная гипотеза и отдельная запись в журнале):
    mr_bars / mom_bars       цена ушла от скользящей средней: ставим на возврат / на продолжение
    mr_daily / mom_daily     то же на дневных свечах (история с 2020 года)
    mr_lowvol                возврат к среднему только при НИЗКОЙ волатильности (спокойный рынок)
    mom_highvol              импульс только при ВЫСОКОЙ волатильности (рынок в движении)
    basis_fade               фьючерс стал необычно дороже спота: ставим на то, что он подешевеет

Как устроена сделка (одна позиция за раз):
  1. По ЗАКРЫТИЮ свечи i считаем сигнал. Он смотрит только в прошлое.
  2. Вход по цене ОТКРЫТИЯ следующей свечи (i+1): раньше узнать о сигнале и исполниться нельзя.
  3. Выход по цене открытия свечи через `hold` свечей после входа.
  4. Из результата вычитается стоимость круга (вход + выход): комиссии и проскальзывание.

Честность:
  * walk-forward: параметры выбираются на одном куске времени, проверяются на следующем;
    сделка целиком (вход и выход) обязана лежать внутри своего куска;
  * сделки, которые пересекают дыру в данных, пропускаются;
  * плацебо-тест: тот же сигнал, сдвинутый во времени на случайный срок;
  * общий t по нескольким инструментам считается с группировкой по времени (clustered):
    монеты двигаются вместе, и десять сделок в один день это не десять независимых наблюдений;
  * каждый перебранный вариант пишется в журнал (hftlab.runs).

Не учтено: плата за финансирование (funding) в этих стратегиях (она есть только за ~92 дня,
см. hftlab.funding), влияние нашей заявки на цену, стоимость плеча. Для дневных стратегий
с удержанием в неделю funding может стоить 10-20 б.п. за сделку: держать в уме.
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
DAY_MS = 86_400_000


# ---------- сигналы: все смотрят только в прошлое ----------

def _z(x: pd.Series, span: int) -> np.ndarray:
    """На сколько стандартных отклонений ряд выше своей EMA."""
    dev = x - x.ewm(span=span, adjust=False).mean()
    win = max(5 * span, 100)
    return (dev / dev.rolling(win, min_periods=win).std()).to_numpy()


def zscore(bars: pd.DataFrame, span: int) -> np.ndarray:
    """z-оценка цены относительно её скользящей средней."""
    return _z(np.log(bars["close"]), span)


def make_side(z: np.ndarray, z_entry: float, flip: bool) -> np.ndarray:
    """+1 купить, -1 продать, 0 ничего. По умолчанию против отклонения (mean reversion)."""
    side = np.zeros(len(z), dtype=np.int8)
    with np.errstate(invalid="ignore"):
        side[z >= z_entry] = -1
        side[z <= -z_entry] = 1
    return -side if flip else side


def sig_price(bars: pd.DataFrame, span: int, z_entry: float, flip: bool = False) -> np.ndarray:
    return make_side(zscore(bars, span), z_entry, flip)


VOL_WIN, RANK_WIN = 24, 720   # волатильность за 24 свечи, её место среди последних 720 свечей


def vol_rank(bars: pd.DataFrame, vol_win: int = VOL_WIN, rank_win: int = RANK_WIN) -> np.ndarray:
    """Место текущей волатильности среди недавних значений: 0 = тише не бывало, 1 = максимум.

    Окна заданы заранее и не подбираются: меньше параметров, меньше шансов подогнать под данные.
    """
    ret = np.log(bars["close"]).diff()
    vol = ret.rolling(vol_win, min_periods=vol_win).std()
    return vol.rolling(rank_win, min_periods=rank_win).rank(pct=True).to_numpy()


def sig_price_vol(bars: pd.DataFrame, span: int, z_entry: float, flip: bool, regime: str) -> np.ndarray:
    """Сигнал по цене, который работает только в заданном режиме волатильности."""
    side = sig_price(bars, span, z_entry, flip)
    rank = vol_rank(bars)
    with np.errstate(invalid="ignore"):
        ok = rank <= 1 / 3 if regime == "low" else rank >= 2 / 3
    side[~ok] = 0   # NaN в начале ряда тоже даёт False: пока режим неизвестен, не торгуем
    return side


def sig_basis(bars: pd.DataFrame, span: int, z_entry: float) -> np.ndarray:
    """Базис = на сколько фьючерс дороже спота. Необычно высокий базис: продаём фьючерс."""
    return make_side(_z(bars["basis_bps"], span), z_entry, flip=False)


GRID = {"span": [12, 48, 192], "z_entry": [1.5, 2.0, 3.0], "hold": [6, 12, 36]}
DAILY_GRID = {"span": [5, 10, 20], "z_entry": [1.0, 1.5, 2.0], "hold": [1, 3, 7]}

STRATEGIES: dict[str, dict[str, Any]] = {
    "mr_bars": {
        "signal": lambda b, **p: sig_price(b, flip=False, **p), "grid": GRID, "bar": "15m", "min_n": 30,
        "hypothesis": "цена, ушедшая от скользящей средней на z сигм, возвращается за часы (mean reversion на свечах)",
    },
    "mom_bars": {
        "signal": lambda b, **p: sig_price(b, flip=True, **p), "grid": GRID, "bar": "15m", "min_n": 30,
        "hypothesis": "цена, ушедшая от скользящей средней на z сигм, продолжает движение за часы (импульс на свечах)",
    },
    "mr_daily": {
        "signal": lambda b, **p: sig_price(b, flip=False, **p), "grid": DAILY_GRID, "bar": "1D", "min_n": 15,
        "hypothesis": "дневная цена, ушедшая от скользящей средней на z сигм, возвращается за 1-7 дней",
    },
    "mom_daily": {
        "signal": lambda b, **p: sig_price(b, flip=True, **p), "grid": DAILY_GRID, "bar": "1D", "min_n": 15,
        "hypothesis": "дневная цена, ушедшая от скользящей средней на z сигм, продолжает движение 1-7 дней",
    },
    "mr_lowvol": {
        "signal": lambda b, **p: sig_price_vol(b, flip=False, regime="low", **p), "grid": GRID, "bar": "1H",
        "min_n": 15,
        "hypothesis": "возврат к средней работает только в спокойном рынке (волатильность в нижней трети за 30 дней)",
    },
    "mom_highvol": {
        "signal": lambda b, **p: sig_price_vol(b, flip=True, regime="high", **p), "grid": GRID, "bar": "1H",
        "min_n": 15,
        "hypothesis": "импульс работает только в бурном рынке (волатильность в верхней трети за 30 дней)",
    },
    "basis_fade": {
        "signal": lambda b, **p: sig_basis(b, **p), "grid": GRID, "bar": "15m", "min_n": 30, "needs": "basis",
        "hypothesis": "когда бессрочный фьючерс необычно дороже спота (базис выше нормы на z сигм), он дешевеет",
    },
}


# ---------- данные ----------

def spot_name(inst: str) -> str:
    """BTC-USDT-SWAP -> BTC-USDT (спот той же монеты)."""
    return inst.removesuffix("-SWAP")


def load_bars(root: str, inst: str, bar: str, needs: str | None = None) -> tuple[pd.DataFrame, int]:
    """Свечи нужного размера и число дыр в исходных данных. Для basis добавляет колонку basis_bps."""
    base_bar = "1D" if bar == "1D" else "5m"
    base = load(root, inst, base_bar)
    bars = resample(base, base_bar, bar)
    n_gaps = len(gaps(base, base_bar))
    if needs == "basis":
        spot = resample(load(root, spot_name(inst), base_bar), base_bar, bar)
        m = bars.merge(spot[["ts_ms", "close"]].rename(columns={"close": "spot_close"}), on="ts_ms", how="inner")
        m["basis_bps"] = (m["close"] / m["spot_close"] - 1) * 1e4
        bars = m.reset_index(drop=True)
    return bars, n_gaps


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
    t_sig, sides, gross, t_in, t_out = [], [], [], [], []
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
        t_in.append(ts[e])
        t_out.append(ts[x])
        sides.append(side[i])
        gross.append(side[i] * (op[x] - op[e]) / op[e] * 1e4)
        free = x
    return {"t_signal": np.array(t_sig, dtype=np.int64), "t_entry": np.array(t_in, dtype=np.int64),
            "t_exit": np.array(t_out, dtype=np.int64), "side": np.array(sides, dtype=np.int8),
            "gross_bps": np.array(gross, dtype=float)}


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
        "gross": g, "t_signal": tr["t_signal"],
    }


def placebo_p(bars: pd.DataFrame, side: np.ndarray, hold: int, lo: int, hi: int, bar_ms: int,
              observed: float | None, n_shifts: int = 200, seed: int = 0) -> float | None:
    """Доля сдвигов сигнала во времени, давших средний результат не хуже настоящего (односторонний p)."""
    span = hi - lo
    min_shift = min(288, span // 4)       # сдвиг заметный, чтобы совпадения с настоящим сигналом были случайны
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
    bars: pd.DataFrame, bar_ms: int, signal: Callable[..., np.ndarray], grid: dict[str, list],
    cost_bps: float, n_chunks: int = 4, min_n: int = 30,
    on_result: Callable[[str, dict, dict, tuple[int, int]], None] | None = None,
) -> list[dict[str, Any]]:
    """Кусок k: выбираем лучшие параметры по результату после издержек. Кусок k+1: проверяем только их.

    signal(bars, **параметры без hold) возвращает side для каждой свечи.
    """
    edges = np.linspace(0, len(bars), n_chunks + 1).astype(int)
    cache: dict[tuple, np.ndarray] = {}   # сигнал не зависит от hold: считаем один раз

    def side_for(p: dict) -> np.ndarray:
        sp = {k: v for k, v in p.items() if k != "hold"}
        key = tuple(sorted(sp.items()))
        if key not in cache:
            cache[key] = signal(bars, **sp)
        return cache[key]

    out = []
    for k in range(n_chunks - 1):
        tr_rng, te_rng = (edges[k], edges[k + 1]), (edges[k + 1], edges[k + 2])
        best, best_score = None, -math.inf
        for p in grid_points(grid):
            m = evaluate(bars, side_for(p), p["hold"], *tr_rng, bar_ms, cost_bps)
            if on_result:
                on_result("train", p, m, tr_rng)
            if m["n"] >= min_n and m["mean_net_bps"] is not None and m["mean_net_bps"] > best_score:
                best, best_score = p, m["mean_net_bps"]
        if best is None:
            out.append({"fold": k + 1, "params": None})
            continue
        m = evaluate(bars, side_for(best), best["hold"], *te_rng, bar_ms, cost_bps)
        m["placebo_p"] = placebo_p(bars, side_for(best), best["hold"], *te_rng, bar_ms, m["mean_gross_bps"])
        if on_result:
            on_result(f"wf-{k + 1}", best, m, te_rng)
        out.append({"fold": k + 1, "params": best, "train_net": best_score, "test": m})
    return out


# ---------- сводка по нескольким инструментам ----------

def clustered_t(values: np.ndarray, cluster: np.ndarray) -> float | None:
    """t-статистика среднего, где наблюдения внутри одной группы считаются зависимыми.

    Обычный t верит, что все сделки независимы. Но десять монет в один и тот же день
    двигаются вместе: это ближе к одной сделке, чем к десяти. Здесь сделки собираются
    в группы по времени, и разброс оценивается между группами.
    """
    n = len(values)
    if n < 2:
        return None
    mean = values.mean()
    _, inv = np.unique(cluster, return_inverse=True)
    k = inv.max() + 1
    if k < 2:
        return None
    resid = np.bincount(inv, weights=values - mean, minlength=k)   # сумма отклонений в каждой группе
    var = (resid ** 2).sum() / n ** 2 * k / (k - 1)
    return float(mean / math.sqrt(var)) if var > 0 else None


def pooled_summary(per_inst: dict[str, dict[str, np.ndarray]], cluster_ms: int = DAY_MS,
                   family: int = 1) -> dict[str, Any]:
    """Тестовые сделки по инструментам и вместе.

    per_inst: инструмент -> {"gross": результаты сделок, "t": время сигнала}.
    family: сколько стратегий проверяется в этой серии. Порог значимости общего t
    ужесточается по Бонферрони: уровень 5% делится на число проверок.
    """
    rows = []
    for inst, d in per_inst.items():
        s = summarize(d["gross"])
        rows.append({"inst": inst, "n": s["n"], "mean": s["mean"], "t": s["t"]})
    parts = [d for d in per_inst.values() if len(d["gross"])]
    allg = np.concatenate([d["gross"] for d in parts]) if parts else np.array([])
    allt = np.concatenate([d["t"] for d in parts]) if parts else np.array([])
    tot = summarize(allg)
    k = max(len(per_inst), 1)
    return {
        "rows": rows, "n": tot["n"], "mean": tot["mean"], "t_naive": tot["t"],
        "t": clustered_t(allg, allt // cluster_ms) if len(allg) else None,
        "t_threshold_inst": NormalDist().inv_cdf(1 - 0.05 / (2 * k)),
        "t_threshold_pooled": NormalDist().inv_cdf(1 - 0.05 / (2 * max(family, 1))),
        "k": k, "family": family,
    }


# ---------- запуск ----------

def _fmt(v, nd=2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _day(ms: float) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser(description="Медленные стратегии на свечах, walk-forward")
    ap.add_argument("--bars", default="bars", help="папка со свечами (hftlab.candles)")
    ap.add_argument("--inst", action="append", help="инструмент, можно несколько раз")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="mr_bars")
    ap.add_argument("--bar", choices=sorted(BAR_MS), help="размер свечи (по умолчанию свой у каждой стратегии)")
    ap.add_argument("--cost-bps", type=float, default=20.0,
                    help="стоимость круга (вход+выход) в б.п.: 2 x (комиссия + проскальзывание)")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--min-n", type=int, help="минимум сделок на train, чтобы вариант мог победить")
    ap.add_argument("--family", type=int, default=1,
                    help="сколько стратегий в этой серии проверок (для порога значимости)")
    ap.add_argument("--db", default="results/runs.sqlite")
    ap.add_argument("--no-log", action="store_true", help="не писать в журнал (только отладка кода)")
    ap.add_argument("--quiet", action="store_true", help="печатать только сводку по инструментам")
    a = ap.parse_args()
    insts = a.inst or ["BTC-USDT-SWAP"]
    spec = STRATEGIES[a.strategy]
    bar = a.bar or spec["bar"]
    bar_ms = BAR_MS[bar]
    min_n = a.min_n or spec["min_n"]
    cluster_ms = 7 * DAY_MS if bar == "1D" else DAY_MS

    db = None if a.no_log else RunDB(a.db)
    if db:
        db.ensure_strategy(a.strategy, hypothesis=spec["hypothesis"])

    per_inst: dict[str, dict[str, np.ndarray]] = {}
    for inst in insts:
        bars, n_gaps = load_bars(a.bars, inst, bar, spec.get("needs"))
        days = (bars["ts_ms"].iloc[-1] - bars["ts_ms"].iloc[0]) / DAY_MS
        if not a.quiet:
            print(f"{inst} {bar}: {len(bars)} свечей, {days:.0f} дней, дыр в исходных данных: {n_gaps}, "
                  f"стратегия {a.strategy}, стоимость круга {a.cost_bps:g} б.п.")

        def log(split: str, params: dict, m: dict, rng: tuple[int, int], bars=bars, inst=inst) -> None:
            if db is None:
                return
            metrics = {k: v for k, v in m.items() if k not in ("gross", "t_signal")}
            metrics.update(unit="bps", kind="bars", bar=bar, cost_bps=a.cost_bps, n_trades=m["n"],
                           pnl_gross=m["sum_gross_bps"], pnl_net=m["sum_net_bps"], fees=m["fees_bps"],
                           execution="signal at bar close, fill at next bar open, exit hold bars later")
            db.log_run(a.strategy, split=split, params={**params, "bar": bar}, metrics=metrics,
                       data_from=_day(bars["ts_ms"].iloc[rng[0]]), data_to=_day(bars["ts_ms"].iloc[rng[1] - 1]),
                       instruments=[inst])

        res = walk_forward_bars(bars, bar_ms, spec["signal"], spec["grid"], a.cost_bps, a.chunks, min_n,
                                on_result=log)
        g_parts, t_parts = [], []
        if not a.quiet:
            print("\n  fold  params                                         train_net  n   gross   t   plac_p   net")
        for r in res:
            if r["params"] is None:
                if not a.quiet:
                    print(f"  {r['fold']:>4}  (на train не набралось {min_n} сделок ни у одного варианта)")
                continue
            t = r["test"]
            g_parts.append(t["gross"])
            t_parts.append(t["t_signal"])
            if not a.quiet:
                print(f"  {r['fold']:>4}  {str(r['params']):<46} {_fmt(r['train_net']):>8} {t['n']:>4} "
                      f"{_fmt(t['mean_gross_bps']):>7} {_fmt(t['t'], 1):>5} {_fmt(t['placebo_p']):>7} "
                      f"{_fmt(t['mean_net_bps']):>6}")
        per_inst[inst] = {"gross": np.concatenate(g_parts) if g_parts else np.array([]),
                          "t": np.concatenate(t_parts) if t_parts else np.array([], dtype=np.int64)}

    ps = pooled_summary(per_inst, cluster_ms, a.family)
    print(f"\n{a.strategy}, свечи {bar}, стоимость круга {a.cost_bps:g} б.п.: только тестовые куски walk-forward")
    print("  инструмент              сделок   до издержек, б.п.     t")
    for r in ps["rows"]:
        print(f"  {r['inst']:<22} {r['n']:>7} {_fmt(r['mean']):>13} {_fmt(r['t'], 1):>12}")
    if len(insts) > 1:
        print(f"  {'ВМЕСТЕ':<22} {ps['n']:>7} {_fmt(ps['mean']):>13} {_fmt(ps['t'], 1):>12}"
              f"   (без учёта связи между монетами было бы {_fmt(ps['t_naive'], 1)})")
        print(f"  Порог для отдельного инструмента ({ps['k']} проверок): |t| > {ps['t_threshold_inst']:.1f}")
    print(f"  Порог для общего результата (стратегий в серии: {ps['family']}): |t| > {ps['t_threshold_pooled']:.1f}")
    mean = ps["mean"]
    if mean is None:
        print("  Сделок на тестовых кусках нет.")
    else:
        print("  После издержек на сделку: "
              + ", ".join(f"{c:.0f} б.п. круг: {mean - c:+.1f}" for c in COSTS_BPS))
        if mean > 0:
            print(f"  Безубыточная стоимость круга: {mean:.1f} б.п.")
        else:
            print("  До издержек результат не положительный.")
    if db:
        n = db.count_runs(a.strategy)
        print(f"\nВсего вариантов в журнале по {a.strategy}: {n}. Верьте только тестовым кускам, "
              "а не лучшему из перебранных.")
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
