"""Этап 2, шаг 2: проверка сигналов через markout и walk-forward.

    python -m hftlab.research processed --inst BTC-USDT-SWAP --strategy mr_ema
    python -m hftlab.research processed --inst BTC-USDT-SWAP --strategy micro_drift

Идея проверки (markout): в момент сигнала запоминаем mid, смотрим mid через
100 мс / 1 с / 5 с / 30 с. Если сигнал "покупай" и цена в среднем выросла,
у сигнала есть предсказательная сила. Это ещё НЕ прибыль: дальше вычитаем издержки.

Что здесь сделано для честности:
  * сигнал в строке t использует только данные до t включительно (EMA и std считаются назад);
  * будущая цена берётся только внутри того же непрерывного куска (seg), через дыры не смотрим;
  * считаем только НАЧАЛО сигнала (onset) и не чаще одного раза за горизонт: соседние
    строки с тем же сигналом это одно и то же наблюдение, а не сто независимых;
  * параметры выбираются на одном отрезке времени (train), а оцениваются на следующем (test);
  * каждый вариант параметров пишется в журнал (hftlab.runs), включая проигравшие.

Издержки "по рынку" (taker): вход и выход пересекают спред и платят taker-комиссию.
Пассивное исполнение (maker) здесь оценить нельзя: неизвестно, исполнят ли нас.
Это задача симулятора (hftlab.sim).
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from hftlab.features import load
from hftlab.runs import RunDB

# Базовый тариф OKX для фьючерсов (обычный пользователь), в базисных пунктах (1 б.п. = 0.01%).
# ПРОВЕРЬТЕ свой тариф в личном кабинете: от него зависит весь вывод.
MAKER_FEE_BPS = 2.0
TAKER_FEE_BPS = 5.0

HORIZONS_MS = (100, 1000, 5000, 30000)
ROW_MS = 100  # стакан приходит примерно раз в 100 мс


# ---------- кирпичики ----------

def forward_mid(b: pd.DataFrame, horizon_ms: int) -> np.ndarray:
    """mid через horizon_ms после каждой строки. NaN, если до конца куска меньше горизонта."""
    out = np.full(len(b), np.nan)
    ts, mid = b["ts_ms"].to_numpy(), b["mid"].to_numpy()
    for idx in b.groupby("seg", sort=False).indices.values():
        t = ts[idx]
        j = np.searchsorted(t, t + horizon_ms, side="left")
        ok = j < len(idx)
        out[idx[ok]] = mid[idx[j[ok]]]
    return out


def onsets(side: np.ndarray, ts_ms: np.ndarray, seg: np.ndarray, cooldown_ms: int) -> np.ndarray:
    """Индексы строк, где сигнал НАЧАЛСЯ, не чаще одного раза за cooldown_ms."""
    prev = np.roll(side, 1)
    prev[0] = 0
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    cand = np.flatnonzero((side != 0) & ((side != prev) | new_seg))
    keep, last = [], -math.inf
    for i in cand:
        if ts_ms[i] - last >= cooldown_ms:
            keep.append(i)
            last = ts_ms[i]
    return np.array(keep, dtype=np.int64)


def summarize(x: np.ndarray) -> dict[str, float | int | None]:
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "t": None}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd > 0 else None
    return {"n": n, "mean": mean, "t": t}


# ---------- сигналы: возвращают side для каждой строки (+1 купить, -1 продать, 0 ничего) ----------

def sig_mr_ema(b: pd.DataFrame, halflife_s: float, z_entry: float) -> np.ndarray:
    """Mean reversion: цена ушла от своей EMA дальше z_entry стандартных отклонений -> ставим на возврат."""
    hl_rows = max(1.0, halflife_s * 1000 / ROW_MS)
    g = b.groupby("seg", sort=False)["mid"]
    ema = g.transform(lambda s: s.ewm(halflife=hl_rows, adjust=False).mean())
    dev = (b["mid"] - ema) / b["mid"] * 1e4
    win = int(hl_rows * 10)
    sd = dev.groupby(b["seg"], sort=False).transform(lambda s: s.rolling(win, min_periods=win).std())
    z = (dev / sd).to_numpy()
    side = np.zeros(len(b), dtype=np.int8)
    side[z >= z_entry] = -1   # цена выше среднего: ждём возврата вниз
    side[z <= -z_entry] = 1
    return side


def sig_micro_drift(b: pd.DataFrame, imb_entry: float, depth: int = 1) -> np.ndarray:
    """Не mean reversion, а эталон для сравнения: перекос стакана предсказывает ближайший сдвиг цены."""
    imb = b["imb1" if depth == 1 else "imb5"].to_numpy()
    side = np.zeros(len(b), dtype=np.int8)
    side[imb >= imb_entry] = 1
    side[imb <= -imb_entry] = -1
    return side


STRATEGIES: dict[str, dict[str, Any]] = {
    "mr_ema": {
        "fn": sig_mr_ema,
        "hypothesis": "mid отклоняется от EMA дальше z сигм и возвращается (mean reversion)",
        "grid": {"halflife_s": [1.0, 5.0, 20.0], "z_entry": [1.5, 2.0, 3.0]},
    },
    "micro_drift": {
        "fn": sig_micro_drift,
        "hypothesis": "перекос объёма на лучших уровнях предсказывает ближайший сдвиг mid (эталон, не MR)",
        "grid": {"imb_entry": [0.5, 0.7, 0.9], "depth": [1, 5]},
    },
}


# ---------- оценка ----------

def evaluate(b: pd.DataFrame, side: np.ndarray, horizons_ms=HORIZONS_MS) -> dict[str, Any]:
    """Markout сигнала по горизонтам. Всё в б.п. на одну сделку."""
    ts, seg, mid = b["ts_ms"].to_numpy(), b["seg"].to_numpy(), b["mid"].to_numpy()
    spread_bps = (b["spread"] / b["mid"] * 1e4).to_numpy()
    out: dict[str, Any] = {"horizons": {}}
    for h in horizons_ms:
        idx = onsets(side, ts, seg, cooldown_ms=h)
        if len(idx) == 0:
            out["horizons"][f"{h}ms"] = {"n": 0, "gross_bps": None, "t": None, "net_taker_bps": None}
            continue
        gross = side[idx] * (forward_mid(b, h)[idx] - mid[idx]) / mid[idx] * 1e4
        s = summarize(gross)
        # вход и выход по рынку: полспреда + полспреда и две taker-комиссии
        cost = float(np.nanmean(spread_bps[idx])) + 2 * TAKER_FEE_BPS
        out["horizons"][f"{h}ms"] = {
            "n": s["n"], "gross_bps": s["mean"], "t": s["t"],
            "net_taker_bps": None if s["mean"] is None else s["mean"] - cost,
        }
    return out


def grid_points(grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def time_folds(b: pd.DataFrame, n_chunks: int) -> list[pd.DataFrame]:
    """Разрезать по времени на n_chunks равных кусков. Без перемешивания."""
    edges = np.linspace(b["ts_ms"].iloc[0], b["ts_ms"].iloc[-1] + 1, n_chunks + 1)
    return [b[(b["ts_ms"] >= lo) & (b["ts_ms"] < hi)].reset_index(drop=True)
            for lo, hi in zip(edges[:-1], edges[1:])]


def markout_scorer(horizon_ms: int) -> Callable[[pd.DataFrame, np.ndarray], dict[str, Any]]:
    """Оценщик для walk_forward: средний markout (до издержек) на горизонте horizon_ms."""
    def score(chunk: pd.DataFrame, side: np.ndarray) -> dict[str, Any]:
        h = evaluate(chunk, side, (horizon_ms,))["horizons"][f"{horizon_ms}ms"]
        return {"n": h["n"], "score": h["gross_bps"], **h}
    return score


def walk_forward(
    b: pd.DataFrame, fn: Callable[..., np.ndarray], grid: dict[str, list],
    scorer: Callable[[pd.DataFrame, np.ndarray], dict[str, Any]],
    n_chunks: int = 4, min_n: int = 30,
    on_result: Callable[[str, dict, dict, pd.DataFrame], None] | None = None,
) -> list[dict[str, Any]]:
    """Кусок k: выбираем лучшие параметры. Кусок k+1: честно проверяем именно их.

    scorer(chunk, side) возвращает словарь с ключами "n" (число сделок) и "score"
    (чем больше, тем лучше; None = оценить нельзя) плюс любые метрики.
    on_result(split, params, metrics, chunk) вызывается для КАЖДОГО варианта (для журнала).
    """
    chunks = time_folds(b, n_chunks)
    results = []
    for k in range(n_chunks - 1):
        train, test = chunks[k], chunks[k + 1]
        best, best_score = None, -math.inf
        for params in grid_points(grid):
            m = scorer(train, fn(train, **params))
            if on_result:
                on_result("train", params, m, train)
            if m["n"] >= min_n and m["score"] is not None and m["score"] > best_score:
                best, best_score = params, m["score"]
        if best is None:
            results.append({"fold": k + 1, "params": None, "train": None, "test": None})
            continue
        m = scorer(test, fn(test, **best))
        if on_result:
            on_result(f"wf-{k + 1}", best, m, test)
        results.append({"fold": k + 1, "params": best, "train": best_score, "test": m})
    return results


# ---------- запуск ----------

def _fmt(v, nd=3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _day(ts_ms: float) -> str:
    return pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser(description="Markout и walk-forward для сигнала")
    ap.add_argument("processed", nargs="?", default="processed")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="mr_ema")
    ap.add_argument("--horizon-ms", type=int, default=1000, help="горизонт для выбора параметров")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--db", default="results/runs.sqlite")
    ap.add_argument("--no-log", action="store_true", help="не писать в журнал (только для отладки кода)")
    a = ap.parse_args()

    b = load(a.processed, a.inst)
    spec = STRATEGIES[a.strategy]
    hours = (b["ts_ms"].iloc[-1] - b["ts_ms"].iloc[0]) / 3.6e6
    print(f"{a.inst}: {len(b)} строк, {hours:.2f} ч, стратегия {a.strategy}")
    if hours < 24:
        print("! данных меньше суток: это проверка кода, а не вывод о стратегии")

    db = None if a.no_log else RunDB(a.db)
    if db:
        db.ensure_strategy(a.strategy, hypothesis=spec["hypothesis"])

    def log(split: str, params: dict, h: dict, chunk: pd.DataFrame) -> None:
        if db is None or chunk.empty:
            return
        n = h["n"]
        metrics = {
            "unit": "bps", "horizon_ms": a.horizon_ms, "n_trades": n,
            "mean_gross_bps": h["gross_bps"], "t_stat": h["t"],
            "mean_net_taker_bps": h["net_taker_bps"],
            "pnl_gross": None if h["gross_bps"] is None else h["gross_bps"] * n,
            "pnl_net": None if h["net_taker_bps"] is None else h["net_taker_bps"] * n,
            "fees": 2 * TAKER_FEE_BPS * n,
            "execution": "taker-taker, mid-to-mid markout",
            "hours": (chunk["ts_ms"].iloc[-1] - chunk["ts_ms"].iloc[0]) / 3.6e6,
        }
        db.log_run(a.strategy, split=split, params=params, metrics=metrics,
                   data_from=_day(chunk["ts_ms"].iloc[0]), data_to=_day(chunk["ts_ms"].iloc[-1]),
                   instruments=[a.inst])

    print(f"\nWalk-forward, горизонт {a.horizon_ms} мс, кусков {a.chunks}:")
    print("  fold  params                               train_gross  test_n  test_gross  test_t  test_net_taker")
    tests = []
    for r in walk_forward(b, spec["fn"], spec["grid"], markout_scorer(a.horizon_ms), a.chunks,
                          on_result=log):
        if r["params"] is None:
            print(f"  {r['fold']:>4}  (на train не набралось сигналов)")
            continue
        t = r["test"]
        tests.append(t)
        print(f"  {r['fold']:>4}  {str(r['params']):<36} {_fmt(r['train']):>11}  {t['n']:>6}  "
              f"{_fmt(t['gross_bps']):>10}  {_fmt(t['t'], 1):>6}  {_fmt(t['net_taker_bps']):>14}")

    print(f"\nИздержки по рынку: спред + 2 x {TAKER_FEE_BPS} б.п. Если test_gross меньше этого,"
          " taker-вариант убыточен; пассивный вариант проверяет симулятор (hftlab.sim).")
    if db:
        print(f"Всего вариантов в журнале по {a.strategy}: {db.count_runs(a.strategy)}")
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
