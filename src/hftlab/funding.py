"""Плата за финансирование (funding) бессрочных фьючерсов: данные и две проверки.

    python -m hftlab.funding fetch  --inst BTC-USDT-SWAP --inst ETH-USDT-SWAP
    python -m hftlab.funding fade   --inst BTC-USDT-SWAP --inst ETH-USDT-SWAP
    python -m hftlab.funding carry  --inst BTC-USDT-SWAP --inst ETH-USDT-SWAP

Что такое funding. У бессрочного фьючерса нет даты поставки, и его цену привязывают к споту
платежами между трейдерами: раз в 8 часов держатели длинных позиций платят держателям коротких
(ставка положительная) или наоборот (отрицательная). Обычная ставка около 0.01% за 8 часов.

Данные: OKX отдаёт историю только за ~3 месяца. Поэтому `fetch` ДОПИСЫВАЕТ файл
bars/<инструмент>/funding.parquet: если запускать его раз в месяц, история будет расти.

Проверка 1, `fade` (гипотеза о цене). Необычно высокая ставка значит, что толпа стоит в длинных
позициях с плечом. Гипотеза: после этого цена чаще падает. Правило задано заранее и не подбирается:
ставка выше своей недавней нормы на 2 сигмы (норма по 30 прошлым выплатам): продаём фьючерс на 24 часа;
ниже на 2 сигмы: покупаем. В результат включён сам funding, который мы платим или получаем.

Проверка 2, `carry` (не предсказание, а арифметика). Покупаем монету на споте и продаём столько же
фьючерсом. Цена нас не волнует: что потеряли на одном, заработали на другом. Доход это funding,
который платят короткой позиции. Расходы: четыре сделки на круг (спот и фьючерс, вход и выход).
Риски, которых в расчёте нет: ставка может стать отрицательной надолго; короткую позицию во фьючерсе
могут ликвидировать при резком росте, если мало залога; биржа может зависнуть или закрыться.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from hftlab.candles import BAR_MS, load, resample
from hftlab.research import summarize
from hftlab.runs import RunDB
from hftlab.slow import DAY_MS, _day, _fmt, clustered_t, spot_name

BASE = "https://www.okx.com"
GetPage = Callable[[str, int | None], list[dict]]

# Правило fade: задано до запуска (docs/PREREG-2026-11-11.md), не подбирается
FADE_WIN, FADE_Z, FADE_HOLD_H = 30, 2.0, 24
# Правило carry: войти, когда средняя ставка за 3 выплаты выше 1 б.п.; выйти, когда ниже нуля
CARRY_LOOKBACK, CARRY_ENTER_BPS, CARRY_EXIT_BPS = 3, 1.0, 0.0


# ---------- данные ----------

def default_get(inst: str, after: int | None, retries: int = 6) -> list[dict]:
    url = f"{BASE}/api/v5/public/funding-rate-history?instId={inst}&limit=100"
    if after is not None:
        url += f"&after={after}"
    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hftlab"})
            with urllib.request.urlopen(req, timeout=20) as r:
                body = json.load(r)
            if body.get("code") == "0":
                return body["data"]
            err = f"OKX code {body.get('code')}: {body.get('msg')}"
        except Exception as e:  # noqa: BLE001 - сеть бывает капризной, повторяем
            err = repr(e)
        if attempt == retries - 1:
            raise RuntimeError(f"не удалось получить funding {inst}: {err}")
        time.sleep(delay)
        delay = min(delay * 2, 20)
    return []


def fetch_all(inst: str, get: GetPage = default_get, sleep_s: float = 0.25, max_pages: int = 60) -> pd.DataFrame:
    """Вся доступная история ставок: ts_ms (момент выплаты) и rate (доля, 0.0001 = 1 б.п.)."""
    rows: list[dict] = []
    after: int | None = None
    for _ in range(max_pages):
        page = get(inst, after)
        if not page:
            break
        rows.extend(page)
        oldest = int(page[-1]["fundingTime"])
        if after is not None and oldest >= after:
            break
        after = oldest
        time.sleep(sleep_s)
    df = pd.DataFrame(
        [(int(r["fundingTime"]), float(r.get("realizedRate") or r["fundingRate"])) for r in rows],
        columns=["ts_ms", "rate"],
    )
    return df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)


def path_for(root: str | Path, inst: str) -> Path:
    return Path(root) / inst / "funding.parquet"


def update(root: str | Path, inst: str, get: GetPage = default_get, sleep_s: float = 0.25) -> pd.DataFrame:
    """Скачать доступное и дописать к уже сохранённому (старые записи не теряются)."""
    p = path_for(root, inst)
    new = fetch_all(inst, get, sleep_s)
    if p.exists():
        new = pd.concat([pd.read_parquet(p), new], ignore_index=True)
    df = new.drop_duplicates("ts_ms", keep="last").sort_values("ts_ms").reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def load_funding(root: str | Path, inst: str) -> pd.DataFrame:
    p = path_for(root, inst)
    if not p.exists():
        raise FileNotFoundError(f"нет {p}: сначала python -m hftlab.funding fetch --inst {inst}")
    return pd.read_parquet(p)


# ---------- проверка 1: ставка как сигнал ----------

def funding_z(rate: np.ndarray, win: int = FADE_WIN) -> np.ndarray:
    """Насколько текущая ставка выше нормы по `win` ПРЕДЫДУЩИМ выплатам (текущая в норму не входит)."""
    s = pd.Series(rate)
    prev = s.shift(1)
    mean = prev.rolling(win, min_periods=win).mean()
    std = prev.rolling(win, min_periods=win).std()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.array((s - mean) / std, dtype=float)   # копия: у pandas массив бывает только для чтения
    z[~np.isfinite(z)] = np.nan
    return z


def fade_trades(bars_1h: pd.DataFrame, f: pd.DataFrame, win: int = FADE_WIN, z_entry: float = FADE_Z,
                hold_h: int = FADE_HOLD_H) -> dict[str, np.ndarray]:
    """Сделки правила fade на часовых свечах. Результат в б.п.: движение цены плюс funding за время сделки."""
    hour = BAR_MS["1H"]
    ts, op = bars_1h["ts_ms"].to_numpy(), bars_1h["open"].to_numpy()
    idx = {int(t): i for i, t in enumerate(ts)}
    ft, fr = f["ts_ms"].to_numpy(), f["rate"].to_numpy()
    z = funding_z(fr, win)
    t_sig, sides, price, fund = [], [], [], []
    free_ts = -1
    for k in range(len(ft)):
        if not np.isfinite(z[k]) or abs(z[k]) < z_entry:
            continue
        s = -1 if z[k] > 0 else 1                      # ставка высокая: толпа в длинных, продаём
        t_in = int(ft[k]) + hour                        # узнали ставку в момент выплаты, входим часом позже
        t_out = t_in + hold_h * hour
        if t_in < free_ts or t_in not in idx or t_out not in idx:
            continue                                    # позиция занята, дыра в свечах или конец данных
        e, x = idx[t_in], idx[t_out]
        if x - e != hold_h:
            continue                                    # внутри сделки не хватает свечей
        paid = fr[(ft > t_in) & (ft <= t_out)].sum()    # выплаты, которые застали в позиции
        t_sig.append(int(ft[k]))
        sides.append(s)
        price.append(s * (op[x] - op[e]) / op[e] * 1e4)
        fund.append(-s * paid * 1e4)                    # длинная позиция платит положительную ставку
        free_ts = t_out
    price_a, fund_a = np.array(price, dtype=float), np.array(fund, dtype=float)
    return {"t_signal": np.array(t_sig, dtype=np.int64), "side": np.array(sides, dtype=np.int8),
            "price_bps": price_a, "funding_bps": fund_a, "gross_bps": price_a + fund_a}


# ---------- проверка 2: спот против фьючерса ----------

def basis_at(perp_5m: pd.DataFrame, spot_5m: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Базис (фьючерс дороже спота на столько б.п.) по последней свече, закрытой к каждому моменту."""
    m = perp_5m[["ts_ms", "close"]].merge(spot_5m[["ts_ms", "close"]], on="ts_ms", suffixes=("_p", "_s"))
    close_ts = m["ts_ms"].to_numpy() + BAR_MS["5m"]
    b = ((m["close_p"] / m["close_s"] - 1) * 1e4).to_numpy()
    i = np.searchsorted(close_ts, times, side="right") - 1
    out = np.where(i >= 0, b[np.clip(i, 0, len(b) - 1)], np.nan)
    stale = (times - close_ts[np.clip(i, 0, len(b) - 1)]) > 6 * BAR_MS["5m"]   # старше получаса: не верим
    out[stale | (i < 0)] = np.nan
    return out


def carry_backtest(f: pd.DataFrame, basis_bps: np.ndarray, cost_bps: float, rule: str = "threshold",
                   lookback: int = CARRY_LOOKBACK, enter_bps: float = CARRY_ENTER_BPS,
                   exit_bps: float = CARRY_EXIT_BPS) -> dict[str, Any]:
    """Купили спот, продали фьючерс. Результат в б.п. от суммы одной ноги.

    rule="always": войти после первой выплаты и держать до конца.
    rule="threshold": войти, когда средняя ставка за `lookback` выплат выше enter_bps; выйти, когда ниже exit_bps.
    Решение принимается после выплаты k, доход начинается с выплаты k+1.
    Базис на входе и выходе учитывается: если фьючерс подешевел относительно спота, короткой позиции это в плюс.
    """
    r = f["rate"].to_numpy() * 1e4
    n = len(r)
    avg = pd.Series(r).rolling(lookback, min_periods=lookback).mean().to_numpy()
    in_pos, b_in = False, 0.0
    funding = basis_pnl = 0.0
    trips = periods_in = 0
    for k in range(n):
        if in_pos:
            funding += r[k]
            periods_in += 1
        last = k == n - 1
        want_in = (not last) and (True if rule == "always" else (
            np.isfinite(avg[k]) and (avg[k] > enter_bps if not in_pos else avg[k] >= exit_bps)))
        if in_pos and not want_in:
            if np.isfinite(basis_bps[k]):
                basis_pnl += b_in - basis_bps[k]
            in_pos = False
            trips += 1
        elif not in_pos and want_in and np.isfinite(basis_bps[k]):
            in_pos, b_in = True, float(basis_bps[k])
    days = (f["ts_ms"].iloc[-1] - f["ts_ms"].iloc[0]) / DAY_MS if n > 1 else 0.0
    gross = funding + basis_pnl
    net = gross - cost_bps * trips
    return {
        "periods": n, "days": days, "periods_in": periods_in, "round_trips": trips,
        "funding_bps": funding, "basis_bps": basis_pnl, "gross_bps": gross, "cost_bps": cost_bps * trips,
        "net_bps": net,
        # годовых от суммы одной ноги; на весь задействованный капитал (спот + залог) примерно вдвое меньше
        "net_apr_pct": net / 100 * 365 / days if days > 0 else None,
        "mean_rate_bps": float(r.mean()) if n else None,
        "share_positive": float((r > 0).mean()) if n else None,
    }


# ---------- запуск ----------

def cmd_fetch(a: argparse.Namespace) -> int:
    for inst in a.inst:
        df = update(a.bars, inst)
        days = (df["ts_ms"].iloc[-1] - df["ts_ms"].iloc[0]) / DAY_MS
        print(f"{inst}: {len(df)} выплат, {days:.0f} дней, {_day(df['ts_ms'].iloc[0])} .. {_day(df['ts_ms'].iloc[-1])}")
    return 0


def cmd_fade(a: argparse.Namespace) -> int:
    name = "funding_fade"
    db = None if a.no_log else RunDB(a.db)
    if db:
        db.ensure_strategy(name, hypothesis="необычно высокая ставка funding (толпа в длинных с плечом) "
                                            "предшествует падению цены за 24 часа, и наоборот")
    print(f"{name}: окно {FADE_WIN} выплат, порог z = {FADE_Z}, удержание {FADE_HOLD_H} ч, "
          f"стоимость круга {a.cost_bps:g} б.п. Параметры не подбирались.")
    print("  инструмент              сделок   цена   funding   всего до издержек     t")
    g_all, t_all = [], []
    for inst in a.inst:
        f = load_funding(a.bars, inst)
        bars = resample(load(a.bars, inst, "5m"), "5m", "1H")
        tr = fade_trades(bars, f)
        g = tr["gross_bps"]
        s = summarize(g)
        g_all.append(g)
        t_all.append(tr["t_signal"])
        pm = float(tr["price_bps"].mean()) if len(g) else None
        fm = float(tr["funding_bps"].mean()) if len(g) else None
        print(f"  {inst:<22} {s['n']:>6} {_fmt(pm):>7} {_fmt(fm):>8} {_fmt(s['mean']):>14} {_fmt(s['t'], 1):>10}")
        if db and len(f) > 1:
            db.log_run(name, split="test",
                       params={"win": FADE_WIN, "z_entry": FADE_Z, "hold_h": FADE_HOLD_H},
                       metrics={"unit": "bps", "kind": "funding_fade", "n_trades": s["n"],
                                "mean_gross_bps": s["mean"], "t": s["t"], "mean_price_bps": pm,
                                "mean_funding_bps": fm, "cost_bps": a.cost_bps,
                                "pnl_gross": float(g.sum()), "pnl_net": float((g - a.cost_bps).sum()),
                                "fees": a.cost_bps * s["n"]},
                       data_from=_day(f["ts_ms"].iloc[0]), data_to=_day(f["ts_ms"].iloc[-1]),
                       instruments=[inst], note="правило задано заранее, без подбора параметров")
    g = np.concatenate(g_all) if g_all else np.array([])
    t = np.concatenate(t_all) if t_all else np.array([], dtype=np.int64)
    s = summarize(g)
    tc = clustered_t(g, t // DAY_MS) if len(g) else None
    print(f"  {'ВМЕСТЕ':<22} {s['n']:>6} {'':>7} {'':>8} {_fmt(s['mean']):>14} {_fmt(tc, 1):>10}"
          f"   (без учёта связи между монетами: {_fmt(s['t'], 1)})")
    if s["mean"] is not None:
        print(f"  После стоимости круга {a.cost_bps:g} б.п.: {s['mean'] - a.cost_bps:+.1f} б.п. на сделку")
    if db:
        db.close()
    return 0


def cmd_carry(a: argparse.Namespace) -> int:
    db = None if a.no_log else RunDB(a.db)
    hyp = {"carry_always": "спот куплен, фьючерс продан, держим всегда: доход от funding покрывает издержки",
           "carry_threshold": "то же, но входим только при высокой недавней ставке (выше 1 б.п. за 8 ч)"}
    if db:
        for k, v in hyp.items():
            db.ensure_strategy(k, hypothesis=v)
    print(f"carry: купили спот, продали фьючерс. Стоимость круга (4 сделки): {a.cost_bps:g} б.п. "
          "Проценты годовых считаются от суммы одной ноги.")
    print("  инструмент            дней  ставка/8ч  плюс%   | держать всегда: б.п.  год% | по порогу: кругов "
          "в поз.%  б.п.  год%")
    for inst in a.inst:
        f = load_funding(a.bars, inst)
        basis = basis_at(load(a.bars, inst, "5m"), load(a.bars, spot_name(inst), "5m"), f["ts_ms"].to_numpy())
        res = {rule: carry_backtest(f, basis, a.cost_bps, rule) for rule in ("always", "threshold")}
        al, th = res["always"], res["threshold"]
        share_in = th["periods_in"] / th["periods"] * 100 if th["periods"] else 0.0
        print(f"  {inst:<20} {al['days']:>5.0f} {_fmt(al['mean_rate_bps']):>9} {al['share_positive'] * 100:>6.0f}"
              f"   | {al['net_bps']:>19.0f} {_fmt(al['net_apr_pct'], 1):>5} | {th['round_trips']:>17}"
              f" {share_in:>7.0f} {th['net_bps']:>5.0f} {_fmt(th['net_apr_pct'], 1):>5}")
        if db:
            for rule, m in res.items():
                db.log_run(f"carry_{rule}", split="test",
                           params={"rule": rule, "lookback": CARRY_LOOKBACK, "enter_bps": CARRY_ENTER_BPS,
                                   "exit_bps": CARRY_EXIT_BPS, "cost_bps": a.cost_bps},
                           metrics={**m, "unit": "bps", "kind": "carry", "n_trades": m["round_trips"],
                                    "pnl_gross": m["gross_bps"], "pnl_net": m["net_bps"], "fees": m["cost_bps"]},
                           data_from=_day(f["ts_ms"].iloc[0]), data_to=_day(f["ts_ms"].iloc[-1]),
                           instruments=[inst, spot_name(inst)],
                           note="арифметика одного периода, не прогноз; риски ликвидации и биржи не учтены")
    print("  Это фактический результат одного периода, а не обещание: ставка меняется вместе с настроением рынка.")
    if db:
        db.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Funding: данные, сигнал fade и арифметика carry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, cost in (("fetch", cmd_fetch, None), ("fade", cmd_fade, 20.0), ("carry", cmd_carry, 30.0)):
        p = sub.add_parser(name)
        p.add_argument("--inst", action="append", required=True)
        p.add_argument("--bars", default="bars")
        if cost is not None:
            p.add_argument("--cost-bps", type=float, default=cost, help="стоимость круга в б.п.")
            p.add_argument("--db", default="results/runs.sqlite")
            p.add_argument("--no-log", action="store_true")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
