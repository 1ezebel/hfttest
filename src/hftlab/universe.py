"""Где вообще есть шанс: фьючерсы OKX, у которых спред шире комиссий.

    python -m hftlab.universe                 # таблица на экран
    python -m hftlab.universe --min-volume 5  # только с оборотом от 5 млн USDT в сутки

Зачем: круг "купил пассивно, продал пассивно" стоит 2 x maker-комиссия (при базовом тарифе
4 б.п.). Заработать на нём можно максимум спред. У BTC спред около 0.01 б.п.: забирать нечего.
У инструментов с крупным шагом цены относительно самой цены спред не может быть уже одного
тика, и если тик шире комиссий, у пассивной стратегии появляется арифметический шанс.

Колонки:
  tick_bps     один шаг цены в б.п. (минимально возможный спред)
  spread_bps   спред прямо сейчас (один снимок, не среднее)
  vol_musd     оборот за 24 ч, млн USDT
  edge_bps     spread_bps минус 2 x maker: что осталось бы от идеального круга

Это только арифметика входа в игру, а не стратегия: широкий тик означает длинную очередь
на лучшей цене и сильный adverse selection. Проверять всё равно рекордером и симулятором.
Данные публичные, ключи не нужны.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from hftlab.research import MAKER_FEE_BPS

BASE = "https://www.okx.com"


def fetch(path: str) -> list[dict]:
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "hftlab"})
    with urllib.request.urlopen(req, timeout=20) as r:
        body = json.load(r)
    if body.get("code") != "0":
        raise RuntimeError(f"OKX: {body.get('msg')}")
    return body["data"]


def build_table(instruments: list[dict], tickers: list[dict], maker_fee_bps: float = MAKER_FEE_BPS) -> list[dict]:
    spec = {i["instId"]: i for i in instruments if i.get("state") == "live" and i.get("settleCcy") == "USDT"}
    rows = []
    for t in tickers:
        i = spec.get(t["instId"])
        try:
            bid, ask, last = float(t["bidPx"]), float(t["askPx"]), float(t["last"])
            tick, ct_val = float(i["tickSz"]), float(i["ctVal"])
            vol_contracts = float(t["vol24h"])
        except (TypeError, ValueError, KeyError):
            continue  # нет котировок или инструмент не подходит
        if bid <= 0 or ask <= bid or last <= 0:
            continue
        mid = (bid + ask) / 2
        spread_bps = (ask - bid) / mid * 1e4
        rows.append({
            "inst": t["instId"], "price": last,
            "tick_bps": tick / mid * 1e4, "spread_bps": spread_bps,
            "spread_ticks": round((ask - bid) / tick),
            "vol_musd": vol_contracts * ct_val * last / 1e6,
            "edge_bps": spread_bps - 2 * maker_fee_bps,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Фьючерсы OKX: спред против комиссий")
    ap.add_argument("--min-volume", type=float, default=20.0, help="минимальный оборот за 24 ч, млн USDT")
    ap.add_argument("--min-tick-bps", type=float, default=2 * MAKER_FEE_BPS,
                    help="минимальный тик в б.п. (по умолчанию 2 x maker)")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    rows = build_table(fetch("/api/v5/public/instruments?instType=SWAP"),
                       fetch("/api/v5/market/tickers?instType=SWAP"))
    liquid = [r for r in rows if r["vol_musd"] >= a.min_volume]
    wide = sorted((r for r in liquid if r["tick_bps"] >= a.min_tick_bps), key=lambda r: -r["vol_musd"])
    print(f"USDT-фьючерсов: {len(rows)}, с оборотом от {a.min_volume:g} млн: {len(liquid)}, "
          f"из них с тиком от {a.min_tick_bps:g} б.п.: {len(wide)}")
    print(f"{'инструмент':<22}{'цена':>12}{'tick_bps':>10}{'spread_bps':>12}{'тиков':>7}{'vol_musd':>10}{'edge_bps':>10}")
    for r in wide[:a.top]:
        print(f"{r['inst']:<22}{r['price']:>12.6g}{r['tick_bps']:>10.2f}{r['spread_bps']:>12.2f}"
              f"{r['spread_ticks']:>7}{r['vol_musd']:>10.1f}{r['edge_bps']:>10.2f}")
    ref = sorted(liquid, key=lambda r: -r["vol_musd"])[:3]
    print("\nДля сравнения, самые торгуемые:")
    for r in ref:
        print(f"{r['inst']:<22}{r['price']:>12.6g}{r['tick_bps']:>10.2f}{r['spread_bps']:>12.2f}"
              f"{r['spread_ticks']:>7}{r['vol_musd']:>10.1f}{r['edge_bps']:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
