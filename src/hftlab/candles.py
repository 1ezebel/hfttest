"""Исторические свечи OKX (публичный REST, ключи не нужны) для медленных стратегий.

    python -m hftlab.candles --inst BTC-USDT-SWAP --inst ETH-USDT-SWAP --days 180

Зачем: на горизонтах в минуты и часы не нужен поток стакана раз в 100 мс. Хватает свечей,
а их история доступна сразу, не нужно неделю ждать записи. Это публичные рыночные данные
для исследования: торговли и обхода чьих-либо ограничений здесь нет.

Файлы: bars/<инструмент>/<интервал>.parquet, колонки:
    ts_ms       начало свечи (мс, UTC)
    open, high, low, close
    vol         объём в контрактах
    quote_vol   объём в USDT

Берутся только ЗАКРЫТЫЕ свечи (confirm == 1): последняя, ещё формирующаяся свеча
меняется и попала бы в данные из будущего.

Повторный запуск докачивает только новое. Дыры внутри истории не чинятся: их видно
по отчёту, а бэктест всё равно пропускает сделки, которые их пересекают.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

BASE = "https://www.okx.com"
BAR_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
          "1H": 3_600_000, "4H": 14_400_000}
COLS = ["ts_ms", "open", "high", "low", "close", "vol", "quote_vol"]

GetPage = Callable[[str, str, int | None], list[list[str]]]


def default_get(inst: str, bar: str, after: int | None, retries: int = 6) -> list[list[str]]:
    """Одна страница (до 100 свечей, от новых к старым). `after`: вернуть свечи старше этого ts."""
    url = f"{BASE}/api/v5/market/history-candles?instId={inst}&bar={bar}&limit=100"
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
            raise RuntimeError(f"не удалось получить свечи {inst} {bar}: {err}")
        time.sleep(delay)
        delay = min(delay * 2, 20)
    return []


def parse_rows(rows: list[list[str]]) -> pd.DataFrame:
    """Сырые строки OKX -> таблица. Незакрытые свечи отбрасываются."""
    closed = [r for r in rows if len(r) >= 9 and r[8] == "1"]
    df = pd.DataFrame(
        [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7]))
         for r in closed],
        columns=COLS,
    )
    return df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)


def fetch_range(inst: str, bar: str, start_ms: int, get: GetPage = default_get,
                sleep_s: float = 0.11) -> pd.DataFrame:
    """Свечи от start_ms до сейчас. Страницы идут от новых к старым, пока не дойдём до start_ms."""
    rows: list[list[str]] = []
    after: int | None = None
    while True:
        page = get(inst, bar, after)
        if not page:
            break
        rows.extend(page)
        oldest = int(page[-1][0])
        if oldest <= start_ms or (after is not None and oldest >= after):
            break  # дошли до нужной даты или биржа больше ничего не отдаёт
        after = oldest
        time.sleep(sleep_s)  # лимит OKX: 20 запросов за 2 секунды
    df = parse_rows(rows)
    return df[df["ts_ms"] >= start_ms].reset_index(drop=True)


def path_for(root: str | Path, inst: str, bar: str) -> Path:
    return Path(root) / inst / f"{bar}.parquet"


def load(root: str | Path, inst: str, bar: str = "5m") -> pd.DataFrame:
    p = path_for(root, inst, bar)
    if not p.exists():
        raise FileNotFoundError(f"нет {p}: сначала python -m hftlab.candles --inst {inst}")
    return pd.read_parquet(p)


def update(root: str | Path, inst: str, bar: str, days: float, get: GetPage = default_get,
           now_ms: int | None = None, sleep_s: float = 0.11) -> pd.DataFrame:
    """Скачать историю за `days` дней или докачать новое. Возвращает всю таблицу."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    start = now_ms - int(days * 86_400_000)
    p = path_for(root, inst, bar)
    old = pd.read_parquet(p) if p.exists() else None
    if old is not None and len(old):
        start = max(start, int(old["ts_ms"].iloc[-1]) + 1)
    new = fetch_range(inst, bar, start, get, sleep_s)
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def resample(df: pd.DataFrame, base_bar: str, bar: str) -> pd.DataFrame:
    """Укрупнить свечи (5m -> 15m, 1H). Неполные группы отбрасываются, чтобы не было кривых свечей."""
    ratio, rem = divmod(BAR_MS[bar], BAR_MS[base_bar])
    if rem or ratio < 1:
        raise ValueError(f"{bar} не делится на {base_bar}")
    if ratio == 1:
        return df.copy()
    g = (df["ts_ms"] // BAR_MS[bar]).rename("g")
    agg = df.groupby(g).agg(
        ts_ms=("ts_ms", "first"), open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), vol=("vol", "sum"), quote_vol=("quote_vol", "sum"), n=("ts_ms", "size"),
    )
    # группа полная, если в ней ровно ratio свечей и первая стоит на границе интервала
    full = (agg["n"] == ratio) & (agg["ts_ms"] % BAR_MS[bar] == 0)
    return agg[full].drop(columns="n").reset_index(drop=True)


def gaps(df: pd.DataFrame, bar: str) -> list[tuple[int, int]]:
    """Дыры в истории: (ts после которого пропуск, сколько свечей не хватает)."""
    d = np.diff(df["ts_ms"].to_numpy())
    idx = np.flatnonzero(d != BAR_MS[bar])
    return [(int(df["ts_ms"].iloc[i]), int(d[i] // BAR_MS[bar] - 1)) for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="Исторические свечи OKX -> parquet")
    ap.add_argument("--inst", action="append", help="instId, можно несколько раз")
    ap.add_argument("--bar", default="5m", choices=sorted(BAR_MS))
    ap.add_argument("--days", type=float, default=180)
    ap.add_argument("--out", default="bars")
    a = ap.parse_args()
    for inst in a.inst or ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]:
        t0 = time.time()
        df = update(a.out, inst, a.bar, a.days)
        g = gaps(df, a.bar)
        lo, hi = (pd.Timestamp(df["ts_ms"].iloc[i], unit="ms", tz="UTC") for i in (0, -1))
        print(f"{inst} {a.bar}: {len(df)} свечей, {lo:%Y-%m-%d} .. {hi:%Y-%m-%d %H:%M}, "
              f"дыр: {len(g)} (не хватает {sum(n for _, n in g)}), {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
