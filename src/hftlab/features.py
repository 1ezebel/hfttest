"""Этап 2, шаг 1: из сырой записи делаем таблицы для исследования.

    python -m hftlab.features data --out processed

На выходе для каждого инструмента и дня (UTC) два parquet-файла:
    processed/<inst>/<дата>.book.parquet     одна строка на каждое сообщение стакана (~100 мс)
    processed/<inst>/<дата>.trades.parquet   одна строка на сделку

Сырые данные не меняются: всё производное лежит в processed/ и может быть
пересчитано заново в любой момент.

Колонки book:
    ts_ms       биржевое время сообщения (по нему считаем горизонты markout)
    recv_ns     наше время получения
    seg         номер непрерывного куска. Меняется после реконнекта, пропуска (gap) или
                тишины дольше MAX_QUIET_MS. Скользящие окна и markout НЕЛЬЗЯ считать
                через границу seg: там дыра в данных.
    bid, ask, bid_sz, ask_sz        лучший бид/аск и объёмы на них
    mid, spread, microprice
    imb1, imb5  дисбаланс объёма на 1 и 5 уровнях: от -1 (давит продавец) до +1
    ofi         order flow imbalance за шаг: насколько прибавилось спроса на лучшем биде
                минус насколько прибавилось предложения на лучшем аске (Cont, Kukanov, Stoikov)
    buy_vol, sell_vol, n_trades     сделки с прошлой строки (buy = агрессор покупал)

Про look-ahead: строка с временем ts_ms содержит только то, что было известно
к этому моменту. Сделки привязываются к первой строке стакана с ts_ms >= времени сделки.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hftlab.check_recording import collect_by_instrument
from hftlab.okx_book import OrderBook, Status
from hftlab.storage import iter_lines

MAX_QUIET_MS = 1500  # стакан шлёт обновления каждые ~100 мс; тишина дольше = дыра

BOOK_COLS = ["ts_ms", "recv_ns", "brk", "bid", "bid_sz", "ask", "ask_sz", "b5", "a5"]
TRADE_COLS = ["ts_ms", "recv_ns", "px", "sz", "side", "trade_id"]


def parse(lines: Iterable[tuple[int, str]], inst_id: str = "?") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Проход по записи: сырые строки -> (стакан по сообщениям, сделки). Без признаков."""
    book = OrderBook(inst_id)
    brk = True  # следующая строка начинает новый непрерывный кусок
    book_rows: list[tuple] = []
    trade_rows: list[tuple] = []
    for recv_ns, payload in lines:
        if payload.startswith("#"):
            if payload.startswith("#CONNECT"):
                book = OrderBook(inst_id)
                brk = True
            continue  # GAP ловим сами по статусу стакана; AUDIT/RESUB здесь не нужны
        msg = json.loads(payload)
        channel = msg.get("arg", {}).get("channel")
        if channel == "trades":
            for t in msg["data"]:
                trade_rows.append((int(t["ts"]), recv_ns, float(t["px"]), float(t["sz"]),
                                   1 if t["side"] == "buy" else -1, int(t["tradeId"])))
        elif channel == "books":
            for d in msg["data"]:
                status = book.apply(msg["action"], d)
                if status is not Status.OK:
                    brk = True
                    continue
                bids, asks = book.top(5)
                if not bids or not asks or bids[0][0] >= asks[0][0]:
                    brk = True  # пустая сторона или пересечённый стакан: строку не пишем
                    continue
                book_rows.append((
                    int(d["ts"]), recv_ns, brk,
                    bids[0][0], bids[0][1], asks[0][0], asks[0][1],
                    sum(s for _, s in bids), sum(s for _, s in asks),
                ))
                brk = False
    b = pd.DataFrame(book_rows, columns=BOOK_COLS)
    t = pd.DataFrame(trade_rows, columns=TRADE_COLS)
    t = t.drop_duplicates("trade_id").sort_values("ts_ms", kind="stable").reset_index(drop=True)
    return b, t


def add_features(b: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Добавить признаки к таблице стакана. Все признаки смотрят только назад."""
    b = b.copy()
    if b.empty:
        return b
    quiet = b["ts_ms"].diff() > MAX_QUIET_MS
    back = b["ts_ms"].diff() < 0  # время пошло назад: тоже не доверяем
    b["seg"] = (b["brk"] | quiet | back).cumsum().astype("int32")
    b = b.drop(columns="brk")

    b["mid"] = (b["bid"] + b["ask"]) / 2
    b["spread"] = b["ask"] - b["bid"]
    b["microprice"] = (b["bid"] * b["ask_sz"] + b["ask"] * b["bid_sz"]) / (b["bid_sz"] + b["ask_sz"])
    b["imb1"] = (b["bid_sz"] - b["ask_sz"]) / (b["bid_sz"] + b["ask_sz"])
    b["imb5"] = (b["b5"] - b["a5"]) / (b["b5"] + b["a5"])
    b = b.drop(columns=["b5", "a5"])

    # OFI: сравниваем лучший уровень с предыдущей строкой того же куска
    g = b.groupby("seg", sort=False)
    pb, pbs, pa, pas = g["bid"].shift(), g["bid_sz"].shift(), g["ask"].shift(), g["ask_sz"].shift()
    e_bid = (b["bid"] >= pb) * b["bid_sz"] - (b["bid"] <= pb) * pbs
    e_ask = (b["ask"] <= pa) * b["ask_sz"] - (b["ask"] >= pa) * pas
    b["ofi"] = (e_bid - e_ask).fillna(0.0)  # первая строка куска: сравнивать не с чем

    # Сделки: каждая попадает в первую строку стакана с ts_ms >= времени сделки
    n = len(b)
    buy = np.zeros(n)
    sell = np.zeros(n)
    cnt = np.zeros(n, dtype=np.int64)
    if len(t):
        idx = np.searchsorted(b["ts_ms"].to_numpy(), t["ts_ms"].to_numpy(), side="left")
        ok = idx < n
        idx, sz, side = idx[ok], t["sz"].to_numpy()[ok], t["side"].to_numpy()[ok]
        np.add.at(buy, idx[side > 0], sz[side > 0])
        np.add.at(sell, idx[side < 0], sz[side < 0])
        np.add.at(cnt, idx, 1)
    b["buy_vol"], b["sell_vol"], b["n_trades"] = buy, sell, cnt
    return b.reset_index(drop=True)


def build(files: list[Path], inst_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    def chain():
        for f in files:
            yield from iter_lines(f)
    b, t = parse(chain(), inst_id)
    return add_features(b, t), t


def _day(ts_ms: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")


def write_by_day(b: pd.DataFrame, t: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for df, kind in ((b, "book"), (t, "trades")):
        if df.empty:
            continue
        for day, part in df.groupby(_day(df["ts_ms"]), sort=True):
            p = out_dir / f"{day}.{kind}.parquet"
            part.to_parquet(p, index=False)
            written.append(p)
    return written


def load(processed: str | Path, inst: str, kind: str = "book",
         date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    """Загрузить обработанные дни одним DataFrame. Даты включительно, формат YYYY-MM-DD."""
    files = sorted((Path(processed) / inst).glob(f"*.{kind}.parquet"))
    picked = [f for f in files
              if (date_from is None or f.name[:10] >= date_from)
              and (date_to is None or f.name[:10] <= date_to)]
    if not picked:
        raise FileNotFoundError(f"нет файлов {kind} для {inst} в {processed}")
    df = pd.concat([pd.read_parquet(f) for f in picked], ignore_index=True)
    if kind == "book" and len(picked) > 1:
        # номера seg в разных днях независимы: делаем сквозную нумерацию
        day_change = _day(df["ts_ms"]) != _day(df["ts_ms"]).shift()
        df["seg"] = ((df["seg"] != df["seg"].shift()) | day_change).cumsum().astype("int32")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Сырая запись -> parquet с признаками")
    ap.add_argument("root", nargs="?", default="data")
    ap.add_argument("--out", default="processed")
    ap.add_argument("--since", help="обрабатывать дни начиная с YYYY-MM-DD (стакан восстановится "
                                    "на первом snapshot, то есть до 6 часов в начале могут пропасть)")
    a = ap.parse_args()
    groups = collect_by_instrument(Path(a.root))
    if not groups:
        print("файлов записи не найдено")
        return 1
    for inst_dir, files in groups.items():
        if a.since:
            files = [f for f in files if f.parent.name >= a.since]
        if not files:
            continue
        b, t = build(files, inst_dir.name)
        written = write_by_day(b, t, Path(a.out) / inst_dir.name)
        hours = (b["ts_ms"].iloc[-1] - b["ts_ms"].iloc[0]) / 3.6e6 if len(b) else 0.0
        segs = b["seg"].nunique() if len(b) else 0
        print(f"{inst_dir.name}: {len(b)} строк стакана, {len(t)} сделок, {hours:.2f} ч, "
              f"непрерывных кусков: {segs}, файлов: {len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
