"""Публичные данные Gate (фьючерсы к USDT): дневные свечи и история funding. Ключи не нужны.

    python -m hftlab.gate universe                         # какие монеты проходят правило отбора
    python -m hftlab.gate fetch --inst ZEC-USDT-SWAP --inst LTC-USDT-SWAP

Зачем отдельный источник: вторая биржа нужна, чтобы проверять идеи на монетах, которых не было
в первой серии, и чтобы не зависеть от одной площадки. У Gate история funding есть с 2020 года
(у OKX только ~92 дня), поэтому плату за финансирование здесь можно честно вычесть из сделок.

Файлы лежат в той же структуре, что и для OKX, только в другой папке:
    bars_gate/<инструмент>/1D.parquet        ts_ms, open, high, low, close, vol, quote_vol
    bars_gate/<инструмент>/funding.parquet   ts_ms, rate
Имена инструментов приведены к виду OKX (ZEC-USDT-SWAP), чтобы остальной код не менялся.
Незакрытая свеча текущего дня отбрасывается: она ещё меняется.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd

BASE = "https://api.gateio.ws/api/v4/futures/usdt"
DAY_S = 86_400
START_S = 1_546_300_800          # 01.01.2019: раньше фьючерсов к USDT на Gate не было
WINDOW_DAYS = 1900               # биржа отдаёт не больше 2000 свечей за запрос
COLS = ["ts_ms", "open", "high", "low", "close", "vol", "quote_vol"]
Get = Callable[[str], Any]


def default_get(path: str, retries: int = 6) -> Any:
    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(BASE + path, headers={"User-Agent": "hftlab", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - сеть бывает капризной, повторяем
            if attempt == retries - 1:
                raise RuntimeError(f"Gate не ответил на {path}: {e!r}") from None
            time.sleep(delay)
            delay = min(delay * 2, 20)


def contract_of(inst: str) -> str:
    """ZEC-USDT-SWAP -> ZEC_USDT (имя контракта на Gate)."""
    return inst.removesuffix("-SWAP").replace("-", "_")


def inst_of(contract: str) -> str:
    return contract.replace("_", "-") + "-SWAP"


def fetch_daily(inst: str, get: Get = default_get, now_s: int | None = None, sleep_s: float = 0.15) -> pd.DataFrame:
    now_s = int(time.time()) if now_s is None else now_s
    today = now_s // DAY_S * DAY_S
    rows: list[dict] = []
    lo = START_S
    while lo < today:
        hi = min(lo + WINDOW_DAYS * DAY_S, today)
        rows.extend(get(f"/candlesticks?contract={contract_of(inst)}&interval=1d&from={lo}&to={hi}") or [])
        lo = hi + DAY_S
        time.sleep(sleep_s)
    df = pd.DataFrame(
        [(int(r["t"]) * 1000, float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]),
          float(r["v"]), float(r.get("sum") or 0)) for r in rows if int(r["t"]) < today],
        columns=COLS,
    )
    return df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)


def fetch_funding(inst: str, get: Get = default_get, sleep_s: float = 0.15, max_pages: int = 400) -> pd.DataFrame:
    """Вся история ставок: страницы идут от новых к старым, параметр `to` сдвигает окно назад."""
    rows: list[dict] = []
    to: int | None = None
    for _ in range(max_pages):
        page = get(f"/funding_rate?contract={contract_of(inst)}&limit=1000" + (f"&to={to}" if to else ""))
        if not page:
            break
        rows.extend(page)
        oldest = min(int(r["t"]) for r in page)
        if to is not None and oldest > to:
            break
        to = oldest - 1
        time.sleep(sleep_s)
    df = pd.DataFrame([(int(r["t"]) * 1000, float(r["r"])) for r in rows], columns=["ts_ms", "rate"])
    return df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)


def universe(get: Get = default_get, created_by_s: int = 1_609_459_200, min_volume: float = 1e6,
             exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Старые ликвидные контракты: созданы до даты, не снимаются с торгов, оборот за сутки не меньше порога."""
    vol = {r["contract"]: float(r.get("volume_24h_quote") or 0) for r in get("/tickers")}
    out = []
    for c in get("/contracts"):
        name = c["name"]
        if not 0 < int(c.get("create_time") or 0) <= created_by_s or c.get("in_delisting"):
            continue
        if inst_of(name) in exclude or vol.get(name, 0) < min_volume:
            continue
        out.append({"inst": inst_of(name), "created_s": int(c["create_time"]), "volume": vol[name]})
    return sorted(out, key=lambda r: -r["volume"])


def save(df: pd.DataFrame, root: str | Path, inst: str, name: str) -> Path:
    p = Path(root) / inst / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="Публичные данные Gate: свечи и funding")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--inst", action="append", required=True)
    f.add_argument("--out", default="bars_gate")
    f.add_argument("--no-funding", action="store_true")
    u = sub.add_parser("universe")
    u.add_argument("--exclude", action="append", default=[])
    a = ap.parse_args()
    if a.cmd == "universe":
        for r in universe(exclude=tuple(a.exclude)):
            print(f"{r['inst']:<18} создан {time.strftime('%Y-%m-%d', time.gmtime(r['created_s']))}  "
                  f"оборот {r['volume'] / 1e6:.1f} млн USDT")
        return 0
    for inst in a.inst:
        d = fetch_daily(inst)
        save(d, a.out, inst, "1D.parquet")
        msg = f"{inst}: {len(d)} дневных свечей"
        if len(d):
            lo, hi = (pd.Timestamp(d["ts_ms"].iloc[i], unit="ms") for i in (0, -1))
            miss = int((d["ts_ms"].diff().dropna() != DAY_S * 1000).sum())
            msg += f", {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}, дыр: {miss}"
        if not a.no_funding:
            fr = fetch_funding(inst)
            save(fr, a.out, inst, "funding.parquet")
            msg += f"; funding: {len(fr)} выплат" + (
                f" с {pd.Timestamp(fr['ts_ms'].iloc[0], unit='ms'):%Y-%m-%d}" if len(fr) else "")
        print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
