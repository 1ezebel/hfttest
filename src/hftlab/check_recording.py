"""Проверка записи: прогоняет файлы через стакан и печатает отчёт.

    python -m hftlab.check_recording data
    python -m hftlab.check_recording data/okx/BTC-USDT-SWAP/2026-11-10/14.jsonl.gz

Что смотрим:
  * сколько сообщений по каналам, сколько переподключений;
  * gaps_detected (нашли при повторном проходе) должно совпадать с gap_events
    (записаны рекордером на лету), иначе в логике есть расхождение;
  * crossed: стакан пересёкся (bid >= ask) = признак порчи данных, должно быть 0;
  * latency recv-ts: наше время получения минус биржевое. Отрицательные значения
    значат, что часы на машине убегают относительно биржи (clock drift).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from hftlab.okx_book import OrderBook, Status
from hftlab.storage import iter_lines


@dataclass
class Report:
    channels: Counter = field(default_factory=Counter)
    snapshots: int = 0
    connects: int = 0
    gap_events: int = 0  # записал рекордер
    gaps_detected: int = 0  # нашли при повторном проходе
    crossed: int = 0
    first_ns: int | None = None
    last_ns: int | None = None
    latencies_ms: list[float] = field(default_factory=list)


def replay(lines: Iterable[tuple[int, str]], inst_id: str = "?") -> Report:
    rep = Report()
    book = OrderBook(inst_id)
    for recv_ns, payload in lines:
        if rep.first_ns is None:
            rep.first_ns = recv_ns
        rep.last_ns = recv_ns

        if payload.startswith("#"):
            kind = payload[1:].split(" ", 1)[0]
            if kind == "CONNECT":
                rep.connects += 1
                book = OrderBook(inst_id)
            elif kind == "GAP":
                rep.gap_events += 1
            continue

        msg = json.loads(payload)
        channel = msg.get("arg", {}).get("channel", "?")
        rep.channels[channel] += 1
        if channel != "books":
            continue

        for d in msg["data"]:
            rep.latencies_ms.append(recv_ns / 1e6 - float(d["ts"]))
            if msg["action"] == "snapshot":
                rep.snapshots += 1
            status = book.apply(msg["action"], d)
            if status in (Status.GAP, Status.RESET):
                rep.gaps_detected += 1
            elif status is Status.OK and book.is_crossed():
                rep.crossed += 1
    return rep


def _chain(paths: list[Path]) -> Iterator[tuple[int, str]]:
    for p in paths:
        yield from iter_lines(p)


def collect_by_instrument(target: Path) -> dict[Path, list[Path]]:
    files = [target] if target.is_file() else sorted(target.rglob("*.jsonl.gz"))
    groups: dict[Path, list[Path]] = {}
    for f in files:
        groups.setdefault(f.parents[1], []).append(f)  # .../<inst>/<date>/<hour>.jsonl.gz
    return {k: sorted(v) for k, v in groups.items()}


def print_report(name: str, rep: Report) -> None:
    dur = ((rep.last_ns - rep.first_ns) / 1e9) if rep.first_ns else 0.0
    print(f"== {name}")
    print(f"   duration:        {dur / 3600:.2f} h")
    print(f"   messages:        {dict(rep.channels)}")
    print(f"   snapshots:       {rep.snapshots}   connects: {rep.connects}")
    print(f"   gap_events:      {rep.gap_events}   gaps_detected: {rep.gaps_detected}")
    print(f"   crossed books:   {rep.crossed}")
    lat = rep.latencies_ms
    if len(lat) >= 2:
        q = statistics.quantiles(lat, n=100)
        print(
            f"   latency recv-ts: min {min(lat):.1f}  p50 {q[49]:.1f}  p99 {q[98]:.1f}  max {max(lat):.1f} ms"
        )
        if min(lat) < 0:
            print("   ! отрицательная задержка: часы машины расходятся с биржей (проверь NTP)")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    groups = collect_by_instrument(Path(argv[1]))
    if not groups:
        print("файлов записи не найдено")
        return 1
    for inst_dir, files in groups.items():
        print_report(inst_dir.name, replay(_chain(files), inst_dir.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
