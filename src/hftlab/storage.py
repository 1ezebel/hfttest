"""Хранение сырых сообщений: по одному gzip-файлу на инструмент и час (UTC).

Формат строки:   <recv_ns>\t<сырой текст сообщения как пришёл от биржи>\n
События рекордера: <recv_ns>\t#CONNECT | #GAP gap | #GAP reset | ...

Почему сырой текст, а не разобранный JSON:
  * данные неизменяемы: если найдём баг в парсере, переразберём заново;
  * дешевле по CPU: не нужно сериализовать обратно;
  * recv_ns (наше время получения) хранится отдельно от биржевого `ts` внутри
    сообщения, это даёт Exchange Time vs Receive Time для анализа задержек.

gzip открывается в режиме добавления ("ab"): каждый запуск/час это новый
gzip-member, читаются они как один поток. flush() делает точку синхронизации,
так что при аварийном завершении теряется только хвост после последнего flush.
"""
from __future__ import annotations

import gzip
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def hour_path(root: str | Path, exchange: str, key: str, ts_ns: int) -> Path:
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
    return Path(root) / exchange / key / dt.strftime("%Y-%m-%d") / f"{dt:%H}.jsonl.gz"


def format_line(recv_ns: int, text: str) -> bytes:
    return f"{recv_ns}\t{text}\n".encode("utf-8")


def format_event(recv_ns: int, kind: str, detail: str = "") -> bytes:
    text = f"#{kind}" + (f" {detail}" if detail else "")
    return format_line(recv_ns, text)


def parse_line(line: str) -> tuple[int, str] | None:
    recv, sep, payload = line.rstrip("\n").partition("\t")
    if not sep or not payload:
        return None
    try:
        return int(recv), payload
    except ValueError:
        return None


class HourlyWriter:
    def __init__(self, root: str | Path, exchange: str = "okx", compresslevel: int = 6) -> None:
        self.root = Path(root)
        self.exchange = exchange
        self.compresslevel = compresslevel
        self._files: dict[str, tuple[Path, gzip.GzipFile]] = {}

    def _write(self, key: str, recv_ns: int, data: bytes) -> None:
        path = hour_path(self.root, self.exchange, key, recv_ns)
        current = self._files.get(key)
        if current is None or current[0] != path:
            if current is not None:
                current[1].close()
            path.parent.mkdir(parents=True, exist_ok=True)
            current = (path, gzip.open(path, "ab", compresslevel=self.compresslevel))
            self._files[key] = current
        current[1].write(data)

    def write_raw(self, key: str, recv_ns: int, text: str) -> None:
        self._write(key, recv_ns, format_line(recv_ns, text))

    def write_event(self, key: str, recv_ns: int, kind: str, detail: str = "") -> None:
        self._write(key, recv_ns, format_event(recv_ns, kind, detail))

    def flush(self) -> None:
        for _, f in self._files.values():
            f.flush()

    def close(self) -> None:
        for _, f in self._files.values():
            f.close()
        self._files.clear()


def iter_lines(path: str | Path) -> Iterator[tuple[int, str]]:
    """Читает файл записи. Оборванный хвост (аварийный выход) не считается ошибкой."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        try:
            for line in f:
                parsed = parse_line(line)
                if parsed is not None:
                    yield parsed
        except (EOFError, gzip.BadGzipFile):
            return
