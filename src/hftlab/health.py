"""Проверка "жив ли рекордер": свежий ли последний файл по каждому инструменту.

    python -m hftlab.health data            # код возврата 0 = всё хорошо, 1 = проблема
    python -m hftlab.health data --max-age 120 --min-free-gb 2

Рекордер сбрасывает данные на диск раз в 5 секунд, поэтому файл, который не менялся
дольше пары минут, означает: процесс умер, завис или потерял сеть.
Код возврата удобен для cron/планировщика: при 1 можно слать себе уведомление.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def newest_file(inst_dir: Path) -> Path | None:
    files = list(inst_dir.rglob("*.jsonl.gz"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def check(root: str | Path, max_age_s: float = 120.0, min_free_gb: float = 2.0,
          now: float | None = None) -> list[str]:
    """Возвращает список проблем (пустой = здоров)."""
    root = Path(root)
    now = time.time() if now is None else now
    problems: list[str] = []
    inst_dirs = [d for ex in root.iterdir() if ex.is_dir() for d in ex.iterdir() if d.is_dir()] \
        if root.is_dir() else []
    if not inst_dirs:
        return [f"в {root} нет ни одной записи"]
    for d in sorted(inst_dirs):
        f = newest_file(d)
        if f is None:
            problems.append(f"{d.name}: нет файлов")
            continue
        age = now - f.stat().st_mtime
        if age > max_age_s:
            problems.append(f"{d.name}: последний файл не менялся {age:.0f} с (порог {max_age_s:.0f})")
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < min_free_gb:
        problems.append(f"мало места на диске: {free_gb:.1f} ГБ (порог {min_free_gb})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка здоровья рекордера")
    ap.add_argument("root", nargs="?", default="data")
    ap.add_argument("--max-age", type=float, default=120.0, help="секунд без записи = проблема")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    a = ap.parse_args()
    problems = check(a.root, a.max_age, a.min_free_gb)
    for p in problems:
        print("ПРОБЛЕМА:", p)
    if not problems:
        print("OK: запись идёт")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
