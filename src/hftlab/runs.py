"""Журнал бэктестов: каждая проверка гипотезы пишется в SQLite.

Зачем отдельная база, а не заметки руками:
  * пишется кодом, поэтому ничего не теряется и не "округляется в голове";
  * пишутся ВСЕ прогоны, включая провальные. Это нужно для честного учёта
    multiple testing: если проверили 40 вариантов параметров, лучший из них
    почти наверняка выглядит хорошо случайно. Число прогонов на стратегию
    видно всегда (`count_runs`);
  * у каждого прогона есть git commit и флаг "были незакоммиченные правки",
    чтобы через месяц можно было воспроизвести результат.

Obsidian получает данные отсюда через `hftlab.vault_export` (только чтение базы).

Пример:
    from hftlab.runs import RunDB
    db = RunDB("results/runs.sqlite")
    db.ensure_strategy("mr_microprice", hypothesis="microprice возвращается к EMA за 1-5 с")
    run_id = db.log_run(
        "mr_microprice",
        split="oos",                       # train | test | oos | wf-<k>
        params={"ema_ms": 500, "z_entry": 2.0},
        metrics={"n_trades": 812, "pnl_gross": 31.2, "pnl_net": -4.8,
                 "fees": 36.0, "markout_bps": {"100ms": 0.4, "1s": 0.9, "5s": 0.2}},
        data_from="2026-11-01", data_to="2026-11-07",
        instruments=["BTC-USDT-SWAP"],
        note="после комиссий в минус",
    )
"""
from __future__ import annotations

import json
import math
import re
import secrets
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Имя стратегии становится именем файла в Obsidian, поэтому только безопасные символы
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SPLIT_RE = re.compile(r"^(train|test|oos|wf-\d+)$")
STRATEGY_STATUSES = ("idea", "testing", "alive", "dead")
RUN_STATUSES = ("new", "suspect", "ok", "invalid")
# Метрики, которые выносятся в отдельные колонки (по ним удобно сортировать)
PROMOTED = ("n_trades", "pnl_gross", "pnl_net", "fees")

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    name        TEXT PRIMARY KEY,
    hypothesis  TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'idea',
    conclusion  TEXT NOT NULL DEFAULT '',
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    strategy     TEXT NOT NULL REFERENCES strategies(name),
    created_utc  TEXT NOT NULL,
    split        TEXT NOT NULL,
    data_from    TEXT NOT NULL DEFAULT '',
    data_to      TEXT NOT NULL DEFAULT '',
    instruments  TEXT NOT NULL DEFAULT '[]',
    params       TEXT NOT NULL DEFAULT '{}',
    metrics      TEXT NOT NULL DEFAULT '{}',
    n_trades     INTEGER,
    pnl_gross    REAL,
    pnl_net      REAL,
    fees         REAL,
    git_commit   TEXT NOT NULL DEFAULT '',
    git_dirty    INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'new',
    pinned       INTEGER NOT NULL DEFAULT 0,
    note         TEXT NOT NULL DEFAULT '',
    artifacts    TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS runs_by_strategy ON runs(strategy, created_utc);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def git_state(cwd: str | Path | None = None) -> tuple[str, bool]:
    """(короткий commit, есть ли незакоммиченные правки). Вне git: ("", False)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip() != ""
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "", False


def _to_json(value: Any, what: str) -> str:
    # allow_nan=False: NaN/inf в метриках почти всегда баг (деление на ноль сделок),
    # лучше упасть сразу, чем тихо записать мусор
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except ValueError as e:
        raise ValueError(f"{what}: NaN/inf не допускаются ({e})") from None


def _promoted(metrics: dict[str, Any], key: str) -> float | int | None:
    v = metrics.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"metrics[{key!r}] должно быть конечным числом, а не {v!r}")
    return v


class RunDB:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RunDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- стратегии ----------

    def ensure_strategy(self, name: str, hypothesis: str = "") -> None:
        """Создать стратегию, если её нет. Существующую не трогает (кроме пустой гипотезы)."""
        if not NAME_RE.match(name):
            raise ValueError(f"имя стратегии {name!r}: только a-z, 0-9, '_' и '-', до 64 символов")
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO strategies(name, hypothesis, created_utc, updated_utc) "
                "VALUES (?, ?, ?, ?)", (name, hypothesis, now, now),
            )
            if hypothesis:
                self.conn.execute(
                    "UPDATE strategies SET hypothesis = ?, updated_utc = ? "
                    "WHERE name = ? AND hypothesis = ''", (hypothesis, now, name),
                )

    def set_strategy_status(self, name: str, status: str, conclusion: str | None = None) -> None:
        if status not in STRATEGY_STATUSES:
            raise ValueError(f"статус стратегии: один из {STRATEGY_STATUSES}")
        with self.conn:
            cur = self.conn.execute(
                "UPDATE strategies SET status = ?, conclusion = COALESCE(?, conclusion), "
                "updated_utc = ? WHERE name = ?", (status, conclusion, utc_now(), name),
            )
        if cur.rowcount == 0:
            raise KeyError(name)

    def strategies(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM strategies ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    # ---------- прогоны ----------

    def log_run(
        self,
        strategy: str,
        *,
        split: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        data_from: str = "",
        data_to: str = "",
        instruments: list[str] | None = None,
        note: str = "",
        artifacts: list[str] | None = None,
        git: tuple[str, bool] | None = None,
    ) -> str:
        """Записать прогон. Стратегия должна существовать (ensure_strategy)."""
        if not SPLIT_RE.match(split):
            raise ValueError(f"split {split!r}: train | test | oos | wf-<номер>")
        commit, dirty = git if git is not None else git_state()
        run_id = new_run_id()
        row = {
            "run_id": run_id,
            "strategy": strategy,
            "created_utc": utc_now(),
            "split": split,
            "data_from": data_from,
            "data_to": data_to,
            "instruments": _to_json(list(instruments or []), "instruments"),
            "params": _to_json(params, "params"),
            "metrics": _to_json(metrics, "metrics"),
            **{k: _promoted(metrics, k) for k in PROMOTED},
            "git_commit": commit,
            "git_dirty": int(dirty),
            "note": note,
            "artifacts": _to_json(list(artifacts or []), "artifacts"),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        try:
            with self.conn:
                self.conn.execute(f"INSERT INTO runs({cols}) VALUES ({marks})", tuple(row.values()))
                self.conn.execute(
                    "UPDATE strategies SET updated_utc = ?, "
                    "status = CASE status WHEN 'idea' THEN 'testing' ELSE status END "
                    "WHERE name = ?", (row["created_utc"], strategy),
                )
        except sqlite3.IntegrityError:
            raise KeyError(f"стратегии {strategy!r} нет, сначала ensure_strategy()") from None
        return run_id

    def set_run_status(self, run_id: str, status: str, note: str | None = None) -> None:
        """suspect: результат подозрительно хорош, ищем баг; invalid: найден баг, не учитывать."""
        if status not in RUN_STATUSES:
            raise ValueError(f"статус прогона: один из {RUN_STATUSES}")
        with self.conn:
            cur = self.conn.execute(
                "UPDATE runs SET status = ?, note = COALESCE(?, note) WHERE run_id = ?",
                (status, note, run_id),
            )
        if cur.rowcount == 0:
            raise KeyError(run_id)

    def pin_run(self, run_id: str, pinned: bool = True) -> None:
        """Закреплённые прогоны получают свою заметку в Obsidian."""
        with self.conn:
            cur = self.conn.execute("UPDATE runs SET pinned = ? WHERE run_id = ?", (int(pinned), run_id))
        if cur.rowcount == 0:
            raise KeyError(run_id)

    def runs(self, strategy: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runs"
        args: tuple = ()
        if strategy is not None:
            sql += " WHERE strategy = ?"
            args = (strategy,)
        # rowid = порядок записи; по времени нельзя: в одну секунду бывает несколько прогонов
        rows = self.conn.execute(sql + " ORDER BY rowid", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("instruments", "params", "metrics", "artifacts"):
                d[k] = json.loads(d[k])
            d["git_dirty"] = bool(d["git_dirty"])
            d["pinned"] = bool(d["pinned"])
            out.append(d)
        return out

    def count_runs(self, strategy: str) -> int:
        """Сколько раз проверяли стратегию (без invalid). Это и есть число попыток для multiple testing."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM runs WHERE strategy = ? AND status != 'invalid'", (strategy,),
        ).fetchone()[0]
