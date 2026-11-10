"""Экспорт журнала бэктестов (SQLite) в хранилище Obsidian.

Запуск:
    python -m hftlab.vault_export --db results/runs.sqlite --vault "~/Obsidian/MyVault/hft-lab"

`--vault` это папка ВНУТРИ вашего хранилища (vault), куда кладётся всё по проекту.

Что создаётся:
    Home.md                     вход: сводка и ссылки
    Strategies/<имя>.md         заметка на стратегию: гипотеза, вывод, таблица ВСЕХ прогонов
    Runs/<run_id>.md            заметки только на важные прогоны (закреплённые + лучший OOS)
    Dashboards/*.base           панели Obsidian Bases (встроены в Obsidian 1.9+)
    Templates/Дневник.md        шаблон дневника, создаётся один раз
    Journal/                    сюда пишите дневник руками

Правила безопасности для ваших заметок:
  * база только читается;
  * файлы никогда не удаляются;
  * в сгенерированной заметке перезаписываются только свойства (frontmatter) и блок
    между маркерами AUTO. Всё, что вы написали ниже маркера AUTO:END, сохраняется;
  * если файл существует, но маркеров в нём нет (его писали руками), он пропускается.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hftlab.runs import RunDB

AUTO_START = "<!-- AUTO:START (генерируется hftlab.vault_export, правки здесь затрутся) -->"
AUTO_END = "<!-- AUTO:END (ниже пишите свободно) -->"
DEFAULT_MANUAL = "\n## Мои заметки\n\n"
MAX_TABLE_ROWS = 200


# ---------- маленькие помощники ----------

def yaml_value(v: Any) -> str:
    # JSON-строки и списки являются валидным YAML, так проще не ошибиться с экранированием
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(v, ensure_ascii=False)


def frontmatter(props: dict[str, Any]) -> str:
    lines = ["---"] + [f"{k}: {yaml_value(v)}" for k, v in props.items()] + ["---"]
    return "\n".join(lines) + "\n"


def cell(v: Any) -> str:
    """Значение для ячейки markdown-таблицы."""
    if v is None:
        return ""
    if isinstance(v, float):
        v = f"{v:.6g}"
    elif not isinstance(v, str):
        v = json.dumps(v, ensure_ascii=False, sort_keys=True)
    return v.replace("|", "\\|").replace("\n", " ")


def short(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def link(target: str, label: str | None = None) -> str:
    return f"[[{target}|{label}]]" if label else f"[[{target}]]"


def write_if_changed(path: Path, text: str) -> bool:
    """Атомарная запись (через временный файл); не трогаем файл, если содержимое то же."""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return True


def render_note(path: Path, props: dict[str, Any], auto_body: str) -> str | None:
    """Собрать заметку, сохранив ручную часть. None: файл ручной, трогать нельзя."""
    manual = DEFAULT_MANUAL
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if AUTO_END not in old:
            return None
        manual = old.split(AUTO_END, 1)[1]
    return frontmatter(props) + "\n" + AUTO_START + "\n" + auto_body.rstrip() + "\n" + AUTO_END + manual


# ---------- выбор и подсчёты ----------

def valid(runs: list[dict]) -> list[dict]:
    return [r for r in runs if r["status"] != "invalid"]


def best_oos(runs: list[dict]) -> dict | None:
    cands = [r for r in valid(runs) if r["split"] == "oos" and r["pnl_net"] is not None]
    return max(cands, key=lambda r: r["pnl_net"]) if cands else None


def runs_with_notes(runs: list[dict]) -> set[str]:
    ids = {r["run_id"] for r in runs if r["pinned"]}
    b = best_oos(runs)
    if b is not None:
        ids.add(b["run_id"])
    return ids


# ---------- заметки ----------

def strategy_note(s: dict, runs: list[dict], noted: set[str]) -> tuple[dict, str]:
    v = valid(runs)
    b = best_oos(runs)
    props = {
        "type": "hft-strategy",
        "strategy": s["name"],
        "status": s["status"],
        "hypothesis": s["hypothesis"],
        "n_runs": len(v),
        "n_oos": sum(1 for r in v if r["split"] == "oos"),
        "best_oos_pnl_net": b["pnl_net"] if b else None,
        "last_run": runs[-1]["created_utc"] if runs else None,
        "updated": s["updated_utc"],
        "tags": ["hft", "hft/strategy"],
    }
    body = [f"# {s['name']}", "", f"**Статус:** {s['status']}", "",
            "## Гипотеза", "", s["hypothesis"] or "_не записана_", ""]
    if s["conclusion"]:
        body += ["## Вывод", "", s["conclusion"], ""]
    body += ["## Честность", ""]
    if v:
        body.append(
            f"Проверено вариантов: **{len(v)}** (из них OOS: {props['n_oos']}). "
            "Лучший результат из N попыток завышен случайностью тем сильнее, чем больше N."
        )
    else:
        body.append("Прогонов пока нет.")
    dirty = sum(1 for r in v if r["git_dirty"])
    if dirty:
        body.append(f"\n⚠️ Прогонов с незакоммиченным кодом: {dirty}. Их нельзя точно воспроизвести.")
    suspect = sum(1 for r in v if r["status"] == "suspect")
    if suspect:
        body.append(f"\n⚠️ Подозрительных прогонов (ищем баг): {suspect}.")
    body += ["", f"## Прогоны (последние {MAX_TABLE_ROWS})", "",
             "| прогон | split | статус | сделок | pnl_gross | pnl_net | параметры | заметка |",
             "|---|---|---|---|---|---|---|---|"]
    for r in reversed(runs[-MAX_TABLE_ROWS:]):
        rid = link(f"Runs/{r['run_id']}", r["run_id"]) if r["run_id"] in noted else r["run_id"]
        body.append("| " + " | ".join([
            rid.replace("|", "\\|"), r["split"], r["status"], cell(r["n_trades"]),
            cell(r["pnl_gross"]), cell(r["pnl_net"]), cell(short(cell(r["params"]))),
            cell(short(r["note"], 60)),
        ]) + " |")
    return props, "\n".join(body)


def run_note(r: dict, attempt_no: int, attempts_total: int) -> tuple[dict, str]:
    props = {
        "type": "hft-run",
        "run_id": r["run_id"],
        "strategy": link(f"Strategies/{r['strategy']}", r["strategy"]),
        "split": r["split"],
        "status": r["status"],
        "created": r["created_utc"],
        "data_from": r["data_from"],
        "data_to": r["data_to"],
        "instruments": r["instruments"],
        "n_trades": r["n_trades"],
        "pnl_gross": r["pnl_gross"],
        "pnl_net": r["pnl_net"],
        "fees": r["fees"],
        "attempt": f"{attempt_no}/{attempts_total}",
        "git_commit": r["git_commit"],
        "git_dirty": r["git_dirty"],
        "tags": ["hft", "hft/run"],
    }
    body = [
        f"# Прогон {r['run_id']}", "",
        f"Стратегия: {props['strategy']} · split: **{r['split']}** · статус: **{r['status']}**", "",
        f"Попытка {attempt_no} из {attempts_total} по этой стратегии (с учётом всех, кроме invalid).", "",
        f"Данные: {r['data_from'] or '?'} → {r['data_to'] or '?'}, инструменты: {', '.join(r['instruments']) or '?'}",
        f"Код: `{r['git_commit'] or 'вне git'}`" + (" ⚠️ с незакоммиченными правками" if r["git_dirty"] else ""),
        "", "## Параметры", "", "```json", json.dumps(r["params"], ensure_ascii=False, indent=2, sort_keys=True), "```",
        "", "## Метрики", "", "```json", json.dumps(r["metrics"], ensure_ascii=False, indent=2, sort_keys=True), "```",
    ]
    if r["note"]:
        body += ["", "## Заметка при запуске", "", r["note"]]
    if r["artifacts"]:
        body += ["", "## Файлы", ""] + [f"- `{a}`" for a in r["artifacts"]]
    return props, "\n".join(body)


def home_note(strats: list[dict], runs_by: dict[str, list[dict]]) -> tuple[dict, str]:
    total = sum(len(valid(runs_by.get(s["name"], []))) for s in strats)
    props = {"type": "hft-home", "n_strategies": len(strats), "n_runs": total, "tags": ["hft", "moc"]}
    body = ["# hft-lab: журнал исследования", "",
            f"Стратегий: **{len(strats)}**, прогонов всего (без invalid): **{total}**.", "",
            "Панели: [[Dashboards/Стратегии.base|Стратегии]] · [[Dashboards/Прогоны.base|Важные прогоны]]", ""]
    for status in ("alive", "testing", "idea", "dead"):
        group = [s for s in strats if s["status"] == status]
        if group:
            body += [f"## {status} ({len(group)})", ""]
            body += [f"- {link('Strategies/' + s['name'], s['name'])}: {short(s['hypothesis'] or '—', 100)}"
                     for s in group]
            body.append("")
    return props, "\n".join(body)


# Obsidian Bases: свойства заметок адресуются как note.<имя>
STRATEGIES_BASE = """\
filters:
  and:
    - 'note.type == "hft-strategy"'
views:
  - type: table
    name: Стратегии
    order:
      - file.name
      - note.status
      - note.n_runs
      - note.n_oos
      - note.best_oos_pnl_net
      - note.last_run
      - note.hypothesis
    sort:
      - property: note.last_run
        direction: DESC
"""

RUNS_BASE = """\
filters:
  and:
    - 'note.type == "hft-run"'
views:
  - type: table
    name: Важные прогоны
    order:
      - file.name
      - note.strategy
      - note.split
      - note.status
      - note.attempt
      - note.n_trades
      - note.pnl_net
      - note.pnl_gross
      - note.git_dirty
      - note.created
    sort:
      - property: note.created
        direction: DESC
"""

DIARY_TEMPLATE = """\
---
type: hft-journal
date: "{{date:YYYY-MM-DD}}"
tags: [hft, hft/journal]
---
# {{date:YYYY-MM-DD}}

## Что проверял

## Что получилось (ссылки на [[Strategies/...]] и [[Runs/...]])

## Что подозрительно / где может быть баг

## Следующий шаг
"""


# ---------- главный проход ----------

def export(db_path: str | Path, vault: str | Path) -> dict[str, int]:
    vault = Path(vault).expanduser()
    stats = {"written": 0, "unchanged": 0, "skipped_manual": 0}

    def put(path: Path, text: str | None) -> None:
        if text is None:
            stats["skipped_manual"] += 1
            print(f"пропущен (написан руками, нет маркеров AUTO): {path}")
        elif write_if_changed(path, text):
            stats["written"] += 1
        else:
            stats["unchanged"] += 1

    with RunDB(db_path) as db:
        strats = db.strategies()
        runs_by = {s["name"]: db.runs(s["name"]) for s in strats}

    for s in strats:
        runs = runs_by[s["name"]]
        noted = runs_with_notes(runs)
        props, body = strategy_note(s, runs, noted)
        p = vault / "Strategies" / f"{s['name']}.md"
        put(p, render_note(p, props, body))

        v = valid(runs)
        order = {r["run_id"]: i + 1 for i, r in enumerate(v)}
        for r in runs:
            if r["run_id"] in noted:
                props, body = run_note(r, order.get(r["run_id"], 0), len(v))
                p = vault / "Runs" / f"{r['run_id']}.md"
                put(p, render_note(p, props, body))

    props, body = home_note(strats, runs_by)
    p = vault / "Home.md"
    put(p, render_note(p, props, body))

    # Панели генерируются целиком (это конфигурация, а не заметки)
    for name, text in (("Стратегии.base", STRATEGIES_BASE), ("Прогоны.base", RUNS_BASE)):
        put(vault / "Dashboards" / name, text)

    # Шаблон и папку дневника создаём один раз и больше не трогаем
    tpl = vault / "Templates" / "Дневник.md"
    if not tpl.exists():
        put(tpl, DIARY_TEMPLATE)
    (vault / "Journal").mkdir(parents=True, exist_ok=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Экспорт журнала бэктестов в Obsidian")
    ap.add_argument("--db", default="results/runs.sqlite")
    ap.add_argument("--vault", required=True, help="папка внутри хранилища Obsidian")
    a = ap.parse_args()
    if not Path(a.db).exists():
        raise SystemExit(f"базы {a.db} нет: сначала запишите хотя бы один прогон (hftlab.runs)")
    s = export(a.db, a.vault)
    print(f"готово: записано {s['written']}, без изменений {s['unchanged']}, "
          f"пропущено ручных {s['skipped_manual']}")


if __name__ == "__main__":
    main()
