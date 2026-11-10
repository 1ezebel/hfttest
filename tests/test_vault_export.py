from hftlab.runs import RunDB
from hftlab.vault_export import AUTO_END, export

GIT = ("abc1234", True)


def fill(db_path):
    db = RunDB(db_path)
    db.ensure_strategy("mr_a", hypothesis="microprice | EMA")  # '|' не должен ломать таблицу
    db.ensure_strategy("idea_b")
    db.log_run("mr_a", split="train", params={"z": 1}, metrics={"pnl_net": 9.0}, git=GIT)
    worse = db.log_run("mr_a", split="oos", params={"z": 1}, metrics={"pnl_net": -2.0}, git=GIT)
    best = db.log_run("mr_a", split="oos", params={"z": 2}, metrics={"pnl_net": 1.0}, git=GIT)
    db.close()
    return worse, best


def test_export_creates_structure(tmp_path):
    db, vault = tmp_path / "runs.sqlite", tmp_path / "vault"
    worse, best = fill(db)
    export(db, vault)

    note = (vault / "Strategies" / "mr_a.md").read_text(encoding="utf-8")
    assert 'type: "hft-strategy"' in note
    assert "n_runs: 3" in note and "best_oos_pnl_net: 1.0" in note
    assert "незакоммиченным" in note          # предупреждение про git_dirty
    assert f"[[Runs/{best}\\|{best}]]" in note  # лучший OOS получил ссылку на свою заметку
    # заметка только на лучший OOS, а не на все прогоны
    assert [p.stem for p in (vault / "Runs").iterdir()] == [best]
    assert "Попытка 3 из 3" in (vault / "Runs" / f"{best}.md").read_text(encoding="utf-8")
    assert (vault / "Strategies" / "idea_b.md").exists()
    assert (vault / "Dashboards" / "Стратегии.base").exists()
    assert (vault / "Templates" / "Дневник.md").exists()
    assert (vault / "Journal").is_dir()
    assert "[[Strategies/mr_a|mr_a]]" in (vault / "Home.md").read_text(encoding="utf-8")


def test_manual_text_preserved_and_manual_files_untouched(tmp_path):
    db, vault = tmp_path / "runs.sqlite", tmp_path / "vault"
    fill(db)
    export(db, vault)

    p = vault / "Strategies" / "mr_a.md"
    p.write_text(p.read_text(encoding="utf-8") + "мой вывод: спред съедает всё\n", encoding="utf-8")
    manual = vault / "Strategies" / "idea_b.md"
    manual.write_text("написано руками без маркеров", encoding="utf-8")
    tpl = vault / "Templates" / "Дневник.md"
    tpl.write_text("мой шаблон", encoding="utf-8")

    with RunDB(db) as d:
        d.log_run("mr_a", split="oos", params={"z": 3}, metrics={"pnl_net": 0.5}, git=GIT)
    stats = export(db, vault)

    text = p.read_text(encoding="utf-8")
    assert "n_runs: 4" in text
    assert text.split(AUTO_END, 1)[1].endswith("мой вывод: спред съедает всё\n")
    assert manual.read_text(encoding="utf-8") == "написано руками без маркеров"
    assert tpl.read_text(encoding="utf-8") == "мой шаблон"
    assert stats["skipped_manual"] == 1


def test_second_export_is_noop(tmp_path):
    db, vault = tmp_path / "runs.sqlite", tmp_path / "vault"
    fill(db)
    export(db, vault)
    assert export(db, vault)["written"] == 0


def test_chart_artifacts_are_copied_and_embedded(tmp_path):
    db, vault = tmp_path / "results" / "runs.sqlite", tmp_path / "vault"
    chart = tmp_path / "results" / "charts" / "decay.png"
    chart.parent.mkdir(parents=True)
    chart.write_bytes(b"png-bytes")
    with RunDB(db) as d:
        d.ensure_strategy("mr_a")
        rid = d.log_run("mr_a", split="train", params={}, metrics={}, git=GIT,
                        artifacts=["results/charts/decay.png", "results/paper/x.jsonl"])
        d.pin_run(rid)
    export(db, vault)
    assert (vault / "Assets" / "decay.png").read_bytes() == b"png-bytes"
    note = (vault / "Runs" / f"{rid}.md").read_text(encoding="utf-8")
    assert "![[Assets/decay.png]]" in note and "`results/paper/x.jsonl`" in note
    assert export(db, vault)["written"] == 0
