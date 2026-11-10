import math

import pytest

from hftlab.runs import RunDB

GIT = ("abc1234", False)  # чтобы тесты не зависели от того, лежит ли проект в git


def make(tmp_path):
    db = RunDB(tmp_path / "runs.sqlite")
    db.ensure_strategy("mr_test", hypothesis="цена возвращается")
    return db


def test_log_and_read_back(tmp_path):
    db = make(tmp_path)
    rid = db.log_run("mr_test", split="oos", params={"z": 2.0},
                     metrics={"n_trades": 10, "pnl_net": -1.5, "markout_bps": {"1s": 0.3}},
                     instruments=["BTC-USDT-SWAP"], git=GIT)
    [r] = db.runs("mr_test")
    assert r["run_id"] == rid
    assert r["params"] == {"z": 2.0}
    assert r["metrics"]["markout_bps"] == {"1s": 0.3}
    assert (r["n_trades"], r["pnl_net"], r["pnl_gross"]) == (10, -1.5, None)
    assert r["git_commit"] == "abc1234" and r["git_dirty"] is False
    # первый прогон переводит стратегию из idea в testing
    assert db.strategies()[0]["status"] == "testing"


def test_survives_reopen(tmp_path):
    make(tmp_path).log_run("mr_test", split="train", params={}, metrics={}, git=GIT)
    assert len(RunDB(tmp_path / "runs.sqlite").runs()) == 1


def test_rejects_nan_and_bad_input(tmp_path):
    db = make(tmp_path)
    with pytest.raises(ValueError):
        db.log_run("mr_test", split="oos", params={}, metrics={"pnl_net": math.nan}, git=GIT)
    with pytest.raises(ValueError):
        db.log_run("mr_test", split="oos", params={}, metrics={"sharpe": math.inf}, git=GIT)
    with pytest.raises(ValueError):
        db.log_run("mr_test", split="validation", params={}, metrics={}, git=GIT)
    with pytest.raises(KeyError):
        db.log_run("no_such", split="oos", params={}, metrics={}, git=GIT)
    with pytest.raises(ValueError):
        db.ensure_strategy("Bad Name/../x")
    assert db.runs() == []  # ни одной полузаписи


def test_count_runs_excludes_invalid(tmp_path):
    db = make(tmp_path)
    ids = [db.log_run("mr_test", split=f"wf-{i}", params={"i": i}, metrics={}, git=GIT) for i in range(3)]
    db.set_run_status(ids[0], "invalid", note="look-ahead в признаке")
    assert db.count_runs("mr_test") == 2
    assert db.runs()[0]["note"] == "look-ahead в признаке"


def test_strategy_status_and_pin(tmp_path):
    db = make(tmp_path)
    rid = db.log_run("mr_test", split="oos", params={}, metrics={}, git=GIT)
    db.pin_run(rid)
    db.set_strategy_status("mr_test", "dead", conclusion="после комиссий в минус")
    s = db.strategies()[0]
    assert (s["status"], s["conclusion"]) == ("dead", "после комиссий в минус")
    assert db.runs()[0]["pinned"] is True
    with pytest.raises(ValueError):
        db.set_strategy_status("mr_test", "great")
    with pytest.raises(KeyError):
        db.pin_run("nope")
