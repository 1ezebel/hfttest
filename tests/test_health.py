import os

from hftlab.health import check


def test_health_fresh_stale_and_empty(tmp_path):
    assert check(tmp_path, min_free_gb=0) != []          # пусто = проблема
    f = tmp_path / "okx" / "BTC" / "2026-11-10" / "14.jsonl.gz"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    now = f.stat().st_mtime
    assert check(tmp_path, max_age_s=120, min_free_gb=0, now=now + 10) == []
    [problem] = check(tmp_path, max_age_s=120, min_free_gb=0, now=now + 500)
    assert "BTC" in problem
    assert any("мало места" in p for p in check(tmp_path, min_free_gb=10**9, now=now))
