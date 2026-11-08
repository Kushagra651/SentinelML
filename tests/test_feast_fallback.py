import logging
import os
import socket
from pathlib import Path

import pandas as pd
import pytest

import experiments.feature_lookup as fl


def _redis_up() -> bool:
    try:
        socket.create_connection(
            (os.getenv("REDIS_HOST", "ml_redis"), int(os.getenv("REDIS_PORT", "6379"))),
            1,
        ).close()
        return True
    except OSError:
        return False


def test_falls_back_to_parquet_when_redis_fails(monkeypatch, tmp_path, caplog):
    row = {c: 1 for c in fl.COLS}
    pd.DataFrame([{"person_id": i, **row} for i in range(5)]).to_parquet(
        tmp_path / "train_feast.parquet"
    )
    monkeypatch.setattr(fl, "STORE_DIR", tmp_path)
    monkeypatch.setattr(
        fl, "_from_redis", lambda ids: (_ for _ in ()).throw(OSError("down"))
    )
    with caplog.at_level(logging.WARNING):
        out = fl.get_features([1, 3])
    assert list(out["person_id"]) == [1, 3]
    assert "falling back" in caplog.text


def test_predict_does_not_depend_on_feast():
    src = Path("api/predict.py").read_text().lower()
    assert "feast" not in src and "redis" not in src


@pytest.mark.skipif(not _redis_up(), reason="Redis not reachable")
def test_live_redis_lookup():
    assert len(fl.get_features([5, 6])) == 2
