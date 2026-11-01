import importlib


def test_fail_safe_when_server_down(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # nothing listens
    import experiments.mlflow_utils as mu

    mu = importlib.reload(mu)  # reset the cached preflight
    art = tmp_path / "a.txt"
    art.write_text("x")

    assert mu.setup_tracking("pytest_down") is False
    run_id = mu.log_run(
        params={"a": 1}, metrics={"m": 0.5}, artifacts=[art], tags={"t": "x"}
    )
    assert run_id is None
