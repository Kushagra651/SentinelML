"""tests/test_agent_tools.py: regression tests for two agent tool bugs."""

import pandas as pd

import agent.tools as tools


def test_drift_report_reports_total_features(monkeypatch):
    report = {
        "overall_drifted": True,
        "drifted_features": ["age"],
        "summary": {"total_features_checked": 14, "critical_count": 1},
        "feature_results": [
            {"feature": "age", "psi": 0.3, "severity": "critical", "drifted": True}
        ],
    }
    monkeypatch.setattr(tools, "_latest_json", lambda subdir: dict(report))
    out = tools.get_drift_report.invoke({})
    assert out["total_features_checked"] == 14


def test_prediction_logs_counts_all_rows_and_samples_newest(monkeypatch):
    import api.logger as lg

    n = 600  # more than the old 500-row cap
    df = pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(n)],
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
            "prediction": [i % 2 for i in range(n)],
            "confidence": [0.9] * n,
            "model_version": ["v1"] * n,
            "ground_truth": [None] * n,
        }
    )
    monkeypatch.setattr(lg, "query_logs", lambda start, end, limit=100_000, **kw: df)
    out = tools.get_prediction_logs.invoke({"hours": 6})
    assert out["total_predictions"] == n
    assert out["recent_sample"][0]["request_id"] == "r599"  # newest first
