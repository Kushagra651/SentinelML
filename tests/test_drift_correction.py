import pytest

import monitoring.drift_report as d


def _r(name, p, psi=0.0):
    return d.FeatureDriftResult(
        feature=name,
        dtype="numerical",
        method="ks",
        statistic=0.0,
        p_value=p,
        psi=psi,
        drifted=p < 0.05,
        severity="none",
    )


def _set(monkeypatch, corr, min_psi=0.0):
    monkeypatch.setattr(d, "DRIFT_CORRECTION", corr)
    monkeypatch.setattr(d, "DRIFT_MIN_PSI", min_psi)


def _flags(results):
    return [r.drifted for r in results]


def test_none_leaves_raw_decision(monkeypatch):
    _set(monkeypatch, "none")
    rs = [_r("a", 0.04), _r("b", 0.5)]
    d._apply_correction(rs)
    assert _flags(rs) == [True, False]


def test_bonferroni_suppresses_borderline(monkeypatch):
    _set(monkeypatch, "bonferroni")
    rs = [_r("a", 0.04), _r("b", 0.0001)] + [_r(f"f{i}", 0.5) for i in range(12)]
    d._apply_correction(rs)
    assert rs[0].drifted is False  # 0.04 > 0.05/14
    assert rs[1].drifted is True


def test_bh_rejects_only_smallest(monkeypatch):
    _set(monkeypatch, "bh")
    rs = [_r("a", 0.001), _r("b", 0.01), _r("c", 0.03)]
    rs += [_r(f"f{i}", 0.9) for i in range(11)]
    d._apply_correction(rs)
    assert _flags(rs)[:3] == [True, False, False]


def test_psi_gate_blocks_low_effect(monkeypatch):
    _set(monkeypatch, "none", 0.1)
    rs = [_r("low", 0.001, psi=0.05), _r("high", 0.001, psi=0.2)]
    d._apply_correction(rs)
    assert _flags(rs) == [False, True]


def test_empty_list_is_safe(monkeypatch):
    _set(monkeypatch, "bonferroni", 0.1)
    d._apply_correction([])