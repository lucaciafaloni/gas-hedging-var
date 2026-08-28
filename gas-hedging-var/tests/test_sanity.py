"""Sanity checks for Project 2. Run: python tests/test_sanity.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import data_loaders as dl
import garch_model as gm
import regime_ml as rm
import var_hedging as vh


def test_calibration_tracks_anchors():
    """Calibrated annual means within 35% of the real EIA anchors."""
    df = dl.get_price_data()
    got = df["spot"].groupby(df.index.year).mean()
    real = dl.load_annual().set_index("year")["avg_price_usd_mmbtu"]
    for y in [2023, 2025]:
        assert abs(got[y] - real[y]) / real[y] < 0.35


def test_garch_stationary():
    b = gm.build(); p = b["params"]
    assert p["alpha[1]"] + p["beta[1]"] < 1.0     # covariance-stationary


def test_ml_beats_garch_identification():
    m = rm.build().attrs["metrics"]
    assert m["hmm_state_auc"] >= m["garch_state_auc"]
    assert m["preemptive_alarm_rate_ml"] > m["preemptive_alarm_rate_garch"]


def test_var_ordering():
    v = vh.build()["var_table"]
    # 99% VaR must exceed 95% VaR (same method)
    assert v["VaR_normal_99"] > v["VaR_normal_95"]
    assert v["VaR_garch_99"] > v["VaR_garch_95"]


def test_hedge_reduces_var():
    h = vh.build()["hedge"]
    assert h["var_reduction_static_pct"] > 30
    assert h["es99_dynamic_hedge"] <= h["es99_static_hedge"]   # dynamic better tail


def test_regime_aware_fewer_breaches():
    bt = vh.build()["backtest"]
    assert bt["regime_aware"]["breaches"] < bt["garch"]["breaches"]
    assert bt["regime_aware"]["max_breaches_in_10d"] <= bt["garch"]["max_breaches_in_10d"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS ", fn.__name__)
    print(f"\n{len(fns)}/{len(fns)} checks passed.")
