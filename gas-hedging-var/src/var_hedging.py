"""Value-at-Risk and hedging — the standard risk-desk deliverable.

A hypothetical energy company holds a long physical natural-gas position of
POSITION_MMBTU. We compute 1-day VaR four ways:
  * parametric-normal (textbook baseline),
  * historical simulation,
  * GARCH(1,1)-t (time-varying vol, industry standard),
  * REGIME-AWARE (GARCH vol blended up in ML/HMM-detected stressed regimes).
We backtest coverage (Kupiec POF) to show the regime-aware VaR is better
calibrated, then design a CME NG futures hedge that reduces the VaR, and show a
regime-conditional dynamic hedge improves tail outcomes over a static hedge.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from config import (VAR_LEVELS, POSITION_MMBTU, CME_CONTRACT_MMBTU, TRADING_DAYS)
import regime_ml as rm


def _z(alpha):            # normal quantile for loss tail
    return stats.norm.ppf(alpha)


def build_var_table():
    r = rm.build()
    df = r.join(rm.gm.dl.get_price_data()[["log_ret", "front_future"]],
                how="left")
    df = df.dropna(subset=["log_ret", "spot", "garch_vol"])
    notional = POSITION_MMBTU * df["spot"]          # $ exposure per day
    pnl = notional.values[:-1] * df["log_ret"].values[1:]   # next-day P&L ($)
    pnl = pd.Series(pnl, index=df.index[1:])

    sigma_uncond = df["log_ret"].std()
    out = {}
    for a in VAR_LEVELS:
        z = _z(a)
        # parametric normal (constant vol)
        out[f"VaR_normal_{int(a*100)}"] = float(z * sigma_uncond * notional.mean())
        # historical simulation
        out[f"VaR_hist_{int(a*100)}"] = float(-np.quantile(pnl, 1 - a))
        # GARCH-t (use fitted t dof)
        nu = float(rm.gm.build()["res"].params.get("nu", 6))
        tq = stats.t.ppf(a, nu) * np.sqrt((nu - 2) / nu)
        out[f"VaR_garch_{int(a*100)}"] = float(
            (tq * df["garch_vol"] * notional).mean())
    return df, pnl, notional, out


def _var_series(df, notional, alpha, regime_aware=False):
    """Time-varying VaR series ($). Regime-aware scales vol up with ML stressed
    probability (up to +60% in fully-stressed regimes)."""
    nu = float(rm.gm.build()["res"].params.get("nu", 6))
    tq = stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)
    vol = df["garch_vol"].copy()
    if regime_aware:
        scale = 1.0 + 0.35 * df["ml_prob"].clip(0, 1).fillna(0)
        vol = vol * scale
    return tq * vol * notional


def backtest(df, pnl, notional, alpha=0.99):
    """Kupiec proportion-of-failures test for GARCH-only vs regime-aware VaR."""
    results = {}
    for label, ra in [("garch", False), ("regime_aware", True)]:
        var = _var_series(df, notional, alpha, regime_aware=ra).reindex(pnl.index)
        breaches = (pnl < -var).sum()
        n = pnl.notna().sum()
        rate = breaches / n
        p = alpha_complement = 1 - alpha
        # Kupiec POF likelihood-ratio
        if 0 < breaches < n:
            lr = -2 * (np.log(((1-p)**(n-breaches)) * (p**breaches))
                       - np.log(((1-rate)**(n-breaches)) * (rate**breaches)))
            pval = 1 - stats.chi2.cdf(lr, 1)
        else:
            lr, pval = float("nan"), float("nan")
        # breach clustering: max run of breaches within any 10-day window
        br = (pnl < -var).astype(int)
        clustered = int(br.rolling(10).sum().max())
        results[label] = dict(expected_rate=round(p, 4),
                              observed_rate=round(float(rate), 4),
                              breaches=int(breaches), n=int(n),
                              max_breaches_in_10d=clustered,
                              kupiec_LR=round(float(lr), 3),
                              kupiec_p=round(float(pval), 3))
    return results


def hedge(df, notional):
    """Minimum-variance CME NG futures hedge, and a regime-conditional dynamic
    hedge. Returns hedge ratio, VaR reduction, and tail-loss comparison."""
    d = df.dropna(subset=["log_ret"]).copy()
    dP = POSITION_MMBTU * d["spot"].shift(1) * d["log_ret"]      # physical $ P&L
    dF = CME_CONTRACT_MMBTU * d["front_future"].shift(1) * \
        np.log(d["front_future"]).diff()                        # $ P&L per contract
    dP, dF = dP.dropna(), dF.reindex(dP.dropna().index)
    valid = dP.notna() & dF.notna()
    dP, dF = dP[valid], dF[valid]

    beta = np.cov(dP, dF)[0, 1] / np.var(dF)                    # contracts to short
    hedged_static = dP - beta * dF
    # regime-conditional: scale hedge up 30% in stressed regimes
    scale = 1.0 + 0.30 * d["ml_prob"].reindex(dP.index).clip(0, 1).fillna(0)
    hedged_dynamic = dP - beta * scale * dF

    def var99(x): return float(-np.quantile(x, 0.01))
    def es99(x):  return float(-x[x <= np.quantile(x, 0.01)].mean())
    res = dict(
        min_var_hedge_contracts=round(float(beta), 1),
        var99_unhedged=round(var99(dP), 0),
        var99_static_hedge=round(var99(hedged_static), 0),
        var99_dynamic_hedge=round(var99(hedged_dynamic), 0),
        var_reduction_static_pct=round((1 - var99(hedged_static)/var99(dP))*100, 1),
        var_reduction_dynamic_pct=round((1 - var99(hedged_dynamic)/var99(dP))*100, 1),
        es99_unhedged=round(es99(dP), 0),
        es99_static_hedge=round(es99(hedged_static), 0),
        es99_dynamic_hedge=round(es99(hedged_dynamic), 0),
        hedge_effectiveness_r2=round(1 - np.var(hedged_static)/np.var(dP), 3),
    )
    return res, dict(dP=dP, static=hedged_static, dynamic=hedged_dynamic)


def build():
    df, pnl, notional, var_tbl = build_var_table()
    bt = backtest(df, pnl, notional, 0.99)
    hres, hpaths = hedge(df, notional)
    return dict(var_table=var_tbl, backtest=bt, hedge=hres,
                df=df, pnl=pnl, paths=hpaths)


if __name__ == "__main__":
    b = build()
    print("=== 1-day VaR ($, long", f"{POSITION_MMBTU:,} MMBtu) ===")
    for k, v in b["var_table"].items():
        print(f"  {k:20s}: ${v:,.0f}")
    print("\n=== Backtest (99% VaR, Kupiec) ===")
    for k, v in b["backtest"].items():
        print(f"  {k}: {v}")
    print("\n=== Hedging ===")
    for k, v in b["hedge"].items():
        print(f"  {k}: {v}")
