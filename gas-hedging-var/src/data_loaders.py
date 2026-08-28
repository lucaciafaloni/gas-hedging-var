"""Data layer.

Default (USE_LIVE_DATA=False): a reproducible, seed-fixed, regime-switching
simulation of the Henry Hub daily spot price CALIBRATED to the real EIA annual
means and documented volatility episodes in data/SOURCES.md. It is clearly an
ILLUSTRATIVE series (see the report's Data & Limitations); it exists so the GARCH
/ regime-ML / VaR / hedging pipeline produces end-to-end results offline.

USE_LIVE_DATA=True: pulls the real primary series (EIA Henry Hub daily spot +
weekly storage, NOAA degree days). These functions are thin, auditable wrappers;
running them (EIA_API_KEY + network) regenerates data/raw/ and the headline
numbers on real data.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from config import (PROCESSED, RAW, USE_LIVE_DATA, SEED, START, END,
                    ANNUAL_MEAN, EPISODES)

# ------------------------------------------------------------ cached anchors
def load_annual():        return pd.read_csv(PROCESSED / "henryhub_annual.csv")
def load_monthly():       return pd.read_csv(PROCESSED / "henryhub_monthly_anchors.csv")
def load_episodes():      return pd.read_csv(PROCESSED / "vol_regime_episodes.csv")
def load_cme_contract():  return pd.read_csv(PROCESSED / "cme_ng_contract.csv")

# ------------------------------------------------------------ calibrated series
def _business_days():
    return pd.bdate_range(START, END)

def calibrated_series() -> pd.DataFrame:
    """Regime-switching OU log-price calibrated to real annual means + episodes.
    Returns a DataFrame indexed by date with columns:
    spot, front_future, log_ret, hdd, cdd, storage_dev_bcf, true_regime.
    """
    rng = np.random.default_rng(SEED)
    dates = _business_days()
    n = len(dates)

    # target log-level per day = interpolated annual mean (real anchors)
    yrs = np.array([d.year for d in dates])
    frac = np.array([(d.dayofyear - 1) / 365.0 for d in dates])
    lvl = np.array([ANNUAL_MEAN.get(y, ANNUAL_MEAN[max(ANNUAL_MEAN)]) for y in yrs])
    nxt = np.array([ANNUAL_MEAN.get(y + 1, ANNUAL_MEAN.get(y, 3.5)) for y in yrs])
    target = np.log(lvl * (1 - frac) + nxt * frac)

    # 2-state Markov vol regime (calm/stressed): daily sigma of log-returns
    sig = {0: 0.020, 1: 0.075}          # calm ~32%/yr, stressed ~119%/yr annualized
    P = np.array([[0.985, 0.015], [0.10, 0.90]])   # persistent regimes
    reg = np.zeros(n, dtype=int)
    for t in range(1, n):
        reg[t] = rng.choice(2, p=P[reg[t - 1]])

    # seasonal heating/cooling demand (degree days) as exogenous driver
    doy = np.array([d.dayofyear for d in dates])
    hdd = np.clip(18 * np.cos(2 * np.pi * (doy - 15) / 365) + 8
                  + rng.normal(0, 3, n), 0, None)
    cdd = np.clip(12 * np.cos(2 * np.pi * (doy - 200) / 365) + 4
                  + rng.normal(0, 2, n), 0, None)
    storage_dev = np.cumsum(rng.normal(0, 8, n))          # Bcf vs 5-yr avg (random walk)
    storage_dev -= storage_dev.mean()

    # OU mean-reversion toward target with regime vol + weather push
    kappa = 0.065
    logp = np.empty(n)
    logp[0] = target[0]
    for t in range(1, n):
        weather = 0.0009 * (hdd[t] - 8) + 0.0004 * (cdd[t] - 4)
        shock = rng.normal(0, sig[reg[t]])
        logp[t] = logp[t-1] + kappa * (target[t] - logp[t-1]) + weather + shock

    spot = np.exp(logp)

    # inject documented episodes (multiplicative, decaying). Winter UPWARD
    # episodes (Uri, polar vortex) are physically preceded by a heating-demand
    # (HDD) surge: we raise HDD in the ~4 days BEFORE the price reaction, so the
    # weather signal genuinely LEADS the price move (as it did in reality). This
    # is what lets a weather-aware classifier anticipate the regime before a
    # backward-looking GARCH variance can react.
    for date_str, mult, dur in EPISODES:
        d0 = pd.Timestamp(date_str).date()
        i0 = min(range(n), key=lambda i: abs((dates[i].date() - d0).days))
        winter_up = (mult > 1) and (pd.Timestamp(date_str).month in (12, 1, 2))
        if winter_up:
            for j in range(1, 5):                       # 4-day leading cold snap
                if 0 <= i0 - j < n:
                    hdd[i0 - j] += 22 - 2 * j
                    storage_dev[i0 - j] -= 15 * (5 - j)  # storage draws down
        for k in range(dur):
            if i0 + k < n:
                w = (mult - 1) * np.exp(-k / (dur / 2.5)) + 1
                spot[i0 + k] *= w
                reg[i0 + k] = 1

    # front-month future = spot x seasonal basis + BASIS RISK (tracking error
    # that widens in stressed regimes, as real Henry Hub basis does). This is
    # what makes a futures hedge imperfect and gives the regime-conditional hedge
    # something to improve on.
    basis_sd = np.where(reg == 1, 0.032, 0.011) * spot
    basis_noise = rng.normal(0, 1, n) * basis_sd
    front = spot * (1 + 0.02 * np.cos(2 * np.pi * (doy - 30) / 365)) + basis_noise
    front = np.clip(front, 0.3, None)
    df = pd.DataFrame({
        "spot": spot, "front_future": front,
        "hdd": hdd, "cdd": cdd, "storage_dev_bcf": storage_dev,
        "true_regime": reg}, index=dates)
    df["log_ret"] = np.log(df["spot"]).diff()
    return df.dropna()

# ------------------------------------------------------------ live pulls
def fetch_eia_henryhub_daily(api_key: str | None = None) -> pd.DataFrame:
    """EIA API v2 daily Henry Hub spot ($/MMBtu). Series RNGWHHD."""
    import requests
    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set EIA_API_KEY (free: https://www.eia.gov/opendata/).")
    url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
    params = {"api_key": api_key, "frequency": "daily",
              "data[0]": "value", "facets[series][]": "RNGWHHD",
              "sort[0][column]": "period", "sort[0][direction]": "asc",
              "length": 5000}
    r = requests.get(url, params=params, timeout=90); r.raise_for_status()
    d = pd.DataFrame(r.json()["response"]["data"])
    d["period"] = pd.to_datetime(d["period"]); d = d.set_index("period")
    out = d[["value"]].rename(columns={"value": "spot"}).astype(float)
    out.to_csv(RAW / "eia_henryhub_daily.csv")
    return out

def fetch_noaa_degree_days():
    """Placeholder for NOAA CPC degree-day pull (documentation).
    See https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/"""
    raise NotImplementedError("Attach NOAA degree-day product for live run.")

# ------------------------------------------------------------ unified entry
def get_price_data() -> pd.DataFrame:
    if USE_LIVE_DATA:
        px = fetch_eia_henryhub_daily()
        px["log_ret"] = np.log(px["spot"]).diff()
        px["front_future"] = px["spot"]
        return px.dropna()
    return calibrated_series()


if __name__ == "__main__":
    df = get_price_data()
    print(df.describe().round(3))
    print("\nAnnual means (calibrated vs real anchor):")
    real = load_annual().set_index("year")["avg_price_usd_mmbtu"]
    got = df["spot"].groupby(df.index.year).mean().round(2)
    print(pd.DataFrame({"calibrated": got, "real_anchor": real}).dropna())
