"""Classical volatility core: GARCH(1,1) — the industry-standard risk-desk model.

Fits a GARCH(1,1) with Student-t innovations to Henry Hub daily log-returns and
returns the conditional volatility series and one-day-ahead forecast. This is the
baseline the ML regime layer is measured against.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from arch import arch_model

import data_loaders as dl
from config import TRADING_DAYS


def fit_garch(returns: pd.Series):
    """GARCH(1,1)-t on percent log-returns (arch convention: scale x100)."""
    r = returns.dropna() * 100.0
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / 100.0          # back to return units
    cond_vol.name = "garch_vol"
    return res, cond_vol


def garch_annualized(cond_vol: pd.Series) -> pd.Series:
    return cond_vol * np.sqrt(TRADING_DAYS)


def one_day_forecast(res) -> float:
    """Next-day conditional volatility (return units)."""
    f = res.forecast(horizon=1, reindex=False)
    return float(np.sqrt(f.variance.values[-1, 0]) / 100.0)


_CACHE = {}

def build() -> dict:
    if "b" in _CACHE:
        return _CACHE["b"]
    df = dl.get_price_data()
    res, cond_vol = fit_garch(df["log_ret"])
    out = df.join(cond_vol)
    _CACHE["b"] = dict(df=out, res=res, cond_vol=cond_vol,
                       next_vol=one_day_forecast(res),
                       params=res.params.round(4).to_dict())
    return _CACHE["b"]


if __name__ == "__main__":
    b = build()
    print(b["res"].summary().tables[1])
    print("\nnext-day vol (daily):", round(b["next_vol"], 4),
          " annualized:", round(b["next_vol"] * np.sqrt(TRADING_DAYS), 3))
    cv = b["cond_vol"]
    print("cond vol (annualized) mean/min/max:",
          round(cv.mean()*np.sqrt(252),2), round(cv.min()*np.sqrt(252),2),
          round(cv.max()*np.sqrt(252),2))
