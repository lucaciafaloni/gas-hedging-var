"""Project 2 report figures. Palette = validated dataviz default (light surface)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from config import FIGDIR, POSITION_MMBTU
import data_loaders as dl
import garch_model as gm
import regime_ml as rm
import var_hedging as vh

BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e3"
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "figure.dpi": 130})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_price_regime():
    df = dl.get_price_data()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(df.index, df["spot"], color=INK2, lw=0.8)
    stressed = df["true_regime"] == 1 if "true_regime" in df else None
    if stressed is not None:
        ax.fill_between(df.index, 0, df["spot"].max()*1.05,
                        where=stressed, color=RED, alpha=0.10, step="mid",
                        label="stressed regime")
    ax.set_ylabel("Henry Hub spot ($/MMBtu)")
    ax.set_title("Calibrated Henry Hub series with volatility regimes\n"
                 "(illustrative; anchored to EIA annual means & documented episodes)",
                 color=INK, fontweight="bold", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _save(fig, "p2_fig1_price_regime")


def fig_garch_vol():
    b = gm.build()
    cv = b["cond_vol"] * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(cv.index, cv.values, color=BLUE, lw=1.0)
    ax.axhline(cv.mean(), color=INK2, ls=":", lw=1, label=f"mean {cv.mean():.0%}")
    ax.set_ylabel("GARCH(1,1) cond. vol (annualized)")
    ax.set_title("Conditional volatility spikes at documented stress episodes",
                 color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    _save(fig, "p2_fig2_garch_vol")


def fig_regime_detect():
    r = rm.build()
    m = r.attrs["metrics"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    # left: AUCs
    labels = ["HMM", "ML-GBM", "GARCH\nthreshold"]
    vals = [m["hmm_state_auc"], m["ml_state_auc"], m["garch_state_auc"]]
    a1.bar(labels, vals, color=[AQUA, ORANGE, INK2])
    for i, v in enumerate(vals):
        a1.text(i, v+0.005, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
    a1.set_ylim(0.7, 0.95); a1.set_ylabel("Regime-ID AUC")
    a1.set_title("Regime identification", color=INK, fontweight="bold",
                 loc="left", fontsize=10)
    # right: pre-emptive alarm rate
    a2.bar(["ML (weather/storage\n-aware)", "GARCH"],
           [m["preemptive_alarm_rate_ml"], m["preemptive_alarm_rate_garch"]],
           color=[ORANGE, INK2])
    for i, v in enumerate([m["preemptive_alarm_rate_ml"],
                           m["preemptive_alarm_rate_garch"]]):
        a2.text(i, v+0.01, f"{v:.0%}", ha="center", fontsize=10, color=INK2)
    a2.set_ylim(0, 0.85); a2.set_ylabel("Onsets flagged pre-emptively")
    a2.set_title(f"Early warning ({m['n_onsets']} regime onsets)", color=INK,
                 fontweight="bold", loc="left", fontsize=10)
    _save(fig, "p2_fig3_regime_detect")


def fig_var_backtest():
    b = vh.build()
    df, pnl = b["df"], b["pnl"]
    notional = POSITION_MMBTU * df["spot"]
    var_g = vh._var_series(df, notional, 0.99, False).reindex(pnl.index)
    var_r = vh._var_series(df, notional, 0.99, True).reindex(pnl.index)
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.plot(pnl.index, pnl/1e6, color=INK2, lw=0.5, alpha=0.7, label="daily P&L")
    ax.plot(pnl.index, -var_g/1e6, color=BLUE, lw=1.1, label="GARCH 99% VaR")
    ax.plot(pnl.index, -var_r/1e6, color=RED, lw=1.1, label="Regime-aware 99% VaR")
    br = pnl < -var_g
    ax.scatter(pnl.index[br], pnl[br]/1e6, s=12, color=RED, zorder=5,
               label="GARCH breaches")
    ax.set_ylim(-8, 4)      # zoom on the VaR band; rare episode spikes clip
    ax.set_ylabel("$ million"); ax.set_title(
        "99% VaR backtest: regime-aware VaR removes breach clustering\n"
        "(extreme episode P&L clipped; e.g. Uri Feb-2021 ≈ -$60M)",
        color=INK, fontweight="bold", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    _save(fig, "p2_fig4_var_backtest")


def fig_hedge():
    b = vh.build()
    p = b["paths"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bins = np.linspace(np.percentile(p["dP"]/1e6, 0.3),
                       np.percentile(p["dP"]/1e6, 99.7), 60)
    ax.hist(p["dP"]/1e6, bins=bins, color=INK2, alpha=0.45, label="unhedged")
    ax.hist(p["static"]/1e6, bins=bins, color=BLUE, alpha=0.55, label="static hedge")
    ax.hist(p["dynamic"]/1e6, bins=bins, color=ORANGE, alpha=0.55,
            label="regime-dynamic hedge")
    h = b["hedge"]
    ax.set_xlabel("Daily P&L ($ million)")
    ax.set_ylabel("days")
    ax.set_title(f"CME NG futures hedge cuts 99% VaR by "
                 f"{h['var_reduction_static_pct']:.0f}% (R²={h['hedge_effectiveness_r2']})",
                 color=INK, fontweight="bold", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5)
    _save(fig, "p2_fig5_hedge")


def fig_importances():
    r = rm.build()
    imp = r.attrs["importances"].head(8).sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.barh(imp.index, imp.values, color=BLUE)
    for i, v in enumerate(imp.values):
        ax.text(v+0.004, i, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("GBM feature importance")
    ax.set_title("Regime-classifier drivers (rv10=realized vol; hdd/storage=exogenous)",
                 color=INK, fontweight="bold", loc="left", fontsize=9.5)
    _save(fig, "p2_fig6_importances")


def build_all():
    fig_price_regime(); fig_garch_vol(); fig_regime_detect()
    fig_var_backtest(); fig_hedge(); fig_importances()
    print(f"P2 figures -> {FIGDIR}")


if __name__ == "__main__":
    build_all()
