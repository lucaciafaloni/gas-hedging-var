"""ML volatility-regime detection — the component GARCH tends to lag.

WHY ML IS NEEDED HERE:
GARCH(1,1) volatility is a smooth, backward-looking function of past squared
returns; it *reacts* to a shock only after the shock enters the variance
recursion, so it systematically LAGS the onset of a stressed regime (and is slow
to stand down afterwards). Regime shifts in gas are driven by observable,
non-linear conditions — cold-snap heating demand, storage deficits, supply
disruptions — that a classifier can read *contemporaneously*. We therefore add:

  (1) a Gaussian Hidden Markov Model (the classical regime-switching approach) on
      returns + realized vol -> unsupervised calm/stressed probability, and
  (2) a gradient-boosting classifier predicting the stressed regime from lagged
      returns, realized vol, and weather/storage drivers.

Both are benchmarked against a GARCH-threshold rule on their ability to identify
the (known) stressed regime and on detection LEAD TIME.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from config import ML_LAGS, RANDOM_STATE, HIGH_VOL_QUANTILE, N_REGIMES
import garch_model as gm


def _features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["ret"] = df["log_ret"]
    x["absret"] = df["log_ret"].abs()
    x["rv10"] = df["log_ret"].rolling(10).std()
    x["rv20"] = df["log_ret"].rolling(20).std()
    for L in range(1, ML_LAGS + 1):
        x[f"ret_l{L}"] = df["log_ret"].shift(L)
        x[f"absret_l{L}"] = df["log_ret"].abs().shift(L)
    x["hdd"] = df["hdd"]; x["cdd"] = df["cdd"]
    x["storage_dev"] = df["storage_dev_bcf"]
    return x


def fit_hmm(df: pd.DataFrame) -> pd.Series:
    obs = np.column_stack([df["log_ret"].values,
                           df["log_ret"].rolling(10).std().bfill().values])
    hmm = GaussianHMM(n_components=N_REGIMES, covariance_type="full",
                      n_iter=200, random_state=RANDOM_STATE)
    hmm.fit(obs)
    post = hmm.predict_proba(obs)
    # stressed state = the one with larger return variance
    stressed = int(np.argmax([hmm.covars_[k][0, 0] for k in range(N_REGIMES)]))
    return pd.Series(post[:, stressed], index=df.index, name="hmm_stressed_prob")


_CACHE = {}

def build():
    if "r" in _CACHE:
        return _CACHE["r"]
    b = gm.build()
    df = b["df"].copy()

    # ground-truth stressed state (known regime in the calibrated data; on live
    # data, replace with a realized-vol threshold label).
    if "true_regime" in df:
        state = df["true_regime"].astype(int)
    else:
        rv = df["log_ret"].rolling(10).std()
        state = (rv > rv.quantile(HIGH_VOL_QUANTILE)).astype(int)
    df["state"] = state
    # EARLY-WARNING target: is a stressed regime present within the next H days?
    # This is the task where an exogenous-driver classifier (weather/storage) can
    # beat a backward-looking GARCH variance.
    H = 3
    y = pd.Series(
        [int(state.iloc[i + 1:i + 1 + H].max()) if i + 1 < len(state) else 0
         for i in range(len(state))], index=state.index, name="y")

    X = _features(df)
    data = X.join(y.rename("y")).dropna()
    Xv, yv = data.drop(columns="y"), data["y"]

    # ---- gradient boosting, walk-forward out-of-sample predictions ----
    tss = TimeSeriesSplit(n_splits=5)
    oos = pd.Series(index=Xv.index, dtype=float)
    for tr, te in tss.split(Xv):
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         learning_rate=0.05,
                                         random_state=RANDOM_STATE)
        clf.fit(Xv.iloc[tr], yv.iloc[tr])
        oos.iloc[te] = clf.predict_proba(Xv.iloc[te])[:, 1]
    oos = oos.dropna()

    # full-fit for feature importances
    clf_full = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05,
                                          random_state=RANDOM_STATE).fit(Xv, yv)
    importances = pd.Series(clf_full.feature_importances_, index=Xv.columns
                            ).sort_values(ascending=False).round(3)

    # ---- HMM unsupervised regime prob ----
    hmm_prob = fit_hmm(df).reindex(oos.index)

    # ---- GARCH-threshold baseline ----
    gv = df["garch_vol"].reindex(oos.index)
    garch_score = (gv - gv.min()) / (gv.max() - gv.min())

    ytrue = yv.reindex(oos.index)                # forward early-warning target
    state_oos = df["state"].reindex(oos.index)   # contemporaneous truth
    hmm_f = hmm_prob.fillna(hmm_prob.mean())
    metrics = dict(
        # (1) regime IDENTIFICATION: classify the contemporaneous stressed state
        ml_state_auc=round(roc_auc_score(state_oos, oos), 3),
        hmm_state_auc=round(roc_auc_score(state_oos, hmm_f), 3),
        garch_state_auc=round(roc_auc_score(state_oos, garch_score.fillna(0)), 3),
        # (2) EARLY WARNING: predict a stressed regime within the next 3 days
        ml_earlywarn_auc=round(roc_auc_score(ytrue, oos), 3),
        garch_earlywarn_auc=round(roc_auc_score(ytrue, garch_score.fillna(0)), 3),
    )

    # ---- detection lead time at state onsets (per-signal 80th-pct alarm) ----
    metrics["mean_lead_days_ml_vs_garch"] = _lead_time(state_oos, oos, gv)
    # ---- pre-emptive alarm rate: onsets flagged at/before onset day ----
    metrics.update(_alarm_rates(state_oos, oos, gv))

    res = pd.DataFrame({"y_true": ytrue, "state": state_oos, "ml_prob": oos,
                        "hmm_prob": hmm_prob, "garch_score": garch_score,
                        "garch_vol": gv, "spot": df["spot"].reindex(oos.index)})
    res.attrs["metrics"] = metrics
    res.attrs["importances"] = importances
    _CACHE["r"] = res
    return res


def _onsets(state):
    y = np.asarray(state.values)
    return np.where((y[1:] == 1) & (y[:-1] == 0))[0] + 1


def _lead_time(state, ml_prob, garch_vol):
    """Average days by which the ML alarm precedes the GARCH alarm around each
    regime onset. Each signal uses ITS OWN 80th-percentile as the alarm level
    (a fair, per-signal threshold). Positive => ML fires earlier."""
    ml = ml_prob.values; g = garch_vol.values
    ml_thr = np.nanquantile(ml, 0.80); g_thr = np.nanquantile(g, 0.80)
    leads = []
    for o in _onsets(state):
        w0, w1 = max(0, o - 15), min(len(ml), o + 6)
        ml_hit = next((i for i in range(w0, w1) if ml[i] >= ml_thr), None)
        g_hit = next((i for i in range(w0, w1) if g[i] >= g_thr), None)
        if ml_hit is not None and g_hit is not None:
            leads.append(g_hit - ml_hit)
    return round(float(np.mean(leads)), 2) if leads else float("nan")


def _alarm_rates(state, ml_prob, garch_vol):
    """Fraction of onsets for which each signal's alarm fires at or before the
    onset day (pre-emptive detection)."""
    ml = ml_prob.values; g = garch_vol.values
    ml_thr = np.nanquantile(ml, 0.80); g_thr = np.nanquantile(g, 0.80)
    ons = _onsets(state); pre_ml = pre_g = 0
    for o in ons:
        w0 = max(0, o - 15)
        if any(ml[i] >= ml_thr for i in range(w0, o + 1)):
            pre_ml += 1
        if any(g[i] >= g_thr for i in range(w0, o + 1)):
            pre_g += 1
    k = max(1, len(ons))
    return {"preemptive_alarm_rate_ml": round(pre_ml / k, 2),
            "preemptive_alarm_rate_garch": round(pre_g / k, 2),
            "n_onsets": int(len(ons))}


if __name__ == "__main__":
    r = build()
    print("=== Regime-detection metrics ===")
    for k, v in r.attrs["metrics"].items():
        print(f"  {k}: {v}")
    print("\n=== GBM feature importances (top 8) ===")
    print(r.attrs["importances"].head(8).to_string())
