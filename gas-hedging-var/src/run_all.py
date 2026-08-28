"""Project 2 end-to-end pipeline. `python src/run_all.py`.
Reproduces all results (calibrated mode) and figures. Set USE_LIVE_DATA=True in
config.py to run on live EIA/NOAA data."""
from __future__ import annotations
import json
import pandas as pd
from config import OUTPUTS, USE_LIVE_DATA
import garch_model as gm
import regime_ml as rm
import var_hedging as vh
import figures


def main():
    print(f"[data mode] {'LIVE primary sources' if USE_LIVE_DATA else 'calibrated-illustrative'}")
    g = gm.build()
    pd.Series(g["params"]).to_csv(OUTPUTS / "garch_params.csv")

    r = rm.build()
    with open(OUTPUTS / "regime_metrics.json", "w") as f:
        json.dump(r.attrs["metrics"], f, indent=2)
    r.attrs["importances"].to_csv(OUTPUTS / "regime_feature_importances.csv")

    v = vh.build()
    with open(OUTPUTS / "var_hedging_results.json", "w") as f:
        json.dump({"var_table": v["var_table"], "backtest": v["backtest"],
                   "hedge": v["hedge"]}, f, indent=2)

    figures.build_all()

    print("\n============== HEADLINE RESULTS (Project 2) ==============")
    print("Regime detection:", r.attrs["metrics"])
    print("VaR (99%, $5M MMBtu long):",
          {k: round(val) for k, val in v["var_table"].items() if "99" in k})
    print("Hedge:", v["hedge"])
    print("=========================================================")


if __name__ == "__main__":
    main()
