"""Project 2 configuration: paths, data mode, and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
FIGDIR = ROOT / "report" / "figures"
for _p in (OUTPUTS, FIGDIR, RAW):
    _p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ data mode
# False -> reproducible calibrated-illustrative series (default; runs offline).
# True  -> pull live primary series (EIA Henry Hub daily + weekly storage, NOAA
#          degree days) via data_loaders.fetch_*(). Requires EIA_API_KEY + network.
USE_LIVE_DATA = False

SEED = 7
START = "2016-01-01"
END = "2026-06-30"

# Real calibration anchors (EIA Today-in-Energy; see data/SOURCES.md).
ANNUAL_MEAN = {2016: 2.52, 2017: 2.99, 2018: 3.15, 2019: 2.56, 2020: 2.03,
               2021: 3.89, 2022: 6.45, 2023: 2.57, 2024: 2.26, 2025: 3.52,
               2026: 3.50}
# Documented regime episodes -> (date, multiplicative shock to price, days).
EPISODES = [
    ("2021-02-15", 3.2, 6),    # Winter Storm Uri spike
    ("2022-08-20", 1.5, 20),   # summer 2022 peak
    ("2025-01-20", 2.2, 8),    # Jan-2025 polar vortex
    ("2020-06-01", 0.72, 30),  # COVID demand collapse (downward)
    ("2024-03-01", 0.68, 25),  # 2024 record low (downward)
]

# ------------------------------------------------------------------ risk / hedge
TRADING_DAYS = 252
VAR_LEVELS = [0.95, 0.99]
POSITION_MMBTU = 5_000_000        # hypothetical utility long physical exposure
CME_CONTRACT_MMBTU = 10_000       # NG futures contract unit (CME rulebook ch.220)

# ------------------------------------------------------------------ ML / regime
N_REGIMES = 2                     # calm vs stressed
ML_LAGS = 5
HIGH_VOL_QUANTILE = 0.80          # label: top-20% realized-vol days = "stressed"
RANDOM_STATE = 42
