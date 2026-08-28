# Data Provenance — Project 2 (Natural-Gas Hedging / VaR)

Primary/official sources only (EIA, NOAA, CME Group). Access date **2026-08-19**.

> **IMPORTANT — data mode.** This repository is built to run on the **live primary
> series** via `src/data_loaders.py` (`USE_LIVE_DATA=True`): EIA Henry Hub daily
> spot + weekly storage, and NOAA heating/cooling degree days. The sandbox in which
> the repo was assembled cannot reach those hosts directly, so the **in-repo working
> price series is a reproducible, seed-fixed simulation CALIBRATED to the real
> published statistics below** (annual means, record low, documented volatility
> episodes) and is clearly labelled *illustrative* throughout. The GARCH, regime-ML,
> VaR and hedging **methodology and code are the real, verifiable contribution**;
> the headline numbers are regenerated on live data by setting `USE_LIVE_DATA=True`.
> This is disclosed in the report (Data & Limitations) — no real datum is fabricated
> and no calibration point is used without the citation below.

## 1. Henry Hub annual average spot price ($/MMBtu)
File: `data/processed/henryhub_annual.csv`. Source: EIA "Today in Energy".
- 2023 = **$2.57** (62% drop from 2022) — https://www.eia.gov/todayinenergy/detail.php?id=61183
- 2024 = **$2.21** inflation-adjusted record low (nominal ≈ $2.26); 2024 monthly ranged
  $3.25 (Jan) to all-time-low $1.51 (Mar) — https://www.eia.gov/todayinenergy/detail.php?id=64184
- 2025 = **$3.52** (+56% vs 2024); 2025 daily range **$2.65–$9.86** —
  https://www.eia.gov/todayinenergy/detail.php?id=66984
- 2026 forecast ≈ **$3.50**; 2027 forecast ≈ **$4.60** (+33%, LNG-driven) —
  https://www.eia.gov/todayinenergy/detail.php?id=67004
- 2022 ≈ $6.45 (context; implied ~$6.76 from the 62% drop statement).
- Full daily history (live pull): https://www.eia.gov/dnav/ng/hist/rngwhhdm.htm

## 2. Documented volatility-regime episodes (calibration + narrative)
File: `data/processed/vol_regime_episodes.csv`. Real events used to calibrate the
regime structure of the illustrative series: COVID demand collapse (2020 lows ~$1.5);
**Winter Storm Uri, Feb 2021** (Henry Hub daily spot record spike); summer-2022 peak
(~$9–10); 2024 record low ($1.51 monthly); Jan-2025 polar-vortex spikes ($9.86 daily
high, Northeast citygates to $16+). Sources: EIA Today-in-Energy items above.

## 3. Weekly natural-gas storage (demand/supply driver)
Live pull: EIA Weekly Natural Gas Storage Report —
https://ir.eia.gov/ngs/ngs.html (working gas in storage vs 5-year average).
EIA STEO notes inventories falling below the 5-yr average through the forecast,
"lower storage → higher prices" (id=67004).

## 4. Weather (demand driver): heating/cooling degree days
Live pull: NOAA CPC / NCEI degree-day products —
https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/
Used as an exogenous demand feature in the regime-ML model.

## 5. CME Henry Hub Natural Gas (NG) futures — contract specification
File: `data/processed/cme_ng_contract.csv`. Source: CME Group / NYMEX Rulebook
Chapter 220 — https://www.cmegroup.com/rulebook/NYMEX/2/220.pdf and
https://www.cmegroup.com/markets/energy/natural-gas/natural-gas.contractSpecs.html
- Contract unit **10,000 MMBtu**; quotation **$/MMBtu**; tick **$0.001 = $10/contract**;
  **physical settlement** at Henry Hub; **all calendar months** listed; trading
  terminates the **3rd business day before the 1st of the delivery month**.
- Market depth: ~400k contracts/day, 1.7M open interest (contract overview page).

## DATA GAPS (flagged, not silently filled)
1. **Full daily price history** is not bulk-downloadable in the build sandbox; the
   working series is calibrated-illustrative (see banner). Live pull provided.
2. **Real-time futures curve / option-implied vols** are not extracted; the hedge
   uses the CME contract SPEC (real) with the illustrative spot/futures basis.
   `USE_LIVE_DATA` + a CME/where-available quote feed replaces this for production.
