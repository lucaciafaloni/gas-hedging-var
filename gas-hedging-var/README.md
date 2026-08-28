# Natural-gas Value-at-Risk and hedging (Henry Hub)

Full analysis and results: [report/report.pdf](report/report.pdf)

Questions this project looks at:
- How large is the daily risk of a natural-gas position, and how much does a futures hedge reduce it?
- Can a machine-learning model detect volatility-regime shifts earlier than a standard GARCH model?

Approach: a GARCH-based Value-at-Risk framework with a CME futures hedge, plus an ML
layer (hidden Markov model / gradient boosting) for regime detection.

Run: `python src/run_all.py` (dependencies in `requirements.txt`). The repo runs on a
sample calibrated to published EIA statistics; see the report and `data/SOURCES.md`.
