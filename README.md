# LSMR — Liquid Staking Mean Reversion

Research and backtesting code for a **cross-venue relative-value strategy on liquid
staking tokens** (LSTs).

A liquid staking token is a transferable claim on staked Ether, redeemable against
its protocol at an exchange rate that moves slowly and almost deterministically.
Its secondary-market price should therefore track Ether up to that rate, but it
does not track it *continuously*: Ether's volatility forces constant re-quoting of
the reference leg while the thinner LST books adjust with a lag. The result is a
stream of small, transient mispricings.

This repository does three things with that observation:

1. **Models** the *price delta* between a reference and an alternative leg as an
   AR–FIGARCH process with Student-*t* innovations and, where the diagnostics
   justify it, a Generalised Pareto tail (`model_researches/`).
2. **Backtests** a delta-neutral, probability-conditioned mean-reversion strategy
   driven by those models on tick-level L2 order books, with venue-specific taker
   fees, signal-to-execution latency and lookahead control (`backtest/`).
3. **Writes it up** as a research paper (`LSMR.pdf`).

## Conventions

| Term | Meaning |
| --- | --- |
| `M1` / `TOKEN1` | the **reference** market and the token traded on it |
| `M2` / `TOKEN2` | the **alternative** market and the token traded on it |
| *combination* | one modelled/traded unit, keyed `M1_TOKEN1_M2_TOKEN2` (e.g. `EXB_STETHUSDT_EXD_ETHUSDT.P`) |
| *price delta* | $(M_1 - M_2)/M_1 \times 10^4$, i.e. the cross-venue price difference in bps of the reference leg |
| `EXA`–`EXE` | anonymised venue codes; the mapping to real venue names lives in `model_researches/market_codes.json`, which is **not** committed |
| `.P` suffix | perpetual contract; no suffix means spot |


## Layout

```
model_researches/   statistical analysis and fitted models
backtest/           backtesting pipeline and results
paper/              the research paper and its figures
```

### `model_researches/` — statistics and models

| File | What it is |
| --- | --- |
| `lsmr_models_researches.ipynb` | the full research notebook, and the source of every test statistic quoted in the paper. Contains models training and out-of-sample validation. |
| `model_observation_chart.py` | `plot_obs_charts` — fan charts: two-regime running means with evolving Student-*t* and GPD quantile bands, refitted at every checkpoint |

### `backtest/` — pipeline and results

| File | What it is |
| --- | --- |
| `lsmr_backtest_run.ipynb` | the entry point: instantiate the pipeline, load the fitted models, override parameters, run one process per combination, display the PnL charts |
| `src/backtest_pipelines.py` | `LiquidStackingMeanReversion` — multiprocessed backtesting pipeline |
| `src/backtest_cores.py` | Core of the backtesting strategy|
| `src/backtest_stats.py` | Produces the backtesting statistics |
| `src/hbtasset.py` | `asset_const` — builds an `hftbacktest` asset from `exchanges_details.json` (tick size, lot size, queue/exchange/fee/latency models) |
| `src/utils.py` | L2 conversion to the engine's event format, snapshot creation, fee lookup, nearest-timestamp alignment across books |
| `results/hftrecords/*.npz` | raw engine records per leg |
| `results/trades_slippage/*_slippage.csv` | per-entry signal vs. executed prices and spread impact, both legs |
| `results/graphs/*.png` | cumulative-PnL charts |


### Paper

| File | What it is |
| --- | --- |
| `LSMR.pdf` | the reasearch paper |