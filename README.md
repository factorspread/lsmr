# LSMR — Liquid Staking Mean Reversion

Research and backtesting code for a **cross-venue relative-value strategy on liquid
staking tokens** (LSTs).

A liquid staking token is a transferable claim on staked Ether, redeemable against
its protocol at an exchange rate that moves slowly and almost deterministically.
Its secondary-market price should therefore track Ether up to that rate, but it
does not track it *continuously*: Ether's volatility forces constant re-quoting of
the reference leg while the thinner LST books adjust with a lag. The result is a
stream of small, transient mispricings.

## Results at a glance

On the statistical side, the hypothesis holds. All 57 combinations that pass the
economic-amplitude screen are stationary (ADF rejects at 1%). Their mean-reversion
half-lives run from 11 to 102 minutes, their variance has long memory
(FIGARCH *d* ∈ [0.21, 0.46]), and their innovations are heavy-tailed Student-*t*
(degrees of freedom as low as 2.66).

### Strategy results

The backtest covers 12 combinations over a common ≈ 30-day window, with
2,000 quote units deployed per combination.

| Metric | Value |
| --- | ---: |
| Combinations profitable | 9/12 (75%) |
| Mean / median return per combination | +48.05% / +3.51% |
| Aggregate PnL | +11,532.8 quote units on 24,000 deployed |
| Entry signals executed | 5,239 |
| Hit rate: mean / median (range) | 19.4% / 20.2% (0.0% to 50.1%) |
| Sharpe: median (range) | 2.75 (−16.33 to 6.15), positive in 9/11 |
| Sortino: median (range) | 37.45 (−179.00 to 615.59) |
| Max drawdown: mean / median / worst | −53.3% / −3.8% / −464.5% |

| Family | *n* | Profitable | Mean / median return | Median Sharpe | Median Sortino | Median hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `STETHUSDT` (rebasing LST) | 6 | 6/6 | +25.2% / +3.5% | 3.39 | 175.06 | 14.4% |
| `METHUSDT` (accrual LST) | 5 | 3/5 | +86.0% / +52.7% | 2.80 | 22.81 | 32.4% |
| `RPLUSDT` (no-redemption control) | 1 | 0/1 | −4.4% | −16.33 | −155.45 | 16.5% |

**Takeaway.** The edge is positive but heavily skewed: the mean return is an order
of magnitude above the median, and most of the PnL comes from a few large
mispricings. Profits are concentrated in combinations that have a redemption
channel. The no-redemption control loses money even though it is statistically
the best-behaved process in the study. Round-trip taker costs of 22–44 bps
amount to 0.5–6.4 standard deviations of the price delta, so the opportunity
lives mostly in the tails of the distribution. Thin LST books limit the
strategy's capacity.

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
| `lsmr_backtest_run.ipynb` | Runs the backtesting pipeline |
| `src/backtest_pipelines.py` | Multiprocessed backtesting pipeline |
| `src/backtest_cores.py` | Core of the backtesting strategy|
| `src/backtest_stats.py` | Produces the backtesting statistics |
| `results/hftrecords/*.npz` | raw engine records per leg |
| `results/trades_slippage/*_slippage.csv` | per-entry signal vs. executed prices and spread impact, both legs |
| `results/graphs/*.png` | cumulative-PnL charts |


### Paper

| File | What it is |
| --- | --- |
| `LSMR.pdf` | the reasearch paper |

*Note: The venues' anonymized datasets are available on request*