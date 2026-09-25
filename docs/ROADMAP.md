# Research and Engineering Roadmap

## Phase 1 — reproducible analytical core

- [x] Market-data ingestion and PostgreSQL persistence
- [x] Security, benchmark, factor, tail-risk, and portfolio analytics
- [x] Deterministic synthetic-data tests
- [x] Logistic-regression research workflow
- [x] Chronological train/validation/test separation
- [x] Walk-forward evaluation and stability summaries
- [x] Locked development environment and continuous integration

## Phase 2 — backtest realism

- [ ] Add explicit transaction-cost, spread, slippage, and execution-delay models
- [ ] Add position sizing, turnover, exposure, and cash accounting
- [ ] Add benchmark and risk-free-rate configuration
- [ ] Add corporate-action and missing-data diagnostics
- [ ] Add leakage tests for features, targets, thresholds, and signal timing

**Evidence gate:** synthetic fixtures demonstrate that costs, timing, and portfolio accounting behave as specified across edge cases.

## Phase 3 — reproducible experiments

- [ ] Add versioned experiment configuration and random seeds
- [ ] Record dataset windows, symbols, feature definitions, and dependency versions
- [ ] Add model cards and experiment reports
- [ ] Compare simple baselines before more complex models
- [ ] Add nested or purged validation where the research question requires it

**Evidence gate:** another researcher can reproduce each reported experiment from versioned code, configuration, and legally accessible input data.

## Phase 4 — broader validation

- [ ] Test across market regimes and multiple non-overlapping periods
- [ ] Evaluate sensitivity to thresholds, costs, and universe construction
- [ ] Measure stability across symbols and sectors
- [ ] Add delisted-security and survivorship-bias controls where data permits
- [ ] Separate exploratory, validation, and final holdout analyses

**Evidence gate:** conclusions remain appropriately qualified after cost, sensitivity, and independent holdout analysis.

## Phase 5 — research operations

- [ ] Add environment-based PostgreSQL configuration and migration tooling
- [ ] Add scheduled data-quality checks
- [ ] Add reproducible report generation and artifact retention
- [ ] Define controls required before any paper-trading integration

Live trading is explicitly out of scope until data, risk, security, operational, and independent-validation requirements are defined and satisfied.
