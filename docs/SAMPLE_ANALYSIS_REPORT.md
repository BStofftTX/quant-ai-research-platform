# Sample Analysis Report

## Purpose

This report is a reproducible demonstration artifact for the Quant-AI Research
Platform. It is generated from deterministic synthetic OHLC data and is not a
claim about live-market performance.

## Demonstrated workflow

1. Generate a seeded synthetic close-price series.
2. Build lagged return and rolling-volatility features.
3. Train a logistic-regression model on an earlier time window.
4. Optionally train the PyTorch classifier on the same walk-forward splits.
5. Compare accuracy, F1, strategy return, buy-and-hold return, excess return,
   and maximum drawdown.

## Interpretation

The app displays result tables and charts at runtime rather than storing a
hard-coded performance claim. Reviewers should compare both models on identical
splits, inspect the worst split, and treat synthetic results as a pipeline check
only.

## Limitations

- Synthetic data does not represent investable market behavior.
- The research code does not model every tax, liquidity, borrowing, market
  impact, corporate-action, or survivorship issue.
- The PyTorch path is optional and CPU-oriented in this demonstration.
- No trading, brokerage, or investment decision should be based on this report.
