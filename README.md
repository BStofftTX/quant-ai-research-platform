# Quant-AI Research Platform

A Python-based quantitative research and machine-learning platform for ingesting, storing, analyzing, modeling, and comparing financial market data.

The project is being developed as a hands-on quantitative engineering and AI research platform with an emphasis on modular architecture, statistical analysis, portfolio analytics, machine learning, reproducibility, backtesting, and automated testing.

## Current Capabilities

### Market Data

- Historical market-data ingestion using `yfinance`
- Multi-security analysis
- Benchmark-aligned return analysis
- PostgreSQL-backed market-data persistence
- CSV export of ranked security comparisons

### Performance Analytics

- Total return
- Compound annual growth rate (CAGR)
- Average daily return
- Daily and annualized volatility
- Benchmark-relative performance

### Risk & Risk-Adjusted Metrics

- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Downside deviation
- Historical Value at Risk (VaR) at 95% and 99%
- Historical Expected Shortfall (ES) at 95% and 99%

### Statistical & Benchmark Analysis

- Correlation
- Beta
- R-squared
- Annualized alpha
- One-factor regression analysis
- Regression significance metrics
- Statistical abnormal-return detection
- Rolling volatility, correlation, and beta

### Portfolio Analytics

- Equal-weight portfolio construction
- Custom portfolio weights
- Portfolio return calculation
- Compounded portfolio total return
- Annualized portfolio return
- Annualized portfolio volatility
- Portfolio Sharpe ratio
- Portfolio maximum drawdown
- Integrated portfolio analysis reports
- Portfolio weight validation

### Machine-Learning Research

The modeling layer provides an end-to-end financial ML research workflow including:

- Financial time-series feature engineering
- Forward-return target creation
- Model dataset construction
- Train/test splitting
- Train/validation/test separation
- Feature/target separation
- Logistic-regression classification
- Prediction generation
- Prediction-probability generation
- Probability-threshold trading signals
- Classification threshold comparison
- Validation-based threshold selection
- Strategy backtesting
- Model prediction evaluation
- Validated ML backtesting

### Walk-Forward Research

The platform supports time-aware out-of-sample model evaluation through:

- Walk-forward split generation
- Walk-forward model backtesting
- Multi-period research execution
- Strategy versus buy-and-hold comparison
- Walk-forward result summaries
- Profitable-split rate
- Benchmark-beating rate
- Median excess return
- Best and worst excess-return analysis
- Walk-forward stability reporting

### Structured Security Profiles

Analytical results are organized into four logical groups:

- **Performance**
- **Risk**
- **Risk-adjusted performance**
- **Market behavior**

## Architecture

```text
                    Market Data
                         |
                         v
                     ingest.py
                         |
                         v
                       etl.py
                         |
             +-----------+-----------+
             |                       |
             v                       v
        PostgreSQL            market_analysis.py
        database.py                  |
        sql/schema.sql               |
                                     +--> Performance analytics
                                     +--> Risk analytics
                                     +--> Benchmark analysis
                                     +--> Regression analysis
                                     +--> Portfolio analytics
                                     |
                                     v
                                  modeling.py
                                     |
                                     +--> Feature engineering
                                     +--> Logistic regression
                                     +--> Prediction probabilities
                                     +--> Threshold selection
                                     +--> ML backtesting
                                     +--> Walk-forward validation
                                     |
                                     v
                          Research Results / Reports