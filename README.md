# Quant-AI Research Platform

A Python-based quantitative research platform for ingesting, storing, analyzing, and comparing financial market data.

This project is being developed as a hands-on quantitative engineering and AI research platform, with an emphasis on modular architecture, statistical analysis, risk measurement, reproducibility, and automated testing.

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

### Structured Security Profiles
The platform organizes analytical results into four logical groups:

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
    +------> PostgreSQL
    |         database.py
    |         sql/schema.sql
    |
    v
market_analysis.py
    |
    +------> Performance metrics
    +------> Risk metrics
    +------> Benchmark analysis
    +------> Regression analysis
    +------> Abnormal-return detection
    +------> Security profiles
    |
    v
Ranked Analysis / CSV Output
```

The project separates data acquisition, transformation, persistence, and analysis so that individual components can evolve independently.

## Project Structure

```text
quant-ai-research-platform/
├── sql/
│   └── schema.sql
├── src/
│   └── quant_ai_research_platform/
│       ├── __init__.py
│       ├── database.py
│       ├── etl.py
│       ├── ingest.py
│       └── market_analysis.py
├── tests/
│   └── test_market_analysis.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Technology Stack

- Python 3.12+
- NumPy
- pandas
- SciPy
- yfinance
- PostgreSQL
- psycopg
- pytest
- uv

## Testing

The analytics layer includes automated tests covering core statistical, benchmark, regression, rolling-risk, tail-risk, and structured-profile functionality.

Run the test suite with:

```bash
uv run pytest -q
```

## Development Approach

The platform is being developed incrementally with an emphasis on:

- modular separation of concerns
- test-driven feature development
- reproducible environments
- Git-based version control
- quantitative risk and performance analysis
- extensibility for future AI/ML research

## Roadmap

Planned areas of exploration include:

- portfolio-level analytics
- expanded factor modeling
- additional benchmark and risk models
- feature engineering for financial time series
- machine-learning research workflows
- model evaluation and backtesting
- visualization and reporting
- AI-assisted quantitative research

## Status

**Active development — 2026**

The current release represents the quantitative-analysis foundation of a broader research platform.

## Author

**W. Bruce Stofft**

MS Computer Science — Artificial Intelligence  
MBA — Management

Founder, MacroStofft LLC

Background spanning artificial intelligence, machine learning, systems analysis, technical program leadership, and enterprise technology.
