# Contributing

Contributions should improve quantitative correctness, reproducibility, testing, documentation, or research methodology.

## Before contributing

1. Search existing issues before opening a new one.
2. Open or reference an issue for substantial work.
3. Use synthetic or legally redistributable data in tests and examples.
4. Do not submit brokerage credentials, database passwords, proprietary datasets, or personal financial information.

## Development setup

```bash
uv sync --locked --dev
```

Run all local quality gates:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pip-audit
```

## Research standards

- Use chronological splits for time-series prediction.
- Document features, targets, lags, holding periods, and signal timing.
- Select thresholds and hyperparameters without using the final test period.
- Include realistic costs and constraints before making performance claims.
- Report negative results, sensitivity, uncertainty, and known data limitations.
- Prefer simple baselines before adding model complexity.

## Pull requests

- Use a focused branch such as `feat/description`, `fix/description`, or `docs/description`.
- Explain the hypothesis or engineering need, implementation, and validation.
- State any financial, data-licensing, leakage, or reproducibility implications.
- Keep commits small and meaningful; never manufacture activity or backdate work.
