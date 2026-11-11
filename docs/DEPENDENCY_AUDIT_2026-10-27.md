# Dependency audit — 2026-10-27

## Scope

Two separate dependency profiles were checked. They must not be conflated.

| Profile | Result | Deployment relevance |
| --- | --- | --- |
| Full local development lock (`uv run pip-audit`) | 43 scanner occurrences across four packages | Local development maintenance work; not used by the hosted demo |
| Streamlit Cloud requirements (`pip-audit -r streamlit_app/requirements.txt`) | No known vulnerabilities | Hosted-demo profile |

The earlier count of 42 represents distinct advisory identifiers. The current
scanner output contains 43 occurrences because `PYSEC-2026-4320` is listed
twice for `certifi`.

## Full local development profile

| Package | Version | Scanner occurrences | Distinct severity | Available fix |
| --- | ---: | ---: | --- | --- |
| certifi | 2026.7.22 | 2 (same advisory duplicated) | Critical: 1 | 2026.10.19 |
| filelock | 4.0.3 | 3 | High: 1; Low: 2 | 4.2.1 |
| Pillow | 12.3.0 | 37 | Critical: 4; High: 22; Medium: 11 | 13.0.0 |
| setuptools | 84.0.0 | 1 | Low: 1 | 84.1.0 |

Distinct-advisory total: **Critical 5, High 23, Medium 11, Low 3**.
The duplicate certifi occurrence makes the scanner-occurrence total **Critical
6, High 23, Medium 11, Low 3**.

Severity was resolved from the linked GitHub security advisories. The duplicated
`PYSEC-2026-4320` maps to `GHSA-6q48-6c7w-9gw2` (Critical).

## Cloud profile verification

A fresh CPython 3.12.14 environment was created with `uv venv`, then populated
only with `uv pip install --python .venv-cloud/bin/python -r
streamlit_app/requirements.txt`.

- Resolved packages: 39
- `import app`: passed
- Torch present: no
- yfinance present: no
- `pip` executable seeded by `uv venv`: no (expected)
- Cloud requirements audit: no known vulnerabilities

The clean Cloud environment resolved `certifi 2026.10.23` and `Pillow 13.0.0`.
It did not include `filelock` or `setuptools` as application dependencies.

## Decision

Do not blindly upgrade the full local lock. Upgrade it in a separate reviewed
maintenance change, test the full development profile, and reassess any optional
PyTorch dependency impact. The isolated Streamlit Cloud profile is not blocked
by these local-profile findings.
