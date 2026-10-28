# Dependency Maintenance — 2026-10-27

## Scope

This maintenance branch starts from `origin/main` and updates only resolved
transitive dependencies in `uv.lock`. It does not modify the Streamlit
deployment branch, application code, or project dependency declarations.

## Baseline and result

`origin/main` initially produced six audit occurrences across three packages:

- `certifi` 2026.7.22 → 2026.10.23
- `filelock` 4.0.3 → 4.2.4
- `msgpack` 1.2.2 → 1.2.4

`setuptools` was also refreshed from 84.0.0 to 84.2.1 as part of the locked
environment refresh. `Pillow` is not present in the `origin/main` lockfile,
so it was not added merely to address findings from a different environment.

The post-update `uv run pip-audit` result reported no known vulnerabilities.

## Verification

- `uv run pytest -p no:cacheprovider -q`: 70 passed
- `uv run ruff check .`: passed
- `uv run ruff format --check .`: passed
- PyTorch optional path: imported successfully (`torch` 2.14.1)

## Important audit boundary

The previously documented 42-advisory result belongs to the separate
full-development/deployment environment, not this `origin/main` maintenance
baseline. It needs its own reviewed remediation scope; this branch does not
claim to resolve findings outside its lockfile.
