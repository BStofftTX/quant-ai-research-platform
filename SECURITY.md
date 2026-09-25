# Security Policy

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature for this repository. Do not disclose exploitable details in a public issue.

Include:

- the affected component and commit or version
- steps to reproduce using synthetic or public test data
- the potential impact
- any suggested mitigation

## Sensitive information

Do not commit or attach:

- brokerage or market-data-provider credentials
- PostgreSQL credentials or connection strings containing secrets
- proprietary or restricted market datasets
- personal financial or account information
- exported production database contents

Use environment variables or an appropriate secret manager for credentials. This research repository must not be treated as a production trading environment.
