# Branch: aws-deploy

## Purpose
Implement AWS deployment infrastructure for paper and live trading as described in AWS_DEPLOYMENT_SCOPE.

## Source Plans
- docs/AWS_DEPLOYMENT_SCOPE.md

## Scope
- Add ECS task definitions for worker, ops-api, mcp-server, and optional temporal.
- Add Terraform modules for VPC, ECS, RDS, ALB, IAM, Secrets Manager, and CloudWatch.
- Add Secrets Manager wiring and config safety checks for live trading.
- Add CI/CD workflows for paper and live deployments (approval gates, rollback).

## Out of Scope / Deferred
- Non-AWS runtime code changes unrelated to deployment wiring.
- Multi-wallet or trading logic changes.

## Key Files
- infra/ecs/task-definitions/*.json
- infra/terraform/*.tf
- app/core/config.py (Secrets Manager wiring)
- pyproject.toml (boto3 dependency)
- .github/workflows/deploy*.yml

## Dependencies / Coordination
- Coordinate with multi-wallet branch if Secrets Manager keys overlap.
- Keep app/runtime behavior unchanged except for optional Secrets Manager toggle.

## Acceptance Criteria
- Task definitions exist for all required services.
- Terraform configs cover VPC, ECS, RDS, ALB, IAM, Secrets, CloudWatch.
- Secrets Manager integration works with explicit live trading safety check.
- CI/CD workflows implement paper vs live separation and approval gates.

## Test Plan (required before commit)
- (Terraform) cd infra/terraform && terraform fmt -recursive
- (Terraform) cd infra/terraform && terraform init
- (Terraform) cd infra/terraform && terraform validate

If infra validation cannot run (missing credentials), obtain user-run output and paste it below before committing.

## Human Verification (required)
- Review Terraform plan output and task definitions for security/cost implications.
- Confirm live trading safety check requires LIVE_TRADING_ACK before orders can be placed.
- Paste plan summary or review notes in the Human Verification Evidence section.

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b aws-deploy

# Work, then review changes
git status
git diff

# Stage changes
git add infra/ecs/task-definitions \
  infra/terraform \
  app/core/config.py \
  pyproject.toml \
  .github/workflows

# Run tests (must succeed or be explicitly approved by user with pasted output)
cd infra/terraform && terraform fmt -recursive
cd infra/terraform && terraform init
cd infra/terraform && terraform validate

# Commit ONLY after test evidence is captured below
git commit -m "AWS deploy: ECS, Terraform, secrets, CI/CD"
```

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

