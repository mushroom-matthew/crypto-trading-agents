.PHONY: init db-up db-down migrate test reconcile

init:
	uv sync

db-up:
	docker compose up -d db
	uv run alembic upgrade head

db-down:
	docker compose stop db

migrate:
	uv run alembic revision --autogenerate -m "$(name)"

test:
	uv run pytest

reconcile:
	uv run python -m app.cli.main reconcile run
