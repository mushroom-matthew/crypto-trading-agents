# Repository Guidelines

## Project Structure & Module Organization
Core trading agents live in `agents/` (broker, execution, judge, shared utilities). Durable MCP tools sit in `tools/`. `mcp_server/` exposes the FastAPI surface; `worker/` boots Temporal workers; `main.py` offers a lightweight CLI entry point. Tests reside in `tests/`, and `run_stack.sh` plus `docker-compose.yml` orchestrate the local stack. Keep logs and artefacts under `logs/` or a new subfolder rather than mixing them with source.

## Build, Test, and Development Commands
- `uv sync` installs Python dependencies declared in `pyproject.toml`.
- `uv run ./run_stack.sh` launches Temporal, the MCP server, and sample agents in tmux panes.
- `uv run python main.py` runs the broker agent loop without Temporal services.
- `uv run pytest [-k pattern]` executes unit tests; add `-vv` when debugging.
Regenerate `uv.lock` only when intentionally updating dependencies.

## Coding Style & Naming Conventions
Target Python 3.11 with four-space indentation. Use `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants (see `agents/constants.py`). Favour async workflows; wrap blocking calls with `asyncio.to_thread`. Add type hints and concise docstrings where intent is not obvious, and log via `agents.logging_utils.setup_logging()` to retain structured, colorised output.

## Testing Guidelines
Pytest powers the suite in `tests/`. Name new files `test_<feature>.py` and mirror the module layout. When extending Temporal workflows or MCP tools, mock external services and verify deterministic behaviour. Guard judge scoring, risk checks, and ledger math with targeted unit tests before wiring them into agent clients.

## Commit & Pull Request Guidelines
Write commits in the present tense and keep subjects under ~72 characters, following the succinct style in the existing history (`update readme and move CAN variables to constants.py`). PRs should include a clear summary, any linked issues, configuration changes, and relevant screenshots or log snippets. Always run `uv run pytest` (and `uv run ./run_stack.sh` if the change touches orchestration) before requesting review.

## Environment & Secrets
Store `OPENAI_API_KEY`, Coinbase credentials, Temporal settings, and other secrets in a local `.env`. Update `.env.example` when adding variables and call out new requirements in PR descriptions. Never commit real keys or production endpoints.

## Agent Operating Principles
- Always begin substantial tasks with a lightweight written plan that breaks work into PR-sized chunks; confirm or adjust with reviewers before diving in.
- Close each chunk by proposing the exact git commit message you intend to use and pause for human review/approval before proceeding.
- When issues persist after an initial fix, proactively suggest concrete workarounds (toggling flags, clearing caches, alternative flows, etc.) instead of retrying the same approach.
- Log unexpected fallbacks or degraded modes and surface them explicitly to downstream operators so they can decide whether to continue.
- Instrument every LLM call via Langfuse (`agents/langfuse_utils.py`) so audit trails capture prompts, responses, and token usage. Do not add new OpenAI clients without wrapping them in Langfuse spans.
