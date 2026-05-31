.PHONY: help setup test quick clean fetch-training fetch-inbox evaluate dry-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and set up dev environment
	uv sync --all-extras

test: ## Run all tests
	uv run pytest

quick: ## Run fast tests only (skip ML model loading)
	uv run pytest -m "not slow"

fetch-training: ## Fetch labeled messages from Gmail (see docs/gmail-setup.md)
	uv run python scripts/fetch_training_data.py

fetch-inbox: ## Fetch recent inbox messages for dry-run classification
	uv run python scripts/fetch_inbox.py

evaluate: ## Run cross-validation evaluation on stored messages
	uv run python scripts/train_and_evaluate.py

dry-run: ## Classify inbox messages without modifying Gmail
	uv run python scripts/dry_run.py

clean: ## Remove build artifacts and virtual environment
	rm -rf .venv __pycache__ src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
