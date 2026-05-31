.PHONY: help setup test quick clean fetch-training fetch-inbox evaluate dry-run classify

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Typical workflow:"
	@echo "  1. make classify         — label new inbox messages"
	@echo "  2. Fix mistakes manually (move to correct label or back to inbox)"
	@echo "  3. make fetch-training   — pick up corrected/new labels"
	@echo "  4. make fetch-inbox      — update skip pool (inbox = don't label)"
	@echo ""
	@echo "Other:"
	@echo "  make evaluate            — check precision/coverage"
	@echo "  make dry-run             — preview without modifying Gmail"

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
	uv run python scripts/dry_run.py --exclude-labels XLC XLE XLCap

classify: ## Classify new inbox messages and apply labels
	uv run python scripts/classify_and_label.py --exclude-labels XLC XLE XLCap

clean: ## Remove build artifacts and virtual environment
	rm -rf .venv __pycache__ src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
