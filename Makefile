.PHONY: help setup test quick clean reauth fetch-training fetch-inbox evaluate dry-run classify watch watch-pubsub embed \
       service-install service-uninstall service-start service-stop service-status service-logs \
       gcp-create gcp-slim gcp-deploy gcp-destroy gcp-start gcp-stop gcp-restart gcp-status gcp-logs gcp-ssh

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

reauth: ## Re-authenticate with Gmail (opens browser for OAuth consent)
	rm -f credentials/token.json
	uv run python -c "from gmail_classifier.auth import get_credentials; get_credentials()"

fetch-training: ## Fetch labeled messages from Gmail (see docs/gmail-setup.md)
	uv run python scripts/fetch_training_data.py

fetch-inbox: ## Fetch recent inbox messages for dry-run classification
	uv run python scripts/fetch_inbox.py

evaluate: ## Run cross-validation evaluation on stored messages
	uv run python scripts/train_and_evaluate.py

dry-run: ## Classify inbox messages without modifying Gmail
	uv run python scripts/dry_run.py --exclude-labels XLC XLE XLCap

classify: ## Classify new inbox messages and apply labels (once)
	uv run python scripts/classify_and_label.py --once --exclude-labels XLC XLE XLCap

watch: ## Run classifier in a loop every 5 minutes (poll mode)
	uv run python scripts/classify_and_label.py --exclude-labels XLC XLE XLCap

watch-pubsub: ## Run classifier with Pub/Sub push notifications
	uv run python scripts/classify_and_label.py --mode pubsub --exclude-labels XLC XLE XLCap

embed: ## Pre-compute embedding cache (for fast startup on GCP)
	uv run python scripts/build_embeddings.py --exclude-labels XLC XLE XLCap

clean: ## Remove build artifacts and virtual environment
	rm -rf .venv __pycache__ src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# --- Service (macOS launchd) ---

service-install: ## Install macOS launchd service (does not auto-start)
	@scripts/service-install.sh

service-uninstall: ## Uninstall macOS launchd service
	@scripts/service-uninstall.sh

service-start: ## Start the launchd service
	@"$$HOME/bin/gmail-classifierctl" start

service-stop: ## Stop the launchd service
	@"$$HOME/bin/gmail-classifierctl" stop

service-status: ## Show launchd service status
	@"$$HOME/bin/gmail-classifierctl" status

service-logs: ## Tail the service log
	@"$$HOME/bin/gmail-classifierctl" logs

# --- GCP Deployment (e2-micro VM) ---

GCP_PROJECT  := classy-498012
GCP_INSTANCE := gmail-classifier
GCP_ZONE     := us-central1-a

gcp-create: ## Create GCP e2-micro VM
	@scripts/gcp-create.sh

gcp-slim: ## Disable non-essential GCP agents to free RAM/CPU
	@scripts/gcp-slim.sh

gcp-deploy: ## Deploy code, data, and credentials to GCP VM
	@scripts/gcp-deploy.sh

gcp-destroy: ## Delete the GCP VM (interactive confirmation)
	@scripts/gcp-destroy.sh

gcp-start: ## Start the classifier service on GCP
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo systemctl start gmail-classifier"

gcp-stop: ## Stop the classifier service on GCP
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo systemctl stop gmail-classifier"

gcp-restart: ## Restart the classifier service on GCP
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo systemctl restart gmail-classifier"

gcp-status: ## Show service status on GCP
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo systemctl status gmail-classifier"

gcp-logs: ## Tail service logs on GCP (last 20 + follow)
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo journalctl -u gmail-classifier -n 20 -f"

gcp-ssh: ## Open SSH session to the GCP VM
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE)
