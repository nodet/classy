.PHONY: help setup test quick clean reauth fetch-training fetch-inbox evaluate experiment-sample-size dry-run classify watch watch-pubsub embed reset-state \
       service-install service-uninstall service-start service-stop service-restart service-status service-logs \
       gcp-create gcp-slim gcp-deploy gcp-deploy-legacy gcp-deploy-state gcp-destroy gcp-start gcp-stop gcp-restart gcp-status gcp-state-status gcp-reset-state gcp-test-alert gcp-logs gcp-ssh

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

experiment-sample-size: ## Learning curve: how many messages/label are needed
	uv run python scripts/experiment_sample_size.py

dry-run: ## Classify inbox messages without modifying Gmail
	uv run python scripts/dry_run.py

classify: ## Classify new inbox messages and apply labels (once)
	uv run python scripts/classify_and_label.py --once

watch: ## Run classifier in a loop every 5 minutes (poll mode)
	uv run python scripts/classify_and_label.py

watch-pubsub: ## Run classifier with Pub/Sub push notifications
	uv run python scripts/classify_and_label.py --mode pubsub

embed: ## Pre-compute embedding cache (for fast startup on GCP)
	uv run python scripts/build_embeddings.py

clean: ## Remove build artifacts and virtual environment
	rm -rf .venv __pycache__ src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

reset-state: ## Delete the local state backend DB (next boot bootstraps fresh)
	@# Backend-scoped: removes ONLY state.db + its SQLite sidecars (and the
	@# rebuild file), never the legacy DBs. The escape hatch "just in case".
	rm -f data/state.db data/state.db-wal data/state.db-shm data/state.db-journal \
	      data/state.rebuild.db data/state.rebuild.db-wal data/state.rebuild.db-shm data/state.rebuild.db-journal
	@echo "Local state.db reset. Next 'state' boot bootstraps from Gmail."

# --- Service (macOS launchd) ---

service-install: ## Install macOS launchd service (does not auto-start)
	@scripts/service-install.sh

service-uninstall: ## Uninstall macOS launchd service
	@scripts/service-uninstall.sh

service-start: ## Start the launchd service
	@"$$HOME/bin/gmail-classifierctl" start

service-stop: ## Stop the launchd service
	@"$$HOME/bin/gmail-classifierctl" stop

service-restart: ## Restart the launchd service (atomic; avoids stop+start race)
	@"$$HOME/bin/gmail-classifierctl" restart

service-status: ## Show launchd service status
	@"$$HOME/bin/gmail-classifierctl" status

service-logs: ## Tail the service log
	@"$$HOME/bin/gmail-classifierctl" logs

# --- GCP Deployment (e2-micro VM) ---

GCP_PROJECT  := classy-498012
GCP_INSTANCE := gmail-classifier
GCP_ZONE     := us-central1-a
GCP_INSTALL_DIR := /opt/gmail-classifier

gcp-create: ## Create GCP e2-micro VM
	@scripts/gcp-create.sh

gcp-slim: ## Disable non-essential GCP agents to free RAM/CPU
	@scripts/gcp-slim.sh

gcp-deploy: gcp-deploy-legacy ## Deploy (alias for gcp-deploy-legacy; the historical default)

gcp-deploy-legacy: ## Deploy the legacy backend (code + 3 DBs + credentials)
	@scripts/gcp-deploy.sh legacy

gcp-deploy-state: ## Deploy the state backend (code + credentials only; VM bootstraps from Gmail)
	@scripts/gcp-deploy.sh state

gcp-destroy: ## Delete the GCP VM (interactive confirmation)
	@scripts/gcp-destroy.sh

gcp-start: ## Start the classifier service on GCP
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo systemctl start gmail-classifier"

gcp-stop: ## Stop the classifier service on GCP
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo systemctl stop gmail-classifier"

gcp-restart: ## Restart the classifier service on GCP
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo systemctl restart gmail-classifier"

gcp-status: ## Show service status on GCP (incl. deployed code version)
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="echo \"Deployed version: \$$(cat /opt/gmail-classifier/.deployed_version 2>/dev/null || echo unknown)\"; echo; sudo systemctl status gmail-classifier"

gcp-state-status: ## Show the state backend's state.db report on GCP (no service impact)
	@# Reads state.db read-only via --report (no Gmail call, no model load), run
	@# as the service user so it can read the file under the strict install dir.
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo -u gmail-classifier -H bash -c 'cd $(GCP_INSTALL_DIR) && \$$HOME/.local/bin/uv run --locked -- python scripts/classify_and_label.py --report'"

gcp-test-alert: ## Send a test crash-alert email from the VM (verifies auth + send path)
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo -u gmail-classifier -H bash -c 'cd $(GCP_INSTALL_DIR) && \$$HOME/.local/bin/uv run --locked -- python scripts/classify_and_label.py --test-alert'"

gcp-reset-state: ## Delete the VM's state.db and leave the service stopped
	@# Backend-scoped, mirrors `make reset-state`: stop, then remove state.db +
	@# sidecars ONLY (legacy DBs untouched). Leaves the service STOPPED -- like
	@# gcp-stop -- so the operator starts it deliberately (make gcp-start); the
	@# next boot then bootstraps fresh from Gmail.
	@# Fail-closed with && (not ;): if the stop fails we do NOT delete the DB, so
	@# a delete never races a live writer. A manual `systemctl stop` is a clean
	@# stop, so the unit's Restart=on-failure does not re-launch it.
	@# rm -v names each file actually removed (nothing for absent sidecars), and
	@# the trailing echo confirms the service is left stopped -- so the log shows
	@# exactly what the reset did, not just that the command ran.
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo systemctl stop gmail-classifier && sudo rm -vf $(GCP_INSTALL_DIR)/data/state.db $(GCP_INSTALL_DIR)/data/state.db-wal $(GCP_INSTALL_DIR)/data/state.db-shm $(GCP_INSTALL_DIR)/data/state.db-journal $(GCP_INSTALL_DIR)/data/state.rebuild.db $(GCP_INSTALL_DIR)/data/state.rebuild.db-wal $(GCP_INSTALL_DIR)/data/state.rebuild.db-shm $(GCP_INSTALL_DIR)/data/state.rebuild.db-journal && echo \"state.db reset; service is now \$$(systemctl is-active gmail-classifier || true). Run 'make gcp-start' to bootstrap fresh.\""

gcp-logs: ## Tail service logs on GCP (last 20 + follow)
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE) --command="sudo journalctl -u gmail-classifier -n 20 -f"

gcp-ssh: ## Open SSH session to the GCP VM
	@gcloud compute ssh $(GCP_INSTANCE) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE)
