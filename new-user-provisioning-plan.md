# Plan: New-user provisioning — make "clone and deploy" true for someone who isn't the author

Date: 2026-06-30
Status: proposed (follow-on to the stateless-bootstrap plan)

## Relationship to the stateless-bootstrap plan

This plan **leans on `stateless-bootstrap-plan.md` being implemented first** and does not
duplicate its scope. The bootstrap plan removes one axis of install friction — the local
`training.db`/`inbox_sample.db`/`embeddings.db` build and the DB upload — so that a fresh
deploy is "code + credentials → start, the VM bootstraps from Gmail." That is necessary but
**not sufficient** for a *new* user (someone other than the author) to install the service.

What remains is provisioning and configuration friction the bootstrap plan deliberately
leaves alone:

- The bootstrap plan assumes a GCP project with Pub/Sub topic/subscription/IAM **already
  exist**. Today they exist only because the author created them by hand, and their
  identifiers are **hardcoded to the author's account**.
- It assumes `credentials/` is already populated with a working OAuth client + token.
- It assumes `config.toml` reflects the user's mailbox.

So this plan picks up exactly where the bootstrap plan stops: turning an author-specific,
manually-provisioned setup into something a stranger can stand up. Sequencing: implement the
bootstrap plan, then this. Several items here (the startup-legibility logging, the
`gcp-state-status` surfacing) reference modules and behavior the bootstrap plan introduces
(`watch()`-first cold path, discovered-label set, maturity gate), so they cannot land before it.

## The core problem

After the bootstrap plan ships, deploy is "code + credentials → start" — but the *code* is
wired to the author's GCP project `classy-498012` in at least eight places, and the GCP-side
resources it talks to must be created by hand. A new user cannot clone and deploy; they must
find-and-replace an opaque project id across source and shell scripts, and manually create
cloud resources with no guidance beyond prose in `docs/gmail-setup.md`.

Hardcoded author-specific identifiers found today:

| Location | What is hardcoded |
|---|---|
| `scripts/classify_and_label.py:33-34` | `PUBSUB_TOPIC` / `PUBSUB_SUBSCRIPTION` (full `projects/classy-498012/...` paths) |
| `Makefile:84` | `GCP_PROJECT := classy-498012` |
| `scripts/gcp-create.sh:6`, `gcp-deploy.sh:6`, `gcp-destroy.sh:6`, `gcp-slim.sh:6` | `GCP_PROJECT="classy-498012"` |
| `README.md:168,175,225` | literal `classy-498012` in setup steps |
| `tests/test_pubsub.py` | the subscription path (test fixtures — update to match the config-driven value) |

## Approach

Four pieces, ranked by how much they decide whether a stranger can run this at all. Items 1–2
are the gate; 3–4 are quality-of-life that compound with the bootstrap plan's safety model.

### 1. De-hardcode the GCP project + automate Pub/Sub provisioning (highest leverage)

**Config.** Extend `config.toml` with a `[gcp]` section and read it through the existing
`config.py` loader (which already exists for `[labels].excluded` and is the designated
"tune without touching code" surface). New keys:

```toml
[gcp]
project   = ""              # GCP project id (required for GCP deploy)
zone      = "us-central1-a"
instance  = "gmail-classifier"
topic     = "gmail-notifications"          # short name; full path derived
subscription = "gmail-notifications-sub"   # short name; full path derived
```

- Add `config.py` accessors: `gcp_project()`, `gcp_zone()`, `gcp_instance()`,
  `pubsub_topic_path()`, `pubsub_subscription_path()` (the latter two derive the full
  `projects/<project>/topics/<topic>` form). Keep the empty-string-means-unset convention so
  local-only / non-GCP use never trips over it.
- `scripts/classify_and_label.py` — replace the `PUBSUB_TOPIC` / `PUBSUB_SUBSCRIPTION` module
  constants with calls to the config accessors. (These are read at startup, so no hot-path
  cost.)
- `Makefile` + `scripts/gcp-*.sh` — source `project`/`zone`/`instance` from config instead of
  the literal. Simplest: a tiny `scripts/gcp-env.sh` that shells out to
  `python -c "from gmail_classifier.config import ..."` (or parses the toml) and exports the
  vars the other scripts already expect, so the eight call sites collapse to one source of
  truth. The Makefile `GCP_PROJECT`/`GCP_ZONE`/`GCP_INSTANCE` become `$(shell ...)` reads of
  the same.
- `tests/test_pubsub.py` — drive the expected subscription path from the config accessor (or a
  fixture-injected value) rather than asserting the author's literal string.

**Provisioning.** Add an idempotent `make gcp-bootstrap-pubsub` target (and document the raw
`gcloud` block as a fallback) that, given the configured project, performs the steps the author
once did by hand:

- enable the Gmail API and Pub/Sub API on the project,
- create the topic (`topic` from config) if absent,
- create a **never-expiring pull subscription** (`subscription` from config) if absent,
- grant `gmail-api-push@system.gserviceaccount.com` the **Pub/Sub Publisher** role on the topic
  (the binding Gmail's `watch()` requires to publish notifications).

Each step is check-then-create so re-running is safe. This is the single change that turns
"find-and-replace the author's project id and read a wiki to click through the console" into
"set your project in `config.toml`, run one target."

### 2. `make doctor` preflight (catches the silent-misconfig failures)

OAuth client setup cannot be automated — Google requires console clicks (consent screen,
desktop-app client type, scopes, test-user allow-listing). But the *failure modes* can be made
loud and actionable instead of a confusing runtime stack trace. Add a `make doctor` /
`scripts/doctor.py` that checks, and for each failure prints the specific fix command:

- `credentials/client_secret.json` exists and is a **desktop-app** OAuth client (not web).
- A token is present and refreshable (or: "run `make fetch-training` once to do the OAuth flow").
- The granted scopes cover what the service needs (read + modify).
- `config.toml` has a non-empty `[gcp].project`.
- The configured Pub/Sub **topic and subscription exist** and are reachable, and the
  `gmail-api-push` publisher IAM binding is present.
- `gcloud` is installed and authenticated to the configured project.

`doctor` is read-only (no mutations) and exits non-zero on any failure, so it doubles as a
CI/pre-deploy gate. It is the highest-value safety net for the OAuth + IAM steps that item 1
can't fully automate.

### 3. Startup legibility (leans directly on the bootstrap plan)

The bootstrap plan makes first boot a 10–20 min **read-only** warmup during which, by design,
nothing is labeled. For a new user that long silent stretch reads as "broken." Make the
service explain itself — these hook into bootstrap-plan code paths, so they land *after* it:

- On the cold path, log a startup banner: bootstrapping, **read-only until ~N examples/label**,
  will not touch existing mail, ordering is round-robin. (Uses the maturity targets the
  bootstrap plan defines.)
- Log the **discovered label set and the effective exclusions** once the cold path's
  `list_user_labels()` runs, so the user sees what the service will and won't classify *before*
  it acts. The bootstrap plan already discovers labels at startup; this just surfaces them.
- Surface `make gcp-state-status` (introduced by the bootstrap plan) prominently in the README
  as the canonical "is it working yet?" command — bootstrap progress, per-label counts, maturity.

### 4. Ship a neutral `config.toml` + README pass

- `config.toml` currently ships with the **author's** labels (`XLC`, `XLE`, `XLCap`) in
  `[labels].excluded` (`config.toml:11`). Replace with an **empty** `excluded = []` plus a
  comment explaining what to put there, and the empty `[gcp]` section from item 1. A new user's
  first edit should be obvious and theirs, not a cleanup of the author's leftovers.
- README pass (note the bootstrap plan *already* schedules a README pass for the deploy-steps
  changes — coordinate, don't double-edit): replace the literal `classy-498012` occurrences
  (`README.md:168,175,225`) with the configured-value placeholder, fold in the
  `make gcp-bootstrap-pubsub` and `make doctor` steps, and reframe Quick-start so the
  config-and-provision steps precede deploy.

## Out of scope

- Anything the bootstrap plan already owns (derived-state persistence, the deploy tarball
  contents, the `state.db` lifecycle, `gcp-reset-state`, the README deploy-steps rewrite for
  "no DB upload").
- Fully automating the OAuth consent-screen / client creation — Google requires manual console
  steps; `doctor` (item 2) verifies the result instead.
- A web/installer UI, multi-user support, or secret-manager integration — single-user,
  single-project tool; out of proportion.

## Verification

- Config: `gcp_project()` / `pubsub_*_path()` accessors round-trip from a fixture `config.toml`;
  empty/unset project yields the documented "unset" behavior; `classify_and_label.py` reads the
  paths from config (no remaining `classy-498012` literal in `src/` or `scripts/`). A test
  asserts the repo contains no hardcoded `classy-498012` outside `config.toml`/docs.
- Provisioning: `make gcp-bootstrap-pubsub` is idempotent (second run is a no-op, exits 0) —
  testable against the `gcloud` calls with a recording fake / dry-run flag.
- `doctor`: with a deliberately broken setup (missing client_secret, empty project, absent
  subscription) it exits non-zero and names the fix for each; with a good setup it exits 0.
  Pure-function checks unit-tested with fakes; no live cloud calls in the test path.
- Legibility: cold-path startup logs the read-only banner and the discovered label set +
  exclusions (assertable in the bootstrap dispatch tests once that code exists).
- End-to-end (manual, documented): a second GCP project + a different Google account can go
  from `git clone` to a running, bootstrapping service using only `config.toml` edits and the
  documented targets — no source edits.
