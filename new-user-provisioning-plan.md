# Plan: New-user provisioning — make "clone and deploy" true for someone who isn't the author

Date: 2026-06-30 (revised 2026-07-07)
Status: proposed

## Context

The service now has a single storage backend (`state`): one derived `state.db`
that bootstraps from Gmail on first run. A fresh deploy is "code + credentials
-> start, the VM bootstraps from Gmail." No local training databases need to be
built or uploaded.

That removes one axis of install friction, but is **not sufficient** for a *new*
user (someone other than the author) to install the service. What remains is
provisioning and configuration friction:

- The GCP project with Pub/Sub topic/subscription/IAM was created by hand, and
  the identifiers are **hardcoded to the author's account**.
- `credentials/` must be populated with a working OAuth client + token.
- `config.toml` reflects the author's mailbox.
- The OAuth setup documentation is easy to get subtly wrong: the service needs
  exact scopes, and Google's Testing-mode refresh-token expiry can surprise
  users on a long-running VM.

## The core problem

The *code* is wired to the author's GCP project `classy-498012` in several
places, and the GCP-side resources it talks to must be created by hand. A new
user cannot clone and deploy; they must find-and-replace an opaque project id
across source and shell scripts, manually create cloud resources, and follow
OAuth setup prose that is currently too easy to misapply.

Hardcoded author-specific identifiers found today:

| Location | What is hardcoded |
|---|---|
| `scripts/classify_and_label.py:33-34` | `PUBSUB_TOPIC` / `PUBSUB_SUBSCRIPTION` (full `projects/classy-498012/...` paths) |
| `Makefile:67` | `GCP_PROJECT := classy-498012` |
| `scripts/gcp-create.sh:6`, `gcp-deploy.sh:6`, `gcp-destroy.sh:6` | `GCP_PROJECT="classy-498012"` |
| `README.md` | literal `classy-498012` in setup steps |
| `tests/test_pubsub.py` | the subscription path (test fixtures) |

## Approach

Four pieces, ranked by how much they decide whether a stranger can run this at
all. Items 1-2 are the gate; 3-4 are quality-of-life.

### 1. De-hardcode the GCP project + automate GCP/Pub/Sub provisioning (highest leverage)

**Config.** Extend `config.toml` with a `[gcp]` section and read it through the
existing `config.py` loader (which already exists for `[labels].excluded`). New
keys:

```toml
[gcp]
project   = ""              # project id, not display name/number; required for deploy
zone      = "us-central1-a"  # keep default in an e2-micro free-tier region
instance  = "gmail-classifier"
topic     = "gmail-notifications"          # short name; full path derived
subscription = "gmail-notifications-sub"   # short name; full path derived
```

- Add `config.py` accessors: `gcp_project()`, `gcp_zone()`, `gcp_instance()`,
  `pubsub_topic_path()`, `pubsub_subscription_path()`, and `require_gcp_config()`.
- Keep the empty-string-means-unset convention so local dev/test paths (which
  never touch GCP) don't trip over unset GCP keys. The service's real mode must
  call `require_gcp_config()` and fail with one clear message naming the missing
  key.
- `scripts/classify_and_label.py` — replace the `PUBSUB_TOPIC` /
  `PUBSUB_SUBSCRIPTION` module constants with calls to the config accessors.
- `Makefile` + `scripts/gcp-*.sh` — source `project`/`zone`/`instance` from
  config instead of the literal. Simplest: a tiny `scripts/gcp-env.sh` that uses
  only stdlib Python (`tomllib`; no project dependencies beyond Python 3.11) and
  exports the vars the other scripts already expect.
- `tests/test_pubsub.py` — drive the expected subscription path from the config
  accessor (or a fixture-injected value) rather than asserting the author's
  literal string.

**Provisioning.** Add an idempotent `make gcp-bootstrap` target that, given the
configured project, performs the steps the author once did by hand:

- enable the Gmail API, Pub/Sub API, and Compute Engine API on the project,
- create the topic (`topic` from config) if absent,
- create a **never-expiring pull subscription** (`subscription` from config) if
  absent,
- grant `gmail-api-push@system.gserviceaccount.com` the **Pub/Sub Publisher**
  role on the topic (the binding Gmail's `watch()` requires).

Each step is check-then-create so re-running is safe. The target should **not**
create the GCP project or link billing; project creation, billing, and choosing a
globally-unique project id stay manual because those depend on the user's Google
account. The target should detect missing billing / disabled project access when
possible and print the exact manual step to perform.

This is the single change that turns "find-and-replace the author's project id
and click through the console" into "set your project in `config.toml`, run one
target."

### 2. `make doctor` preflight (catches the silent-misconfig failures)

OAuth client setup cannot be fully automated — Google requires console clicks.
But the *failure modes* can be made loud and actionable. Add a `make doctor` /
`scripts/doctor.py` that checks the deploy/GCP preflight and, for each failure,
prints the specific fix command or setup step. Keep a `make doctor-local` mode
for dev/test that skips GCP resource checks:

- Local prerequisites: `uv` exists, Python >= 3.11, dependencies installed.
- `gcloud` is installed, authenticated, and pointed at the configured project.
- Billing is enabled or at least detectable as required for Compute Engine.
- `credentials/client_secret.json` exists and is a **desktop-app** OAuth client
  (not web). If the file exposes a `project_id`, it matches `[gcp].project`.
- A token is present and refreshable (or: "run `make reauth`").
- Granted scopes match the code's required scopes (`gmail.modify` + `pubsub`).
- `config.toml` has a non-empty `[gcp].project` and valid `zone`/`instance`.
- The configured Pub/Sub **topic and subscription exist** and the
  `gmail-api-push` publisher IAM binding is present.
- (Optional) if a `state.db` is present, its `gmail_account_id` matches the
  authorized account and `last_processed_at` is sane.
- The Gmail account has at least one trainable user label after exclusions.

`doctor` is read-only (no mutations) and exits non-zero on hard failures, so it
doubles as a CI/pre-deploy gate.

Important OAuth behavior to surface: for personal Gmail accounts using an
External app in Google's **Testing** publishing status, refresh tokens expire
after seven days. The setup docs should tell users what that means before they
put the service on a VM:

- Workspace users can often use an **Internal** app.
- Personal Gmail users should understand the tradeoff between Testing mode,
  In-production/unverified mode, the unverified-app warning, and Google's
  100-user cap. For a single-user self-hosted install, the practical path is
  usually "the user owns their own project and authorizes their own app."
- Changing OAuth scopes or publishing mode means re-authorizing; the fastest
  recovery is `make reauth`.

### 3. Startup legibility

The cold bootstrap is a 10-20 min **read-only** warmup during which nothing is
labeled. For a new user that long silent stretch reads as "broken." Make the
service explain itself:

- On the cold path, log a startup banner: bootstrapping, **read-only until ~N
  examples/label**, will not touch existing mail.
- Log the **discovered label set and the effective exclusions** once the cold
  path's `list_user_labels()` runs.
- If zero trainable labels are discovered, fail clearly instead of starting a
  permanently idle service.
- If labels exist but too few have enough examples, keep running but log which
  labels are below the minimum and what the user can do.
- Surface `make gcp-state-status` prominently in the README as the canonical "is
  it working yet?" command.

### 4. Ship neutral config + docs/README pass

- `config.toml` currently ships with the **author's** labels in
  `[labels].excluded`. Replace with an **empty** `excluded = []` plus a comment
  explaining what to put there, and the empty `[gcp]` section from item 1.
- README pass: replace literal `classy-498012` occurrences with the
  configured-value placeholder, fold in `make gcp-bootstrap` and `make doctor`
  steps, and reframe Quick-start so config-and-provision steps precede deploy.
- `docs/gmail-setup.md` pass: update OAuth consent-screen scope instructions to
  match `auth.py` (`gmail.modify` + `pubsub`). Tell users to create the OAuth
  client in the configured GCP project. Add troubleshooting for
  `insufficient authentication scopes` / `invalid_grant`.
- Frame setup docs around the **one supported way to operate the service: the
  always-on GCP deploy.** Running locally is documented only as a **test/debug
  path** in a clearly-labeled subsection.
- Add a short cost/safety note: `e2-micro` is free-tier-eligible in US zones by
  default, but users must still have billing enabled.

## Out of scope

- Fully automating GCP project creation or billing linking — project ids are
  global, billing setup is account-specific.
- Fully automating the OAuth consent-screen / client creation — Google requires
  manual console steps; `doctor` verifies the result instead.
- OAuth app verification for a public multi-user app. This remains a
  single-user/self-hosted tool.
- A web/installer UI, multi-user support, or secret-manager integration.

## Verification

- Config: `gcp_project()` / `pubsub_*_path()` accessors round-trip from a
  fixture `config.toml`; empty/unset project yields the documented "unset"
  behavior; `require_gcp_config()` fails with actionable messages;
  `classify_and_label.py` reads the paths from config (no remaining
  `classy-498012` literal in `src/` or `scripts/`).
- Shell integration: `scripts/gcp-env.sh` works with only Python 3.11 stdlib,
  handles missing config cleanly, and exports the same values the Makefile and
  `gcp-*.sh` scripts use.
- Provisioning: `make gcp-bootstrap` is idempotent (second run is a no-op, exits
  0) — testable against the `gcloud` calls with a recording fake / dry-run flag.
- `doctor`: with a deliberately broken setup (missing `uv`, old read-only token,
  missing `client_secret`, client secret from wrong project, empty project,
  absent subscription) it exits non-zero and names the fix; with a good setup it
  exits 0. `doctor-local` skips GCP resource checks. Pure-function checks
  unit-tested with fakes; no live cloud calls in the test path.
- OAuth docs: setup instructions list the same scopes as `auth.py`.
- Legibility: cold-path startup logs the read-only banner and the discovered
  label set + exclusions. Empty trainable-label set fails clearly; too-few-
  examples state logs a warning.
- End-to-end (manual, documented): a second GCP project + a different Google
  account can go from `git clone` to a running, bootstrapping service with only
  `config.toml` edits and the documented targets — no source edits, no author-
  specific labels or project ids.
