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
- The existing OAuth setup documentation is still easy to get subtly wrong: the service needs
  the exact scopes the code asks for, and a long-running service should not leave users
  surprised by Google's Testing-mode refresh-token expiry.

So this plan picks up exactly where the bootstrap plan stops: turning an author-specific,
manually-provisioned setup into something a stranger can stand up. Sequencing: implement the
bootstrap plan, then this. Several items here (the startup-legibility logging, the
`gcp-state-status` surfacing) reference modules and behavior the bootstrap plan introduces
(`watch()`-first cold path, discovered-label set, maturity gate), so they cannot land before it.

## The core problem

After the bootstrap plan ships, deploy is "code + credentials → start" — but the *code* is
wired to the author's GCP project `classy-498012` in at least eight places, and the GCP-side
resources it talks to must be created by hand. A new user cannot clone and deploy; they must
find-and-replace an opaque project id across source and shell scripts, manually create cloud
resources, and follow OAuth setup prose that is currently too easy to misapply.

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

### 1. De-hardcode the GCP project + automate GCP/Pub/Sub provisioning (highest leverage)

**Config.** Extend `config.toml` with a `[gcp]` section and read it through the existing
`config.py` loader (which already exists for `[labels].excluded` and is the designated
"tune without touching code" surface). New keys:

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
- Keep the empty-string-means-unset convention so local-only / non-GCP use never trips over it.
  Accessors may return `""` for unset values, but any GCP command or Pub/Sub mode should call
  `require_gcp_config()` and fail with one clear message that names the missing key.
- `scripts/classify_and_label.py` — replace the `PUBSUB_TOPIC` / `PUBSUB_SUBSCRIPTION` module
  constants with calls to the config accessors. (These are read at startup, so no hot-path
  cost.)
- `Makefile` + `scripts/gcp-*.sh` — source `project`/`zone`/`instance` from config instead of
  the literal. Simplest: a tiny `scripts/gcp-env.sh` that uses only stdlib Python (`tomllib`;
  no project dependencies beyond Python 3.11) and exports the vars the other scripts already
  expect, so the eight call sites collapse to one source of truth. The Makefile
  `GCP_PROJECT`/`GCP_ZONE`/`GCP_INSTANCE` become `$(shell ...)` reads of the same source.
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

Broaden the user-facing command slightly so README can say one thing:

```text
make gcp-bootstrap
```

`gcp-bootstrap` should call `gcp-bootstrap-pubsub` and also enable the Compute Engine API needed
by `make gcp-create`. It should **not** try to create the GCP project or link billing; project
creation, billing, and choosing a globally-unique project id stay manual because those depend
on the user's Google account and billing setup. The target should detect missing billing /
disabled project access when possible and print the exact manual step to perform.

Each step is check-then-create so re-running is safe. This is the single change that turns
"find-and-replace the author's project id and read a wiki to click through the console" into
"set your project in `config.toml`, run one target."

### 2. `make doctor` preflight (catches the silent-misconfig failures)

OAuth client setup cannot be fully automated — Google requires console clicks (consent screen,
desktop-app client type, scopes, publishing status / test-user allow-listing). But the
*failure modes* can be made loud and actionable instead of a confusing runtime stack trace.
Add a `make doctor` / `scripts/doctor.py` that checks, and for each failure prints the specific
fix command or setup step:

- Local prerequisites: `uv` exists, Python is ≥3.11, dependencies are installed or `make setup`
  is the next action.
- `gcloud` is installed, authenticated, and pointed at the configured project.
- Billing is enabled or at least detectable as required for Compute Engine; if the caller lacks
  billing permissions, print a warning rather than a false failure.
- `credentials/client_secret.json` exists and is a **desktop-app** OAuth client (not web).
- A token is present and refreshable (or: "run `make reauth` to do the OAuth flow").
- The granted scopes match the code's required scopes exactly enough for current behavior:
  Gmail modify plus Pub/Sub. A token produced under old docs that only requested
  `gmail.readonly` should fail loudly and tell the user to delete `credentials/token.json` or
  run `make reauth` after fixing the consent-screen scopes.
- `config.toml` has a non-empty `[gcp].project` for GCP mode, and the configured
  `zone`/`instance` are valid strings.
- The configured Pub/Sub **topic and subscription exist** and are reachable, and the
  `gmail-api-push` publisher IAM binding is present.
- The Gmail account has at least one trainable user label after exclusions; warn if most labels
  have fewer than the classifier's minimum example count.

`doctor` is read-only (no mutations) and exits non-zero on hard failures, so it doubles as a
CI/pre-deploy gate. OAuth publishing status may not be reliably inspectable from local files or
`gcloud`; in that case `doctor` should print a checklist warning instead of pretending to
verify it.

Important OAuth-doc behavior to surface: for personal Gmail accounts using an External app in
Google's **Testing** publishing status, refresh tokens can expire after seven days. The setup
docs should tell users what that means before they put the service on a VM:

- Workspace users can often use an **Internal** app.
- Personal Gmail users should understand the tradeoff between Testing mode,
  In-production/unverified mode, the unverified-app warning, and Google's 100-user cap. For a
  single-user self-hosted install, the practical path is usually "the user owns their own
  project and authorizes their own app," but the docs must be explicit about the token-lifetime
  consequence.
- Changing OAuth scopes or publishing mode means re-authorizing; the fastest recovery is
  `make reauth`.

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
- If zero trainable labels are discovered, fail clearly instead of starting a permanently idle
  service.
- If labels exist but too few have enough examples to become classifier-eligible, keep running
  but log the precise issue: which labels are below the minimum and what the user can do (label
  more examples, reduce exclusions, or accept low coverage).
- Surface `make gcp-state-status` (introduced by the bootstrap plan) prominently in the README
  as the canonical "is it working yet?" command — bootstrap progress, per-label counts,
  maturity, pending-new count, last history cursor.

### 4. Ship neutral config + docs/README pass

- `config.toml` currently ships with the **author's** labels (`XLC`, `XLE`, `XLCap`) in
  `[labels].excluded` (`config.toml:11`). Replace with an **empty** `excluded = []` plus a
  comment explaining what to put there, and the empty `[gcp]` section from item 1. A new user's
  first edit should be obvious and theirs, not a cleanup of the author's leftovers.
- Keep the implementation simple for now: a neutral tracked `config.toml` is acceptable. If
  local edits start creating merge friction later, move to `config.example.toml` + gitignored
  `config.toml` + `make init-config`; that is a follow-up, not required for the first new-user
  pass.
- README pass (note the bootstrap plan *already* schedules a README pass for the deploy-steps
  changes — coordinate, don't double-edit): replace the literal `classy-498012` occurrences
  (`README.md:168,175,225`) with the configured-value placeholder, fold in the
  `make gcp-bootstrap`, `make gcp-bootstrap-pubsub`, and `make doctor` steps, and reframe
  Quick-start so the config-and-provision steps precede deploy.
- `docs/gmail-setup.md` pass: update the OAuth consent-screen scope instructions to match
  `auth.py` (`gmail.modify` + `pubsub`), not the current read-only-only guidance. Add a
  troubleshooting entry for `insufficient authentication scopes` / `invalid_grant` that says
  when to run `make reauth`.
- Split the setup docs into two clear paths: **local dry run** and **always-on GCP service**.
  The local path should not make users think GCP is mandatory; the GCP path should not make
  users think local `training.db` / `inbox_sample.db` are deploy prerequisites after the
  bootstrap plan lands.
- Add a short cost/safety note: `e2-micro` should remain in a free-tier-eligible US zone by
  default, but users must still have billing enabled and are responsible for checking their own
  GCP billing page.

## Out of scope

- Anything the bootstrap plan already owns (derived-state persistence, the deploy tarball
  contents, the `state.db` lifecycle, `gcp-reset-state`, the README deploy-steps rewrite for
  "no DB upload").
- Fully automating GCP project creation or billing linking — project ids are global, billing
  setup is account-specific, and failures are easier to explain than to paper over.
- Fully automating the OAuth consent-screen / client creation — Google requires manual console
  steps; `doctor` verifies the result instead.
- OAuth app verification for a public multi-user app. This remains a single-user/self-hosted
  tool where each user can own their own project and OAuth client.
- A web/installer UI, multi-user support, or secret-manager integration — single-user,
  single-project tool; out of proportion.

## Verification

- Config: `gcp_project()` / `pubsub_*_path()` accessors round-trip from a fixture `config.toml`;
  empty/unset project yields the documented "unset" behavior; `require_gcp_config()` fails with
  actionable messages; `classify_and_label.py` reads the paths from config (no remaining
  `classy-498012` literal in `src/` or `scripts/`). A test asserts the repo contains no
  hardcoded `classy-498012` outside this plan file and the prior `*-plan.md` history (README
  and `docs/` are cleaned to the configured-value placeholder, not exempted).
- Shell integration: `scripts/gcp-env.sh` works with only Python 3.11 stdlib, handles missing
  config cleanly, and exports the same values the Makefile and `gcp-*.sh` scripts use.
- Provisioning: `make gcp-bootstrap-pubsub` is idempotent (second run is a no-op, exits 0) —
  testable against the `gcloud` calls with a recording fake / dry-run flag. `make
  gcp-bootstrap` also enables Compute Engine and leaves project creation/billing manual.
- `doctor`: with a deliberately broken setup (missing `uv`, old read-only-only token, missing
  `client_secret`, empty project, absent subscription) it exits non-zero and names the fix for
  each; with a good setup it exits 0. Pure-function checks unit-tested with fakes; no live
  cloud calls in the test path.
- OAuth docs: setup instructions list the same scopes as `auth.py`; a test or checklist
  prevents docs from drifting back to `gmail.readonly` only.
- Legibility: cold-path startup logs the read-only banner and the discovered label set +
  exclusions (assertable in the bootstrap dispatch tests once that code exists). Empty
  trainable-label set fails clearly; too-few-examples state logs a warning.
- End-to-end (manual, documented): a second GCP project + a different Google account can go
  from `git clone` to a running, bootstrapping service using only `config.toml` edits and the
  documented targets — no source edits, no local training DB upload, no author-specific labels
  or project ids.
