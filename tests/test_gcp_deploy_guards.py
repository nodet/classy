"""Guard-scope tests for scripts/gcp-deploy.sh (Phase 7 deploy).

The deploy script's preflight guards are pure local-filesystem checks that run
*before* the first `gcloud` call, so we can drive them deterministically without
a VM. These pin the coexistence-invariant behavior the plan calls out under
"Unit -- backend isolation / State deploy guard scope":

- legacy still fails fast when its required DBs are absent;
- state succeeds (past the guards) with only credentials present, no legacy DBs;
- credentials are required by both backends;
- an unknown backend argument is rejected.

We copy the script into a temp "project" whose layout it derives from its own
location (`dirname $0/..`), and put a stub `gcloud` on PATH that exits non-zero.
A guard rejection exits with its message before ever calling gcloud; a guard
*pass* proceeds to the first `vm_run "id ..."`, whose stub failure sends the
script into its "First deploy detected" branch and then aborts under `set -e`.
So the presence of the guard message vs. "First deploy detected" tells us
exactly where the script stopped.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEPLOY_SRC = REPO / "scripts" / "gcp-deploy.sh"


def _make_project(tmp_path, *, files):
    """Build a temp project dir with a copy of the deploy script and a stub
    gcloud that always fails. ``files`` maps relative path -> contents (empty
    string still creates a non-empty file so `-s` guards pass)."""
    proj = tmp_path / "proj"
    (proj / "scripts").mkdir(parents=True)
    shutil.copy(DEPLOY_SRC, proj / "scripts" / "gcp-deploy.sh")

    for rel, content in files.items():
        p = proj / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content or "x")  # non-empty so `[[ -s ]]` is satisfied

    # Stub gcloud: always non-zero, so the first vm_run fails and we never touch
    # a real VM. Placed in a bin/ dir we prepend to PATH.
    binp = tmp_path / "bin"
    binp.mkdir()
    stub = binp / "gcloud"
    stub.write_text("#!/bin/bash\nexit 1\n")
    stub.chmod(0o755)
    return proj, binp


def _run(proj, binp, backend):
    env = dict(os.environ, PATH=f"{binp}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", str(proj / "scripts" / "gcp-deploy.sh"), backend],
        capture_output=True, text=True, env=env,
    )


CREDS = {
    "credentials/token.json": "{}",
    "credentials/client_secret.json": "{}",
}


def test_unknown_backend_rejected(tmp_path):
    proj, binp = _make_project(tmp_path, files=CREDS)
    r = _run(proj, binp, "bogus")
    assert r.returncode != 0
    assert "backend must be 'legacy' or 'state'" in r.stdout


def test_legacy_fails_fast_without_dbs(tmp_path):
    # Credentials present but no training.db -> legacy guard rejects.
    proj, binp = _make_project(tmp_path, files=CREDS)
    r = _run(proj, binp, "legacy")
    assert r.returncode != 0
    assert "training.db missing" in r.stdout
    # Rejected at the guard, never reached the VM setup path.
    assert "First deploy detected" not in r.stdout


def test_state_passes_guards_with_only_credentials(tmp_path):
    # No legacy DBs at all -- the state backend must still get past the guards
    # (its whole point is a code+credentials-only deploy).
    proj, binp = _make_project(tmp_path, files=CREDS)
    r = _run(proj, binp, "state")
    # Got past every guard: no DB-guard message, and it advanced to the VM setup
    # branch (where the stub gcloud then fails).
    assert "training.db missing" not in r.stdout
    assert "inbox_sample.db missing" not in r.stdout
    assert "embeddings.db missing" not in r.stdout
    assert "First deploy detected" in r.stdout
    assert "no DB upload" not in r.stdout  # aborts before the sync section


def test_credentials_required_by_both_backends(tmp_path):
    for backend in ("legacy", "state"):
        proj, binp = _make_project(tmp_path / backend, files={
            # Legacy DBs present, but no credentials.
            "data/training.db": "x",
            "data/inbox_sample.db": "x",
            "data/embeddings.db": "x",
        })
        r = _run(proj, binp, backend)
        assert r.returncode != 0
        assert "token.json missing" in r.stdout, backend
