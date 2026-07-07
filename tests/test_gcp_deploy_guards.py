"""Guard-scope tests for scripts/gcp-deploy.sh.

The deploy script's preflight guards are pure local-filesystem checks that run
*before* the first `gcloud` call, so we can drive them deterministically without
a VM. These pin:

- deploy succeeds (past the guards) with only credentials present;
- credentials are required.

We copy the script into a temp "project" whose layout it derives from its own
location (`dirname $0/..`), and put a stub `gcloud` on PATH that exits non-zero.
A guard rejection exits with its message before ever calling gcloud; a guard
*pass* proceeds to the first `vm_run "id ..."`, whose stub failure sends the
script into its "First deploy detected" branch and then aborts under `set -e`.
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
    gcloud that always fails."""
    proj = tmp_path / "proj"
    (proj / "scripts").mkdir(parents=True)
    shutil.copy(DEPLOY_SRC, proj / "scripts" / "gcp-deploy.sh")

    for rel, content in files.items():
        p = proj / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content or "x")

    binp = tmp_path / "bin"
    binp.mkdir()
    stub = binp / "gcloud"
    stub.write_text("#!/bin/bash\nexit 1\n")
    stub.chmod(0o755)

    # stub git so `git describe` doesn't fail when run outside a repo
    git_stub = binp / "git"
    git_stub.write_text("#!/bin/bash\necho unknown\n")
    git_stub.chmod(0o755)

    return proj, binp


def _run(proj, binp):
    env = dict(os.environ, PATH=f"{binp}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", str(proj / "scripts" / "gcp-deploy.sh")],
        capture_output=True, text=True, env=env,
    )


CREDS = {
    "credentials/token.json": "{}",
    "credentials/client_secret.json": "{}",
}


def test_passes_guards_with_credentials(tmp_path):
    proj, binp = _make_project(tmp_path, files=CREDS)
    r = _run(proj, binp)
    # Got past every guard and advanced to the VM setup branch
    assert "First deploy detected" in r.stdout


def test_credentials_required(tmp_path):
    proj, binp = _make_project(tmp_path, files={})
    r = _run(proj, binp)
    assert r.returncode != 0
    assert "token.json missing" in r.stdout
