# Plan: macOS launchd service install via Make

## Context

The project has a detailed guide (`mac_uv_launchd_service_plan.md`) describing how to run the classifier as a macOS launchd service, but all values are placeholders. The user wants a `make service-install` target that generates and installs the concrete scripts/plist with real values, plus a `make service-uninstall` to remove them. Additionally, SIGTERM handling is needed so launchd can stop the service gracefully.

## Files to modify

| File | Change |
|------|--------|
| `/workspace/Makefile` | Add service-install, service-uninstall, service-status, service-logs targets |
| `/workspace/scripts/classify_and_label.py` | Add SIGTERM handler + crash email alert |
| `/workspace/src/gmail_classifier/gmail_client.py` | Add `send_message()` method for crash alerts |
| `/workspace/README.md` | Add clone-to-running walkthrough |

## Generated files (at install time, on user's Mac)

- `~/bin/gmail-classifier-runner` — runner script
- `~/bin/gmail-classifierctl` — control script (start/stop/restart/reload/status/logs/rotate-log/enable/disable)
- `~/Library/LaunchAgents/com.xnodet.gmail-classifier.plist` — LaunchAgent

## 1. SIGTERM handling in classify_and_label.py

Add `import signal` at the top. Register a SIGTERM handler that raises `SystemExit` so the existing `try/except KeyboardInterrupt` catches it cleanly:

```python
import signal

def _sigterm_handler(signum, frame):
    raise SystemExit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print(f"\n{datetime.now().strftime('%H:%M:%S')} Stopped.")
        sys.exit(0)
```

This is simpler than a threading.Event approach — SIGTERM delivers during `time.sleep()` or gRPC `pull()` and unwinds the stack. The worst-case delay is the pull timeout (60s), but in practice Python signal delivery interrupts blocking calls. If needed, we can add `ExitTimeOut` to the plist (default is 20s for agents).

## 2. Makefile targets

### Variables

```makefile
SERVICE_LABEL   := com.xnodet.gmail-classifier
SERVICE_RUNNER  := $(HOME)/bin/gmail-classifier-runner
SERVICE_CTL     := $(HOME)/bin/gmail-classifierctl
SERVICE_PLIST   := $(HOME)/Library/LaunchAgents/$(SERVICE_LABEL).plist
SERVICE_LOG     := $(HOME)/Library/Logs/$(SERVICE_LABEL).log
EXCLUDE_LABELS  := XLC XLE XLCap
```

UV path detected at install time via `command -v uv`.

### `service-install`

1. Guard: fail if not macOS (`uname != Darwin`)
2. Guard: fail if `data/training.db` missing/empty ("Run 'make fetch-training' first")
3. Guard: fail if `data/inbox_sample.db` missing/empty ("Run 'make fetch-inbox' first")
4. Detect uv path, fail if not found
5. `mkdir -p ~/bin ~/Library/LaunchAgents`
6. Generate runner script (zsh, fills in UV path, project dir, exclude labels, log path)
7. Generate plist (XML with concrete values)
8. Generate control script (zsh, the myservicectl equivalent)
9. `chmod +x` runner and ctl, `chmod 644` plist
10. `plutil -lint` on plist
11. Print: "Installed. Run 'gmail-classifierctl start' or 'make service-start'"

Does NOT auto-start the service — user decides when.

### Crash behavior and notifications

- `KeepAlive: true` means launchd restarts the process if it exits unexpectedly
- launchd throttles crash-loops (10s delay between restarts by default)
- The log is append-only — Python tracebacks are preserved across restarts
- The runner prints a timestamped "starting" line on each launch, so restart frequency is visible in the log
- `gmail-classifierctl logs` (or `make service-logs`) tails the log for analysis

**Crash-loop detection (runner script):**

Before launching Python, the runner checks if the log's last "starting" line is less than 60s old. If so, it posts a macOS Notification Center alert:
```zsh
# Detect crash loop: if last start was <60s ago, notify
last_start=$(grep -o '\[[^]]*\] starting' "$LOG" | tail -1 | tr -d '[]' | sed 's/ starting//')
if [[ -n "$last_start" ]]; then
    last_epoch=$(/bin/date -j -f "%Y-%m-%dT%H:%M:%SZ" "$last_start" "+%s" 2>/dev/null || echo 0)
    now_epoch=$(/bin/date "+%s")
    if (( now_epoch - last_epoch < 60 )); then
        osascript -e 'display notification "Service is crash-looping. Check: gmail-classifierctl logs" with title "gmail-classifier"'
    fi
fi
```

**Crash email alert (Python script):**

In `classify_and_label.py`, wrap `main()` with an outer exception handler that sends an email to the user via the Gmail API before re-raising:
```python
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print(f"\n{datetime.now().strftime('%H:%M:%S')} Stopped.")
        sys.exit(0)
    except Exception as e:
        # Attempt to send crash alert email before exiting
        try:
            _send_crash_alert(e)
        except Exception:
            pass  # don't mask the original error
        raise
```

The `_send_crash_alert(e)` function:
- Uses the existing credentials (from `credentials/` dir) to authenticate
- Sends a short email to the user's own Gmail: subject "gmail-classifier crashed", body = traceback
- Fails silently if auth/network is broken (the traceback still goes to the log)
- Adds `gmail_client.py` helper: `send_message(to, subject, body)` (thin wrapper around `users.messages.send`)

### `service-uninstall`

1. `launchctl bootout gui/$(id -u)/$(SERVICE_LABEL)` (ignore errors)
2. Remove runner, plist, ctl
3. Print note about log file (leave it for user)

### Convenience targets

- `service-start`: `gmail-classifierctl start`
- `service-stop`: `gmail-classifierctl stop`
- `service-status`: `gmail-classifierctl status`
- `service-logs`: `gmail-classifierctl logs`

### Generation approach

Use `cat > file <<'EOF'` (single-quoted heredoc to avoid shell expansion), then `sed -i` to substitute placeholders (`__UV_PATH__`, `__PROJECT_DIR__`, `__EXCLUDE_LABELS__`, `__LABEL__`, `__LOG__`). This avoids Makefile `$$` escaping issues with the plist XML and zsh script content.

## 3. Key details for generated files

**Runner** (`~/bin/gmail-classifier-runner`):
- `#!/bin/zsh`, `set -euo pipefail`
- Sets HOME, PYTHONUNBUFFERED=1, UV_CACHE_DIR
- `cd $PROJECT_DIR`
- `exec >> "$LOG" 2>&1`
- Prints startup timestamp
- `exec "$UV" run --locked -- python -u scripts/classify_and_label.py --mode pubsub --exclude-labels XLC XLE XLCap`

**Plist** (`~/Library/LaunchAgents/com.xnodet.gmail-classifier.plist`):
- Label, ProgramArguments (runner path), WorkingDirectory
- EnvironmentVariables: HOME, PYTHONUNBUFFERED, UV_CACHE_DIR
- RunAtLoad: true, KeepAlive: true, ProcessType: Background
- StandardOutPath/StandardErrorPath: log (fallback)

**Control script** (`~/bin/gmail-classifierctl`):
- Subcommands: start, stop, restart, reload, status, logs, truncate-log, rotate-log, enable, disable
- Matches the myservicectl design from the guide

## 4. README: clone-to-running walkthrough

Add a section to `README.md` (or create it if it doesn't exist yet — it's currently untracked) with a clear step-by-step from scratch:

```
## Quick start: git clone to running service

1. Clone and install dependencies
   git clone <repo-url>
   cd gmail-classifier
   make setup

2. Set up GCP credentials
   - Create OAuth2 credentials (see docs/gmail-setup.md)
   - Place client_secret.json in credentials/
   - Create Pub/Sub topic + subscription (see docs/gmail-setup.md)

3. Authenticate
   make fetch-training    # triggers OAuth flow on first run

4. Fetch training data
   make fetch-training    # downloads labeled emails
   make fetch-inbox       # downloads inbox as skip examples

5. Verify it works interactively
   make watch-pubsub      # Ctrl+C to stop

6. Install as macOS service
   make service-install   # generates runner, plist, control script

7. Start the service
   gmail-classifierctl start
   gmail-classifierctl logs   # watch output
```

Keep it concise — link to `mac_uv_launchd_service_plan.md` for detailed launchd explanation, and `docs/gmail-setup.md` for GCP setup.

## 5. Verification

1. `make test` — existing tests still pass (SIGTERM change is minimal)
2. On macOS: `make service-install` generates all three files, `plutil -lint` passes
3. `gmail-classifierctl start` → service starts, log file populated
4. `gmail-classifierctl status` → shows running PID
5. `gmail-classifierctl stop` → service stops, "Stopped." appears in log
6. `make service-uninstall` → files removed, service stopped
7. Sending SIGTERM to the python process directly → clean exit with "Stopped." message
