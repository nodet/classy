# macOS launchd service for a long-running Python project managed by uv

This guide describes how to run a long-running Python service on macOS 26.5.1 using `launchd` and `uv`.

Target behavior:

- Start automatically when you log in.
- Optionally start at system boot before login, if you choose the LaunchDaemon variant.
- Run a Python project through `uv` so dependencies come from `pyproject.toml` and `uv.lock`.
- Treat the Python script as a long-running service that should not exit on its own.
- Restart automatically if the process crashes or exits unexpectedly.
- Combine standard output and standard error into one easy-to-read log file.
- Provide simple `start`, `stop`, `restart`, `reload`, `status`, and `logs` commands.

The recommended default is a per-user LaunchAgent. Use a system LaunchDaemon only if the service truly needs to start before any user logs in.

---

## 1. Recommended architecture

Use macOS `launchd` rather than cron, shell login items, or a custom background loop. `launchd` is the native service manager for macOS agents and daemons.

Recommended per-user layout:

```text
/Users/YOU/Projects/myservice/
  pyproject.toml
  uv.lock
  src/...

/Users/YOU/bin/myservice-runner
/Users/YOU/bin/myservicectl
/Users/YOU/Library/LaunchAgents/com.example.myservice.plist
/Users/YOU/Library/Logs/com.example.myservice.log
```

The LaunchAgent starts at login, runs as your user, and can use a `uv` installation located in your home directory or installed through Homebrew.

Use these placeholders throughout the guide:

```text
YOU                     Your macOS short username
com.example.myservice   The unique launchd label for the service
/Users/YOU/Projects/myservice
                        The Python project directory
/opt/homebrew/bin/uv    The absolute path to uv; adjust this for your Mac
python -u -m your_package
                        The Python command to run your service
```

Find the correct `uv` path with:

```bash
command -v uv
```

Common `uv` locations are:

```text
/opt/homebrew/bin/uv        Apple Silicon Homebrew
/usr/local/bin/uv           Intel Homebrew
/Users/YOU/.local/bin/uv    uv standalone installer
```

Use absolute paths in `launchd` files and scripts. Do not rely on the `PATH` you see in an interactive Terminal session.

---

## 2. Prepare and test the Python project

From Terminal:

```bash
cd /Users/YOU/Projects/myservice
uv sync --locked
uv run --locked -- python -u -m your_package
```

Notes:

- `uv sync --locked` verifies that the project environment can be created from the existing lockfile.
- `uv run --locked -- python -u -m your_package` runs the project without allowing `uv.lock` to be changed.
- `python -u` forces unbuffered output so log messages appear promptly.
- If your service is a script instead of a module, use a command like:

```bash
uv run --locked -- python -u path/to/script.py
```

Do not continue until the command works interactively from Terminal.

---

## 3. Create one combined log file

Because the service is long-running, the simplest operational model is one append-only log file:

```text
/Users/YOU/Library/Logs/com.example.myservice.log
```

Create the directory and file:

```bash
mkdir -p /Users/YOU/Library/Logs
: > /Users/YOU/Library/Logs/com.example.myservice.log
chmod 644 /Users/YOU/Library/Logs/com.example.myservice.log
```

For strict stdout/stderr combination, use a small wrapper script that redirects both streams through the same file descriptor:

```bash
exec >> "$LOG" 2>&1
```

That is preferable to relying only on two separate `launchd` keys pointing at the same path, because the wrapper explicitly makes stderr a duplicate of stdout after stdout has been opened in append mode.

---

## 4. Create the runner script

Create `/Users/YOU/bin/myservice-runner`:

```bash
mkdir -p /Users/YOU/bin
nano /Users/YOU/bin/myservice-runner
```

Paste this script and adjust the placeholder values:

```zsh
#!/bin/zsh
set -euo pipefail

LABEL="com.example.myservice"
UV="/opt/homebrew/bin/uv"
PROJECT_DIR="/Users/YOU/Projects/myservice"
LOG="/Users/YOU/Library/Logs/${LABEL}.log"

export HOME="/Users/YOU"
export PYTHONUNBUFFERED="1"
export UV_CACHE_DIR="/Users/YOU/Library/Caches/uv"

mkdir -p "$(dirname "$LOG")"
mkdir -p "$UV_CACHE_DIR"

cd "$PROJECT_DIR"

# Combine stdout and stderr into one append-only log.
exec >> "$LOG" 2>&1

echo "[$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')] starting ${LABEL}"

# Replace this command with your actual long-running service entry point.
exec "$UV" run --locked -- python -u -m your_package
```

Make it executable:

```bash
chmod +x /Users/YOU/bin/myservice-runner
```

Test it manually:

```bash
/Users/YOU/bin/myservice-runner
```

Since this is a long-running service, stop the manual test with `Control-C`. Then inspect the log:

```bash
tail -n 100 /Users/YOU/Library/Logs/com.example.myservice.log
```

---

## 5. Create the LaunchAgent plist

Create the LaunchAgent file:

```bash
mkdir -p /Users/YOU/Library/LaunchAgents
nano /Users/YOU/Library/LaunchAgents/com.example.myservice.plist
```

Paste this plist and adjust the placeholder values:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.example.myservice</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOU/bin/myservice-runner</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/YOU/Projects/myservice</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>/Users/YOU</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>UV_CACHE_DIR</key>
        <string>/Users/YOU/Library/Caches/uv</string>
    </dict>

    <!-- Start when the LaunchAgent is loaded, normally at login. -->
    <key>RunAtLoad</key>
    <true/>

    <!-- This is a long-running service. Keep it continuously running. -->
    <key>KeepAlive</key>
    <true/>

    <!-- Optional but appropriate for a non-interactive background service. -->
    <key>ProcessType</key>
    <string>Background</string>

    <!-- Fallback capture for messages before the runner redirects its own streams. -->
    <key>StandardOutPath</key>
    <string>/Users/YOU/Library/Logs/com.example.myservice.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/YOU/Library/Logs/com.example.myservice.log</string>
</dict>
</plist>
```

Important points:

- `KeepAlive` is intentionally enabled because this service is expected to run continuously.
- Do not use `StartInterval` or `StartCalendarInterval` for this design. Those are for periodic jobs, not a process that should stay alive.
- Do not make the Python process daemonize, fork into the background, close stdio, or redirect itself to `/dev/null`. Let `launchd` supervise the process.
- The runner script combines stdout and stderr into one log file. The plist also points both launchd stream paths at the same log as a fallback.

Validate and secure the plist:

```bash
plutil -lint /Users/YOU/Library/LaunchAgents/com.example.myservice.plist
chmod 644 /Users/YOU/Library/LaunchAgents/com.example.myservice.plist
```

For a plist in `~/Library/LaunchAgents`, the file should be owned by the user who loads it and should not be group-writable or world-writable.

---

## 6. Start, stop, restart, and inspect manually

Set convenient shell variables:

```bash
LABEL="com.example.myservice"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
TARGET="$DOMAIN/$LABEL"
LOG="$HOME/Library/Logs/$LABEL.log"
```

Start or install the LaunchAgent:

```bash
launchctl bootstrap "$DOMAIN" "$PLIST"
```

Check status:

```bash
launchctl print "$TARGET"
```

Watch the combined log:

```bash
tail -F "$LOG"
```

Restart the running service process:

```bash
launchctl kickstart -k "$TARGET"
```

Stop the service:

```bash
launchctl bootout "$TARGET"
```

For this design, use `bootout` to stop the service. Do not merely kill the process or use legacy `launchctl stop`; because `KeepAlive` is true, `launchd` may immediately start it again.

Reload after editing the plist:

```bash
plutil -lint "$PLIST"
launchctl bootout "$TARGET" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
```

---

## 7. Create an easy control command

Create `/Users/YOU/bin/myservicectl`:

```bash
nano /Users/YOU/bin/myservicectl
```

Paste this script and adjust the label if needed:

```zsh
#!/bin/zsh
set -euo pipefail

LABEL="com.example.myservice"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
TARGET="$DOMAIN/$LABEL"
LOG="$HOME/Library/Logs/$LABEL.log"

is_loaded() {
  launchctl print "$TARGET" >/dev/null 2>&1
}

case "${1:-}" in
  start)
    if is_loaded; then
      launchctl kickstart "$TARGET"
    else
      launchctl bootstrap "$DOMAIN" "$PLIST"
    fi
    ;;

  stop)
    if is_loaded; then
      launchctl bootout "$TARGET"
    else
      echo "$LABEL is not loaded"
    fi
    ;;

  restart)
    if is_loaded; then
      launchctl kickstart -k "$TARGET"
    else
      launchctl bootstrap "$DOMAIN" "$PLIST"
    fi
    ;;

  reload)
    plutil -lint "$PLIST"
    if is_loaded; then
      launchctl bootout "$TARGET"
    fi
    launchctl bootstrap "$DOMAIN" "$PLIST"
    ;;

  status)
    launchctl print "$TARGET"
    ;;

  logs)
    tail -F "$LOG"
    ;;

  truncate-log)
    : > "$LOG"
    ;;

  rotate-log)
    was_loaded=0
    if is_loaded; then
      was_loaded=1
      launchctl bootout "$TARGET"
    fi

    timestamp="$(/bin/date '+%Y%m%d-%H%M%S')"
    if [[ -f "$LOG" ]]; then
      mv "$LOG" "$LOG.$timestamp"
    fi
    : > "$LOG"

    if [[ "$was_loaded" == "1" ]]; then
      launchctl bootstrap "$DOMAIN" "$PLIST"
    fi
    ;;

  enable)
    launchctl enable "$TARGET"
    if ! is_loaded; then
      launchctl bootstrap "$DOMAIN" "$PLIST"
    fi
    ;;

  disable)
    if is_loaded; then
      launchctl bootout "$TARGET"
    fi
    launchctl disable "$TARGET"
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|reload|status|logs|truncate-log|rotate-log|enable|disable}" >&2
    exit 2
    ;;
esac
```

Make it executable:

```bash
chmod +x /Users/YOU/bin/myservicectl
```

Use it like this:

```bash
myservicectl start
myservicectl status
myservicectl logs
myservicectl restart
myservicectl stop
myservicectl reload
myservicectl rotate-log
```

Command meanings:

```text
start         Load the LaunchAgent if needed; otherwise ask launchd to start it.
stop          Unload the LaunchAgent so KeepAlive cannot respawn it.
restart       Kill the running instance and start a fresh one.
reload        Re-validate the plist, unload it, and load it again.
status        Print launchd's view of the service.
logs          Follow the combined stdout/stderr log.
truncate-log  Empty the current log file without restarting the service.
rotate-log    Stop the service, rename the old log, create a new log, and restart.
enable        Re-enable the service if it was disabled.
disable       Stop and disable the service until re-enabled.
```

---

## 8. Operational notes for a long-running service

### KeepAlive behavior

`KeepAlive` is the key that makes this a continuously running service. If the Python process exits unexpectedly, `launchd` starts it again. This is exactly what you want for a service that should never stop on its own.

If the script exits immediately in a loop, `launchd` may throttle restarts. In that case, inspect the combined log first:

```bash
tail -n 200 /Users/YOU/Library/Logs/com.example.myservice.log
```

Then inspect launchd state:

```bash
launchctl print "gui/$(id -u)/com.example.myservice"
```

### Graceful shutdown

When you run `myservicectl stop`, `launchd` unloads the job and terminates the process. Your Python code should handle `SIGTERM` and shut down cleanly.

A minimal Python pattern is:

```python
import signal
import sys
import time

running = True

def handle_stop(signum, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handle_stop)
signal.signal(signal.SIGINT, handle_stop)

while running:
    # Do service work here.
    time.sleep(1)

# Close files, flush queues, stop workers, etc.
sys.exit(0)
```

### Log growth

Because stdout and stderr are combined into one file, the log can grow indefinitely. Use one of these periodically:

```bash
myservicectl truncate-log
```

or:

```bash
myservicectl rotate-log
```

`rotate-log` is safer if you want to archive the previous log and force the service to reopen a new file.

---

## 9. Optional: start at boot before login with a LaunchDaemon

Use this only if login-time startup is insufficient.

A LaunchDaemon lives here:

```text
/Library/LaunchDaemons/com.example.myservice.plist
```

Differences from the LaunchAgent design:

- The plist must be owned by `root:wheel`.
- The plist should be mode `644`.
- You control it with `sudo launchctl ... system/...`.
- Add `UserName` so the Python service runs as your normal user instead of root.
- Consider moving the log to `/Library/Logs/com.example.myservice.log` and making it writable by the service user.

Example LaunchDaemon plist:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.example.myservice</string>

    <key>UserName</key>
    <string>YOU</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOU/bin/myservice-runner</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/YOU/Projects/myservice</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>/Users/YOU</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>UV_CACHE_DIR</key>
        <string>/Users/YOU/Library/Caches/uv</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>ProcessType</key>
    <string>Background</string>

    <key>StandardOutPath</key>
    <string>/Users/YOU/Library/Logs/com.example.myservice.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/YOU/Library/Logs/com.example.myservice.log</string>
</dict>
</plist>
```

Install and start the LaunchDaemon:

```bash
sudo cp /path/to/com.example.myservice.plist /Library/LaunchDaemons/com.example.myservice.plist
sudo chown root:wheel /Library/LaunchDaemons/com.example.myservice.plist
sudo chmod 644 /Library/LaunchDaemons/com.example.myservice.plist
sudo plutil -lint /Library/LaunchDaemons/com.example.myservice.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.example.myservice.plist
```

Control it:

```bash
sudo launchctl print system/com.example.myservice
sudo launchctl kickstart -k system/com.example.myservice
sudo launchctl bootout system/com.example.myservice
```

If you use `/Library/Logs` instead of your user log directory, create and permit the log file before starting:

```bash
sudo touch /Library/Logs/com.example.myservice.log
sudo chown YOU:staff /Library/Logs/com.example.myservice.log
sudo chmod 644 /Library/Logs/com.example.myservice.log
```

Then update `LOG`, `StandardOutPath`, and `StandardErrorPath` to use `/Library/Logs/com.example.myservice.log`.

---

## 10. Troubleshooting checklist

### The service does not start

Validate the plist:

```bash
plutil -lint ~/Library/LaunchAgents/com.example.myservice.plist
```

Run the wrapper directly:

```bash
/Users/YOU/bin/myservice-runner
```

Check the combined log:

```bash
tail -n 200 ~/Library/Logs/com.example.myservice.log
```

Check launchd status:

```bash
launchctl print "gui/$(id -u)/com.example.myservice"
```

### uv cannot be found

Use an absolute path in `myservice-runner`:

```bash
command -v uv
```

Then update:

```zsh
UV="/absolute/path/to/uv"
```

### The service starts in Terminal but not from launchd

Typical causes:

- The plist has a typo or invalid XML.
- The runner script is not executable.
- The `uv` path is wrong.
- The project path is wrong.
- The service depends on environment variables that exist in your shell but not under `launchd`.
- The script accesses files protected by macOS privacy controls, such as Desktop, Documents, Downloads, removable volumes, or network volumes.

Make all required environment variables explicit in either the plist or the runner script.

### The service keeps restarting

Because `KeepAlive` is true, any unexpected exit triggers a restart. Look at the combined log first:

```bash
tail -n 200 ~/Library/Logs/com.example.myservice.log
```

Then run the same command manually from the project directory:

```bash
cd /Users/YOU/Projects/myservice
/opt/homebrew/bin/uv run --locked -- python -u -m your_package
```

Fix the underlying Python error before starting the LaunchAgent again.

---

## 11. References

- Apple Support: Script management with launchd in Terminal on Mac: https://support.apple.com/guide/terminal/script-management-with-launchd-apdc6c1077b-5d5d-4d35-9c19-60f2397b2369/mac
- `launchd.plist(5)` keys including `ProgramArguments`, `KeepAlive`, `WorkingDirectory`, `EnvironmentVariables`, `StandardOutPath`, and `StandardErrorPath`: https://keith.github.io/xcode-man-pages/launchd.plist.5.html
- `launchctl(1)` subcommands including `bootstrap`, `bootout`, `kickstart`, `enable`, `disable`, and `print`: https://keith.github.io/xcode-man-pages/launchctl.1.html
- uv CLI reference for `uv run`, `--locked`, and related options: https://docs.astral.sh/uv/reference/cli/
