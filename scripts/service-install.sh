#!/bin/zsh
set -euo pipefail

# Generate and install macOS launchd service files for gmail-classifier.

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: service-install requires macOS"
    exit 1
fi

UV_PATH=$(command -v uv) || true
if [[ -z "$UV_PATH" ]]; then
    echo "Error: uv not found in PATH"
    exit 1
fi

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

if [[ ! -s "$PROJECT_DIR/data/training.db" ]]; then
    echo "Error: data/training.db missing or empty. Run 'make fetch-training' first."
    exit 1
fi

if [[ ! -s "$PROJECT_DIR/data/inbox_sample.db" ]]; then
    echo "Error: data/inbox_sample.db missing or empty. Run 'make fetch-inbox' first."
    exit 1
fi

LABEL="com.xnodet.gmail-classifier"
RUNNER="$HOME/bin/gmail-classifier-runner"
CTL="$HOME/bin/gmail-classifierctl"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/$LABEL.log"

mkdir -p "$HOME/bin" "$HOME/Library/LaunchAgents"

# --- Runner script ---
echo "Generating: $RUNNER"
cat > "$RUNNER" <<EOF
#!/bin/zsh
set -euo pipefail

LABEL="$LABEL"
UV="$UV_PATH"
PROJECT_DIR="$PROJECT_DIR"
LOG="$LOG"

export HOME="$HOME"
export PYTHONUNBUFFERED="1"
export UV_CACHE_DIR="$HOME/Library/Caches/uv"

mkdir -p "\$(dirname "\$LOG")"
mkdir -p "\$UV_CACHE_DIR"

cd "\$PROJECT_DIR"

exec >> "\$LOG" 2>&1

# Detect crash loop: if last start was <60s ago, notify
last_start=\$(grep -o '\[[^]]*\] starting' "\$LOG" | tail -1 | tr -d '[]' | sed 's/ starting//') || true
if [[ -n "\$last_start" ]]; then
    last_epoch=\$(/bin/date -j -f "%Y-%m-%dT%H:%M:%SZ" "\$last_start" "+%s" 2>/dev/null || echo 0)
    now_epoch=\$(/bin/date "+%s")
    if (( now_epoch - last_epoch < 60 )); then
        osascript -e 'display notification "Service is crash-looping. Check: gmail-classifierctl logs" with title "gmail-classifier"'
    fi
fi

echo "[\$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')] starting \${LABEL}"

exec "\$UV" run --locked -- python -u scripts/classify_and_label.py --mode pubsub
EOF
chmod +x "$RUNNER"

# --- LaunchAgent plist ---
echo "Generating: $PLIST"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>

    <key>ProgramArguments</key>
    <array>
        <string>$RUNNER</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>$HOME</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>UV_CACHE_DIR</key>
        <string>$HOME/Library/Caches/uv</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>ProcessType</key>
    <string>Background</string>

    <key>StandardOutPath</key>
    <string>$LOG</string>

    <key>StandardErrorPath</key>
    <string>$LOG</string>
</dict>
</plist>
EOF
chmod 644 "$PLIST"

# --- Control script ---
echo "Generating: $CTL"
cat > "$CTL" <<'CTLEOF'
#!/bin/zsh
set -euo pipefail

LABEL="com.xnodet.gmail-classifier"
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
    if is_loaded; then
      launchctl print "$TARGET"
    else
      echo "Service is stopped."
    fi
    ;;

  logs)
    tail -n 20 -F "$LOG"
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
CTLEOF
chmod +x "$CTL"

# --- Validate ---
plutil -lint "$PLIST"

echo ""
echo "Installed successfully."
echo "  Runner: $RUNNER"
echo "  Plist:  $PLIST"
echo "  Ctl:    $CTL"
echo "  Log:    $LOG"
echo ""
echo "To start: gmail-classifierctl start  (or: make service-start)"
