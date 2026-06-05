#!/bin/zsh
set -euo pipefail

# Remove macOS launchd service files for gmail-classifier.

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: service-uninstall requires macOS"
    exit 1
fi

LABEL="com.xnodet.gmail-classifier"
RUNNER="$HOME/bin/gmail-classifier-runner"
CTL="$HOME/bin/gmail-classifierctl"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/$LABEL.log"
DOMAIN="gui/$(id -u)"
TARGET="$DOMAIN/$LABEL"

# Stop the service if running
if launchctl print "$TARGET" >/dev/null 2>&1; then
    echo "Stopping service..."
    launchctl bootout "$TARGET" 2>/dev/null || true
fi

# Remove generated files
for f in "$RUNNER" "$CTL" "$PLIST"; do
    if [[ -f "$f" ]]; then
        echo "Removing: $f"
        rm "$f"
    fi
done

echo ""
echo "Uninstalled. Log file kept at: $LOG"
echo "Remove it manually if no longer needed."
