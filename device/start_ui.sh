#!/usr/bin/env bash
# Starts the NailScan server and opens the interface.
#   ./start_ui.sh            normal app window
#   ./start_ui.sh --kiosk    full screen, no browser controls (for the touchscreen)
cd "$(dirname "$(readlink -f "$0")")"
[ -d venv ] || { echo "Not installed yet. Run: bash install.sh"; exit 1; }
PORT="${NAILSCAN_PORT:-8000}"
URL="http://localhost:$PORT"

# stop a previous NailScan server if one is still running
if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then kill "$(cat .server.pid)"; sleep 1; fi
./venv/bin/python server.py > server.log 2>&1 &
SERVER=$!
echo $SERVER > .server.pid
trap 'kill $SERVER 2>/dev/null; rm -f .server.pid' EXIT

echo "Starting NailScan..."
for _ in $(seq 1 60); do
  curl -s -o /dev/null "$URL/api/status" && break
  kill -0 $SERVER 2>/dev/null || { echo "Server failed to start. See server.log:"; tail -20 server.log; exit 1; }
  sleep 0.5
done

BROWSER="$(command -v chromium-browser || command -v chromium || true)"
if [ -n "$BROWSER" ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  if [ "${1:-}" = "--kiosk" ]; then MODE=(--kiosk "$URL"); else MODE=(--app="$URL" --start-maximized); fi
  "$BROWSER" "${MODE[@]}" --noerrdialogs --disable-infobars --no-first-run \
    --user-data-dir="$HOME/.config/nailscan-browser" --password-store=basic
else
  echo "NailScan is running. Open $URL in a browser (or http://$(hostname -I | awk '{print $1}'):$PORT from a phone)."
  echo "Press Ctrl+C to stop."
  wait $SERVER
fi
