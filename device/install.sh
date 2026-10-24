#!/usr/bin/env bash
# RaktSetu / NailScan - one-command installer for Raspberry Pi 5 (Raspberry Pi OS 64-bit)
# Usage:  bash install.sh               (add --autostart to open NailScan full screen at boot)
set -e
cd "$(dirname "$(readlink -f "$0")")"
APP_DIR="$(pwd)"
VENV="$APP_DIR/venv"

say()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m!! $*\033[0m"; }

say "NailScan installer - folder: $APP_DIR"

if [ "${SKIP_APT:-0}" != "1" ]; then
  say "1/5  Installing system packages (needs your password once)"
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-pip python3-dev v4l-utils curl
  # allow this user to use the webcam without sudo
  sudo usermod -aG video "$USER" || true
fi

say "2/5  Creating virtual environment: $VENV"
python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

say "3/5  Installing Python libraries (numpy, pillow, opencv, flask)"
pip install "numpy" "pillow" "opencv-python-headless" "flask"

say "4/5  Installing the TensorFlow Lite runtime"
if pip install ai-edge-litert; then
  echo "Installed ai-edge-litert"
elif pip install tflite-runtime; then
  echo "Installed tflite-runtime"
else
  warn "Lightweight runtime not available for this Python - installing full TensorFlow (bigger, slower to install)"
  pip install tensorflow
fi

say "5/5  Self-test: loading the model and running one prediction"
python - <<'PY'
import numpy as np, os, sys
from PIL import Image
sys.path.insert(0, os.getcwd())
from predict_pi import AnemiaClassifier
clf = AnemiaClassifier("models/anemia_model_float32.tflite", "models/model_config.json")
prob, label, ms = clf.predict(Image.fromarray(np.full((300, 300, 3), 200, np.uint8)))
print(f"Model OK - test inference {ms:.0f} ms (dummy image, result not meaningful)")
PY

chmod +x "$APP_DIR"/*.sh

say "Adding NailScan to the app menu and desktop"
DESKTOP_FILE="[Desktop Entry]
Type=Application
Name=NailScan
Comment=Nail-bed anemia screening
Exec=$APP_DIR/start_ui.sh
Icon=$APP_DIR/static/icon.svg
Terminal=false
Categories=Science;Education;"
mkdir -p "$HOME/.local/share/applications"
echo "$DESKTOP_FILE" > "$HOME/.local/share/applications/nailscan.desktop"
if [ -d "$HOME/Desktop" ]; then
  echo "$DESKTOP_FILE" > "$HOME/Desktop/NailScan.desktop"
  chmod +x "$HOME/Desktop/NailScan.desktop"
  command -v gio >/dev/null && gio set "$HOME/Desktop/NailScan.desktop" metadata::trusted true 2>/dev/null || true
fi

if [ "${1:-}" = "--autostart" ]; then
  mkdir -p "$HOME/.config/autostart"
  echo "${DESKTOP_FILE/start_ui.sh/start_ui.sh --kiosk}" > "$HOME/.config/autostart/nailscan.desktop"
  echo "NailScan will open full screen every time the Pi starts."
fi

say "Installed successfully!"
cat <<MSG

Open the NailScan interface:
  double-click the NailScan icon on the desktop (or in the app menu)
  or run:  ./start_ui.sh          (add --kiosk for full screen)

Command-line tools, from this folder ($APP_DIR):

  ./check_camera.sh            take a webcam photo + colour/clipping report (no prediction)
  ./run.sh --webcam            take a webcam photo and classify it
  ./run.sh --image photo.jpg   classify a saved photo
  ./run.sh --list-controls     show the webcam's adjustable settings

If the webcam gives a permission error, reboot once (needed after joining the 'video' group).
MSG
