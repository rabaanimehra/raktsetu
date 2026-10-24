#!/usr/bin/env bash
# Runs NailScan inside its virtual environment. Example: ./run.sh --webcam
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -d venv ]; then echo "Not installed yet. Run: bash install.sh"; exit 1; fi
exec ./venv/bin/python predict_pi.py "$@"
