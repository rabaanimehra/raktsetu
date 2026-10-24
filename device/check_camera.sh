#!/usr/bin/env bash
# Takes one webcam photo with locked settings and prints a colour / red-clipping report.
cd "$(dirname "$(readlink -f "$0")")"
exec ./run.sh --webcam --check "$@"
