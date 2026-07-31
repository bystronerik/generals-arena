#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" -u "$DIR/main.py"
