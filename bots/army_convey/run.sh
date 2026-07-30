#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python -u "$DIR/main.py"
