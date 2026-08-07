#!/usr/bin/env bash
# Resolve this script's directory so `python main.py` works no matter where
# the caller invoked us from (matchup.py sets cwd to the script dir, but
# absolute paths stay correct if someone runs the script directly).
DIR="$(cd "$(dirname "$0")" && pwd)"

# Pin every math backend to one thread BEFORE the interpreter starts. OpenMP and
# BLAS size their pools at library init, so this cannot be done from Python.
#
# Two bot subprocesses on one host would otherwise each spawn one thread per
# core. Measured on an 11-core M3 Pro: under that oversubscription a 4-leaf
# batch costs up to 154 ms — more than the whole 140 ms turn — so admission
# denied leaf_batch on 99.1% of checks and search completed zero simulations on
# 100% of normal turns. Pinned, search decides ~75% of moves.
#
# This also makes play match calibration: training/morpheus/measure_online.py
# pins one thread, so an unpinned bot was running on numbers measured for a
# machine it never ran on. See docs/bots/morpheus/thread-pinning.md.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${PYTHON:-python3}" -u "$DIR/main.py"
