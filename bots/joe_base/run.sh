#!/usr/bin/env bash
# Resolve this script's directory so `python main.py` works no matter where
# the caller invoked us from (matchup.py sets cwd to the script dir, but
# absolute paths stay correct if someone runs the script directly).
DIR="$(cd "$(dirname "$0")" && pwd)"

# Pin every math backend to one thread BEFORE the interpreter starts (same
# rationale as the morpheus launcher: pools size themselves at library init,
# and two bot subprocesses on one host would otherwise oversubscribe every
# core; a literal path here would also drag that launcher into this bot's
# content-hash closure via the shell-reference scan). XLA gets the same
# treatment — these are the flags the Phase 2 x86
# latency measurement ran under (docs/research/measurements/
# joe-phase2-cpu-latency.md), so play matches its budget evidence.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Inference is jax-CPU by decision (plan section 5); never grab a GPU.
export JAX_PLATFORMS=cpu

exec "${PYTHON:-python3}" -u "$DIR/main.py"
