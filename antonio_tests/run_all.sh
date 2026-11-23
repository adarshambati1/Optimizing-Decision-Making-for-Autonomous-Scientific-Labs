#!/usr/bin/env bash
# Tests, offline solves, the 250-seed benchmark and the figures. About 15 minutes on 10 cores.
set -e
cd "$(dirname "$0")/.."
python3 antonio_tests/tests/test_solvers.py
python3 antonio_tests/solve_all.py
python3 antonio_tests/benchmark.py --seeds 250
python3 antonio_tests/plots.py
