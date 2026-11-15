#!/usr/bin/env bash
# Tests, the benchmark (50 seeds), and the figure. A few minutes on 4 cores.
set -e
cd "$(dirname "$0")"
python3 tests/test_core.py
python3 experiments/benchmark.py --seeds 50
python3 experiments/plots.py
