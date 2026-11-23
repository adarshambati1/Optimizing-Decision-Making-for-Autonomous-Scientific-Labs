#!/usr/bin/env bash
# Tests, the offline log, training, the 250-seed real-dataset benchmark and the figures.
# About 15 minutes on 10 cores.
set -e
cd "$(dirname "$0")/../.."
python3 antonio_tests/offline_rl/tests/test_logger.py
python3 antonio_tests/offline_rl/tests/test_methods.py
python3 antonio_tests/offline_rl/logger.py
python3 antonio_tests/offline_rl/train.py
python3 antonio_tests/offline_rl/benchmark.py --seeds 250
python3 antonio_tests/offline_rl/plots.py
