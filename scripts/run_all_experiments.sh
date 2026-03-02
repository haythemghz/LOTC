#!/usr/bin/env bash
# Run all LOTC experiments across datasets and seeds.
# Usage: bash scripts/run_all_experiments.sh [N_SEEDS]

set -euo pipefail

N_SEEDS=${1:-10}
CONFIG_DIR="src/experiments/configs"

echo "============================================"
echo "  LOTC — Full Experiment Suite"
echo "  Seeds per experiment: $N_SEEDS"
echo "============================================"

for config in "$CONFIG_DIR"/*.yaml; do
    name=$(basename "$config" .yaml)
    if [ "$name" = "smoke_test" ]; then
        continue
    fi
    echo ""
    echo ">>> Running experiment: $name"
    python -m src.experiments.run_experiment --config "$config" --seeds "$N_SEEDS"
done

echo ""
echo "============================================"
echo "  All experiments complete."
echo "  Generating figures..."
echo "============================================"
python scripts/make_figures.py

echo "Done. Results in experiments/results/, figures in paper/figures/"
