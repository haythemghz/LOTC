# Reproducibility script for LOTC
Write-Host "Starting LOTC Benchmarks..." -ForegroundColor Cyan

# 1. Synthetic
Write-Host "Running Synthetic Benchmarks..." -ForegroundColor Yellow
python -m src.experiments.run_experiment --config src/experiments/configs/synthetic/moons.yaml --seed 42
python -m src.experiments.run_experiment --config src/experiments/configs/synthetic/circles.yaml --seed 42
python -m src.experiments.run_experiment --config src/experiments/configs/synthetic/unbalanced.yaml --seed 42

# 2. Real-World
Write-Host "Running Real-World Benchmarks..." -ForegroundColor Yellow
python -m src.experiments.run_experiment --config src/experiments/configs/real_world/fmnist.yaml --seed 42
python -m src.experiments.run_experiment --config src/experiments/configs/real_world/cifar10.yaml --seed 42

Write-Host "All experiments complete. Results stored in experiments/results/" -ForegroundColor Green
