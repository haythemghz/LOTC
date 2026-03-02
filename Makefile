.PHONY: test smoke train-synthetic train-all figures paper clean help

help:  ## Show this help
	@echo "LOTC — Learned Optimal Transport Clustering"
	@echo ""
	@echo "Targets:"
	@echo "  test            Run all unit tests"
	@echo "  smoke           Run quick smoke test (CPU, ~30s)"
	@echo "  train-synthetic Run synthetic blobs experiment"
	@echo "  train-all       Run all experiments (10 seeds each)"
	@echo "  figures         Generate paper figures"
	@echo "  paper           Compile LaTeX paper"
	@echo "  reproducer      Reproduce core results (smoke + synthetic + figures)"
	@echo "  clean           Remove generated outputs"

test:  ## Run unit tests
	pytest tests/ -v --tb=short

smoke:  ## Quick smoke test on tiny blobs
	python -m src.experiments.run_experiment \
		--config src/experiments/configs/smoke_test.yaml --seeds 1

train-synthetic:  ## Run synthetic blobs experiment (10 seeds)
	python -m src.experiments.run_experiment \
		--config src/experiments/configs/synthetic_blobs.yaml --seeds 10

train-all:  ## Run all experiments
	bash scripts/run_all_experiments.sh 10

figures:  ## Generate paper figures from results
	python scripts/make_figures.py

paper:  ## Compile LaTeX paper
	cd paper && latexmk -pdf main.tex

reproducer:  ## Reproduce core results
	$(MAKE) smoke
	$(MAKE) train-synthetic
	$(MAKE) figures

clean:  ## Remove generated outputs
	rm -rf experiments/results/
	rm -rf paper/figures/*.png
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
