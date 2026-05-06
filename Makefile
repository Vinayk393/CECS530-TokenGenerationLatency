MODEL ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
PEAK_BW ?= 120

.PHONY: all bench graphs smoke verify clean help \
	bench-01 bench-02 bench-03 bench-04 bench-05 \
	bench-06 bench-07-modeled bench-07-llamacpp bench-08 bench-09

all: bench graphs

bench: bench-01 bench-02 bench-03 bench-04 bench-05 bench-06 bench-07-modeled bench-08 bench-09

bench-01:
	python benchmarks/01_ttft_vs_prompt_length.py --model $(MODEL)

bench-02:
	python benchmarks/02_per_token_latency_vs_context.py --model $(MODEL)

bench-03:
	python benchmarks/03_e2e_latency_vs_output_length.py --model $(MODEL)

bench-04:
	python benchmarks/04_throughput_vs_prompt_length.py --model $(MODEL) --peak_bw $(PEAK_BW)

bench-05:
	python benchmarks/05_inter_token_latency_timeline.py --model $(MODEL)

bench-06:
	python benchmarks/06_cold_vs_warm_run.py --model $(MODEL)

bench-07-modeled:
	python benchmarks/07_quantization_speedup.py --mode modeled --model $(MODEL)

bench-07-llamacpp:
	python benchmarks/07_quantization_speedup.py --backend llamacpp

bench-08:
	python benchmarks/08_kvcache_size_vs_context.py

bench-09:
	python benchmarks/09_latency_decomposition.py --model $(MODEL) --also_run_scaling

smoke:
	python benchmarks/run_smoke_test.py

graphs:
	mkdir -p graphs
	python analysis/generate_research_graphs.py

verify:
	python benchmarks/run_smoke_test.py
	python -m pytest tests/

clean:
	rm -rf graphs/*.png
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "Usage:"
	@echo "  make smoke              Run fast smoke test (no GGUF needed)"
	@echo "  make bench              Run all 9 benchmarks (downloads ~2.2GB model)"
	@echo "  make bench-07-modeled   Q4/Q8 speedup projected from measured F16 baseline"
	@echo "  make bench-07-llamacpp  Q4/Q8 speedup via llama.cpp (requires GGUF files)"
	@echo "  make graphs             Generate all figures from CSVs"
	@echo "  make verify             Run smoke test + pytest"
	@echo "  make clean              Remove graphs and cache"
	@echo ""
	@echo "Options:"
	@echo "  MODEL=<hf-model-id>   Override model (default: TinyLlama-1.1B)"
	@echo "  PEAK_BW=<GB/s>        Peak bandwidth for BW util calc (M4: 120, M2: 100)"
