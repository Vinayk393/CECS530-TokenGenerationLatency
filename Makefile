MODEL   ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
PEAK_BW ?= 120   # Set to 100 for M2, 120 for M4

.PHONY: all bench graphs clean help \
        bench-01 bench-02 bench-03 bench-04 bench-05 \
        bench-06 bench-07 bench-08 bench-09

# ── Default ───────────────────────────────────────────────────────────────────
all: bench graphs

# ── Full benchmark suite ──────────────────────────────────────────────────────
bench: bench-01 bench-02 bench-03 bench-04 bench-05 bench-06 bench-07 bench-08 bench-09

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

bench-07:
	python benchmarks/07_quantization_speedup.py --backend hf --model $(MODEL)

bench-08:
	python benchmarks/08_kvcache_size_vs_context.py

bench-09:
	python benchmarks/09_latency_decomposition.py --model $(MODEL) --also_run_scaling

# ── Graphs ────────────────────────────────────────────────────────────────────
graphs:
	mkdir -p graphs
	python analysis/generate_research_graphs.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf graphs/*.png
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Token-Generation Latency Benchmarking — Mac M4 vs M2"
	@echo ""
	@echo "  Usage:"
	@echo "    make bench              Run all 9 benchmarks (default model: TinyLlama-1.1B)"
	@echo "    make bench-01           Run only TTFT benchmark"
	@echo "    make graphs             Generate all 9 graphs from CSVs"
	@echo "    make all                Run benchmarks + generate graphs"
	@echo "    make clean              Remove generated graphs and cache"
	@echo ""
	@echo "  Options:"
	@echo "    MODEL=<hf-model-id>     Override model  (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
	@echo "    PEAK_BW=<GB/s>          Set device peak bandwidth  (M4: 120, M2: 100)"
	@echo ""
	@echo "  Examples:"
	@echo "    make bench MODEL=meta-llama/Llama-3.2-1B"
	@echo "    make bench-04 PEAK_BW=100"
	@echo "    make bench-05 MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	@echo ""
