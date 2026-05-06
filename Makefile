MODEL    ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
PEAK_BW  ?= 120        # Set to 100 for M2, 120 for M4
DEVICE   ?= Mac_M4_16GB  # Set to Mac_M2_8GB when running on M2
GGUF_DIR ?= models/

.PHONY: all bench graphs clean smoke verify clean-results help \
        bench-01 bench-02 bench-03 bench-04 bench-05 \
        bench-06 bench-07-modeled bench-07-llamacpp bench-08 bench-09

# ── Default ───────────────────────────────────────────────────────────────────
all: bench graphs

# ── Smoke test (quick grader verification — no GGUF needed) ──────────────────
smoke:
	python benchmarks/run_smoke_test.py

# ── Full benchmark suite ──────────────────────────────────────────────────────
# bench-07 uses the modeled path by default (projects Q4/Q8 from F16 baseline).
# To run with real GGUF files via llama.cpp: use bench-07-llamacpp instead.
bench: bench-01 bench-02 bench-03 bench-04 bench-05 bench-06 bench-07-modeled bench-08 bench-09

bench-01:
	python benchmarks/01_ttft_vs_prompt_length.py \
	    --model $(MODEL) --device_name $(DEVICE)

bench-02:
	python benchmarks/02_per_token_latency_vs_context.py \
	    --model $(MODEL) --device_name $(DEVICE)

bench-03:
	python benchmarks/03_e2e_latency_vs_output_length.py \
	    --model $(MODEL) --device_name $(DEVICE)

bench-04:
	python benchmarks/04_throughput_vs_prompt_length.py \
	    --model $(MODEL) --peak_bw $(PEAK_BW) --device_name $(DEVICE)

bench-05:
	python benchmarks/05_inter_token_latency_timeline.py \
	    --model $(MODEL) --device_name $(DEVICE)

bench-06:
	python benchmarks/06_cold_vs_warm_run.py \
	    --model $(MODEL) --device_name $(DEVICE)

# Modeled path: projects Q4/Q8 speedup from measured F16 baseline.
# Output is labeled measurement_type=modeled. This is the default.
bench-07-modeled:
	python benchmarks/07_quantization_speedup.py \
	    --mode modeled --model $(MODEL) --device_name $(DEVICE)

# llama.cpp path: requires real GGUF files. Produces measured speedup values.
# Usage: make bench-07-llamacpp GGUF_DIR=models/
bench-07-llamacpp:
	python benchmarks/07_quantization_speedup.py \
	    --backend llamacpp \
	    --llama_bench $(GGUF_DIR)/llama-bench \
	    --models Q4=$(GGUF_DIR)/model-Q4_K_M.gguf \
	              Q8=$(GGUF_DIR)/model-Q8_0.gguf \
	              F16=$(GGUF_DIR)/model-F16.gguf \
	    --device_name $(DEVICE)

bench-08:
	python benchmarks/08_kvcache_size_vs_context.py \
	    --device_name $(DEVICE)

bench-09:
	python benchmarks/09_latency_decomposition.py \
	    --model $(MODEL) --also_run_scaling --device_name $(DEVICE)

# ── Graphs ────────────────────────────────────────────────────────────────────
graphs:
	mkdir -p graphs
	python analysis/generate_research_graphs.py

# ── Verify (check CSVs exist then generate graphs) ────────────────────────────
verify:
	python benchmarks/run_smoke_test.py
	python analysis/generate_research_graphs.py --check_only

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf graphs/*.png
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-results:
	rm -rf results/tmp results/smoke_test.json

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Token-Generation Latency Benchmarking — Mac M4 vs M2"
	@echo ""
	@echo "  Usage:"
	@echo "    make smoke              Quick grader test (no model download needed for schema tests)"
	@echo "    make bench              Run all 9 benchmarks (default model: TinyLlama-1.1B)"
	@echo "    make bench-07-modeled   Run quantization benchmark (modeled projection from F16)"
	@echo "    make bench-07-llamacpp  Run quantization via llama.cpp GGUF (requires GGUF files)"
	@echo "    make graphs             Generate all figures from CSVs"
	@echo "    make verify             Check CSVs exist, then generate graphs"
	@echo "    make all                Run benchmarks + generate graphs"
	@echo "    make clean              Remove generated graphs and cache"
	@echo "    make clean-results      Remove temporary result files"
	@echo ""
	@echo "  Options:"
	@echo "    MODEL=<hf-model-id>     Override model  (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
	@echo "    PEAK_BW=<GB/s>          Set device peak bandwidth  (M4: 120, M2: 100)"
	@echo "    DEVICE=<name>           Device label for output paths  (Mac_M4_16GB or Mac_M2_8GB)"
	@echo "    GGUF_DIR=<path>         Path to GGUF model files  (for bench-07-llamacpp)"
	@echo ""
	@echo "  Examples:"
	@echo "    make bench DEVICE=Mac_M2_8GB PEAK_BW=100"
	@echo "    make bench-04 PEAK_BW=100 DEVICE=Mac_M2_8GB"
	@echo "    make bench-07-llamacpp GGUF_DIR=/path/to/models"
	@echo ""
