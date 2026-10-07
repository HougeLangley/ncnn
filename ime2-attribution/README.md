# IME2 attribution run — raw data

Three-arm A/B/C used to attribute the gain between ncnn PR #7032 (Gemm IME2)
and PR #7037 (SDPA IME2 flash prefill).

* `01-attribution-bars.png`     — replicate means, 95% CI, per-kind facets
* `02-attribution-forest.png`   — paired %-change forest plot
* `03-attribution-combined.png` — both, stacked
* `rep<NN>_<arm>.json`          — raw llama-benchy output (per-run `values`)
* `summary.csv`, `paired_tests.csv`, `tidy_reps.csv`, `tidy_runs.csv`
* `ab3_bench.sh`, `ab3_analyze.R` — driver and analysis scripts
* `concurrency-audit-supervisor3.log` — process-table sampling (280 samples, 0 with >1 server)

Environment: SpacemiT K3, A100 cluster cpu8-15, GCC 17.0.0 20260928 +
binutils 2.47.50, Qwen3-0.6B, threads=8, 5 replicates per arm.
