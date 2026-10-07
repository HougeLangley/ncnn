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

## Companion scripts (these depend on the `ncnn_llm` application, not on ncnn itself)

* `check_correctness.sh` — the three-leg byte-identical gate: greedy decoding of
  the same prompt must produce identical bytes from `IME2@A100`, `RVV@A100` and
  `RVV@X100`. The X100 leg first asserts VLEN==256, because that leg silently
  degrading once produced a false pass.
* `x100-run` — the mirror of `ai-run` for the X100 cluster (cpu0-7).

Both are reproduced here for reference; they invoke `k3bench` from the
`ncnn_llm` repository, so they are not runnable from an ncnn checkout alone.
