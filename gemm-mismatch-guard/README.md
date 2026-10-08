# Gemm fp16/fp32 mismatch guard — reproducer, patch and measurements

Companion data for Tencent/ncnn issue #7050 and the fix PR.

* `gemm_repro.cpp` — standalone reproducer (bare `Gemm` layer, transB=1,
  constantB=1). `NO_FP16_STORAGE=1` makes it take the fp32 path, `FP16_IN=1`
  casts the input to fp16 first (what a `Net` does), `BENCH=<n>` times
  `forward()`, `DUMP=<file>` writes the output for byte comparison, and
  `NAIVE=1` uses `ncnn::create_layer_naive("Gemm")` as ground truth.
* `gemm-mismatch-guard.patch` — the one-file fix.
* `raw.csv`, `summary.csv`, `paired_tests.csv`, `tidy_reps.csv` — paired A/B of
  the unpatched and patched `libncnn` (8 shape x mode configs, 5 replicates,
  order alternated per replicate), analysed with `analyze.R`.
* `01-guard-cost.png`, `02-guard-cost-delta.png` — the same, plotted.

Environment: SpacemiT K3, A100 cluster `cpu8-15` (VLEN=1024), SpacemiT GCC
17.0.0 20260928 + binutils 2.47.50.20260928, ncnn `16bdc2d`.

Result: no statistically significant cost. The paired latency difference is
between -1.2 % and +2.6 % with all Benjamini-Hochberg adjusted p-values
>= 0.13, and the guard sits outside the kernel loop.
