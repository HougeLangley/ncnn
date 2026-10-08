#!/bin/bash
# Systematic failure-mode sweep: bare Gemm, fp32 input, use_fp16_storage default.
# For every shape we run, for both builds:
#   fp32 run  (NO_FP16_STORAGE=1) -> reference output + must succeed
#   default run                   -> classify crash / wrong / correct / clean_error
set -u
B=/home/ubuntu/k3-ncnn/bench-rebase
OUT=$B/deepdive
CSV=$OUT/sweep.csv
: > "$CSV"
echo "build,M,N,K,fp32_exit,default_exit,fp32_ok,verdict,forward_ret,logged" >> "$CSV"

NS="1024 4096 16384 32768 49152 65536 70000 74000 75700 75800 80000 98304 131072 151936"
KS="256 1024 2048"
MS="1 32"

for M in $MS; do for K in $KS; do for N in $NS; do
  # reference on the unpatched build
  NO_FP16_STORAGE=1 DUMP=/tmp/sw_ref.bin ai-run taskset -c 8-15 $B/issue/gr_orig $M $N $K 8 1 >/tmp/sw_r.txt 2>&1
  ref_exit=$?
  for build in orig fixed; do
    NO_FP16_STORAGE=1 DUMP=/tmp/sw_f32.bin ai-run taskset -c 8-15 $B/issue/gr_$build $M $N $K 8 1 >/tmp/sw_f.txt 2>&1
    f32_exit=$?
    ok="na"
    if [ $f32_exit -eq 0 ] && [ -f /tmp/sw_f32.bin ] && [ -f /tmp/sw_ref.bin ]; then
      if cmp -s /tmp/sw_f32.bin /tmp/sw_ref.bin; then ok="yes"; else ok="no"; fi
    fi
    DUMP=/tmp/sw_def.bin ai-run taskset -c 8-15 $B/issue/gr_$build $M $N $K 8 1 >/tmp/sw_d.txt 2>&1
    def_exit=$?
    ret=$(grep -o "forward() returned [-0-9]*" /tmp/sw_d.txt | head -1 | grep -o '[-0-9]*$')
    logged=$(grep -q "Gemm: layer was built for fp16 input" /tmp/sw_d.txt && echo yes || echo no)
    if [ $def_exit -eq 139 ]; then verdict="crash"
    elif [ $def_exit -ne 0 ]; then verdict="exit_$def_exit"
    elif [ "${ret:-0}" -ne 0 ]; then verdict="clean_error"
    elif [ -f /tmp/sw_def.bin ] && [ -f /tmp/sw_ref.bin ] && cmp -s /tmp/sw_def.bin /tmp/sw_ref.bin; then verdict="correct"
    else verdict="silent_wrong"; fi
    echo "$build,$M,$N,$K,$f32_exit,$def_exit,$ok,$verdict,${ret:-0},$logged" >> "$CSV"
    rm -f /tmp/sw_def.bin /tmp/sw_f32.bin
  done
  echo "  M=$M K=$K N=$N done ($(wc -l < $CSV) rows)"
done; done; done
rm -f /tmp/sw_ref.bin /tmp/sw_r.txt /tmp/sw_f.txt /tmp/sw_d.txt
echo "########## SWEEP DONE rows=$(($(wc -l < $CSV) - 1)) ##########"
