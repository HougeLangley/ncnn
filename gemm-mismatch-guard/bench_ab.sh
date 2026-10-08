#!/bin/bash
# Paired A/B of the unpatched vs patched libncnn: does the added
# elembits guard cost anything on the paths it does not affect?
set -u
B=/home/ubuntu/k3-ncnn/bench-rebase
OUT=$B/fixbench
REPS=${1:-5}
CSV=$OUT/raw.csv
: > "$CSV"
echo "variant,shape,mode,rep,ms,gflops" >> "$CSV"

# shape list: "<M> <N> <K>"
SHAPES=("1 151936 1024" "128 2048 1024" "1 4096 1024" "512 2048 1024")

run_one() {
  local variant=$1 shape=$2 mode=$3 rep=$4
  local bin=$B/issue/gr_$variant
  local envs=()
  [ "$mode" = "fp16in" ] && envs=(env FP16_IN=1)
  [ "$mode" = "fp32" ] && envs=(env NO_FP16_STORAGE=1)
  local out
  out=$(ai-run taskset -c 8-15 ${envs[@]+"${envs[@]}"} "$bin" $shape 8 1 2>/dev/null \
        | BENCH=5 xargs -0 true 2>/dev/null; \
        BENCH=5 ai-run taskset -c 8-15 ${envs[@]+"${envs[@]}"} "$bin" $shape 8 1 2>/dev/null | grep '^BENCH')
  echo "$out"
}

for rep in $(seq 1 "$REPS"); do
  for shape in "${SHAPES[@]}"; do
    for mode in fp16in fp32; do
      # alternate the order every replicate to cancel drift
      if [ $((rep % 2)) -eq 1 ]; then ORDER="orig fixed"; else ORDER="fixed orig"; fi
      for v in $ORDER; do
        envs=(); [ "$mode" = "fp16in" ] && envs=(FP16_IN=1); [ "$mode" = "fp32" ] && envs=(NO_FP16_STORAGE=1)
        line=$(BENCH=5 ai-run taskset -c 8-15 env "${envs[@]+"${envs[@]}"}" \
                 $B/issue/gr_$v $shape 8 1 2>/dev/null | grep '^BENCH')
        ms=$(echo "$line" | sed -n 's/.*ms=\([0-9.]*\).*/\1/p')
        gf=$(echo "$line" | sed -n 's/.*GFLOPs=\([0-9.]*\).*/\1/p')
        [ -n "$ms" ] && echo "$v,\"$shape\",$mode,$rep,$ms,$gf" >> "$CSV"
      done
    done
  done
  echo "replicate $rep done"
done
echo "=== rows: $(($(wc -l < "$CSV") - 1)) ==="
