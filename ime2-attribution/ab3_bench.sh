#!/bin/bash
# ab3_bench.sh -- THREE-arm A/B/C to attribute the gain between two separate PRs.
#
#   arm "off"   : build-ime2-off  (Gemm IME2 OFF, SDPA IME2 OFF)   <- upstream default (fp16sa)
#   arm "gemm"  : build-ime2-on + NCNN_SDPA_IME2=0 (Gemm IME2 ON, SDPA IME2 OFF)
#                                                                    <- isolates ncnn PR #7032
#   arm "both"  : build-ime2-on  (Gemm IME2 ON, SDPA IME2 OFF->ON) <- + the SDPA/RMSNorm work (PR #7037)
#
# Attribution:  effect(PR #7032) = gemm - off ;  effect(SDPA work) = both - gemm
#
# Same discipline as ab_bench.sh: fresh server process per session, counterbalanced
# order, preflight refusal on any competing benchmark, trap cleanup on every exit path.
#
# usage: ab3_bench.sh <replicates>
set -u

REPS=${1:-5}
SRC=/home/ubuntu/k3-ncnn/bench-rebase/ncnn_llm
MODEL=/home/ubuntu/k3-ncnn/ncnn_llm/assets/qwen3_0.6b
OUT=/home/ubuntu/k3-ncnn/bench-rebase/results3
THREADS=8
PP="128 512"
TG="32 128"
RUNS=3
LLAMA_BENCHY=/home/ubuntu/.local/bin/llama-benchy
mkdir -p "$OUT"
export OMP_WAIT_POLICY=ACTIVE

SRV_PID=""
cleanup() {
    if [ -n "$SRV_PID" ]; then
        kill -9 "$SRV_PID" 2>/dev/null
        for c in $(ps -eo pid,ppid --no-headers 2>/dev/null | awk -v p="$SRV_PID" '$2==p {print $1}'); do
            kill -9 "$c" 2>/dev/null
        done
        SRV_PID=""
    fi
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

preflight() {
    local bad
    bad=$(ps -eo pid,args --no-headers | awk -v me="$$" '
        $1 != me && ($0 ~ /build-ime2-(on|off)\/ncnn_llm_server +--model/ || $0 ~ /llama[-_]benchy +--base-url/) && $0 !~ /awk/ {print $1}' | tr '\n' ' ')
    if [ -n "${bad// /}" ]; then
        echo "!!! PREFLIGHT FAILED: another benchmark is already running (pids: $bad)"
        echo "    Refusing to start -- concurrent runs corrupt the shared IME2 matrix unit."
        exit 2
    fi
    echo "preflight ok: no competing server / llama-benchy"
}
preflight

wait_ready() {
    local port=$1 n=0
    while [ $n -lt 300 ]; do
        curl -sf --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && return 0
        sleep 1; n=$((n+1))
    done
    return 1
}

run_one() {
    local arm=$1 rep=$2
    local port=$((19100 + RANDOM % 700))
    local bin=$SRC/build-ime2-on/ncnn_llm_server
    local envs=()
    case "$arm" in
        off)  bin=$SRC/build-ime2-off/ncnn_llm_server ;;
        gemm) envs=(env NCNN_SDPA_IME2=0) ;;          # Gemm IME2 only  (PR #7032)
        both) ;;                                       # Gemm + SDPA IME2
        *) echo "unknown arm $arm"; return 1 ;;
    esac
    local tag
    tag=$(printf "rep%02d_%s" "$rep" "$arm")
    local srvlog=$OUT/${tag}.server.log

    echo "=== [$tag] starting arm=$arm on port $port ==="
    ai-run taskset -c 8-15 ${envs[@]+"${envs[@]}"} "$bin" --model "$MODEL" --threads "$THREADS" \
        --port "$port" --name qwen3-0.6b > "$srvlog" 2>&1 &
    SRV_PID=$!

    if ! wait_ready "$port"; then
        echo "!!! [$tag] server failed to become ready"; tail -20 "$srvlog"; cleanup; return 1
    fi
    echo "    ready (pid $SRV_PID)"

    "$LLAMA_BENCHY" \
        --base-url "http://127.0.0.1:$port/v1" \
        --model qwen3-0.6b \
        --tokenizer "$MODEL" \
        --pp $PP --tg $TG \
        --runs "$RUNS" \
        --skip-coherence \
        --latency-mode none \
        --emit-progress "$OUT/${tag}.progress.jsonl" \
        --format json \
        --save-result "$OUT/${tag}.json" \
        > "$OUT/${tag}.stdout.log" 2>&1
    local rc=$?
    echo "    llama-benchy exit=$rc"

    cleanup
    if curl -sf --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
        echo "    WARNING: port $port still answering after cleanup"
    fi
    sleep 8
    return $rc
}

echo "############ 3-ARM A/B/C START $(date -Is) reps=$REPS ############"
FAILED=0
for r in $(seq 1 "$REPS"); do
    # rotate the arm order every replicate so no arm is systematically first/last
    case $((r % 3)) in
        1) ORDER="off gemm both" ;;
        2) ORDER="gemm both off" ;;
        0) ORDER="both off gemm" ;;
    esac
    echo "--- replicate $r order: $ORDER (rotated) ---"
    for a in $ORDER; do
        run_one "$a" "$r" || { echo "!!! replicate $r arm $a FAILED"; FAILED=$((FAILED+1)); }
    done
done
echo "############ 3-ARM DONE $(date -Is) failed_sessions=$FAILED ############"
echo "json files: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
