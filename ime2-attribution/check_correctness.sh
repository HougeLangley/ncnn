#!/bin/bash
# 验证 ncnn IME2 路径与 RVV 路径输出逐字节一致（同一模型、同一 prompt、贪心解码）
# 乱码防护的第一道闸门：IME2 若引入数值错误，这里立刻现形。
set -u
ROOT=/home/ubuntu/k3-ncnn/ncnn_llm
OUT=/home/ubuntu/k3-compare
export LD_LIBRARY_PATH=/opt/riscv-spacemit/lib:${LD_LIBRARY_PATH:-}

PROMPT="Please write a short poem about summer."

# 1) IME2 build @ A100 (IME2 激活) —— 保留生成文本
ai-run "$ROOT/build-ime/k3bench" --model "$ROOT/assets/qwen3_0.6b" \
    --threads 8 --max-tokens 64 --repeat 1 --warmup 0 --prompt "$PROMPT" \
    > "$OUT/logs/correct_ime2_a100.log" 2>&1

# 2) RVV build @ A100 (同一硅片，纯 RVV)
ai-run "$ROOT/build-rvv/k3bench" --model "$ROOT/assets/qwen3_0.6b" \
    --threads 8 --max-tokens 64 --repeat 1 --warmup 0 --prompt "$PROMPT" \
    > "$OUT/logs/correct_rvv_a100.log" 2>&1

# 2.5) 先确认 "X100 那一腿" 真的跑在 X100 上（VLEN=256）；否则闸门会假通过
VLEN_X100=$(x100-run "$OUT/bench/vlen_probe" 2>/dev/null | grep -o '[0-9]\+' | head -1)
if [ "$VLEN_X100" != "256" ]; then
    echo "!! X100 腿未生效（实测 VLEN=${VLEN_X100:-未知}，期望 256）—— 结果不可作为黄金参考，闸门中止"
    exit 2
fi

# 3) RVV build @ X100 (另一簇，黄金参考)
x100-run "$ROOT/build-rvv/k3bench" --model "$ROOT/assets/qwen3_0.6b" \
    --threads 8 --max-tokens 64 --repeat 1 --warmup 0 --prompt "$PROMPT" \
    > "$OUT/logs/correct_rvv_x100.log" 2>&1

extract() { sed -n '/---- GENERATED TEXT ----/,/---- END ----/p' "$1" | sed '1d;$d'; }

extract "$OUT/logs/correct_ime2_a100.log" > "$OUT/logs/text_ime2.txt"
extract "$OUT/logs/correct_rvv_a100.log" > "$OUT/logs/text_rvv_a100.txt"
extract "$OUT/logs/correct_rvv_x100.log" > "$OUT/logs/text_rvv_x100.txt"

echo "== IME2(A100) vs RVV(A100):"
if cmp -s "$OUT/logs/text_ime2.txt" "$OUT/logs/text_rvv_a100.txt"; then echo "  BYTE-IDENTICAL ✓"; else echo "  DIFFER ✗"; diff "$OUT/logs/text_ime2.txt" "$OUT/logs/text_rvv_a100.txt" | head; fi
echo "== IME2(A100) vs RVV(X100 黄金参考):"
if cmp -s "$OUT/logs/text_ime2.txt" "$OUT/logs/text_rvv_x100.txt"; then echo "  BYTE-IDENTICAL ✓"; else echo "  DIFFER ✗"; diff "$OUT/logs/text_ime2.txt" "$OUT/logs/text_rvv_x100.txt" | head; fi

echo; echo "---- IME2 生成的文本 ----"; cat "$OUT/logs/text_ime2.txt"
