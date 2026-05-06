#!/usr/bin/env bash
# run.sh – Entry point for the LoRA CUDA Optimization Agent
#
# The evaluation system runs:
#   bash run.sh
# from this directory.  After completion (or at 30-min timeout), it reads
# ./optimized_lora.cu for benchmarking.
#
# Environment variables (optional):
#   OPENAI_API_KEY   – your OpenAI API key (enables LLM-driven optimization)
#   OPENAI_MODEL     – model to use (default: gpt-4o)
#   LORA_TIME_BUDGET – agent time budget in seconds (default: 1500 = 25 min)
#   LORA_EVAL_D      – matrix dimension for local benchmarking (default: 4096)

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo " LoRA CUDA Optimization Agent"
echo " $(date)"
echo "============================================================"

# ── Ensure openai Python package is available ────────────────────────────────
if ! python3 -c "import openai" 2>/dev/null; then
    echo "[run.sh] Installing openai package …"
    pip install --quiet openai
fi

# ── Verify CUDA / PyTorch availability ──────────────────────────────────────
python3 - <<'PYCHECK'
import sys, torch
print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
if not torch.cuda.is_available():
    print("ERROR: CUDA GPU not available.", file=sys.stderr)
    sys.exit(1)
PYCHECK

# ── Run the agent ────────────────────────────────────────────────────────────
TIME_BUDGET="${LORA_TIME_BUDGET:-1500}"
EVAL_D="${LORA_EVAL_D:-4096}"
MODEL="${OPENAI_MODEL:-gpt-4o}"

echo ""
echo "[run.sh] Starting agent  (budget=${TIME_BUDGET}s, d=${EVAL_D}, model=${MODEL})"
echo ""

python3 agent.py \
    --time-budget "${TIME_BUDGET}" \
    --model       "${MODEL}" \
    --d           "${EVAL_D}"

echo ""
echo "[run.sh] Agent finished."
echo "[run.sh] Final kernel: ./optimized_lora.cu"
echo "============================================================"
