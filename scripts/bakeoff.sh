#!/usr/bin/env bash
# Pick the base model with the harness, not with vibes: serve each candidate on its own GPU
# (fp16, as it will run on V100) and run the agent benchmark zero-shot on N held-out topics.
#   N=8 scripts/bakeoff.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
CANDIDATES=(${CANDIDATES:-microsoft/Phi-4-mini-instruct Qwen/Qwen3-4B-Instruct-2507 ibm-granite/granite-4.2-3b HuggingFaceTB/SmolLM3-3B})
mkdir -p logs eval_runs/bakeoff
servers=(); evals=()
trap 'kill "${servers[@]}" 2>/dev/null || true' EXIT
i=0
for m in "${CANDIDATES[@]}"; do
  port=$((8200+i))
  GPUS=$i PORT=$port NAME=student MAX_LEN=16384 scripts/serve_vllm.sh "$m" > "logs/bakeoff_$i.log" 2>&1 &
  servers+=($!)
  i=$((i+1))
done
i=0
for m in "${CANDIDATES[@]}"; do
  port=$((8200+i)); name=$(basename "$m")
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do sleep 10; done
  LN_STUDENT_URL="http://127.0.0.1:$port/v1" python -m lecturenotes.eval.benchmark \
      --config configs/agent/student_vllm.yaml --out "eval_runs/bakeoff/$name" --n "${N:-8}" --concurrency 2 \
      > "logs/bakeoff_eval_$name.log" 2>&1 &
  evals+=($!)
  i=$((i+1))
done
wait "${evals[@]}" || true
python -m lecturenotes.eval.benchmark --compare eval_runs/bakeoff/*
