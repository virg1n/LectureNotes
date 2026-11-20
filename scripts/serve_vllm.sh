#!/usr/bin/env bash
# Serve a model with vLLM's OpenAI API on V100s (fp16; no bf16 on sm_70).
#   GPUS=0 PORT=8000 NAME=student scripts/serve_vllm.sh checkpoints/sft-merged
#   GPUS=0,1,2,3 TP=4 PORT=8001 NAME=teacher scripts/serve_vllm.sh Qwen/Qwen3-32B
#   LORA=checkpoints/sft-lora scripts/serve_vllm.sh microsoft/Phi-4-mini-instruct   # base + adapter
set -euo pipefail
MODEL=${1:?model path or HF id}
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
ARGS=(--model "$MODEL" --served-model-name "${NAME:-student}" --port "${PORT:-8000}" --dtype half
      --tensor-parallel-size "${TP:-1}" --max-model-len "${MAX_LEN:-16384}"
      --gpu-memory-utilization "${GPU_UTIL:-0.90}" --disable-log-requests)
if [ -n "${LORA:-}" ]; then
  ARGS+=(--enable-lora --max-lora-rank 64 --lora-modules "${NAME:-student}=$LORA")
  ARGS[3]="${NAME:-student}-base"
fi
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" "${@:2}"
