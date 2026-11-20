#!/usr/bin/env bash
# End-to-end runbook on the 8 x V100 node. Run stages individually:  scripts/pipeline.sh <stage>
#   data     collect arXiv / GitHub / DaTikZ sources                      (CPU, network, hours)
#   process  normalise + compile-filter sections with the harness         (CPU, ~20 workers)
#   tasks    real-doc tasks, error-injection repairs, TikZ figure tasks   (CPU)
#   teacher  serve the teacher (TP=4) and distil full agent trajectories  (GPU 0-3 + CPU)
#   sft-data mix everything into data/sft/{train,val,prefs}.jsonl
#   sft      two-stage LoRA SFT (all data, then house-style only) -> checkpoints/sft-merged
#   eval     serve a merged model (MODEL=..., default sft-merged) and run the benchmark
#   grpo     GRPO, fresh LoRA on sft-merged (rollouts on vLLM, training on GRPO_TRAIN_GPUS)
# GRPO trains a *fresh* LoRA on the *merged* SFT policy, so the adapter-disabled reference
# used for the KL term is that policy and not the raw base model.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
TEACHER=${TEACHER:-Qwen/Qwen3-14B}
W=${WORKERS:-20}
# GPUs this run may use (GPU 3 was busy with another user's job on 2026-11-16); override per run
export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,4,5,6,7}
NGPU=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
mkdir -p logs data

case "${1:?stage}" in
data)
  # the arXiv API answers 429 to this node's shared IP -> select from the HF metadata snapshot instead
  [ -s data/raw/arxiv/meta.jsonl ] || python -m lecturenotes.data.sources.arxiv harvest-hf --out data/raw/arxiv/meta.jsonl --licenses "${ARXIV_LICENSES:-permissive}"
  python -m lecturenotes.data.sources.arxiv download --meta data/raw/arxiv/meta.jsonl --out data/raw/arxiv/src \
      --licenses "${ARXIV_LICENSES:-permissive}" --max "${ARXIV_MAX:-4000}"
  python -m lecturenotes.data.sources.github --out data/raw/github --max-repos "${GITHUB_MAX:-2000}" || true
  python -m lecturenotes.data.sources.hf datikz --out data/raw/datikz.jsonl --max 60000 ;;
process)
  python -m lecturenotes.data.process --raw data/raw/arxiv/src data/raw/github/src \
      --out data/processed/sections.jsonl --workers "$W" ;;
tasks)
  python -m lecturenotes.data.synth.tasks real --sections data/processed/sections.jsonl --out data/tasks/real.jsonl
  python -m lecturenotes.data.synth.tasks repair --sections data/processed/sections.jsonl --out data/tasks/repair.jsonl --n 15000 --workers "$W"
  python -m lecturenotes.data.synth.tasks tikz --datikz data/raw/datikz.jsonl --out data/tasks/tikz.jsonl --n 12000 --workers "$W" ;;
teacher)
  GPUS=${TEACHER_GPUS:-0,1} TP=2 PORT=8001 NAME=teacher MAX_LEN=24576 scripts/serve_vllm.sh "$TEACHER" > logs/teacher_vllm.log 2>&1 &
  until curl -sf http://127.0.0.1:8001/health >/dev/null; do sleep 10; done
  LN_TEACHER_URL=http://127.0.0.1:8001/v1 python -m lecturenotes.data.synth.teacher --config configs/agent/teacher.yaml \
      --out data/teacher/runs --n-topics "${N_TOPICS:-400}" --concurrency 6 --with-slides 0.35
  kill %1
  # /public quota is tight (~49 GB free): drop the teacher weights (~30 GB) after distillation
  [ "${KEEP_TEACHER:-0}" = 1 ] || rm -rf "$HF_HOME/hub/models--${TEACHER//\//--}" ;;
sft-data)
  python -m lecturenotes.data.synth.tasks figrepair --figures data/tasks/tikz.jsonl --traces data/teacher/runs \
      --out data/tasks/figrepair.jsonl --n 6000 --workers "$W"
  python -m lecturenotes.data.synth.tasks webquery --traces data/teacher/runs --out data/tasks/webquery.jsonl
  python -m lecturenotes.data.build_sft --tasks data/tasks/*.jsonl --traces data/teacher/runs --out data/sft \
      --mix configs/data/mix.yaml ;;
sft)
  # stage 1: everything (real + synthetic + teacher); stage 2: house style only, lower lr
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node $NGPU -m lecturenotes.training.sft --config configs/train/sft_lora.yaml 2>&1 | tee logs/sft1.log
  python -m lecturenotes.training.merge_lora --base "$BASE_MODEL" --adapter checkpoints/sft-lora --out checkpoints/sft1-merged
  SFT1_MERGED=checkpoints/sft1-merged CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node $NGPU -m lecturenotes.training.sft \
      --config configs/train/sft_stage2.yaml 2>&1 | tee logs/sft2.log
  python -m lecturenotes.training.merge_lora --base checkpoints/sft1-merged --adapter checkpoints/sft2-lora --out checkpoints/sft-merged
  rm -rf checkpoints/sft1-merged ;;  # disk quota: keep only the final merged SFT model
eval)
  MODEL=${MODEL:-checkpoints/sft-merged}
  GPUS=${EVAL_GPU:-0} PORT=8000 NAME=student scripts/serve_vllm.sh "$MODEL" > logs/student_vllm.log 2>&1 &
  srv=$!
  until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 10; done
  python -m lecturenotes.eval.benchmark --config configs/agent/student_vllm.yaml --out "eval_runs/$(basename "$MODEL")"
  kill $srv ;;
grpo)
  GG=${GRPO_TRAIN_GPUS:-4,5,6,7}
  POLICY_BASE=${POLICY_BASE:-checkpoints/sft-merged} CUDA_VISIBLE_DEVICES=$GG torchrun --nproc_per_node "$(echo "$GG" | tr -cd , | wc -c | xargs expr 1 +)" -m lecturenotes.training.grpo --config configs/train/grpo.yaml 2>&1 | tee logs/grpo.log
  python -m lecturenotes.training.merge_lora --base "${POLICY_BASE:-checkpoints/sft-merged}"       --adapter checkpoints/grpo/final --out checkpoints/grpo-merged ;;
*) echo "unknown stage $1"; exit 1 ;;
esac
