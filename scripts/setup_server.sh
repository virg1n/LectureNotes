#!/usr/bin/env bash
# Python environment for 8 x V100 (sm_70). Versions are pinned to a stack known to work on Volta:
#   torch 2.7.1 + cu126 (cu128 wheels dropped sm_70), vLLM 0.10.1.1, transformers 4.57.x, peft 0.17.x
set -euo pipefail
source "$(dirname "$0")/env.sh"
PY=${PY:-python3}
cd "$(dirname "$0")/.."
if [ ! -x "$VENV/bin/python" ]; then
  $PY -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel
pip install -q "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
pip install -q "vllm==0.10.1.1" "transformers==4.57.6" "peft==0.17.1" "accelerate>=1.0" "datasets>=2.20" \
               "huggingface_hub>=0.24" anthropic
pip install -q -e ".[dev]"
# optional: DeepSpeed for full fine-tuning (JIT-builds ops with the system nvcc)
if [ "${WITH_DEEPSPEED:-0}" = "1" ]; then pip install -q "deepspeed>=0.15"; fi
python - <<'PY'
import torch, vllm, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", torch.cuda.get_arch_list())
assert "sm_70" in torch.cuda.get_arch_list(), "this torch build cannot run on V100"
print("gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("vllm", vllm.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
PY
echo "venv ready: source $VENV/bin/activate"
