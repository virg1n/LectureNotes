# LectureNotes

An agent that turns a lecture topic and, optionally, slides into **typeset LaTeX lecture notes** and a **cheat sheet**. The notes include definitions, theorems, examples, formulas and code-generated figures (matplotlib or TikZ). The agent runs on a model under 4B parameters: Qwen3-4B-Instruct-2507 fine-tuned with SFT and then GRPO on V100 GPUs.

```bash
lecturenotes generate --topic "The perceptron" --slides lecture9.pdf --pages 11-21 --length "4 pages" \
                      --cheatsheet-pages 1 --config configs/agent/student_vllm.yaml --out out/perceptron
# -> notes.pdf, notes.tex, figures/, cheatsheet.pdf, report.md, trace.jsonl
```

The full write-up with results and examples is in [`docs/report/REPORT.md`](docs/report/REPORT.md). Sample outputs are in [`examples/out/`](examples/out).

## Idea: a small model, guided by a harness

A 4B model cannot write ten pages of LaTeX that compile in one pass. Instead, it only does short steps that can be checked: a plan, one section, one figure or one repair. A deterministic **harness** checks every step and returns structured feedback:

```
plan (JSON) → section drafts (best-of-n) → verify ⇄ SEARCH/REPLACE repairs → revise (length, coverage)
            → assemble + compile → cheat sheet (fit to N pages)
```

The verifier runs these checks:
- LaTeX lint and autofix;
- an isolated compile of each section, so errors point to lines the model wrote;
- figures rendered in a sandbox, then checked for blank output and overlapping elements;
- web images from Wikimedia Commons, free licenses only;
- an optional VLM relevance judge for figures;
- length, key-point coverage, repetition and filler phrases.

A section or figure that still fails after repairs is cut out, so the document always compiles.

**The same harness has four roles:**

| Role | Code |
|---|---|
| Inference-time verifier and repair loop | `agent/section.py`, `agent/cheatsheet.py`, `harness/verifier.py` |
| Training-data filter | `data/process.py`, `data/build_sft.py` |
| RL reward | `harness/rewards.py`, `training/rl_common.py` |
| Benchmark on 30 held-out topics | `eval/benchmark.py` |

## Training

1. **Data.** The SFT data comes from four sources. Only examples that passed the harness are kept.
   - arXiv lecture notes and tutorials under permissive licenses.
   - DaTikZ, a caption-to-TikZ dataset.
   - Synthetic repair tasks made by injecting errors.
   - Teacher trajectories: Qwen3-14B ran the full agent on 400 topics.
2. **SFT** has two stages, both LoRA r64 (`configs/train/sft_lora.yaml`, `sft_stage2.yaml`). Stage 1 uses all task types; stage 2 uses only the target notes style. The result is merged into `checkpoints/sft-merged`.
3. **GRPO** (`training/grpo.py`, `configs/train/grpo.yaml`) trains a fresh LoRA on the merged SFT model with the harness score as reward. The KL reference is the SFT policy. Rollouts run on vLLM. Because vLLM's LoRA kernels do not build on V100 (sm_70), the adapter is merged and vLLM is restarted every round. The final model is the **round-20** checkpoint, because the reward degraded later in training.

## Results (30 held-out topics, same harness)

| Metric | Base Qwen3-4B | SFT | GRPO (final) |
|---|---|---|---|
| First drafts that compile | 80% | **100%** | **100%** |
| Figures rendered and lint-clean | 44% | 82% | **87%** |
| Length error | 16% | 9% | **7%** |
| Key-point coverage | 94.0% | 94.6% | **95.6%** |
| Mean section reward | 0.74 | 0.80 | **0.82** |
| Repair calls per section | 1.77 | **0.89** | 0.92 |
| Units cut out after failed repairs (per lecture) | 0.9 | **0** | **0** |
| Figures per lecture | ≈3 | ≈8 | ≈8 |
| VLM judge: relevant / irrelevant figures | ≈98% / 0% | 88% / 6% | **89% / 6%** |

## Setup

```bash
# local: tests and an offline demo with a mock model
pip install -e ".[dev]"
pytest -q
lecturenotes generate --topic "Metric spaces" --config configs/agent/mock.yaml --out out/demo
lecturenotes doctor                                   # which TeX/Python features work on this machine

# GPU server (paths set in scripts/env.sh)
scripts/setup_server.sh && scripts/setup_tex.sh
scripts/pipeline.sh data|process|tasks|teacher|sft-data|sft|eval|grpo
GPUS=0 PORT=8000 NAME=student scripts/serve_vllm.sh checkpoints/final-model   # then generate with student_vllm.yaml
```

The model checkpoints are not in this repository.

## Web app

`webapp/` is a small website with no dependencies beyond the Python standard library. You upload slides; it connects to the GPU server over ssh and runs `scripts/web_job.py` there. That script takes a free GPU, starts vLLM with the final model, generates the notes and stops vLLM. The site then shows the notes and cheat sheet. See [`webapp/README.md`](webapp/README.md).

## Layout

```
src/lecturenotes/
  agent/      orchestrator, section state machine, planner, cheatsheet, length budget
  harness/    verifier, SEARCH/REPLACE edits + quarantine, rewards
  latex/      lint, autofix, compiler + log parsing, TeX capability probes, assembler, templates/
  figures/    figure blocks, sandboxed runner, figure lint, style, web images, VLM relevance
  llm/        OpenAI-compatible (vLLM), HF local, Anthropic, mock backends
  ingest/     slides (PDF / PPTX / Beamer / Markdown), slide selection
  prompts/    task prompt templates (identical to the training format)
  data/       sources (arXiv, GitHub, HF), processing, dedup, synthetic tasks, teacher distillation, SFT mix
  training/   sft, grpo, rl_common, merge_lora
  eval/       benchmark, pairwise judge
configs/      agent/, train/, deepspeed/, data/
scripts/      server setup, vLLM serving, pipeline, base-model bake-off, VLM server, web job runner
webapp/       upload site (server.py, index.html, deploy/)
```
