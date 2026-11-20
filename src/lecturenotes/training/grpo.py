"""Group Relative Policy Optimisation with harness rewards, built for 8 x V100.

Topology (default): GPUs 0-3 serve rollouts with vLLM (one server per GPU), GPUs 4-7
train the LoRA policy with DDP. Rank 0 drives each round:

  1. publish the current adapter to the rollout servers (LoRA hot-swap, or merge +
     restart when the vLLM build cannot serve LoRA on this GPU)
  2. sample a task-mixed batch of prompts; G completions per prompt
  3. score every completion with the harness (LaTeX compile, figure sandbox, ...)
     in a CPU process pool
  4. group-normalised advantages; groups without reward variance are dropped
Then all ranks: old/ref log-probs -> clipped policy-gradient loss with a k3 KL
penalty against the adapter-disabled reference, token-level normalisation.

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 -m lecturenotes.training.grpo \
        --config configs/train/grpo.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path

from .common import (PadCollator, TrainConfig, disable_lora_dropout, encode_chat, load_model_and_tokenizer,
                     warn_if_reference_is_raw_base, world_info)
from .rl_common import RewardPool, RLPrompt, VLLMPool, load_prompt_pool, rollout_openai

log = logging.getLogger(__name__)


def sample_batch(pool: dict[str, list[RLPrompt]], mix: dict[str, float], n: int, rng: random.Random) -> list[RLPrompt]:
    tasks = [t for t in mix if pool.get(t)]
    weights = [mix[t] for t in tasks]
    return [rng.choice(pool[rng.choices(tasks, weights)[0]]) for _ in range(n)]


def group_advantages(rewards: list[float], eps: float = 1e-4, std_floor: float = 0.1) -> list[float] | None:
    """Group-normalised advantages. Groups without variance carry no signal (dropped); the std floor
    keeps tiny reward differences (noise) from being blown up to unit-scale advantages."""
    m = sum(rewards) / len(rewards)
    sd = math.sqrt(sum((r - m) ** 2 for r in rewards) / len(rewards))
    if sd < eps:
        return None
    return [(r - m) / max(sd, std_floor) for r in rewards]


def token_logps(model, input_ids, attention_mask, labels):
    """Per-token log-probs of the completion (batch size 1, completion at the end).

    Only the last K positions are projected to the vocabulary (`logits_to_keep`),
    which avoids materialising a (seq x 150k) fp32 logit matrix for the prompt."""
    import torch

    real = int(attention_mask.sum().item())  # drop right padding so the completion really is at the end
    input_ids, attention_mask, labels = input_ids[:, :real], attention_mask[:, :real], labels[:, :real]
    k = int((labels != -100).sum().item()) + 1
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=k)
        logits = out.logits[:, -k:-1].float()
    except TypeError:  # older transformers
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -k:-1].float()
    tgt = labels[:, -k + 1:]
    lp = torch.gather(logits.log_softmax(-1), 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return lp, (tgt != -100).float()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    import torch.distributed as dist

    cfg = TrainConfig.load(a.config)
    x = cfg.extra
    G = int(x.get("group_size", 8))
    prompts_per_round = int(x.get("prompts_per_round", 64))
    rounds = int(x.get("rounds", 40))
    clip = float(x.get("clip_eps", 0.2))
    kl_beta = float(x.get("kl_beta", 0.02))
    ppo_epochs = int(x.get("ppo_epochs", 1))
    temperature = float(x.get("temperature", 1.0))
    mix = x.get("task_mix", {"write_section": 0.45, "repair_latex": 0.2, "figure": 0.15, "repair_figure": 0.1, "plan": 0.1})
    rollout_gpus = [int(g) for g in x.get("rollout_gpus", [0, 1, 2, 3])]
    publish = x.get("publish", "lora")  # "lora" (hot-swap) | "merge" (merge + restart)

    rank, world, local = world_info()
    if world > 1:
        from datetime import timedelta

        # ranks > 0 wait in barrier() while rank 0 runs rollouts + rewards (often > the 10 min NCCL default)
        dist.init_process_group("nccl", timeout=timedelta(hours=float(x.get("dist_timeout_hours", 6))))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warn_if_reference_is_raw_base(cfg)
    model, tok = load_model_and_tokenizer(cfg)
    # dropout would make the training forward differ from the behaviour log-probs (ratio != 1 at step 0)
    disable_lora_dropout(model)
    model.to(dev)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.dtype == "fp16")
    amp_dtype = torch.float16 if cfg.dtype == "fp16" else torch.bfloat16
    collate = PadCollator(tok.pad_token_id)

    vpool = rewards = pool = None
    rng = random.Random(cfg.seed)
    if rank == 0:
        pool = load_prompt_pool(x["prompt_files"], tasks=mix, seed=cfg.seed)
        log.info("prompt pool: %s", {k: len(v) for k, v in pool.items()})
        rewards = RewardPool(int(x.get("reward_workers", 16)))
        vpool = VLLMPool(rollout_gpus, base_port=int(x.get("vllm_port", 8100)),
                         extra_args=(["--enable-lora", "--max-lora-rank", str(cfg.lora_r), "--max-loras", "2"]
                                     if publish == "lora" else []), max_model_len=int(x.get("max_model_len", 12288)))
        if publish == "lora":
            vpool.start(cfg.model, served_name="base")

    for rnd in range(rounds):
        t0 = time.time()
        rdir = out_dir / f"round_{rnd:03d}"
        batch_file = rdir / "batch.jsonl"
        if rank == 0:
            rdir.mkdir(parents=True, exist_ok=True)
            adapter = rdir / "adapter_in"
            model.save_pretrained(adapter)
            if publish == "lora":
                vpool.load_lora(f"policy-{rnd}", str(adapter.resolve()))
                if rnd > 0:
                    vpool.unload_lora(f"policy-{rnd - 1}")
                served = f"policy-{rnd}"
            else:
                from .merge_lora import merge

                merged = rdir / "merged"
                merge(cfg.model, str(adapter), str(merged), dtype="fp16")
                vpool.start(str(merged), served_name="policy")
                served = "policy"
            if rnd > 0:  # ~0.4 GB per round otherwise; checkpoints are saved separately
                shutil.rmtree(out_dir / f"round_{rnd - 1:03d}" / "adapter_in", ignore_errors=True)
            prompts = sample_batch(pool, mix, prompts_per_round, rng)
            outs = asyncio.run(rollout_openai(vpool.urls, served, prompts, G, temperature,
                                              top_p=float(x.get("top_p", 0.95)),
                                              per_server=int(x.get("requests_per_server", 4))))
            items = [(p.task, p.meta, text, fr) for p, comps in zip(prompts, outs) for text, fr in comps]
            scores = rewards.score(items)
            rows, stats = [], {}
            k = 0
            for p, comps in zip(prompts, outs):
                rs = [scores[k + j][0] for j in range(len(comps))]
                k += len(comps)
                st = stats.setdefault(p.task, {"n": 0, "reward": 0.0, "kept_groups": 0})
                st["n"] += len(rs)
                st["reward"] += sum(rs)
                adv = group_advantages(rs)
                if adv is None:
                    continue
                st["kept_groups"] += 1
                for (text, fr), r, ad in zip(comps, rs, adv):
                    rows.append({"messages": p.messages + [{"role": "assistant", "content": text}], "reward": r,
                                 "adv": ad, "task": p.task})
            for t, st in stats.items():
                st["reward"] = round(st["reward"] / max(st["n"], 1), 4)
            with open(batch_file, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info("round %d rollout+reward %.0fs | %s | %d samples", rnd, time.time() - t0, stats, len(rows))
            (rdir / "stats.json").write_text(json.dumps(stats, indent=1))
            if publish == "merge":
                vpool.stop()
                shutil.rmtree(rdir / "merged", ignore_errors=True)
        if world > 1:
            dist.barrier()

        # ------------------------------------------------------------- policy update (all ranks)
        rows = [json.loads(l) for l in open(batch_file, encoding="utf-8")] if batch_file.exists() else []
        mine = rows[rank::world]
        encoded = []
        for r in mine:
            e = encode_chat(tok, r["messages"], cfg.max_len)
            if e is not None:
                e["adv"] = r["adv"]
                encoded.append(e)
        # equalise the number of micro-steps across ranks (DDP needs matching collectives)
        n_local = torch.tensor([len(encoded)], device=dev)
        if world > 1:
            dist.all_reduce(n_local, op=dist.ReduceOp.MIN)
        n_steps = int(n_local.item())
        encoded = encoded[:n_steps]
        if n_steps == 0:
            continue
        accum = int(x.get("grad_accum", 8))
        # token-level normalisation per optimiser step (all ranks' tokens in the accumulation window),
        # so gradient clipping and the learning rate mean the same thing in every round
        n_win = (n_steps + accum - 1) // accum
        win_tok = torch.zeros(n_win, device=dev)
        for i, e in enumerate(encoded):
            win_tok[i // accum] += sum(1 for l in e["labels"] if l != -100)
        if world > 1:
            dist.all_reduce(win_tok)
        win_tok = win_tok.clamp(min=1.0)
        # old (behaviour) and reference log-probs
        model.eval()
        olds, refs = [], []
        with torch.no_grad():
            for e in encoded:
                b = {k: v.to(dev) for k, v in collate([e]).items()}
                with torch.autocast("cuda", dtype=amp_dtype):
                    lp, _ = token_logps(model, b["input_ids"], b["attention_mask"], b["labels"])
                    olds.append(lp.detach())
                    with model.disable_adapter():
                        rlp, _ = token_logps(model, b["input_ids"], b["attention_mask"], b["labels"])
                    refs.append(rlp.detach())
        model.train()
        for ep in range(ppo_epochs):
            opt.zero_grad(set_to_none=True)
            for i, e in enumerate(encoded):
                b = {k: v.to(dev) for k, v in collate([e]).items()}
                sync = (i + 1) % accum == 0 or i + 1 == len(encoded)
                ctx = ddp.no_sync() if (world > 1 and not sync) else _nullctx()
                with ctx:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        lp, mask = token_logps(ddp, b["input_ids"], b["attention_mask"], b["labels"])
                    ratio = torch.exp(lp - olds[i])
                    adv = torch.tensor(e["adv"], device=dev)
                    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
                    d = refs[i] - lp
                    kl = torch.exp(d) - d - 1  # k3 estimator, >= 0
                    loss = ((pg + kl_beta * kl) * mask).sum() / win_tok[i // accum] * world
                    scaler.scale(loss).backward()
                if sync:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
        if rank == 0:
            log.info("round %d done in %.0fs (%d samples/rank)", rnd, time.time() - t0, n_steps)
            if (rnd + 1) % int(x.get("save_every", 5)) == 0 or rnd + 1 == rounds:
                model.save_pretrained(out_dir / f"checkpoint_round_{rnd + 1:03d}")
        if world > 1:
            dist.barrier()

    if rank == 0:
        model.save_pretrained(out_dir / "final")
        tok.save_pretrained(out_dir / "final")
        vpool.stop()
        rewards.close()
    if world > 1:
        dist.destroy_process_group()


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
