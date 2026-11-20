"""Supervised fine-tuning (LoRA or full) with assistant-only loss.

    # LoRA on 8 x V100 (DDP)
    torchrun --nproc_per_node 8 -m lecturenotes.training.sft --config configs/train/sft_lora.yaml
    # full fine-tuning with ZeRO-3
    torchrun --nproc_per_node 8 -m lecturenotes.training.sft --config configs/train/sft_full_zero3.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .common import PadCollator, TrainConfig, build_dataset, load_model_and_tokenizer, read_rows, world_info

log = logging.getLogger(__name__)


def completion_ce(model, input_ids, attention_mask, labels):
    """Summed cross-entropy over supervised positions (completion = suffix) and their count.

    Only the last K positions are projected to the vocabulary (`logits_to_keep`): the default loss
    materialises (seq x vocab) fp32 logits -- 8192 x 200k = 6.5 GB per copy for Phi-4's vocabulary."""
    import torch.nn.functional as F

    total, count = 0.0, 0
    for b in range(input_ids.shape[0]):
        L = int(attention_mask[b].sum().item())
        lab = labels[b : b + 1, :L]
        k = int((lab != -100).sum().item()) + 1
        if k <= 1:
            continue
        out = model(input_ids=input_ids[b : b + 1, :L], attention_mask=attention_mask[b : b + 1, :L], logits_to_keep=k)
        logits = out.logits[:, -k:-1].float()
        tgt = lab[:, -k + 1:]
        total = total + F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100,
                                        reduction="sum")
        count += int((tgt != -100).sum().item())
    return total, count


def make_trainer_cls():
    from transformers import Trainer

    class CompletionLossTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # we return a per-micro-batch token mean; let the Trainer divide by grad-accumulation steps
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            total, count = completion_ce(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
            loss = total / max(count, 1)
            return (loss, {"loss": loss}) if return_outputs else loss

    return CompletionLossTrainer


class ListDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model")
    ap.add_argument("--output_dir")
    ap.add_argument("--train_file")
    ap.add_argument("--max_steps", type=int, default=-1)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    cfg = TrainConfig.load(a.config, model=a.model, output_dir=a.output_dir, train_file=a.train_file)
    set_seed(cfg.seed)
    rank, world, _ = world_info()
    gb = int(cfg.extra.get("global_batch") or 0)
    if gb:  # keep the effective batch fixed whatever number of GPUs happens to be free
        cfg.grad_accum = max(1, gb // (world * cfg.per_device_batch))
        log.info("global batch %d -> grad_accum %d on %d GPU(s)", gb, cfg.grad_accum, world)
    model, tok = load_model_and_tokenizer(cfg)

    train = build_dataset(tok, read_rows(cfg.train_file), cfg.max_len, cfg.task_weights, seed=cfg.seed)
    val = build_dataset(tok, read_rows(cfg.val_file), cfg.max_len) if cfg.val_file and Path(cfg.val_file).exists() else None
    if rank == 0:
        lens = sorted(x["length"] for x in train)
        log.info("train examples: %d (median %d tokens, p95 %d), val: %d", len(train), lens[len(lens) // 2],
                 lens[int(len(lens) * 0.95)], len(val or []))

    args = TrainingArguments(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=cfg.per_device_batch, per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr, warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay, lr_scheduler_type=cfg.lr_scheduler, max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.dtype == "fp16", bf16=cfg.dtype == "bf16",
        logging_steps=cfg.logging_steps, save_steps=cfg.save_steps, save_total_limit=3,
        eval_strategy="steps" if val else "no", eval_steps=cfg.eval_steps if val else None,
        group_by_length=True, length_column_name="length", remove_unused_columns=False,
        dataloader_num_workers=2, report_to=cfg.report_to, deepspeed=cfg.deepspeed,
        ddp_find_unused_parameters=False if world > 1 else None, gradient_checkpointing=False,  # enabled on the model
        seed=cfg.seed, save_safetensors=True, prediction_loss_only=True,
    )
    trainer_cls = make_trainer_cls() if cfg.extra.get("completion_loss", True) else Trainer
    trainer = trainer_cls(model=model, args=args, train_dataset=ListDataset(train),
                          eval_dataset=ListDataset(val) if val else None, data_collator=PadCollator(tok.pad_token_id))
    # resume after a crash / relaunch on a different number of GPUs (global_batch keeps steps comparable)
    ckpts = sorted(Path(cfg.output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        log.info("resuming from %s", ckpts[-1])
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(cfg.output_dir)
    if trainer.is_world_process_zero():
        tok.save_pretrained(cfg.output_dir)
    if rank == 0:
        (Path(cfg.output_dir) / "train_config.json").write_text(json.dumps(cfg.__dict__, indent=1, default=str))
        if torch.cuda.is_available():
            log.info("peak memory: %.1f GB", torch.cuda.max_memory_allocated() / 2 ** 30)


if __name__ == "__main__":
    main()
