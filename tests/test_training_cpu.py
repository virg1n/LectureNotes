"""CPU smoke tests for the training code (tiny model). Opt-in: LN_TEST_TRAIN=1 pytest -m train"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.train
if not os.environ.get("LN_TEST_TRAIN"):
    pytest.skip("set LN_TEST_TRAIN=1 to run training smoke tests (downloads a 135M model)", allow_module_level=True)
torch = pytest.importorskip("torch")
pytest.importorskip("peft")

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    rows = [{"task": "write_section", "group": f"g{i}", "source": "real", "messages": [
        {"role": "system", "content": "You are LectureScribe."},
        {"role": "user", "content": f"Write section {i}.\nLength: about 60 words of prose."},
        {"role": "assistant", "content": f"\\section{{Vectors {i}}}\nA vector space is a set with addition."}]}
        for i in range(8)]
    (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (root / "sft.yaml").write_text(
        f"model: {MODEL}\ntrain_file: {root / 'train.jsonl'}\noutput_dir: {root / 'sft'}\nmax_len: 512\n"
        "per_device_batch: 1\ngrad_accum: 2\ndtype: fp32\nlr: 1.0e-3\nepochs: 1\nuse_lora: true\nlora_r: 8\n"
        "lora_alpha: 16\nlogging_steps: 1\n")
    return root, rows


def test_sft_saves_adapter(workdir):
    root, _ = workdir
    r = subprocess.run([sys.executable, "-m", "lecturenotes.training.sft", "--config", str(root / "sft.yaml"),
                        "--max_steps", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (root / "sft" / "adapter_config.json").exists()


def test_completion_logprobs_match_full_logits(workdir):
    from lecturenotes.training.common import PadCollator, TrainConfig, encode_chat, load_model_and_tokenizer
    from lecturenotes.training.grpo import group_advantages, token_logps

    root, rows = workdir
    cfg = TrainConfig.load(root / "sft.yaml")
    model, tok = load_model_and_tokenizer(cfg, for_training=False)
    b = PadCollator(tok.pad_token_id)([encode_chat(tok, rows[0]["messages"], 512)])
    with torch.no_grad():
        lp, mask = token_logps(model, b["input_ids"], b["attention_mask"], b["labels"])
        full = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1].float().log_softmax(-1)
        tgt = b["labels"][:, 1:]
        ref = torch.gather(full, 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)[tgt != -100]
    assert torch.allclose(lp[mask.bool()], ref, atol=1e-4)
    assert group_advantages([0.2, 0.2]) is None
