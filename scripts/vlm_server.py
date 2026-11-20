"""Minimal OpenAI-compatible chat server for a vision-language judge, on plain transformers.

vLLM's V0 engine (the only one on V100) crashes on Qwen2.5-VL image requests, so the figure-relevance
judge (lecturenotes.figures.relevance) talks to this instead. One request at a time; enough for judging.

    CUDA_VISIBLE_DEVICES=0 python scripts/vlm_server.py --model Qwen/Qwen2.5-VL-7B-Instruct --port 8010
"""
from __future__ import annotations

import argparse
import base64
import io
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--port", type=int, default=8010)
ap.add_argument("--max-pixels", type=int, default=896 * 896)
args = ap.parse_args()

processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda",
                                                    attn_implementation="sdpa").eval()
lock = threading.Lock()
app = FastAPI()


def _convert(messages):
    """OpenAI content parts -> HF chat template parts + list of PIL images."""
    images, out = [], []
    for m in messages:
        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        conv = []
        for p in parts:
            if p["type"] == "image_url":
                url = p["image_url"]["url"]
                images.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
                conv.append({"type": "image"})
            else:
                conv.append({"type": "text", "text": p["text"]})
        out.append({"role": m["role"], "content": conv})
    return out, images


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    msgs, images = _convert(body["messages"])
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images or None, return_tensors="pt").to("cuda")
    with lock, torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(body.get("max_tokens", 128)), do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return {"id": f"vlm-{time.time_ns()}", "object": "chat.completion", "model": body.get("model", "vlm"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
