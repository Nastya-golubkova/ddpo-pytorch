#!/usr/bin/env python3
"""Generate images with base SD1.5 using the same prompt distribution as training (imagenet_animals).

Training samples in ``scripts/train.py`` use ``pipeline_with_logprob`` + patched DDIM (``ddim_step_with_logprob``),
``prompt_embeds`` / ``negative_prompt_embeds`` from the tokenizer (negative = empty string), and **no** ``generator``
— latent + DDIM variance noise come from the **global PyTorch RNG** (same as DDPO).

Using the stock ``StableDiffusionPipeline.__call__`` with per-image ``torch.Generator`` yields **different** noise
and can look visibly different (e.g. different framing / "spread" with ``eta=1``).

Examples::

    python scripts/generate_imagenet_animals_baseline.py --out_dir report/imagenet_animals_baseline --num_samples 64 --seed 42

    python scripts/generate_imagenet_animals_baseline.py --out_dir report/imagenet_animals_all --all_classes --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate.utils import set_seed
from diffusers import DDIMScheduler, StableDiffusionPipeline

import ddpo_pytorch.prompts as prompts
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=str,
        default="report/imagenet_animals_baseline",
        help="Directory for PNGs, prompts.txt, manifest.json",
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    p.add_argument("--num_samples", type=int, default=64, help="Ignored if --all_classes")
    p.add_argument(
        "--all_classes",
        action="store_true",
        help="Generate one image per line in imagenet_animals range (398 prompts)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--torch_seed_per_batch",
        action="store_true",
        help="Before each batch: torch.manual_seed(seed + batch_start_index) and cuda.manual_seed_all (reproducible; still not identical to a full training run's RNG consumption).",
    )
    return p.parse_args()


def _load_imagenet_animal_lines():
    path = Path(prompts.__file__).resolve().parent / "assets" / "imagenet_classes.txt"
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return lines[0:398]


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.all_classes:
        prompt_list = _load_imagenet_animal_lines()
    else:
        # Match training: global seed before sampling prompt_fn (see scripts/train.py set_seed).
        set_seed(args.seed, device_specific=False)
        prompt_list = [prompts.imagenet_animals()[0] for _ in range(args.num_samples)]

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)

    # Same empty-string negative embed as train.py (single forward, repeated over batch).
    device = pipe.device
    tok = pipe.tokenizer
    enc = pipe.text_encoder
    neg_ids = tok(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tok.model_max_length,
    ).input_ids.to(device)
    neg_embed = enc(neg_ids)[0]

    manifest = {
        "pretrained": args.pretrained,
        "prompt_fn": "imagenet_animals",
        "sampler": "pipeline_with_logprob (same as train.py)",
        "generator": None,
        "all_classes": args.all_classes,
        "seed": args.seed,
        "torch_seed_per_batch": args.torch_seed_per_batch,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "eta": args.eta,
        "negative_prompt": "",
        "num_images": len(prompt_list),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(out / "prompts.txt", "w") as f:
        for pr in prompt_list:
            f.write(pr + "\n")

    bs = max(1, args.batch_size)
    idx = 0
    while idx < len(prompt_list):
        if args.torch_seed_per_batch:
            s = args.seed + idx
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        chunk = prompt_list[idx : idx + bs]
        prompt_ids = tok(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tok.model_max_length,
        ).input_ids.to(device)
        prompt_embeds = enc(prompt_ids)[0]
        negative_prompt_embeds = neg_embed.repeat(len(chunk), 1, 1)

        images, _, _, _ = pipeline_with_logprob(
            pipe,
            prompt=None,
            height=args.height,
            width=args.width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            generator=None,
            output_type="pil",
        )
        for j, im in enumerate(images):
            im.save(out / f"{idx + j:04d}.png")
        idx += len(chunk)

    print(f"Wrote {len(prompt_list)} images to {out.resolve()}")


if __name__ == "__main__":
    main()
