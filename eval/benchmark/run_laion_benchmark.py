
from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ddpo_pytorch.prompts as ddpo_prompts

from ddpo_pytorch.aesthetic_scorer import AestheticScorer

CLIP_ID = "openai/clip-vit-large-patch14"


def load_imagenet_animals_prompt_pool() -> list[str]:
    path = Path(ddpo_prompts.__file__).resolve().parent / "assets" / "imagenet_classes.txt"
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    pool = [p for p in lines if p and not p.startswith("#")]
    return pool[0:398]


def load_prompts(path: Path, max_prompts: int | None) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    prompts = [p for p in lines if p and not p.startswith("#")]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"No prompts in {path}")
    return prompts


def resolve_lora_dir(raw: str) -> Path:
    
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("lora_path is empty")
    p = Path(raw).expanduser()
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(REPO_ROOT / p)

    tried: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if r.is_dir():
            return r

    


def attach_lora(pipeline: StableDiffusionPipeline, lora_dir: Path) -> None:
    lora_dir = lora_dir.resolve()
    if not lora_dir.is_dir():
        raise NotADirectoryError(lora_dir)

    lora_attn_procs = {}
    for name in pipeline.unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else pipeline.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = pipeline.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                block_id
            ]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipeline.unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
    pipeline.unet.set_attn_processor(lora_attn_procs)
    path_str = str(lora_dir)
    weight_names = [
        n
        for n in (
            "pytorch_lora_weights.safetensors",
            "pytorch_lora_weights.bin",
        )
        if (lora_dir / n).is_file()
    ]
    try:
        if weight_names:
            try:
                pipeline.unet.load_attn_procs(path_str, weight_name=weight_names[0])
            except TypeError:
                pipeline.unet.load_attn_procs(path_str)
        else:
            pipeline.unet.load_attn_procs(path_str)
    except OSError as e:
        listing = sorted(f.name for f in lora_dir.iterdir()) if lora_dir.is_dir() else []
        preview = listing[:30]
        more = " …" if len(listing) > len(preview) else ""
        raise OSError(
            f"load_attn_procs failed for {lora_dir}.\n"
            f"Directory contents (names): {preview}{more}\n"
            f"Expected something like pytorch_lora_weights.bin or .safetensors "
            f"(output of train.py save_attn_procs).\n"
            f"Original error: {e}"
        ) from e


@torch.inference_mode()
def clip_scores(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Image.Image],
    prompts: list[str],
    device: torch.device,
) -> np.ndarray:
    pix = processor(images=images, return_tensors="pt")["pixel_values"].to(
        device, dtype=torch.float32
    )
    text_out = processor(
        text=prompts, return_tensors="pt", padding=True, truncation=True
    )
    text_ids = text_out["input_ids"].to(device)
    text_mask = text_out["attention_mask"].to(device)

    img_f = clip_model.get_image_features(pixel_values=pix)
    txt_f = clip_model.get_text_features(
        input_ids=text_ids, attention_mask=text_mask
    )
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    s = (img_f * txt_f).sum(dim=-1)
    scores = s.detach().float().cpu().numpy()
    return scores


def jpeg_kb(images: list[Image.Image], quality: int = 95) -> np.ndarray:
    
    out = []
    for im in images:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=quality)
        out.append(buf.tell() / 1000.0)
    return np.array(out, dtype=np.float64)


@torch.inference_mode()
def aesthetic_scores(
    scorer: AestheticScorer, images: list[Image.Image], device: torch.device
) -> np.ndarray:
    _ = device  # scorer is already on device
    rgb = [im.convert("RGB") for im in images]
    s = scorer(rgb)
    return s.detach().float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    default_prompts = Path(__file__).resolve().parent / "prompts" / "laion_eval_prompts.txt"
    p = argparse.ArgumentParser(description="LAION-style T2I benchmark (CLIP + aesthetic + optional JPEG).")
    p.add_argument("--pretrained", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--revision", type=str, default="main")
    p.add_argument(
        "--lora_path",
        type=str,
        default="",
        help="Directory from save_attn_procs (e.g. logs/run/checkpoint_10). Empty = base model only.",
    )
    p.add_argument("--prompts_file", type=str, default=str(default_prompts))
    p.add_argument(
        "--use_training_prompts",
        action="store_true",
        help=(
            "Use the training prompt pool for default config.prompt_fn=imagenet_animals: "
            "all 398 captions from ddpo_pytorch/assets/imagenet_classes.txt (same as random choice space in training). "
            "Ignores --prompts_file. Benchmark logic (pipe, DDIM, metrics) is unchanged."
        ),
    )
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Directory for this eval (results JSON + default preview/). Default: eval/benchmark/runs/<YYYYMMDD_HHMMSS>.",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Path to results JSON. Default: <run_dir>/results.json",
    )
    p.add_argument(
        "--save_images_dir",
        type=str,
        default="",
        help="Override directory for all generated PNGs (+ matching .txt prompts). Default: <run_dir>/images.",
    )
    p.add_argument(
        "--no_save_images",
        action="store_true",
        help="Skip saving the full image set (saves disk); previews (--preview_first_n) still apply if > 0.",
    )
    p.add_argument(
        "--preview_first_n_dir",
        type=str,
        default="",
        help="Preview PNG/txt dir. Default: <run_dir>/preview. Override with a path; use --preview_first_n 0 to disable.",
    )
    p.add_argument(
        "--preview_first_n",
        type=int,
        default=4,
        help="Number of first prompts to save into preview_first_n_dir (default 4).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=("fp16", "bf16", "no"),
    )
    p.add_argument("--skip_aesthetic", action="store_true")
    p.add_argument("--skip_clip", action="store_true")
    p.add_argument(
        "--jpeg-metrics",
        dest="jpeg_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Measure JPEG size at Q=95 (KB) and jpeg_compressibility reward (-KB), "
            "aligned with ddpo_pytorch.rewards. Default: on."
        ),
    )
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.run_dir or "").strip():
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (REPO_ROOT / "eval" / "benchmark" / "runs" / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Eval run directory:\n  {run_dir}")

    if str(args.output_json or "").strip():
        out_path = Path(args.output_json).expanduser()
    else:
        out_path = run_dir / "results.json"

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.use_training_prompts:
        prompts = load_imagenet_animals_prompt_pool()
        if args.max_prompts is not None:
            prompts = prompts[: args.max_prompts]
        prompts_path = None
    else:
        prompts_path = Path(args.prompts_file).expanduser()
        if not prompts_path.is_file():
            raise FileNotFoundError(prompts_path)
        prompts = load_prompts(prompts_path, args.max_prompts)

    dtype = torch.float32
    if device.type == "cuda":
        if args.mixed_precision == "fp16":
            dtype = torch.float16
        elif args.mixed_precision == "bf16":
            dtype = torch.bfloat16
    else:
        dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained,
        revision=args.revision,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if args.lora_path:
        attach_lora(pipe, resolve_lora_dir(args.lora_path))

    pipe.set_progress_bar_config(disable=False)

    clip_model = None
    processor = None
    if not args.skip_clip:
        clip_model = CLIPModel.from_pretrained(CLIP_ID).to(device, dtype=torch.float32)
        processor = CLIPProcessor.from_pretrained(CLIP_ID)
        clip_model.eval()

    scorer = None
    if not args.skip_aesthetic:
        scorer = AestheticScorer(dtype=torch.float32).to(device)
        scorer.eval()

    if args.no_save_images:
        save_dir = None
    elif str(args.save_images_dir or "").strip():
        save_dir = Path(args.save_images_dir).expanduser().resolve()
    else:
        save_dir = (run_dir / "images").resolve()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    preview_n = max(0, int(args.preview_first_n))
    preview_dir = None
    if preview_n > 0:
        if str(args.preview_first_n_dir or "").strip():
            preview_dir = Path(args.preview_first_n_dir).expanduser()
        else:
            preview_dir = run_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    bs = max(1, args.batch_size)

    for start in tqdm(range(0, len(prompts), bs), desc="Batches"):
        batch_prompts = prompts[start : start + bs]
        out = pipe(
            prompt=batch_prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
            output_type="pil",
        )
        images: list[Image.Image] = out.images

        clip_vals = np.full(len(batch_prompts), np.nan, dtype=np.float64)
        if clip_model is not None and processor is not None:
            clip_vals = clip_scores(clip_model, processor, images, batch_prompts, device)

        aes_vals = np.full(len(batch_prompts), np.nan, dtype=np.float64)
        if scorer is not None:
            aes_vals = aesthetic_scores(scorer, images, device)

        jpg_vals = None
        if args.jpeg_metrics:
            jpg_vals = jpeg_kb(images)

        for i, (pr, im) in enumerate(zip(batch_prompts, images)):
            rec = {
                "prompt": pr,
                "clip_score": float(clip_vals[i]),
                "aesthetic": float(aes_vals[i]),
            }
            if jpg_vals is not None:
                kb = float(jpg_vals[i])
                rec["jpeg_kb_q95"] = kb
                rec["jpeg_compressibility_reward"] = -kb
            rows.append(rec)
            idx = start + i
            if preview_dir is not None and idx < preview_n:
                im.save(preview_dir / f"{idx:02d}.png")
                (preview_dir / f"{idx:02d}.txt").write_text(pr + "\n", encoding="utf-8")
            if save_dir is not None:
                im.save(save_dir / f"{idx:05d}.png")
                (save_dir / f"{idx:05d}.txt").write_text(pr + "\n", encoding="utf-8")

    def stat(key: str) -> dict:
        vals = np.array([r[key] for r in rows if not np.isnan(r[key])])
        if vals.size == 0:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "n": int(vals.size),
        }

    summary = {
        "clip_score": stat("clip_score"),
        "aesthetic": stat("aesthetic"),
    }
    if args.jpeg_metrics:
        summary["jpeg_kb_q95"] = stat("jpeg_kb_q95")
        summary["jpeg_compressibility_reward"] = stat("jpeg_compressibility_reward")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = {k: getattr(args, k) for k in vars(args)}
    if args.use_training_prompts:
        cfg_dict["prompts_resolved"] = "imagenet_animals_pool:398"
    else:
        cfg_dict["prompts_resolved"] = str(prompts_path)

    payload = {
        "config": cfg_dict,
        "paths": {
            "run_dir": str(run_dir),
            "output_json": str(out_path.resolve()),
            "preview_dir": str(preview_dir.resolve()) if preview_dir else None,
            "images_dir": str(save_dir.resolve()) if save_dir is not None else None,
        },
        "num_prompts": len(prompts),
        "device": str(device),
        "summary": summary,
        "per_prompt": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    if preview_dir is not None:
        print(f"Saved first {preview_n} previews under {preview_dir.resolve()}")
    if save_dir is not None:
        print(f"Saved all {len(prompts)} images under {save_dir.resolve()}")


if __name__ == "__main__":
    main()
