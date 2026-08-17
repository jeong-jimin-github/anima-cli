#!/usr/bin/env python3
"""Anima-2.9B CLI - Text-to-Image generation tool for agentic AI workflows."""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[anima] {msg}", file=sys.stderr)


def load_model(model_path: str | None, device: str, dtype: str):
    import torch
    from anima_loader import build_pipeline

    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(dtype, torch.bfloat16)

    anima_path = model_path
    if not anima_path:
        from huggingface_hub import hf_hub_download
        log("Downloading Anima-2.9B checkpoint (5.8GB)...")
        anima_path = hf_hub_download(
            "Gazingstars123/Anima-2.9B",
            "Anima-2.9B-preview-v1.safetensors",
        )
        log(f"Checkpoint: {anima_path}")

    target_device = device
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    cpu_offload = target_device == "cuda"

    pipe = build_pipeline(
        anima_path,
        device=target_device,
        torch_dtype=torch_dtype,
        cpu_offload=cpu_offload,
    )

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    log(f"Model loaded on {target_device} ({dtype})")
    return pipe, target_device


SAMPLER_MAP = {
    "euler": "euler",
    "euler_a": "euler_a",
    "dpm++_2m": "dpm++_2m",
    "dpm++_sde": "dpm++_sde",
    "res_multistep": "res_multistep",
    "er_sde": "er_sde",
}

RESOLUTION_PRESETS = {
    "sd": (816, 1216),
    "hd": (1152, 1536),
    "1k": (1024, 1024),
    "2k": (1536, 1536),
    "portrait": (816, 1216),
    "landscape": (1216, 816),
    "square": (1024, 1024),
    "wide": (1536, 1024),
    "tall": (1024, 1536),
    "phone": (720, 1280),
    "wallpaper": (1920, 1088),
}


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse resolution string: preset name or WxH format (rounded to multiple of 16)."""
    lower = value.lower().strip()
    if lower in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[lower]

    def _to_16(n: int) -> int:
        return max(16, round(n / 16) * 16)

    if "x" in lower:
        parts = lower.split("x", 1)
        try:
            w, h = int(parts[0]), int(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError
            return _to_16(w), _to_16(h)
        except (ValueError, IndexError):
            pass

    raise argparse.ArgumentTypeError(
        f"Invalid resolution: '{value}'. Use WxH (e.g. 816x1216) or preset: {', '.join(RESOLUTION_PRESETS.keys())}"
    )


def generate_image(pipe, device: str, args) -> dict:
    import torch
    from PIL import Image

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    if not prompt:
        raise ValueError("No prompt provided. Use --prompt or --prompt-file.")

    seed = args.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    log(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    log(f"Size: {args.width}x{args.height}, Steps: {args.steps}, CFG: {args.cfg}")
    log(f"Sampler: {args.sampler}, Scheduler: {args.scheduler}, Seed: {seed}")

    generator = torch.Generator(device=device).manual_seed(seed)

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
    )
    elapsed = time.time() - t0

    image: Image.Image = result.images[0]

    output_path = args.output
    if not output_path:
        output_path = f"anima_{seed}.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path), "PNG")

    log(f"Saved: {output_path} ({elapsed:.2f}s)")

    return {
        "success": True,
        "path": str(output_path.resolve()),
        "prompt": prompt,
        "seed": seed,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "elapsed_seconds": round(elapsed, 3),
    }


def batch_generate(pipe, device: str, args) -> list[dict]:
    prompts_file = Path(args.batch)
    if not prompts_file.exists():
        raise FileNotFoundError(f"Batch file not found: {prompts_file}")

    lines = [
        line.strip()
        for line in prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    log(f"Batch mode: {len(lines)} prompts")
    results = []

    for i, prompt in enumerate(lines):
        log(f"[{i + 1}/{len(lines)}] Generating...")
        args.prompt = prompt
        args.prompt_file = None
        if args.output and args.batch:
            stem = Path(args.output).stem
            ext = Path(args.output).suffix or ".png"
            args.output = f"{stem}_{i:04d}{ext}"

        result = generate_image(pipe, device, args)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        prog="anima",
        description="Anima-2.9B text-to-image CLI for agentic workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  anima --prompt "1girl, masterpiece, cherry blossoms"
  anima -p "1boy, fantasy armor" -r hd -o hero.png
  anima -p "landscape scene" -r landscape --json
  anima -p "custom size" -r 1536x1024 --steps 40
  anima --prompt-file prompt.txt --json
  anima --batch prompts.txt --output-dir ./batch_output
  anima -p "test" --seed 12345 --cfg 4.5

resolutions:
  sd         812x1216   hd          1152x1536
  1k         1024x1024  2k          1536x1536
  portrait   812x1216   landscape   1216x812
  square     1024x1024  wide        1536x1024
  tall       1024x1536  phone       720x1280
  wallpaper  1920x1080  WxH         custom
""",
    )

    g_prompt = parser.add_argument_group("Prompt")
    g_prompt.add_argument(
        "-p", "--prompt", type=str, default=None, help="Text prompt for generation"
    )
    g_prompt.add_argument(
        "--prompt-file", type=str, default=None, help="Read prompt from file"
    )
    g_prompt.add_argument(
        "-n",
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt (default: standard quality tags)",
    )

    g_gen = parser.add_argument_group("Generation")
    g_gen.add_argument(
        "-r",
        "--resolution",
        type=str,
        default=None,
        help="Resolution: WxH (e.g. 812x1216) or preset name. Overrides -W/-H",
    )
    g_gen.add_argument(
        "-W", "--width", type=int, default=812, help="Image width (default: 812)"
    )
    g_gen.add_argument(
        "-H", "--height", type=int, default=1216, help="Image height (default: 1216)"
    )
    g_gen.add_argument(
        "-s", "--steps", type=int, default=30, help="Inference steps 28-50 (default: 30)"
    )
    g_gen.add_argument(
        "-c", "--cfg", type=float, default=4.0, help="CFG scale 3.5-5 (default: 4.0)"
    )
    g_gen.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=list(SAMPLER_MAP.keys()),
        help="Sampler (default: euler)",
    )
    g_gen.add_argument(
        "--scheduler",
        type=str,
        default="sgm_uniform",
        choices=["sgm_uniform", "beta", "beta57", "linear", "linear_quadratic"],
        help="Scheduler (default: sgm_uniform)",
    )
    g_gen.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: random)"
    )

    g_out = parser.add_argument_group("Output")
    g_out.add_argument(
        "-o", "--output", type=str, default=None, help="Output file path"
    )
    g_out.add_argument("--json", action="store_true", help="Output result as JSON")

    g_batch = parser.add_argument_group("Batch")
    g_batch.add_argument(
        "--batch", type=str, default=None, help="Batch file with one prompt per line"
    )

    g_model = parser.add_argument_group("Model")
    g_model.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model path or HF repo (default: Gazingstars123/Anima-2.9B)",
    )
    g_model.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device (default: auto)",
    )
    g_model.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: bf16)",
    )

    args = parser.parse_args()

    if args.resolution:
        args.width, args.height = parse_resolution(args.resolution)

    if not args.prompt and not args.prompt_file and not args.batch:
        parser.error("Provide --prompt, --prompt-file, or --batch")

    pipe, device = load_model(args.model_path, args.device, args.dtype)

    try:
        if args.batch:
            results = batch_generate(pipe, device, args)
            if args.json:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                for r in results:
                    print(r["path"])
        else:
            result = generate_image(pipe, device, args)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(result["path"])
    except Exception as e:
        error = {"success": False, "error": str(e)}
        if args.json:
            print(json.dumps(error, indent=2), file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
