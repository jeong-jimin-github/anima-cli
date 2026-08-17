"""Load Anima-2.9B single-file checkpoint into the diffusers Anima pipeline.

Anima-2.9B is a 40-layer expanded fine-tune. The community diffusers repo
(CalamitousFelicitousness/Anima-sdnext-diffusers) only ships the 28-layer base,
so we remap the ComfyUI-style single-file keys into a 40-layer
CosmosTransformer3DModel and swap it into the pipeline.
"""

import sys
import torch
from diffusers import DiffusionPipeline
from diffusers.models import CosmosTransformer3DModel


def _remap_comfyui_to_diffusers_key(comfy_key: str) -> str:
    """Map a ComfyUI Anima transformer key to a diffusers CosmosTransformer3DModel key."""
    # Top-level (non-block) keys
    top_map = {
        "net.x_embedder.proj.1.weight": "patch_embed.proj.weight",
        "net.t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
        "net.t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
        "net.t_embedding_norm.weight": "time_embed.norm.weight",
        "net.final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
        "net.final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
        "net.final_layer.linear.weight": "proj_out.weight",
    }
    if comfy_key in top_map:
        return top_map[comfy_key]

    # Block keys: net.blocks.{N}.{module}
    if comfy_key.startswith("net.blocks."):
        parts = comfy_key.split(".")
        n = parts[2]
        module = ".".join(parts[3:])
        block_map = {
            "adaln_modulation_self_attn.1.weight": "norm1.linear_1.weight",
            "adaln_modulation_self_attn.2.weight": "norm1.linear_2.weight",
            "adaln_modulation_cross_attn.1.weight": "norm2.linear_1.weight",
            "adaln_modulation_cross_attn.2.weight": "norm2.linear_2.weight",
            "adaln_modulation_mlp.1.weight": "norm3.linear_1.weight",
            "adaln_modulation_mlp.2.weight": "norm3.linear_2.weight",
            "self_attn.q_norm.weight": "attn1.norm_q.weight",
            "self_attn.k_norm.weight": "attn1.norm_k.weight",
            "self_attn.q_proj.weight": "attn1.to_q.weight",
            "self_attn.k_proj.weight": "attn1.to_k.weight",
            "self_attn.v_proj.weight": "attn1.to_v.weight",
            "self_attn.output_proj.weight": "attn1.to_out.0.weight",
            "cross_attn.q_norm.weight": "attn2.norm_q.weight",
            "cross_attn.k_norm.weight": "attn2.norm_k.weight",
            "cross_attn.q_proj.weight": "attn2.to_q.weight",
            "cross_attn.k_proj.weight": "attn2.to_k.weight",
            "cross_attn.v_proj.weight": "attn2.to_v.weight",
            "cross_attn.output_proj.weight": "attn2.to_out.0.weight",
            "mlp.layer1.weight": "ff.net.0.proj.weight",
            "mlp.layer2.weight": "ff.net.2.weight",
        }
        if module in block_map:
            return f"transformer_blocks.{n}.{block_map[module]}"

    raise KeyError(f"No mapping for key: {comfy_key}")


def _remap_comfyui_to_adapter_key(comfy_key: str) -> str:
    """Map ComfyUI llm_adapter keys to the diffusers AnimaLLMAdapter keys."""
    # net.llm_adapter.embed.weight -> embed.weight
    if comfy_key == "net.llm_adapter.embed.weight":
        return "embed.weight"
    if comfy_key == "net.llm_adapter.norm.weight":
        return "norm.weight"
    if comfy_key == "net.llm_adapter.out_proj.weight":
        return "out_proj.weight"
    if comfy_key == "net.llm_adapter.out_proj.bias":
        return "out_proj.bias"
    if comfy_key.startswith("net.llm_adapter.blocks."):
        # net.llm_adapter.blocks.{N}.{module} -> blocks.{N}.{module}
        return comfy_key[len("net.llm_adapter."):]
    raise KeyError(f"No adapter mapping for key: {comfy_key}")


def load_anima_2_9b_transformer(
    anima_path: str,
    num_layers: int = 40,
    torch_dtype=torch.bfloat16,
):
    """Build a 40-layer CosmosTransformer3DModel from the Anima-2.9B checkpoint."""
    from safetensors.torch import load_file

    sd = load_file(anima_path)

    # Config mirrors the base Anima transformer config, but 40 layers.
    config = CosmosTransformer3DModel.load_config(
        "CalamitousFelicitousness/Anima-sdnext-diffusers",
        subfolder="transformer",
    )
    config["num_layers"] = num_layers

    model = CosmosTransformer3DModel.from_config(config).to(torch_dtype)

    # Build remapped state dict, only keeping transformer-relevant keys
    remapped = {}
    skipped = []
    for k, v in sd.items():
        if k.startswith("net.blocks.") or k in (
            "net.x_embedder.proj.1.weight",
            "net.t_embedder.1.linear_1.weight",
            "net.t_embedder.1.linear_2.weight",
            "net.t_embedding_norm.weight",
            "net.final_layer.adaln_modulation.1.weight",
            "net.final_layer.adaln_modulation.2.weight",
            "net.final_layer.linear.weight",
        ):
            try:
                new_key = _remap_comfyui_to_diffusers_key(k)
                remapped[new_key] = v
            except KeyError:
                skipped.append(k)
        else:
            skipped.append(k)

    # Load
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys (may be intentionally skipped):")
        for u in unexpected[:20]:
            print("  ", u)
    if missing:
        print(f"WARNING: {len(missing)} missing keys:")
        for m in missing[:20]:
            print("  ", m)

    print(f"Loaded transformer: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, {num_layers} layers")
    return model


def load_llm_adapter_weights(pipeline, anima_path: str, torch_dtype=torch.bfloat16):
    """Load llm_adapter weights from the Anima-2.9B checkpoint into the pipeline's adapter."""
    from safetensors.torch import load_file

    sd = load_file(anima_path)
    remapped = {}
    for k, v in sd.items():
        if k.startswith("net.llm_adapter."):
            new_key = _remap_comfyui_to_adapter_key(k)
            remapped[new_key] = v

    missing, unexpected = pipeline.llm_adapter.load_state_dict(remapped, strict=False)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected adapter keys")
    if missing:
        print(f"WARNING: {len(missing)} missing adapter keys: {missing}")
    return pipeline.llm_adapter


def build_pipeline(
    anima_path: str,
    base_repo: str = "CalamitousFelicitousness/Anima-sdnext-diffusers",
    device: str = "cuda",
    torch_dtype=torch.bfloat16,
    cpu_offload: bool = True,
):
    """Load the full Anima-2.9B pipeline by assembling components manually."""
    import importlib.util
    import os
    from transformers import Qwen3Model, AutoTokenizer
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    # Locate cached snapshot of the base repo
    from huggingface_hub import snapshot_download
    base_dir = snapshot_download(base_repo)
    print(f"Base repo: {base_dir}")

    # --- Load the custom pipeline class from the repo's pipeline.py ---
    # Use diffusers' dynamic module loading with a local path to avoid hub path issues.
    from diffusers.utils import get_class_from_dynamic_module
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    import importlib
    spec = importlib.util.spec_from_file_location("anima_pipeline_module", os.path.join(base_dir, "pipeline.py"))
    module = importlib.util.module_from_spec(spec)
    # The pipeline.py may import from modeling_llm_adapter; inject the base_dir into sys.path
    sys_path_backup = list(sys.path)
    sys.path.insert(0, base_dir)
    spec.loader.exec_module(module)
    sys.path[:] = sys_path_backup
    AnimaTextToImagePipeline = module.AnimaTextToImagePipeline

    # --- Load text encoder (Qwen3Model) ---
    print("Loading text encoder (Qwen3-0.6B)...")
    text_encoder = Qwen3Model.from_pretrained(
        os.path.join(base_dir, "text_encoder"),
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
    t5_tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "t5_tokenizer"))

    # --- Load LLM adapter ---
    print("Loading LLM adapter...")
    sys_path_backup = list(sys.path)
    sys.path.insert(0, os.path.join(base_dir, "llm_adapter"))
    llm_spec = importlib.util.spec_from_file_location(
        "anima_llm_adapter_module",
        os.path.join(base_dir, "llm_adapter", "modeling_llm_adapter.py"),
    )
    llm_module = importlib.util.module_from_spec(llm_spec)
    llm_spec.loader.exec_module(llm_module)
    sys.path[:] = sys_path_backup
    AnimaLLMAdapter = llm_module.AnimaLLMAdapter

    llm_adapter = AnimaLLMAdapter.from_pretrained(
        os.path.join(base_dir, "llm_adapter"),
        torch_dtype=torch_dtype,
    )

    # --- Load VAE ---
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(base_dir, "vae"),
        torch_dtype=torch_dtype,
    )

    # --- Scheduler ---
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(base_dir, "scheduler"))

    # --- Build 40-layer transformer from Anima-2.9B ---
    print("Building 40-layer transformer from Anima-2.9B checkpoint...")
    transformer = load_anima_2_9b_transformer(anima_path, num_layers=40, torch_dtype=torch_dtype)

    # --- Assemble pipeline ---
    pipe = AnimaTextToImagePipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        t5_tokenizer=t5_tokenizer,
        llm_adapter=llm_adapter,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )

    # Load llm_adapter weights from Anima-2.9B checkpoint (adapter may have been finetuned)
    print("Loading LLM adapter weights...")
    load_llm_adapter_weights(pipe, anima_path, torch_dtype)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--anima-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pipe = build_pipeline(args.anima_path, device=args.device)
    print("Pipeline ready!")
