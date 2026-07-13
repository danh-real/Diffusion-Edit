# Use the modified diffusers & peft library
import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_repo_root, "custom"), os.path.join(_repo_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml
import torch
from safetensors.torch import load_file
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from peft import LoraConfig, get_peft_model_state_dict


def _remap_saved_state_dict(saved_state_dict, model, adapter_name="default"):
    """Reverse get_peft_model_state_dict's adapter-name stripping against the real model keys.

    get_peft_model_state_dict strips an exact ".{adapter_name}" segment immediately before
    ".weight" (e.g. "x_embedder.lora_A.default.weight" -> "x_embedder.lora_A.weight"), but
    leaves MoE expert keys alone since they're named "default_expert_0" etc, not "default" —
    an inexact match. So each saved key either already matches a model key as-is (experts,
    router) or needs ".{adapter_name}" re-inserted before ".weight" (everything else). This
    checks the real key set instead of assuming one or the other, so it doesn't silently
    produce a malformed double-adapter key path the way peft's own set_peft_model_state_dict
    does when it blindly re-inserts ".{adapter_name}" into already-adapter-tagged keys.
    """
    model_keys = set(model.state_dict().keys())
    remapped = {}
    for k, v in saved_state_dict.items():
        if k in model_keys:
            remapped[k] = v
        else:
            prefix, last = k.rsplit(".", 1)
            candidate = f"{prefix}.{adapter_name}.{last}"
            if candidate not in model_keys:
                raise KeyError(f"Could not map saved key {k!r} to a model key")
            remapped[candidate] = v
    return remapped


def load_moe_lora(pipe, ckpt_dir: str, adapter_name: str = "default"):
    """Reconstruct the MoE-LoRA adapter structure and load trained weights into `pipe.transformer`.

    `ckpt_dir` is the directory passed to OminiModel.save_lora during training
    (e.g. "runs/<save_path>/<run_name>/ckpt/<step>"), which holds
    pytorch_lora_weights.safetensors. Its run's lora_config (r, num_experts,
    expert_rank, ...) — needed to rebuild the same adapter shape before loading
    weights into it — lives in config.yaml one level up, saved once per run by
    train_moe.py. A generic pipe.load_lora_weights(ckpt_dir) will not work here:
    it infers a plain LoraConfig from tensor shapes/key names, which cannot
    recover num_experts/expert_rank/top_k or recognize the expert_0..N/lora_route
    key layout our custom peft fork uses for MoE.
    """
    run_dir = os.path.dirname(os.path.dirname(ckpt_dir))
    with open(os.path.join(run_dir, "config.yaml")) as f:
        run_config = yaml.safe_load(f)
    lora_config = run_config["train"]["lora_config"]

    pipe.transformer.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)

    state_dict = load_file(os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors"))
    state_dict = {k.removeprefix("transformer."): v for k, v in state_dict.items()}
    state_dict = _remap_saved_state_dict(state_dict, pipe.transformer, adapter_name=adapter_name)
    missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading MoE LoRA checkpoint: {unexpected}")


pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

load_moe_lora(pipe, "runs/train_moe_lora+type_emb/<run_name>/ckpt/<step>")

input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5
).images[0]

image.save("edit_moe.jpg")
