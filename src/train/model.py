import lightning as L
from diffusers.pipelines import FluxPipeline, FluxKontextPipeline, FluxKontextInpaintPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
import torch
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import load_file
import os
import prodigyopt

from flux.transformer import tranformer_forward
from flux.condition import Condition
from flux.pipeline_tools import encode_images, encode_images_fill, encode_images_kontext, prepare_text_input
from .rl_utils import set_fixed_routing, clear_fixed_routing


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


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        task_expert_map: dict = None,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config

        self.optimizer_config = optimizer_config
        self.task_expert_map = task_expert_map

        # Kontext-dev's transformer takes only the vanilla 64 packed-latent channels and
        # conditions by appending the reference image as extra sequence tokens. Fill-dev's
        # transformer is channel-extended (384) to take noisy+masked-image+mask concatenated
        # on the channel axis instead — it needs FluxKontextInpaintPipeline's mask_processor,
        # which plain FluxKontextPipeline doesn't have. Peek the checkpoint's transformer
        # config (no weights downloaded) to pick the right pipeline class, since this model.py
        # is shared by both train.py (Fill) and train_moe.py (Kontext).
        transformer_config = FluxTransformer2DModel.load_config(flux_fill_id, subfolder="transformer")
        self.use_sequence_conditioning = transformer_config.get("in_channels") == 64
        pipeline_cls = FluxKontextPipeline if self.use_sequence_conditioning else FluxKontextInpaintPipeline

        # Load the Flux pipeline
        self.flux_kontext_pipe = pipeline_cls.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_kontext_pipe.transformer
        self.text_encoder = self.flux_kontext_pipe.text_encoder
        self.text_encoder_2 = self.flux_kontext_pipe.text_encoder_2
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_kontext_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)
        
        # Freeze the transformer
        self.transformer.requires_grad_(False)

        # Set the trainable parameters
        if self.model_config['train_route_only']:
            self.trainable_params = [p for name, p in self.lora_layers if "lora_route" in name]
        else:
            self.trainable_params = [p for name, p in self.lora_layers]

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # A generic self.flux_kontext_pipe.load_lora_weights(lora_path) won't work here:
            # it infers a plain LoraConfig from tensor shapes/key names, which cannot recover
            # num_experts/expert_rank/top_k or recognize the expert_0..N/lora_route key layout
            # our custom peft fork uses for MoE. Rebuild the adapter from lora_config (the same
            # config the checkpoint was trained with) before loading its raw weights.
            assert lora_config, "lora_config is required to rebuild the adapter shape for lora_path"
            self.transformer.add_adapter(LoraConfig(**lora_config))

            state_dict = load_file(os.path.join(lora_path, "pytorch_lora_weights.safetensors"))
            state_dict = {k.removeprefix("transformer."): v for k, v in state_dict.items()}
            state_dict = _remap_saved_state_dict(state_dict, self.transformer)
            missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys loading LoRA checkpoint: {unexpected}")

            lora_layers = filter(
                lambda p: p[1].requires_grad, self.transformer.named_parameters()
            )
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p[1].requires_grad, self.transformer.named_parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        type(self.flux_kontext_pipe).save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        opt_config = self.optimizer_config
        
        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        cond_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]
        task = batch["task"][0] if "task" in batch else None

        # Force a specific expert for this batch's task if the caller supplied a mapping;
        # otherwise fall back to lora_route's own learned top-k choice.
        if self.task_expert_map and task in self.task_expert_map:
            set_fixed_routing(self.transformer, self.task_expert_map[task])
        else:
            clear_fixed_routing(self.transformer)

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_kontext_pipe, prompts
            )

            if self.use_sequence_conditioning:
                x_0, x_cond, img_ids = encode_images_kontext(self.flux_kontext_pipe, imgs, cond_imgs, prompt_embeds.dtype, prompt_embeds.device)
            else:
                x_0, x_cond, img_ids = encode_images_fill(self.flux_kontext_pipe, imgs, cond_imgs, prompt_embeds.dtype, prompt_embeds.device)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)
                
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        concat_dim = 1 if self.use_sequence_conditioning else 2
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=concat_dim),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        if self.use_sequence_conditioning:
            # extra condition tokens were appended after the target tokens; discard their
            # predictions the same way FluxKontextInpaintPipeline slices noise_pred
            pred = pred[:, : x_0.shape[1]]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss