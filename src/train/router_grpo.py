"""
Router-level GRPO: batched multi-rollout sampling + PPO loss, reward from EditScore.

Scope (deliberately narrower than the EditScore/FlowGRPO paper's trajectory-level RL):
  - The flow-matching sampling PATH is a plain deterministic Euler integration of the
    trained velocity field v_theta -- no SDE / no per-step Gaussian noise term. Time runs
    from t=1 (pure noise, this repo's `x_1`) down to t=0 (data, `x_0`), matching model.py's
    `x_t = (1-t)*x_0 + t*x_1` convention (velocity target `x_1 - x_0` is constant along the
    path, so `x_t = x_t + v * dt` is exact Euler integration of that path, not an approximation).
  - The only stochastic, RL-trainable decision is the MoE ROUTER's expert choice
    (Gumbel-top-k sampling instead of greedy top-k -- see infer_route_weight's
    "stochastic" branch in custom/peft/tuners/lora/layer.py) at a random subset of the T
    sampling steps. Non-selected steps route greedily, so they contribute zero policy
    gradient (mirrors rl_utils.generate_routing_mask's per-token ratio, applied here per
    denoising step instead of per token).
  - All G rollouts for one input start from IDENTICAL initial noise, so the only thing
    that differs between rollouts in a group is which experts got sampled -- this isolates
    the router's causal effect on the final reward from ordinary sampling-noise variance.
  - Reward is terminal only (EditScore on the final decoded image), broadcast equally to
    every stochastic step's log-prob term -- no GAE / per-step credit assignment.
"""
from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from flux.pipeline_tools import decode_latents, encode_images_kontext, prepare_text_input
from .editscore_reward import EditScoreReward, tensor_to_pil
from .rl_utils import (
    _moe_layers,
    _route_linear,
    capture_routing_inputs,
    clear_fixed_routing,
    compute_routing_log_prob,
    compute_routing_log_prob_nograd,
    load_balancing_loss_func,
    router_z_loss_func,
    set_routing_mode,
    set_routing_masks,
)


@dataclass
class RouterGRPOConfig:
    num_rollouts: int = 4            # G
    num_sampling_steps: int = 12     # T (rollout-only; independent of the supervised path)
    stochastic_step_ratio: float = 0.3
    clip_eps: float = 0.2
    rl_coeff: float = 1.0
    supervised_coeff: float = 1.0
    load_balancing_coeff: float = 0.0
    z_coeff: float = 0.0


def _snapshot_actions_cpu(model: torch.nn.Module) -> Dict[int, torch.Tensor]:
    """{layer_id: stored_actions.cpu()} for MoE layers that routed stochastically this step
    (stored_actions is None on layers that routed greedily -- see infer_route_weight)."""
    return {
        id(layer): layer.stored_actions.detach().cpu()
        for layer in _moe_layers(model)
        if layer.stored_actions is not None
    }


def _recompute_aux_losses(
    model: torch.nn.Module,
    g_inputs: Dict[int, torch.Tensor],
    g_actions: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """z-loss / load-balancing loss for one stochastic step, recomputed (with grad) from the
    CPU-captured router inputs -- no re-forward through the transformer needed, same trick as
    rl_utils.compute_routing_log_prob."""
    z_losses, lb_losses = [], []
    device = None
    for layer in _moe_layers(model):
        lid = id(layer)
        if lid not in g_inputs or lid not in g_actions:
            continue
        weight = layer.lora_route[layer.active_adapters[0]].weight
        device = weight.device
        route_logits = grad_checkpoint(_route_linear, g_inputs[lid], weight, use_reentrant=False)
        router_probs = torch.softmax(route_logits, dim=-1)
        actions = g_actions[lid].to(weight.device)
        z_losses.append(router_z_loss_func(route_logits))
        lb_losses.append(load_balancing_loss_func(router_probs, actions))

    if not z_losses:
        zero = torch.zeros((), device=device or "cpu")
        return zero, zero.clone()
    return torch.stack(z_losses).mean(), torch.stack(lb_losses).mean()


@torch.no_grad()
def _sample_rollouts(
    pl_module,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    x_cond: torch.Tensor,
    img_ids: torch.Tensor,
    x0_shape: torch.Size,
    G: int,
    T: int,
    stochastic_step_ratio: float,
    concat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, dict]]:
    """
    Batched G rollouts (batch axis = x.repeat(G, ...), block g occupies rows
    [g*B:(g+1)*B], the layout rl_utils' helpers already assume).

    Returns:
        final_latents: [G*B, N, C] predicted x_0 (data) latent tokens
        old_logprob:   [G*B] routing log-prob at sampling time (no grad; PPO reference),
                        averaged over the stochastic steps that were sampled
        step_captures: {step_idx: {"inputs": {layer_id: Tensor cpu}, "actions": {layer_id: Tensor cpu}}}
                        needed to recompute the *new*-policy log-prob later, with grad,
                        WITHOUT re-running the transformer.
    """
    B = x_cond.shape[0]
    device, dtype = x_cond.device, x_cond.dtype

    prompt_embeds_g = prompt_embeds.repeat(G, 1, 1)
    pooled_g = pooled_prompt_embeds.repeat(G, 1)
    x_cond_g = x_cond.repeat(G, 1, 1)
    # text_ids / img_ids carry no batch dim in this codebase (shared position ids), so they
    # don't need repeating -- see flux/pipeline_tools.py's prepare_text_input / encode_images.

    x1_single = torch.randn(x0_shape, device=device, dtype=dtype)
    x_t = x1_single.repeat(G, 1, 1)

    t_schedule = torch.linspace(1.0, 0.0, T + 1, device=device)
    num_stochastic = max(1, round(T * stochastic_step_ratio))
    stochastic_steps = set(random.sample(range(T), num_stochastic))

    guidance = (
        torch.ones(G * B, device=device, dtype=torch.float32)
        if pl_module.transformer.config.guidance_embeds
        else None
    )

    step_captures: Dict[int, dict] = {}
    logprob_terms: List[torch.Tensor] = []

    was_training = pl_module.transformer.training
    pl_module.transformer.eval()
    try:
        for i in range(T):
            is_stochastic = i in stochastic_steps
            set_routing_mode(pl_module.transformer, "stochastic" if is_stochastic else "greedy")
            set_routing_masks(pl_module.transformer, None)  # all tokens stochastic at a stochastic step

            t_cur = t_schedule[i].repeat(G * B)
            g_inputs: Dict[int, torch.Tensor] = {}
            capture_ctx = capture_routing_inputs(pl_module.transformer, g_inputs) if is_stochastic else nullcontext()
            with capture_ctx:
                v = pl_module.transformer(
                    hidden_states=torch.cat((x_t, x_cond_g), dim=concat_dim),
                    timestep=t_cur,
                    guidance=guidance,
                    pooled_projections=pooled_g,
                    encoder_hidden_states=prompt_embeds_g,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            if pl_module.use_sequence_conditioning:
                v = v[:, : x_t.shape[1]]

            if is_stochastic:
                g_actions = _snapshot_actions_cpu(pl_module.transformer)
                old_lp = compute_routing_log_prob_nograd(pl_module.transformer, g_inputs, g_actions)
                step_captures[i] = {"inputs": g_inputs, "actions": g_actions}
                logprob_terms.append(old_lp)

            dt = t_schedule[i + 1] - t_schedule[i]
            x_t = x_t + v * dt
    finally:
        set_routing_mode(pl_module.transformer, "greedy")
        set_routing_masks(pl_module.transformer, None)
        if was_training:
            pl_module.transformer.train()

    old_logprob = (
        torch.stack(logprob_terms, dim=0).mean(dim=0)
        if logprob_terms
        else torch.zeros(G * B, device=device)
    )
    return x_t, old_logprob, step_captures


def compute_router_grpo_loss(
    pl_module,
    batch: dict,
    reward_model: EditScoreReward,
    cfg: RouterGRPOConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Router-level GRPO loss for one training step: sample G rollouts per input, score the
    decoded images with EditScore, PPO-clip the routing policy against the group-normalized
    advantage. Does NOT include the supervised flow-matching loss -- callers combine this with
    `pl_module.step(batch)` themselves (see model.py's rl_step), since RL always uses free
    routing (task_expert_map is bypassed here) while the supervised pass still respects it.
    """
    if not pl_module.use_sequence_conditioning:
        raise NotImplementedError("Router-GRPO rollout currently only supports Kontext sequence conditioning")

    imgs = batch["image"]
    cond_imgs = batch["condition"]
    prompts = batch["description"]
    B = imgs.shape[0]
    G = cfg.num_rollouts
    assert G >= 2, f"router GRPO needs at least 2 rollouts per group to normalize advantages, got num_rollouts={G}"

    clear_fixed_routing(pl_module.transformer)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            pl_module.flux_kontext_pipe, prompts
        )
        x_0, x_cond, img_ids = encode_images_kontext(
            pl_module.flux_kontext_pipe, imgs, cond_imgs, prompt_embeds.dtype, pl_module.device
        )

    final_latents, old_logprob, step_captures = _sample_rollouts(
        pl_module,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        x_cond,
        img_ids,
        x_0.shape,
        G,
        cfg.num_sampling_steps,
        cfg.stochastic_step_ratio,
        concat_dim=1,
    )

    if not step_captures:
        # No stochastic step got sampled at all (degenerate stochastic_step_ratio /
        # num_sampling_steps combo) -- nothing to score a policy gradient against.
        zero = torch.zeros((), device=pl_module.device)
        return zero, {"reward_mean": 0.0, "reward_std": 0.0, "rl_loss": 0.0, "ratio_mean": 1.0}

    height, width = imgs.shape[-2:]
    with torch.no_grad():
        pil_images = decode_latents(pl_module.flux_kontext_pipe, final_latents, height, width)
    cond_pils = [tensor_to_pil(c) for c in cond_imgs] * G
    prompts_g = list(prompts) * G
    rewards = reward_model.score_batch(cond_pils, pil_images, prompts_g).to(pl_module.device)

    # Group = the same original sample's G rollouts (identical initial noise, see
    # _sample_rollouts) -> per-sample advantage normalization across the G axis.
    rewards_g = rewards.view(G, B)
    advantages = (rewards_g - rewards_g.mean(dim=0, keepdim=True)) / (
        rewards_g.std(dim=0, keepdim=True) + 1e-8
    )
    advantages = advantages.reshape(G * B)

    new_logprob_terms = []
    z_terms, lb_terms = [], []
    need_aux = cfg.load_balancing_coeff > 0 or cfg.z_coeff > 0
    for step_idx, captured in step_captures.items():
        g_inputs, g_actions = captured["inputs"], captured["actions"]
        new_logprob_terms.append(compute_routing_log_prob(pl_module.transformer, g_inputs, g_actions))
        if need_aux:
            z_loss, lb_loss = _recompute_aux_losses(pl_module.transformer, g_inputs, g_actions)
            z_terms.append(z_loss)
            lb_terms.append(lb_loss)
    new_logprob = torch.stack(new_logprob_terms, dim=0).mean(dim=0)

    ratio = torch.exp(new_logprob - old_logprob)
    clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
    rl_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

    total = cfg.rl_coeff * rl_loss
    if need_aux:
        total = total + cfg.z_coeff * torch.stack(z_terms).mean() + cfg.load_balancing_coeff * torch.stack(lb_terms).mean()

    stats = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "rl_loss": rl_loss.item(),
        "ratio_mean": ratio.mean().item(),
    }
    return total, stats
