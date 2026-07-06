"""
GRPO-style RL utilities for LoRA routing.

Reward function interface:
    RewardFn: (pred: Tensor[B,T,C], target: Tensor[B,T,C]) -> Tensor[B]
    Higher reward = better.  Default: negative per-sample MSE.

To plug in a custom reward (e.g. LPIPS, CLIP), pass it as `reward_fn` in
GRPOConfig — the signature is the only contract.
"""
from __future__ import annotations
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ── Reward function ───────────────────────────────────────────────────────────

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def default_reward_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative per-sample MSE over all non-batch dims.  Shape: [B]."""
    dims = list(range(1, pred.ndim))
    return -((pred - target) ** 2).mean(dim=dims)


# ── GRPO config ───────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    train_lora: bool = False
    """Whether to also train LoRA weights"""

    num_rollouts: int = 4
    """Number of stochastic routing samples per batch (G).
    More rollouts → lower variance but G× more no-grad forward passes."""

    rl_coeff: float = 0.05
    """Weight of the RL policy-gradient loss relative to the supervised loss."""

    supervised_coeff: float = 1.0
    """Weight of the supervised diffusion loss."""

    clip_eps: float = 0.2
    """PPO clip range ε.  The probability ratio r = exp(log_new − log_old) is
    clamped to [1−ε, 1+ε] before being multiplied by the advantage.
    log_old is computed from the current policy at the start of each Pass 2
    rollout (no gradient), so r starts at 1 and diverges as gradient updates
    accumulate (e.g. with accumulate_grad_batches > 1 or multi-step updates)."""

    load_balancing_coeff: float = 0.0
    """Weight of the Switch-Transformer load-balancing loss. Penalises uneven
    expert utilisation; set > 0 to prevent routing collapse."""

    z_coeff: float = 0.0
    """Weight of the router z-loss (ST-MoE). Encourages small logit magnitudes
    to improve routing stability."""

    stochastic_ratio: float = 1.0
    """Fraction of tokens per sequence that receive stochastic (Gumbel-noised)
    routing during RL rollouts.  The remaining tokens are routed greedily by
    the actual lora_route scores, and their actions are excluded from the
    policy-gradient loss so gradient flows only through the stochastic subset.
    1.0 = all tokens stochastic (original behaviour).  0.5 = half stochastic."""

    reward_fn: RewardFn = field(default_factory=lambda: default_reward_fn)
    """Reward function.  Swap here for LPIPS, CLIP-score, etc."""


# ── MoE layer helpers ─────────────────────────────────────────────────────────

def _moe_layers(model: nn.Module):
    """Yield every MoE Linear layer (those that carry a routing_mode attr)."""
    for m in model.modules():
        if getattr(m, "moe_lora", False) and hasattr(m, "routing_mode"):
            yield m


def generate_routing_mask(
    B: int, T: int, stochastic_ratio: float, device: torch.device
) -> torch.Tensor:
    """
    Generate a [B, T] boolean mask: True = stochastic routing, False = greedy.

    Exactly round(T * stochastic_ratio) positions per batch item are selected
    for stochastic (Gumbel-noised) routing; the rest follow the deterministic
    lora_route scores.  The same mask is broadcast conceptually to every MoE
    layer, though each layer generates its own shape-matched version at runtime
    (see layer.moe_forward) because different FLUX layer types see different T.
    """
    num_stochastic = max(1, round(T * stochastic_ratio))
    noise = torch.rand(B, T, device=device)
    _, indices = noise.topk(num_stochastic, dim=-1)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask.scatter_(1, indices, True)
    return mask


def set_routing_masks(
    model: nn.Module,
    masks: Optional[Dict[int, torch.Tensor]],
) -> None:
    """Set per-layer routing masks from a {layer_id: [B, T] bool} dict, or clear with None."""
    for layer in _moe_layers(model):
        layer.routing_mask = None if masks is None else masks.get(id(layer))


@contextmanager
def capture_layer_seq_lens(model: nn.Module, storage: Dict[int, int]):
    """Capture {id(layer): T} for each MoE layer during a forward pass.

    Used to learn each layer's sequence length from Pass 1 so that
    generate_global_routing_masks can allocate the global token budget
    correctly before the stochastic rollouts.
    """
    hooks = [
        layer.register_forward_hook(
            lambda m, inp, _: storage.__setitem__(id(m), inp[0].shape[1])
        )
        for layer in _moe_layers(model)
    ]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def generate_global_routing_masks(
    layer_seq_lens: Dict[int, int],
    B: int,
    stochastic_ratio: float,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Generate one mask spanning every MoE layer's token budget, then split per layer.

    All layer sequence lengths are concatenated into a single pool of
    sum(T_i) positions.  Exactly round(total * stochastic_ratio) slots are
    drawn uniformly from this pool and marked stochastic, then the mask is
    split back into per-layer [B, T_i] tensors.

    Because slots are drawn from a shared pool, some layers may receive zero
    stochastic tokens and others may receive more than ratio * T_i — the
    natural variability the caller wants.

    Returns {layer_id: [B, T_i] bool tensor} on `device`.
    """
    layer_ids = list(layer_seq_lens.keys())
    seq_lens  = [layer_seq_lens[lid] for lid in layer_ids]
    total_T   = sum(seq_lens)

    num_stochastic = max(1, round(total_T * stochastic_ratio))
    noise = torch.rand(B, total_T, device=device)
    _, indices = noise.topk(num_stochastic, dim=-1)
    global_mask = torch.zeros(B, total_T, dtype=torch.bool, device=device)
    global_mask.scatter_(1, indices, True)

    per_layer: Dict[int, torch.Tensor] = {}
    offset = 0
    for lid, T in zip(layer_ids, seq_lens):
        per_layer[lid] = global_mask[:, offset : offset + T]
        offset += T
    return per_layer


def set_routing_mode(model: nn.Module, mode: str) -> None:
    """Broadcast routing mode to all MoE layers.  mode ∈ {greedy, stochastic, replay}."""
    for layer in _moe_layers(model):
        layer.routing_mode = mode


def set_fixed_routing(model: nn.Module, expert_idx: int) -> None:
    """Fix all MoE layers to route exclusively to expert_idx (replay mode, one-hot weight).

    Sets imposed_route_weight to a [1, 1, E] one-hot tensor that broadcasts over
    batch and sequence dimensions, then switches routing_mode to 'replay' so the
    layer bypasses the learned router entirely.  Call clear_fixed_routing() after
    the forward/backward to restore normal routing.
    """
    for layer in _moe_layers(model):
        E = layer.num_experts
        route_linear = layer.lora_route[layer.active_adapters[0]]
        weight = torch.zeros(1, 1, E, dtype=route_linear.weight.dtype, device=route_linear.weight.device)
        weight[0, 0, expert_idx] = 1.0
        layer.imposed_route_weight = weight
        layer.routing_mode = "replay"


def clear_fixed_routing(model: nn.Module) -> None:
    """Clear imposed_route_weight and reset routing_mode to greedy on all MoE layers."""
    for layer in _moe_layers(model):
        layer.imposed_route_weight = None
        layer.routing_mode = "greedy"


def set_type_embedding(model: nn.Module, type_emb: Optional[torch.Tensor]) -> None:
    """Broadcast the task/type embedding to all MoE layers, or clear it with None.

    When set, layer.moe_forward routes on this [B, D] embedding instead of its
    own per-token input — one routing decision per sample, applied to every
    token. Requires the layer's lora_route to have been sized via
    LoraConfig.route_input_dim == D.
    """
    for layer in _moe_layers(model):
        layer.type_emb = type_emb


def save_actions(model: nn.Module) -> Dict[int, torch.Tensor]:
    """Snapshot stored_actions and stored_mask from every MoE layer after a stochastic pass.
    All tensors stored on CPU to avoid occupying GPU memory across rollouts."""
    return {
        id(layer): (layer.stored_actions, layer.stored_logits, layer.stored_mask)
        for layer in _moe_layers(model)
        if layer.stored_actions is not None and layer.stored_logits is not None
    }


def save_actions_batched(
    model: nn.Module, G: int, B: int
) -> List[Dict[int, torch.Tensor]]:
    """
    After a batched stochastic forward with batch size G*B, split stored_actions
    (shape [G*B, T, K] on GPU) into G per-rollout dicts of [B, T, K] on CPU.
    """
    result: List[Dict[int, torch.Tensor]] = [{} for _ in range(G)]
    for layer in _moe_layers(model):
        if layer.stored_actions is None:
            continue
        actions = layer.stored_actions  # [G*B, T, K] on GPU
        lid = id(layer)
        for g in range(G):
            result[g][lid] = actions[g * B : (g + 1) * B].cpu()
    return result


# ── Advantage computation ─────────────────────────────────────────────────────

def grpo_advantages(rewards: List[float], eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a list of scalar rewards within the group: (r - mean) / std.
    Returns a 1-D float32 tensor of shape [G].
    """
    t = torch.tensor(rewards, dtype=torch.float32)
    return (t - t.mean()) / (t.std() + eps)


def generate_change_map(seq_length, num_blocks, change_ratio=0.1):
    total_positions = seq_length * num_blocks
    num_change = int(total_positions * change_ratio)

    # 전체 인덱스 목록
    all_positions = list(range(total_positions))
    # 무작위로 num_change개 뽑음
    changed_positions = random.sample(all_positions, num_change)

    # 2D mask
    change_map = [[False]*seq_length for _ in range(num_blocks)]
    for pos in changed_positions:
        block_i = pos // seq_length
        token_j = pos % seq_length
        change_map[block_i][token_j] = True

# ── Policy gradient loss (hook-based, no replay forward) ─────────────────────

@contextmanager
def capture_routing_inputs(model: nn.Module, storage: Dict[int, torch.Tensor]):
    """
    Capture routing-layer hidden-state inputs directly onto CPU.

    Moving to CPU in the hook avoids retaining large GPU tensors (each routing
    layer's input can be [B, T, ffn_dim] ≈ 63 MB in bf16 for FLUX) across
    the whole rollout loop.  Only CPU RAM is consumed, which is much larger.
    """
    hooks = [
        layer.register_forward_hook(
            lambda m, inp, out: storage.__setitem__(id(m), inp[0].detach().cpu())
        )
        for layer in _moe_layers(model)
    ]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def capture_routing_logits(model: nn.Module, storage: Dict[int, torch.Tensor]):
    """
    Capture route_logits WITH grad for each MoE layer during a forward pass.

    Hooks on lora_route[adapter_name] (an nn.Linear) whose output is route_logits.
    Does NOT detach — gradient flows back to lora_route.weight so auxiliary losses
    (z-loss, load-balancing) can update the router.

    Safe with use_reentrant=False gradient checkpointing: hooks are removed before
    backward, so checkpointed recompute does not double-populate storage.
    """
    hooks = []
    for layer in _moe_layers(model):
        adapter_name = layer.active_adapters[0]
        route_linear = layer.lora_route[adapter_name]
        hooks.append(
            route_linear.register_forward_hook(
                lambda m, _, out: storage.__setitem__(id(m), out)
            )
        )
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def compute_aux_losses(
    model: nn.Module,
    routing_logits: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute z-loss and load-balancing loss from logits captured via
    capture_routing_logits. Returns (z_loss, lb_loss) averaged over MoE layers.

    router_probs (full softmax over all experts) and expert_indices (top-k
    selection, detached) are both derived from the captured logits so no extra
    state needs to be stored on the layer.
    """
    z_losses:  List[torch.Tensor] = []
    lb_losses: List[torch.Tensor] = []
    last_logits: Optional[torch.Tensor] = None

    for layer in _moe_layers(model):
        adapter_name = layer.active_adapters[0]
        lid = id(layer.lora_route[adapter_name])
        if lid not in routing_logits:
            continue
        logits = routing_logits[lid]                          # [B, T, E], has grad
        router_probs   = F.softmax(logits, dim=-1)            # [B, T, E]
        expert_indices = logits.detach().topk(layer.top_k, dim=-1).indices  # [B, T, K]

        z_losses.append(router_z_loss_func(logits))
        lb_losses.append(load_balancing_loss_func(router_probs, expert_indices))
        last_logits = logits

    zero = (
        torch.zeros(1, device=last_logits.device, dtype=last_logits.dtype).squeeze()
        if last_logits is not None
        else torch.zeros(1).squeeze()
    )
    z_loss  = torch.stack(z_losses).mean()  if z_losses  else zero
    lb_loss = torch.stack(lb_losses).mean() if lb_losses else zero
    return z_loss, lb_loss


@torch.no_grad()
def compute_routing_log_prob_nograd(
    model: nn.Module,
    g_inputs: Dict[int, torch.Tensor],                          # CPU tensors
    g_actions: Dict[int, torch.Tensor],                         # CPU tensors
    g_masks: Optional[Dict[int, Optional[torch.Tensor]]] = None,  # CPU bool [B,T]
) -> torch.Tensor:
    """
    Compute routing log-prob with NO gradient — used as the old-policy reference.

    Processes one layer at a time: x_cpu is moved to GPU (~63 MB transiently),
    route_logits are computed, then the GPU copy is freed before the next layer.
    Peak GPU overhead: one layer's hidden state.

    When g_masks is provided, log-probs are averaged only over stochastic
    token positions (mask==True); non-stochastic positions are excluded so the
    ratio stays 1 for deterministic decisions.

    Returns a detached scalar (same value as compute_routing_log_prob when
    called with the same weights — the ratio starts at 1 and diverges as
    accumulated gradient updates shift the current policy away from this anchor).
    """
    layers = list(_moe_layers(model))
    if not layers:
        raise RuntimeError("No MoE routing layers found in model.")

    layer_log_probs: List[torch.Tensor] = []
    weight = None
    for layer in layers:
        lid = id(layer)
        if lid not in g_inputs or lid not in g_actions:
            continue

        adapter_name = layer.active_adapters[0]
        weight = layer.lora_route[adapter_name].weight   # [E, D] on GPU
        x_gpu  = g_inputs[lid].to(weight.device)         # [B, T, D], transient GPU copy
        actions = g_actions[lid].to(weight.device)       # [B, T, K]

        route_logits  = F.linear(x_gpu, weight)           # [B, T, E]
        log_probs_all = torch.log_softmax(route_logits, dim=-1)
        selected_lp   = log_probs_all.gather(-1, actions).sum(-1)  # [B, T]

        mask = g_masks.get(lid) if g_masks else None
        if mask is not None:
            mask_f = mask.to(weight.device).float()              # [B, T]
            n_stoch = mask_f.sum(-1).clamp(min=1.0)              # [B]
            selected_lp = (selected_lp * mask_f).sum(-1) / n_stoch  # [B]
        else:
            selected_lp = selected_lp.mean(-1)                   # [B]

        layer_log_probs.append(selected_lp)
        del x_gpu  # free GPU copy immediately

    if not layer_log_probs:
        dev = weight.device if weight is not None else torch.device("cpu")
        dtype = weight.dtype if weight is not None else torch.float32
        return torch.zeros(1, device=dev, dtype=dtype).squeeze()

    return torch.stack(layer_log_probs, dim=0).mean(dim=0).mean()


def _route_linear(x_cpu: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Module-free linear forward used inside a gradient checkpoint.

    x_cpu lives on CPU; weight lives on GPU and requires grad.
    The checkpoint saves x_cpu (CPU, free) and weight (already accounted for)
    instead of x_gpu (large GPU tensor), so x_gpu is only created transiently
    during the checkpoint's forward/backward execution and is freed afterwards.
    """
    return F.linear(x_cpu.to(weight.device), weight)


def compute_routing_log_prob(
    model: nn.Module,
    g_inputs: Dict[int, torch.Tensor],                          # CPU tensors
    g_actions: Dict[int, torch.Tensor],                         # CPU tensors
    g_masks: Optional[Dict[int, Optional[torch.Tensor]]] = None,  # CPU bool [B,T]
) -> torch.Tensor:
    """
    Compute the mean routing log-prob for ONE rollout with minimal GPU pressure.

    Two-level memory optimisation:
    1. g_inputs are on CPU — no large GPU allocation for hidden states.
    2. The linear x→route_logits is wrapped in gradient checkpointing so the
       autograd graph saves x_cpu (CPU) instead of x_gpu (GPU).  During
       rl_loss.backward() each layer's x is moved to GPU transiently (~63 MB
       peak per layer), then freed before the next layer's backward.

    When g_masks is provided, log-probs (and gradients) are restricted to the
    stochastic token positions (mask==True).  Non-stochastic positions were
    routed greedily so their contribution to the policy-gradient is zero.

    Returns a scalar: mean over layers and batch of log P(action | state),
    with gradient flowing only through lora_route.weight (and only through
    stochastic token positions when g_masks is given).
    """
    layers = list(_moe_layers(model))
    if not layers:
        raise RuntimeError("No MoE routing layers found in model.")

    layer_log_probs: List[torch.Tensor] = []
    weight = None
    for layer in layers:
        lid = id(layer)
        if lid not in g_inputs or lid not in g_actions:
            continue

        adapter_name = layer.active_adapters[0]
        weight = layer.lora_route[adapter_name].weight  # [E, D] on GPU, requires_grad
        x_cpu = g_inputs[lid]                           # [B, T, D] on CPU, no grad

        # Checkpointed linear: saves (x_cpu, weight) for recompute, NOT x_gpu.
        # x_gpu ([B,T,D] on GPU) is created inside _route_linear and freed when
        # the checkpoint function returns — it is never part of the saved graph.
        route_logits = grad_checkpoint(
            _route_linear, x_cpu, weight, use_reentrant=False
        )  # [B, T, E], has grad via weight

        actions = g_actions[lid].to(weight.device)      # [B, T, K], tiny int tensor
        log_probs_all = torch.log_softmax(route_logits, dim=-1)
        selected_lp = log_probs_all.gather(-1, actions).sum(-1)  # [B, T]

        mask = g_masks.get(lid) if g_masks else None
        if mask is not None:
            # Zero out non-stochastic positions: gradient only flows through mask==True.
            mask_f = mask.to(weight.device).float()              # [B, T]
            n_stoch = mask_f.sum(-1).clamp(min=1.0)              # [B]
            selected_lp = (selected_lp * mask_f).sum(-1) / n_stoch  # [B]
        else:
            selected_lp = selected_lp.mean(-1)                   # [B]

        layer_log_probs.append(selected_lp)

    if not layer_log_probs:
        dev = weight.device if weight is not None else torch.device("cpu")
        dtype = weight.dtype if weight is not None else torch.float32
        return torch.zeros(1, dtype=dtype, device=dev).squeeze()

    return torch.stack(layer_log_probs, dim=0).mean(dim=0).mean()


def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)

#added
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)