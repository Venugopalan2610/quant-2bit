"""Per-expert activation collector for joint LUT fine-tuning.

Captures (input, output) tensor pairs per expert by patching OlmoeExperts.forward
in the same style as hessian/hooks.py. Subsamples to a fixed token budget per
expert to keep memory bounded.
"""
import os
import types
import gc
import copy
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OlmoeForCausalLM


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HIDDEN_DIR = "cache/hidden_states"


class ExpertActivationCollector:
    """Reservoir-style per-expert (X, Y) collector. Up to budget tokens per expert."""

    def __init__(self, num_experts, hidden_size, budget_per_expert=1024):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.budget = budget_per_expert
        # Lists of CPU tensors so we don't blow GPU memory
        self.inputs  = [[] for _ in range(num_experts)]
        self.outputs = [[] for _ in range(num_experts)]
        self.counts  = [0] * num_experts

    def add(self, expert_idx, x, y):
        """x, y: (n_e, hidden) fp32 on GPU. Subsample down to remaining budget."""
        e = int(expert_idx)
        remaining = self.budget - self.counts[e]
        if remaining <= 0:
            return
        n = x.shape[0]
        if n > remaining:
            # Random subsample
            idx = torch.randperm(n, device=x.device)[:remaining]
            x = x[idx]
            y = y[idx]
        self.inputs[e].append(x.detach().cpu())
        self.outputs[e].append(y.detach().cpu())
        self.counts[e] += x.shape[0]

    def finalize(self):
        """Concatenate per-expert lists into (n_e, hidden) tensors."""
        out = {}
        for e in range(self.num_experts):
            if self.counts[e] == 0:
                out[e] = (None, None)
                continue
            X = torch.cat(self.inputs[e], dim=0)
            Y = torch.cat(self.outputs[e], dim=0)
            out[e] = (X, Y)
        return out


def _patched_experts_forward_capture(self, hidden_states, top_k_index, top_k_weights):
    """Mirror of OlmoeExperts.forward, capturing per-expert (X, Y) pairs."""
    collector = self._activation_collector
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # X = current_state (input to expert MLP)
        x_capture = current_state.to(torch.float32)

        gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        mlp_intermediate = self.act_fn(gate) * up
        current_hidden_states = F.linear(mlp_intermediate, self.down_proj[expert_idx])

        # Y = expert MLP output, BEFORE top_k weighting
        y_capture = current_hidden_states.to(torch.float32)
        collector.add(expert_idx, x_capture, y_capture)

        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


def install_capture_patch(experts_module, collector):
    if hasattr(experts_module, "_orig_forward_capture"):
        raise RuntimeError("Already patched")
    experts_module._orig_forward_capture = experts_module.forward
    experts_module._activation_collector = collector
    experts_module.forward = types.MethodType(_patched_experts_forward_capture, experts_module)
    def restore():
        experts_module.forward = experts_module._orig_forward_capture
        del experts_module._orig_forward_capture
        del experts_module._activation_collector
    return restore


def collect_layer_activations(model, rotary_emb, layer_idx, cfg, device,
                                budget_per_expert=1024, max_shards=None):
    """Re-run hidden states through one layer to collect per-expert (X, Y).

    Reuses cache/hidden_states/layer_{N}_input/ shards if present.
    Otherwise we need to regenerate them — handled by the orchestrator.

    Returns: dict {expert_idx: (X, Y)} with X, Y as CPU tensors of shape (n_e, hidden_size).
    """
    if isinstance(device, str):
        device = torch.device(device)
    in_dir = os.path.join(HIDDEN_DIR, f"layer_{layer_idx:02d}_input")
    if not os.path.exists(in_dir):
        raise FileNotFoundError(
            f"Hidden states for layer {layer_idx} missing at {in_dir}. "
            f"Run regenerate_hidden_states.py first."
        )

    meta = torch.load(os.path.join(in_dir, "meta.pt"), weights_only=True)
    seq_len = meta["seq_len"]
    n_shards = meta["n_shards"]
    if max_shards is not None:
        n_shards = min(n_shards, max_shards)

    # Fresh fp32 GPU copy of layer N
    layer = copy.deepcopy(model.model.layers[layer_idx]).to(
        device=device, dtype=torch.float32,
    )

    collector = ExpertActivationCollector(
        num_experts=cfg.num_experts,
        hidden_size=cfg.hidden_size,
        budget_per_expert=budget_per_expert,
    )
    restore = install_capture_patch(layer.mlp.experts, collector)

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    t_start = time.time()
    try:
        with torch.no_grad():
            for shard_idx in range(n_shards):
                shard_bf16 = torch.load(
                    os.path.join(in_dir, f"shard_{shard_idx:04d}.pt"),
                    weights_only=True,
                )
                B = shard_bf16.shape[0]
                shard_fp32 = shard_bf16.to(device=device, dtype=torch.float32)
                del shard_bf16

                pos_ids_batch = position_ids.expand(B, -1)
                cos, sin = rotary_emb(shard_fp32, pos_ids_batch)

                _ = layer(
                    shard_fp32,
                    attention_mask=None,
                    position_ids=pos_ids_batch,
                    position_embeddings=(cos, sin),
                    use_cache=False,
                )
                del shard_fp32, cos, sin
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # Early exit if all experts are saturated
                if all(c >= budget_per_expert for c in collector.counts):
                    print(f"    all experts saturated at shard {shard_idx + 1}/{n_shards}")
                    break
    finally:
        restore()

    captures = collector.finalize()
    elapsed = time.time() - t_start
    counts = [collector.counts[e] for e in range(cfg.num_experts)]
    print(f"    collected in {elapsed:.1f}s — "
          f"counts: min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.0f}")

    del layer, collector
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return captures


class MoEIOCollector:
    """Reservoir collector for (X_moe, Y_moe) pairs at the MoE block boundary."""

    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.xs = []
        self.ys = []
        self.count = 0

    def add(self, x, y):
        """x, y: (B, T, H) or (N, H) fp32 GPU. Flattens and subsamples."""
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        remaining = self.max_tokens - self.count
        if remaining <= 0:
            return
        n = x.shape[0]
        if n > remaining:
            idx = torch.randperm(n, device=x.device)[:remaining]
            x = x[idx]
            y = y[idx]
        self.xs.append(x.detach().cpu())
        self.ys.append(y.detach().cpu())
        self.count += x.shape[0]

    def finalize(self):
        if self.count == 0:
            return None, None
        X = torch.cat(self.xs, dim=0)
        Y = torch.cat(self.ys, dim=0)
        return X, Y


def collect_moe_io(model, rotary_emb, layer_idx, cfg, device,
                   max_tokens=16384, max_shards=None):
    """Capture (X_moe, Y_moe) at the MoE block boundary for one layer.

    X_moe: input to `layer.mlp` (post_attention_layernorm output), flattened (N, H)
    Y_moe: output of `layer.mlp` (before residual add), flattened (N, H)

    Returns (X_moe, Y_moe) CPU fp32 tensors of shape (N, H) with N <= max_tokens.
    """
    if isinstance(device, str):
        device = torch.device(device)
    in_dir = os.path.join(HIDDEN_DIR, f"layer_{layer_idx:02d}_input")
    if not os.path.exists(in_dir):
        raise FileNotFoundError(
            f"Hidden states for layer {layer_idx} missing at {in_dir}. "
            f"Run regenerate_hidden_states.py first."
        )

    meta = torch.load(os.path.join(in_dir, "meta.pt"), weights_only=True)
    seq_len = meta["seq_len"]
    n_shards = meta["n_shards"]
    if max_shards is not None:
        n_shards = min(n_shards, max_shards)

    layer = copy.deepcopy(model.model.layers[layer_idx]).to(
        device=device, dtype=torch.float32,
    )

    collector = MoEIOCollector(max_tokens=max_tokens)

    def hook(module, inputs, output):
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        collector.add(x_in.to(torch.float32), output.to(torch.float32))

    handle = layer.mlp.register_forward_hook(hook)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    t_start = time.time()
    try:
        with torch.no_grad():
            for shard_idx in range(n_shards):
                if collector.count >= max_tokens:
                    print(f"    saturated at shard {shard_idx}/{n_shards}")
                    break
                shard_bf16 = torch.load(
                    os.path.join(in_dir, f"shard_{shard_idx:04d}.pt"),
                    weights_only=True,
                )
                B = shard_bf16.shape[0]
                shard_fp32 = shard_bf16.to(device=device, dtype=torch.float32)
                del shard_bf16

                pos_ids_batch = position_ids.expand(B, -1)
                cos, sin = rotary_emb(shard_fp32, pos_ids_batch)

                _ = layer(
                    shard_fp32,
                    attention_mask=None,
                    position_ids=pos_ids_batch,
                    position_embeddings=(cos, sin),
                    use_cache=False,
                )
                del shard_fp32, cos, sin
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        handle.remove()

    X, Y = collector.finalize()
    elapsed = time.time() - t_start
    print(f"    collected {collector.count} tokens in {elapsed:.1f}s")

    del layer, collector
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return X, Y


def collect_layer_io(model, rotary_emb, layer_idx, cfg, device,
                     max_shards=None):
    """Capture full decoder-layer I/O as 3D shards (needed for attention).

    Returns:
        shards_X: list of (B, T, H) fp32 CPU tensors  — layer inputs
        shards_Y: list of (B, T, H) fp32 CPU tensors  — layer outputs (post-residuals)
        seq_len: int
    """
    if isinstance(device, str):
        device = torch.device(device)
    in_dir = os.path.join(HIDDEN_DIR, f"layer_{layer_idx:02d}_input")
    if not os.path.exists(in_dir):
        raise FileNotFoundError(
            f"Hidden states for layer {layer_idx} missing at {in_dir}. "
            f"Run regenerate_hidden_states.py first."
        )

    meta = torch.load(os.path.join(in_dir, "meta.pt"), weights_only=True)
    seq_len = meta["seq_len"]
    n_shards = meta["n_shards"]
    if max_shards is not None:
        n_shards = min(n_shards, max_shards)

    layer = copy.deepcopy(model.model.layers[layer_idx]).to(
        device=device, dtype=torch.float32,
    )

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    shards_X, shards_Y = [], []

    t_start = time.time()
    with torch.no_grad():
        for shard_idx in range(n_shards):
            shard_bf16 = torch.load(
                os.path.join(in_dir, f"shard_{shard_idx:04d}.pt"),
                weights_only=True,
            )
            B = shard_bf16.shape[0]
            shard_fp32 = shard_bf16.to(device=device, dtype=torch.float32)
            del shard_bf16

            pos_ids_batch = position_ids.expand(B, -1)
            cos, sin = rotary_emb(shard_fp32, pos_ids_batch)

            y = layer(
                shard_fp32,
                attention_mask=None,
                position_ids=pos_ids_batch,
                position_embeddings=(cos, sin),
                use_cache=False,
            )
            if isinstance(y, tuple):
                y = y[0]

            shards_X.append(shard_fp32.detach().cpu())
            shards_Y.append(y.detach().cpu())
            del shard_fp32, cos, sin, y
            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    total_tokens = sum(s.shape[0] * s.shape[1] for s in shards_X)
    print(f"    collected {len(shards_X)} shards ({total_tokens} tokens) in {elapsed:.1f}s")

    del layer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return shards_X, shards_Y, seq_len