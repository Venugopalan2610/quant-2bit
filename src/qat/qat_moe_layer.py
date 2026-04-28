"""QATMoELayer: experts-only QAT version of OlmoeSparseMoeBlock.

Mirrors the upstream forward:
    router_logits = softmax(F.linear(h, gate_weight))
    top_values, top_index = topk(router_logits, top_k)
    (optionally) top_values /= top_values.sum(-1, keepdim=True)
    for each expert hit:
        y_e = expert(h[token_idx])
        final.index_add_(0, token_idx, y_e * top_values[...])

The router is frozen (copied from the FP16 checkpoint). Only QATExpert
W_latent tensors are trained. Loss flows through the experts that fire for
each batch — which, with top-k=8 and 64 experts, hits all experts in
expectation within a few batches.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.qat.qat_expert import QATExpert


class QATMoELayer(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size,
                 top_k, norm_topk_prob, codebook_gpu,
                 seed=0, reencode_every_n_steps=1,
                 device="cuda", dtype=torch.float32):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        self.gate_weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.experts = nn.ModuleList([
            QATExpert(
                hidden_size=hidden_size, intermediate_size=intermediate_size,
                codebook_gpu=codebook_gpu,
                seed=seed + 100 * e,
                reencode_every_n_steps=reencode_every_n_steps,
                device=device, dtype=dtype,
            )
            for e in range(num_experts)
        ])

    @classmethod
    def from_mlp_block(cls, mlp_block, config, codebook_gpu,
                       seed=0, reencode_every_n_steps=1,
                       device="cuda", dtype=torch.float32):
        """Build from an OlmoeSparseMoeBlock. Copies router; inits experts to FP."""
        num_experts = config.num_experts
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        top_k = config.num_experts_per_tok
        norm_topk_prob = getattr(config, "norm_topk_prob", True)

        layer = cls(
            num_experts=num_experts, hidden_size=hidden_size,
            intermediate_size=intermediate_size, top_k=top_k,
            norm_topk_prob=norm_topk_prob, codebook_gpu=codebook_gpu,
            seed=seed, reencode_every_n_steps=reencode_every_n_steps,
            device=device, dtype=dtype,
        )
        with torch.no_grad():
            layer.gate_weight.copy_(
                mlp_block.gate.weight.detach().to(device=device, dtype=dtype)
            )
            gate_up_all = mlp_block.experts.gate_up_proj.detach().to(
                device=device, dtype=dtype
            )
            down_all = mlp_block.experts.down_proj.detach().to(
                device=device, dtype=dtype
            )
            for e in range(num_experts):
                layer.experts[e].gate_up.load_fp_weights(gate_up_all[e])
                layer.experts[e].down.load_fp_weights(down_all[e])
        return layer

    def forward(self, hidden_states):
        """hidden_states: (N, H) or (B, T, H). Returns same shape."""
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, self.hidden_size)

        router_logits = F.linear(hidden_states, self.gate_weight)
        router_logits = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        top_values, top_index = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        top_values = top_values.to(hidden_states.dtype)

        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            y_e = self.experts[e](current_state)
            weighted = y_e * top_values[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, weighted.to(final.dtype))

        return final.reshape(orig_shape)

    def trainable_parameters(self):
        params = []
        for e in range(self.num_experts):
            params.extend(self.experts[e].trainable_parameters())
        return params

    def reset_step_counters(self):
        for e in range(self.num_experts):
            self.experts[e].gate_up._step_counter = 0
            self.experts[e].down._step_counter = 0

    def prime_cache(self):
        """Populate all expert _W_q_cache tensors from current W_latent values."""
        for e in range(self.num_experts):
            self.experts[e].prime_cache()
            # Viterbi allocates ~1 GB transient per matrix; return it to the
            # allocator between experts to keep headroom for the next one.
            torch.cuda.empty_cache()

    def eval_forward(self, hidden_states):
        """Same dispatch as forward() but uses each QL's cached W_q (no Viterbi)."""
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, self.hidden_size)

        router_logits = F.linear(hidden_states, self.gate_weight)
        router_logits = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        top_values, top_index = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        top_values = top_values.to(hidden_states.dtype)

        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(
            expert_mask.sum(dim=(-1, -2)), 0
        ).nonzero()

        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            y_e = self.experts[e].eval_forward(current_state)
            weighted = y_e * top_values[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, weighted.to(final.dtype))

        return final.reshape(orig_shape)
