"""QATDecoderLayer: full OLMoE decoder layer with QAT on attention + experts.

Wraps an OlmoeDecoderLayer by:
  - replacing self_attn.{q,k,v,o}_proj with QuantizedLinear
  - replacing self.mlp with QATMoELayer
  - freezing everything else (layernorms, q_norm, k_norm, router)

Forward delegates to the underlying OlmoeDecoderLayer.forward, so the attention
path (rotary, SDPA, q/k norm, clip_qkv) matches upstream exactly.
"""
import copy

import torch
import torch.nn as nn

from src.qat.quantized_linear import QuantizedLinear
from src.qat.qat_moe_layer import QATMoELayer


ATTN_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


class QATDecoderLayer(nn.Module):
    def __init__(self, layer, codebook_gpu, config,
                 seed=0, reencode_every_n_steps=1,
                 device="cuda", dtype=torch.float32):
        """`layer` is an OlmoeDecoderLayer already on `device`/`dtype`. We wrap
        it in place: swap attn projections for QuantizedLinear, swap mlp for
        QATMoELayer, freeze everything else.
        """
        super().__init__()
        self.layer = layer

        # --- attention: replace q/k/v/o with QuantizedLinear ---
        for i, name in enumerate(ATTN_PROJ_NAMES):
            fp_lin = getattr(layer.self_attn, name)
            qlin = QuantizedLinear.from_linear(
                fp_lin, codebook_gpu=codebook_gpu,
                seed=seed + 1_000_000 + i * 17,
                reencode_every_n_steps=reencode_every_n_steps,
            )
            setattr(layer.self_attn, name, qlin)

        # --- mlp: replace with QATMoELayer built from the FP block ---
        fp_mlp = layer.mlp
        qmoe = QATMoELayer.from_mlp_block(
            mlp_block=fp_mlp, config=config, codebook_gpu=codebook_gpu,
            seed=seed, reencode_every_n_steps=reencode_every_n_steps,
            device=device, dtype=dtype,
        )
        layer.mlp = qmoe

        # --- freeze non-QAT params: layernorms, q_norm, k_norm ---
        # QuantizedLinear.W_latent and QATExpert weights are the only trainables.
        layer.input_layernorm.weight.requires_grad_(False)
        layer.post_attention_layernorm.weight.requires_grad_(False)
        layer.self_attn.q_norm.weight.requires_grad_(False)
        layer.self_attn.k_norm.weight.requires_grad_(False)
        # bias tensors on attn projs, if any, were copied from FP and should stay
        # learnable unless caller freezes — but OLMoE has attention_bias=False.

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, position_ids=None, **kwargs):
        """Forward the wrapped OlmoeDecoderLayer. Accepts and passes through
        any extra kwargs (past_key_values, cache_position, output_attentions,
        ...) so this wrapper works both standalone (per-layer driver) and
        in-model (end-to-end Path Y, where OlmoeModel.forward passes the
        full HF layer kwarg surface)."""
        kwargs.setdefault("use_cache", False)
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        return self.layer(hidden_states, **kwargs)

    # ----- QAT control surface -----

    def _attn_qls(self):
        return [getattr(self.layer.self_attn, n) for n in ATTN_PROJ_NAMES]

    def prime_cache(self):
        """Populate all W_q caches (attn + experts). Frees Viterbi transient
        between matrices to avoid allocator fragmentation."""
        for ql in self._attn_qls():
            ql.prime_cache()
            torch.cuda.empty_cache()
        self.layer.mlp.prime_cache()

    def reset_step_counters(self):
        for ql in self._attn_qls():
            ql._step_counter = 0
        self.layer.mlp.reset_step_counters()

    def trainable_parameters(self):
        params = []
        for ql in self._attn_qls():
            params.append(ql.W_latent)
        params.extend(self.layer.mlp.trainable_parameters())
        return params

    def _all_qls(self):
        """Yield every QuantizedLinear with a unique name: 4 attn + 2·num_experts
        expert QLs. Names are used as dict keys for snapshot/restore — must be
        distinct or snapshots collide (previous bug: all 4 attn projs shared
        the key "attn", so only o_proj's weights survived)."""
        for proj in ATTN_PROJ_NAMES:
            ql = getattr(self.layer.self_attn, proj)
            yield (f"attn_{proj}", ql)
        for e in range(self.layer.mlp.num_experts):
            ex = self.layer.mlp.experts[e]
            yield (f"e{e:02d}_gate_up", ex.gate_up)
            yield (f"e{e:02d}_down", ex.down)

    def proximal_loss(self):
        """Sum of squared distances between W_latent and current codewords.

        Keeps W_latent from drifting far from the current _W_q_cache between
        reprimes — the drift is what lets reprime overshoot onto a worse
        codeword. Use sum (not mean) so per-parameter gradient is
        2·(W_latent - W_q); balance against MSE with prox_lambda ≈ 0.01."""
        total = 0.0
        for _, ql in self._all_qls():
            total = total + (ql.W_latent - ql._W_q_cache.detach()).pow(2).sum()
        return total

    def snapshot_W_q_cpu(self):
        """Return a CPU fp32 snapshot of all _W_q_cache tensors (~1.6GB).

        Must be fp32, not bf16: round-tripping through bf16 introduces ~1e-3
        relative error per element, which is enough to wreck MSE at 1e-6
        scale (observed val ballooning from 1.3e-6 to 2e-5 when bf16 was
        used). Disk save still writes bf16 — that's a separate path."""
        snap = {}
        for name, ql in self._all_qls():
            snap[name] = ql._W_q_cache.detach().to(
                device="cpu", copy=True
            )
        return snap

    def restore_W_q_from_cpu(self, snap):
        """In-place restore _W_q_cache tensors from a CPU fp32 snapshot."""
        for name, ql in self._all_qls():
            ql._W_q_cache.copy_(snap[name].to(
                dtype=ql._W_q_cache.dtype, device=ql._W_q_cache.device
            ))
