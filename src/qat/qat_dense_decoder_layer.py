"""QATDenseDecoderLayer: wraps a dense Transformer decoder layer with QAT.

Supports Llama-style decoder layers (also Mistral, Qwen2, Phi-3 have the
same structure). Each layer has 7 QuantizedLinear:
  attention: q_proj, k_proj, v_proj, o_proj (4)
  mlp:       gate_proj, up_proj, down_proj  (3)

Contrast with QATDecoderLayer (src/qat/qat_decoder_layer.py) which handles
OLMoE's MoE MLP (4 attn + num_experts*2 expert QLs — up to 132 per layer).
For a 16-layer Llama-3.2-1B, this gives 112 QLs total vs 2112 for OLMoE.
"""
import torch
import torch.nn as nn

from src.qat.quantized_linear import QuantizedLinear


ATTN_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


class QATDenseDecoderLayer(nn.Module):
    """Wrap a dense (non-MoE) decoder layer with QAT-quantized projections.

    `layer` is assumed to have:
      layer.self_attn.{q,k,v,o}_proj  (nn.Linear)
      layer.mlp.{gate,up,down}_proj   (nn.Linear)
      layer.input_layernorm, layer.post_attention_layernorm  (RMSNorm)

    All 7 Linears are replaced in place with QuantizedLinear. Everything
    else (layernorms, biases if any) is frozen.
    """
    def __init__(self, layer, codebook_gpu, config,
                 seed=0, reencode_every_n_steps=1,
                 device="cuda", dtype=torch.float32):
        super().__init__()
        self.layer = layer

        for i, name in enumerate(ATTN_PROJ_NAMES):
            fp_lin = getattr(layer.self_attn, name)
            qlin = QuantizedLinear.from_linear(
                fp_lin, codebook_gpu=codebook_gpu,
                seed=seed + 1_000_000 + i * 17,
                reencode_every_n_steps=reencode_every_n_steps,
            )
            setattr(layer.self_attn, name, qlin)

        for i, name in enumerate(MLP_PROJ_NAMES):
            fp_lin = getattr(layer.mlp, name)
            qlin = QuantizedLinear.from_linear(
                fp_lin, codebook_gpu=codebook_gpu,
                seed=seed + 2_000_000 + i * 31,
                reencode_every_n_steps=reencode_every_n_steps,
            )
            setattr(layer.mlp, name, qlin)

        # Freeze layernorms and any leftover non-QAT params.
        if hasattr(layer, "input_layernorm") and hasattr(layer.input_layernorm, "weight"):
            layer.input_layernorm.weight.requires_grad_(False)
        if hasattr(layer, "post_attention_layernorm") and hasattr(layer.post_attention_layernorm, "weight"):
            layer.post_attention_layernorm.weight.requires_grad_(False)

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, position_ids=None, **kwargs):
        """Pass through kwargs so both standalone and in-model (HF) calls work.

        Casts hidden_states up to match our layer's dtype (fp32 when W_latent
        is fp32 and other layers are bf16/fp16). Casts outputs back to the
        original dtype so downstream bf16/fp16 layers in the same model work.
        """
        kwargs.setdefault("use_cache", False)
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

        input_dtype = hidden_states.dtype
        # Match our layer's fp32 (if different). Uses layernorm weight as proxy
        # for the layer's compute dtype.
        target_dtype = self.layer.input_layernorm.weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)

        out = self.layer(hidden_states, **kwargs)

        # Cast back to the original input dtype for downstream layers
        if isinstance(out, tuple):
            out = tuple(
                o.to(input_dtype) if torch.is_tensor(o) and o.dtype != input_dtype else o
                for o in out
            )
        elif torch.is_tensor(out) and out.dtype != input_dtype:
            out = out.to(input_dtype)
        return out

    # ---------- QAT control surface ----------

    def _all_qls(self):
        """Yield (name, QuantizedLinear) for every QAT-quantized projection."""
        for name in ATTN_PROJ_NAMES:
            yield (f"attn_{name}", getattr(self.layer.self_attn, name))
        for name in MLP_PROJ_NAMES:
            yield (f"mlp_{name}", getattr(self.layer.mlp, name))

    def trainable_parameters(self):
        params = []
        for _, ql in self._all_qls():
            params.append(ql.W_latent)
            # Scalar-faithful arm also trains a per-group clip (empty otherwise).
            if hasattr(ql, "scalar_extra_parameters"):
                params.extend(ql.scalar_extra_parameters())
        return params

    def prime_cache(self):
        """Populate all W_q caches."""
        for _, ql in self._all_qls():
            ql.prime_cache()
            torch.cuda.empty_cache()

    def reset_step_counters(self):
        for _, ql in self._all_qls():
            ql._step_counter = 0

    def snapshot_W_q_cpu(self):
        """Return fp32 CPU snapshot of all _W_q_cache tensors."""
        snap = {}
        for name, ql in self._all_qls():
            snap[name] = ql._W_q_cache.detach().to(device="cpu", copy=True)
        return snap

    def restore_W_q_from_cpu(self, snap):
        for name, ql in self._all_qls():
            ql._W_q_cache.copy_(snap[name].to(
                dtype=ql._W_q_cache.dtype, device=ql._W_q_cache.device
            ))
