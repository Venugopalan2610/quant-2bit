"""Full-model BCJR student for end-to-end KL distillation (Path Y).

Wraps an OlmoeForCausalLM so every decoder layer is a QATDecoderLayer in
BCJR mode. Initializes each layer's W_latent from a v2 BCJR snapshot
(cache/qat_bcjr_full_v2/layer_NN_wq.pt) so training starts from the
Block-AP-quality codes rather than a random place.

Contrast with qat_one_layer (per-layer greedy): here we hold all 16 layers
in memory and backprop end-to-end. Designed for H200-class VRAM
(~140 GB). Will NOT fit on a 4080 — do per-layer smoke tests separately.
"""
import os
import gc

import torch
import torch.nn as nn

from src.qat.qat_decoder_layer import QATDecoderLayer
from src.bcjr.anneal import (
    measure_ptq_residual,
    convert_layer_to_bcjr,
    convert_layer_to_ste,
    set_layer_temperature,
)


def _load_v2_snapshot(snap_dir, layer_idx):
    path = os.path.join(snap_dir, f"layer_{layer_idx:02d}_wq.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"v2 snapshot missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _install_snapshot_into_qat_layer(qat_layer, snap, intermediate_size):
    """Copy v2 snapshot tensors into each QuantizedLinear.W_latent (fp32)."""
    I = intermediate_size
    with torch.no_grad():
        # Attention projs
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"attn_{proj}"
            if key not in snap:
                raise KeyError(f"v2 snapshot missing {key}")
            ql = getattr(qat_layer.layer.self_attn, proj)
            ql.load_fp_weights(snap[key])
        # Experts
        num_experts = qat_layer.layer.mlp.num_experts
        for e in range(num_experts):
            fused = snap.get(f"e{e:02d}_gate_up")
            down = snap.get(f"e{e:02d}_down")
            if fused is None or down is None:
                raise KeyError(f"v2 snapshot missing expert {e}")
            expert = qat_layer.layer.mlp.experts[e]
            expert.gate_up.load_fp_weights(fused)
            expert.down.load_fp_weights(down)


def build_bcjr_student(model, codebook_gpu, v2_snap_dir,
                       seed=0, reencode_every_n_steps=1,
                       bcjr_chunk=16, measure_T0=False, prime=False,
                       verbose=True):
    """Wrap every decoder layer of `model` with QATDecoderLayer in BCJR mode.

    Args:
        model: OlmoeForCausalLM on GPU in fp32 (trainable) or bf16.
        codebook_gpu: shared (65536, 2) fp32 codebook on CUDA.
        v2_snap_dir: path to cache/qat_bcjr_full_v2/
        seed: random seed for sign vectors.
        reencode_every_n_steps: 1 = run BCJR fresh every step (what Path Y
            wants); large values cache the first step's W_q. Default 1.
        bcjr_chunk: BCJR tile chunk size; increase on H200 (16-32) for speed.
        measure_T0: if True, measure per-layer PTQ residual to derive T_0.
            Expensive (runs Viterbi on every QL). Default False — Path Y
            uses a globally chosen T_0 instead.
        prime: if True, run prime_cache() per layer after wrapping. Needed
            only for STE training; BCJR forward recomputes W_q each step
            and doesn't consume the cache. Default False.

    Returns:
        model: same model object, mutated in place. model.model.layers[i]
            are now QATDecoderLayer instances.
        T_per_layer: list of per-layer T_0 if measure_T0=True, else empty.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    T_per_layer = []

    for layer_idx in range(num_layers):
        if verbose:
            print(f"  [student] wrapping layer {layer_idx}/{num_layers}",
                  flush=True)
        fp_layer = model.model.layers[layer_idx]
        qat_layer = QATDecoderLayer(
            fp_layer, codebook_gpu=codebook_gpu, config=cfg,
            seed=seed + layer_idx * 997,
            reencode_every_n_steps=reencode_every_n_steps,
        )
        # Install v2 codes as the starting latent
        snap = _load_v2_snapshot(v2_snap_dir, layer_idx)
        _install_snapshot_into_qat_layer(qat_layer, snap,
                                         intermediate_size=cfg.intermediate_size)
        del snap
        # Swap the module in model.model.layers so the model's forward goes
        # through QATDecoderLayer rather than the original OlmoeDecoderLayer.
        model.model.layers[layer_idx] = qat_layer

        if measure_T0:
            T_0 = measure_ptq_residual(qat_layer)
            T_per_layer.append(T_0)

        if prime:
            qat_layer.prime_cache()

        gc.collect()
        torch.cuda.empty_cache()

    return model, T_per_layer


def flip_all_to_bcjr(model, T_init, bcjr_chunk=16):
    """Flip every QAT decoder layer to BCJR mode at temperature T_init."""
    for layer in model.model.layers:
        if isinstance(layer, QATDecoderLayer):
            convert_layer_to_bcjr(layer, T_init=T_init, bcjr_chunk=bcjr_chunk)


def flip_all_to_ste(model):
    """Flip every QAT decoder layer to STE (hard Viterbi) mode. Call before
    final prime_cache + snapshot save."""
    for layer in model.model.layers:
        if isinstance(layer, QATDecoderLayer):
            convert_layer_to_ste(layer)


def set_global_temperature(model, T):
    """Broadcast T to every QL in every QAT decoder layer."""
    for layer in model.model.layers:
        if isinstance(layer, QATDecoderLayer):
            set_layer_temperature(layer, T)


def prime_all(model):
    """Refresh _W_q_cache in every QAT decoder layer."""
    for layer in model.model.layers:
        if isinstance(layer, QATDecoderLayer):
            layer.prime_cache()


def student_trainable_parameters(model):
    """Yield every W_latent nn.Parameter across all wrapped decoder layers.
    Use this (NOT model.parameters()) so we only attach the optimizer to
    trainable latents and avoid accidentally training the router, lm_head,
    or embeddings."""
    for layer in model.model.layers:
        if isinstance(layer, QATDecoderLayer):
            for p in layer.trainable_parameters():
                yield p


def freeze_non_qat(model):
    """Freeze embeddings, lm_head, final norm, and anything that's not a
    QuantizedLinear.W_latent or QATExpert weight."""
    # Embeddings + lm_head
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    # Re-enable only W_latent / expert latents via the QAT layer helper
    for layer in model.model.layers:
        if isinstance(layer, QATDecoderLayer):
            for p in layer.trainable_parameters():
                p.requires_grad_(True)


def count_trainable(model):
    n_trainable = 0
    n_total = 0
    for p in model.parameters():
        n_total += p.numel()
        if p.requires_grad:
            n_trainable += p.numel()
    return n_trainable, n_total
