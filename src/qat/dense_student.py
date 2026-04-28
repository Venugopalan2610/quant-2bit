"""Full-model BCJR student for dense (Llama-family) LMs.

Wraps every decoder layer of a dense LM with QATDenseDecoderLayer. No v2
snapshot dependency — W_latent initializes directly from the FP weights
via QuantizedLinear.from_linear (which calls load_fp_weights internally).

Use with Llama-3.2-1B, Mistral-7B, Qwen2.5-1.5B, Phi-3, etc.
"""
import gc

import torch

from src.qat.qat_dense_decoder_layer import QATDenseDecoderLayer
from src.bcjr.anneal import (
    convert_layer_to_bcjr,
    convert_layer_to_ste,
    set_layer_temperature,
)


def build_bcjr_student_dense(model, codebook_gpu,
                             seed=0, reencode_every_n_steps=1,
                             bcjr_chunk=16, verbose=True):
    """Wrap every decoder layer of `model` with QATDenseDecoderLayer.

    Args:
        model: LlamaForCausalLM / MistralForCausalLM / etc., on GPU, fp32
            trainable or bf16. The decoder layers are assumed to expose
            .self_attn.{q,k,v,o}_proj and .mlp.{gate,up,down}_proj.
        codebook_gpu: shared (65536, 2) fp32 codebook on CUDA.
        seed: base seed for per-QL sign vectors.
        reencode_every_n_steps: 1 = BCJR every step; >1 caches W_q and uses
            STE grad on intervening steps. Default 1 (full BCJR each step).
        bcjr_chunk: BCJR tile chunk size. 16 is safe on 12 GB; bump to
            32-128 on larger cards for better GPU saturation.

    Returns:
        model, mutated in place.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers

    for layer_idx in range(num_layers):
        if verbose:
            print(f"  [dense-student] wrapping layer {layer_idx}/{num_layers}",
                  flush=True)
        fp_layer = model.model.layers[layer_idx]
        qat_layer = QATDenseDecoderLayer(
            fp_layer, codebook_gpu=codebook_gpu, config=cfg,
            seed=seed + layer_idx * 997,
            reencode_every_n_steps=reencode_every_n_steps,
        )
        model.model.layers[layer_idx] = qat_layer
        gc.collect()
        torch.cuda.empty_cache()

    return model


def flip_all_to_bcjr(model, T_init, bcjr_chunk=16):
    """Flip every QAT dense layer to BCJR mode at temperature T_init."""
    for layer in model.model.layers:
        if isinstance(layer, QATDenseDecoderLayer):
            convert_layer_to_bcjr(layer, T_init=T_init, bcjr_chunk=bcjr_chunk)


def flip_all_to_ste(model):
    for layer in model.model.layers:
        if isinstance(layer, QATDenseDecoderLayer):
            convert_layer_to_ste(layer)


def set_global_temperature(model, T):
    for layer in model.model.layers:
        if isinstance(layer, QATDenseDecoderLayer):
            set_layer_temperature(layer, T)


def prime_all(model):
    for layer in model.model.layers:
        if isinstance(layer, QATDenseDecoderLayer):
            layer.prime_cache()


def student_trainable_parameters(model):
    for layer in model.model.layers:
        if isinstance(layer, QATDenseDecoderLayer):
            for p in layer.trainable_parameters():
                yield p


def freeze_non_qat(model):
    """Freeze everything except QAT W_latent params."""
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    for layer in model.model.layers:
        if isinstance(layer, QATDenseDecoderLayer):
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
