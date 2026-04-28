"""Install BCJR-trained W_q tensors into an OLMoE model.

Phase-5 BCJR QAT saves one dict per layer under cache/qat_bcjr_full/:
    layer_NN_wq.pt  ->  {
        "attn_q_proj": (out, in)        fp32 CPU
        "attn_k_proj": ...
        "attn_v_proj": ...
        "attn_o_proj": ...
        "eEE_gate_up": (2*I, H)         fp32 CPU, gate=[0:I], up=[I:2I]
        "eEE_down":    (H, I)           fp32 CPU
        ...  for EE in 0..num_experts-1
    }

That keying matches QATDecoderLayer._all_qls(). This installer bridges to
QuantTarget (which addresses gate_proj / up_proj as separate I×H matrices).
"""
import os
import time
import torch

from src.models.olmoe_adapter import enumerate_quant_targets


def _snap_tensor_for_target(target, snap, I):
    """Pull the fp32 tensor from the per-layer snap dict that corresponds to
    this QuantTarget, slicing the fused gate_up tensor when needed.
    Returns None if the snap has no matching entry."""
    if target.kind == "attention":
        return snap.get(f"attn_{target.proj}")

    ek = f"e{target.expert_idx:02d}"
    if target.proj == "down_proj":
        return snap.get(f"{ek}_down")
    fused = snap.get(f"{ek}_gate_up")
    if fused is None:
        return None
    if target.proj == "gate_proj":
        return fused[:I, :]
    if target.proj == "up_proj":
        return fused[I:2 * I, :]
    raise KeyError(target.proj)


def install_bcjr_weights(model, bcjr_dir, verbose=True, allow_partial=False):
    """Copy saved BCJR W_q tensors into the model in-place.

    Args:
        model: OlmoeForCausalLM (freshly loaded, any dtype/device)
        bcjr_dir: directory with layer_NN_wq.pt files
        verbose: progress prints
        allow_partial: if True, layers without a snapshot stay FP

    Returns stats dict.
    """
    cfg = model.config
    I = cfg.intermediate_size

    targets = list(enumerate_quant_targets(model))
    by_layer = {}
    for t in targets:
        by_layer.setdefault(t.layer_idx, []).append(t)

    if verbose:
        print(f"Installing BCJR weights from {bcjr_dir} "
              f"({len(targets)} targets across {cfg.num_hidden_layers} layers)")

    n_installed = 0
    n_missing = 0
    missing_layers = []
    t0 = time.time()
    last_print = t0

    for layer_idx in range(cfg.num_hidden_layers):
        snap_path = os.path.join(bcjr_dir, f"layer_{layer_idx:02d}_wq.pt")
        if not os.path.exists(snap_path):
            missing_layers.append(layer_idx)
            n_missing += len(by_layer.get(layer_idx, []))
            continue
        snap = torch.load(snap_path, map_location="cpu", weights_only=True)

        for target in by_layer[layer_idx]:
            w = _snap_tensor_for_target(target, snap, I)
            if w is None:
                n_missing += 1
                continue
            target.set_weight(w)
            n_installed += 1

        del snap
        if verbose and time.time() - last_print > 5.0:
            elapsed = time.time() - t0
            rate = n_installed / max(elapsed, 1e-6)
            eta = (len(targets) - n_installed) / max(rate, 1e-6)
            print(f"  layer {layer_idx:02d}: {n_installed}/{len(targets)} installed "
                  f"({rate:.0f}/s, ETA {eta:.0f}s)")
            last_print = time.time()

    elapsed = time.time() - t0
    if verbose:
        print(f"  Done: {n_installed}/{len(targets)} installed in {elapsed:.0f}s")
        if missing_layers:
            print(f"  Missing layer snapshots: {missing_layers}")

    if n_missing and not allow_partial:
        raise RuntimeError(
            f"{n_missing} BCJR targets missing under {bcjr_dir} "
            f"(layers {missing_layers}). Pass allow_partial=True to keep "
            f"untrained layers at FP."
        )

    return {
        "n_installed": n_installed,
        "n_missing": n_missing,
        "n_targets": len(targets),
        "missing_layers": missing_layers,
        "elapsed_seconds": elapsed,
    }
