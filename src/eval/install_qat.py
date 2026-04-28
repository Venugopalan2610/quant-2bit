"""Install QAT-trained W_q tensors into an OLMoE model.

Parallel to src.eval.install_quantized but loads plain bf16 tensors (saved by
src.qat.qat_save.save_qat_layer) instead of dequantizing from QTIP bitstreams.

Uses the same QuantTarget abstraction so the on-disk layout matches the PTQ
pipeline's: cache/qat/L{NN}/{attn_{q,k,v,o}_proj,expert_{EE}_{gate,up,down}_proj}.pt
"""
import os
import time
import torch

from src.models.olmoe_adapter import enumerate_quant_targets
from src.qat.qat_save import load_qat_tensor


def _qat_path_for_target(target, qat_dir):
    L = target.layer_idx
    if target.kind == "attention":
        fname = f"attn_{target.proj}.pt"
    else:
        fname = f"expert_{target.expert_idx:02d}_{target.proj}.pt"
    return os.path.join(qat_dir, f"L{L:02d}", fname)


def install_qat_weights(model, qat_dir, verbose=True, allow_partial=False):
    """Copy saved QAT W_q tensors into the model in-place.

    Args:
        model: an OlmoeForCausalLM (typically loaded fresh in bf16)
        qat_dir: directory produced by src.qat.qat_all
        verbose: progress prints
        allow_partial: if True, untrained layers stay FP; if False, error out

    Returns stats dict.
    """
    targets = list(enumerate_quant_targets(model))
    if verbose:
        print(f"Installing QAT weights from {qat_dir} ({len(targets)} targets)")

    n_installed = 0
    n_missing = 0
    missing_layers = set()
    t0 = time.time()
    last_print = t0

    for target in targets:
        path = _qat_path_for_target(target, qat_dir)
        if not os.path.exists(path):
            n_missing += 1
            missing_layers.add(target.layer_idx)
            continue
        saved = load_qat_tensor(path)
        target.set_weight(saved["W_q"])  # set_weight handles dtype/device cast
        n_installed += 1
        if verbose and time.time() - last_print > 5.0:
            elapsed = time.time() - t0
            rate = n_installed / elapsed
            eta = (len(targets) - n_installed) / max(rate, 1e-6)
            print(f"  {n_installed}/{len(targets)} installed "
                  f"({rate:.0f}/s, ETA {eta:.0f}s)")
            last_print = time.time()

    elapsed = time.time() - t0
    if verbose:
        print(f"  Done: {n_installed}/{len(targets)} installed in {elapsed:.0f}s")
        if n_missing:
            print(f"  {n_missing} missing (layers: {sorted(missing_layers)})")

    if n_missing and not allow_partial:
        raise RuntimeError(
            f"{n_missing} QAT targets missing under {qat_dir}. "
            f"Pass allow_partial=True to keep untrained layers at FP."
        )

    return {
        "n_installed": n_installed,
        "n_missing": n_missing,
        "n_targets": len(targets),
        "missing_layers": sorted(missing_layers),
        "elapsed_seconds": elapsed,
    }
