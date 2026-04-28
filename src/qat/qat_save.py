"""Save/load QAT-trained W_q tensors per QuantTarget.

Format: one .pt file per QuantTarget under cache/qat/L{NN}/, matching the
layout used by src.eval.install_quantized for PTQ artifacts:
    attn_{q,k,v,o}_proj.pt
    expert_{EE}_{gate,up,down}_proj.pt

Each file stores {"W_q": bf16 tensor, "shape": tuple, "meta": dict}.

Stored in bf16 to keep disk footprint bounded (16 layers × ~400 MB ≈ 6.5 GB).
The fp32 W_q can be re-derived from bf16 with ~1e-3 rel error, which is the
same noise floor as the bf16 eval model — so no loss at install time.

Basis: QuantizedLinear holds W_q in ROTATED basis (full-IP). The saved
artifact here is un-rotated back to the original basis so that the install
path (src.eval.install_quantized) can treat it as a plain nn.Linear weight,
unchanged from pre-IP.
"""
import os
import torch

from src.qat.ste import apply_inverse_rht_gpu


def _attn_path(qat_dir, layer_idx, proj):
    return os.path.join(qat_dir, f"L{layer_idx:02d}", f"attn_{proj}.pt")


def _expert_path(qat_dir, layer_idx, expert_idx, proj):
    return os.path.join(
        qat_dir, f"L{layer_idx:02d}",
        f"expert_{expert_idx:02d}_{proj}.pt",
    )


def _save_tensor(path, W_q, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "W_q": W_q.detach().to(dtype=torch.bfloat16).cpu().contiguous(),
        "shape": tuple(W_q.shape),
        "meta": meta,
    }
    torch.save(payload, path)


def save_qat_layer(qat_layer, layer_idx, qat_dir, intermediate_size):
    """Save all W_q tensors for one trained QATDecoderLayer.

    Fused gate_up of shape (2I, H) is split into:
      - gate_proj = rows [0:I]
      - up_proj   = rows [I:2I]
    to match the existing per-target layout used by install_quantized_weights.

    Returns a dict with per-file paths + byte counts.
    """
    I = int(intermediate_size)
    attn = qat_layer.layer.self_attn
    moe = qat_layer.layer.mlp

    def _unrotate(ql):
        """Return ql._W_q_cache in original basis. In IP mode the cache is
        rotated and we apply inverse RHT; in pre-IP mode the cache is already
        in original basis and we return it as-is."""
        with torch.no_grad():
            W_q = ql._W_q_cache.to(torch.float32)
            if getattr(ql, "ip_mode", False):
                return apply_inverse_rht_gpu(W_q, ql.sign_l, ql.sign_r)
            return W_q

    written = []
    # --- attention projections ---
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        ql = getattr(attn, proj)
        W_q = _unrotate(ql)
        path = _attn_path(qat_dir, layer_idx, proj)
        _save_tensor(path, W_q, {"layer": layer_idx, "kind": "attention", "proj": proj})
        written.append(path)

    # --- experts: fused gate_up split into gate + up; plus down ---
    for e in range(moe.num_experts):
        gu = _unrotate(moe.experts[e].gate_up)  # (2I, H) in original basis
        assert gu.shape[0] == 2 * I, (
            f"L{layer_idx:02d} E{e}: gate_up shape {tuple(gu.shape)} "
            f"doesn't match 2*I={2*I}"
        )
        gate = gu[:I].contiguous()       # (I, H)
        up   = gu[I:].contiguous()       # (I, H)
        down = _unrotate(moe.experts[e].down).contiguous()  # (H, I)

        for proj_name, W_q in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            path = _expert_path(qat_dir, layer_idx, e, proj_name)
            _save_tensor(path, W_q, {
                "layer": layer_idx, "kind": "expert",
                "expert": e, "proj": proj_name,
            })
            written.append(path)

    total_bytes = sum(os.path.getsize(p) for p in written)
    return {
        "layer_idx": layer_idx,
        "n_files": len(written),
        "total_bytes": total_bytes,
    }


def load_qat_tensor(path):
    """Load a saved W_q file. Returns {W_q: bf16 Tensor, shape, meta}."""
    return torch.load(path, weights_only=False)
