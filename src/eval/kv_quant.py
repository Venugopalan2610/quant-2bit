"""KV-cache fake-quantization for the composition-quality eval.

Default = REAL TurboQuant (the user-cloned repo at ./turboquant). We use its
tensor-level `TurboQuantMSE.forward` (random rotation + Lloyd-Max codebook,
quantize→dequantize) — pure torch, no Triton; the Triton/vLLM path is only for
the serving/throughput run. A transparent per-token uniform quantizer is kept
as an explicit, labeled fallback (mode="uniform") for sanity/ablation.

"Fake-quant" = quantize then dequantize in-place, so attention runs in fp; this
measures the QUALITY cost of the KV bitrate, not throughput. Default allocation
matches TurboQuant: 3-bit keys, 2-bit values.
"""
import os
import sys

import torch

from src.qat.scalar_quant import _symmetric_levels, _quantize_levels


# ── REAL TurboQuant import (repo cloned by the user) ─────────────────────────

def _import_turboquant(repo_dir=None):
    """Import the real TurboQuant package from the cloned repo. Returns the
    TurboQuantMSE class. repo_dir defaults to ./turboquant (the repo root that
    contains the `turboquant` package)."""
    repo_dir = repo_dir or os.environ.get("TURBOQUANT_DIR", "turboquant")
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from turboquant import TurboQuantMSE  # noqa: E402
    return TurboQuantMSE


# ── uniform stand-in (fallback only) ─────────────────────────────────────────

def quantize_kv_tensor(x, n_bits):
    """Per-token symmetric uniform N-bit fake-quant along the head dim. A
    transparent fallback — NOT TurboQuant. Used only when mode='uniform'."""
    if n_bits is None or n_bits >= 16:
        return x
    dt = x.dtype
    xf = x.float()
    max_level = float(_symmetric_levels(n_bits)[-1])
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max_level
    return (_quantize_levels(xf / scale, n_bits, ste=False) * scale).to(dt)


# ── attention hooks ──────────────────────────────────────────────────────────

class KVQuantHooks:
    """Fake-quantize K and V at each layer's k_proj/v_proj OUTPUT (a version-
    robust approximation of cache quantization). The projection output is
    (..., n_kv_heads*head_dim); we reshape to per-head vectors of length
    head_dim and quantize each — the granularity TurboQuant operates at.

    mode="turboquant" (default): real TurboQuantMSE at head_dim (codebook_d64
    exists for Llama-1B's head_dim=64). mode="uniform": the stand-in above.

        h = KVQuantHooks(model, key_bits=3, val_bits=2); h.add(); ...; h.remove()
    """
    def __init__(self, model, key_bits=3, val_bits=2, mode="turboquant",
                 head_dim=None, repo_dir=None):
        self.model = model
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.mode = mode
        self.handles = []
        cfg = model.config
        self.head_dim = head_dim or getattr(cfg, "head_dim", None) \
            or cfg.hidden_size // cfg.num_attention_heads
        dev = next(model.parameters()).device
        if mode == "turboquant":
            TQ = _import_turboquant(repo_dir)
            # one quantizer per bitwidth, reused across layers (rotation+codebook
            # are deterministic given dim/bits/seed).
            self._qk = TQ(dim=self.head_dim, bits=key_bits, device=dev, dtype=torch.float32)
            self._qv = TQ(dim=self.head_dim, bits=val_bits, device=dev, dtype=torch.float32)
        elif mode != "uniform":
            raise ValueError(f"bad mode {mode!r}")

    def _quant(self, out, which):
        hd = self.head_dim
        shp = out.shape
        v = out.reshape(*shp[:-1], shp[-1] // hd, hd)        # (..., n_heads, head_dim)
        if self.mode == "turboquant":
            q = (self._qk if which == "k" else self._qv).forward(v.float())
            v = q.to(out.dtype)
        else:
            v = quantize_kv_tensor(v, self.key_bits if which == "k" else self.val_bits)
        return v.reshape(shp)

    def _mk(self, which):
        def hook(_m, _i, out):
            return self._quant(out, which)
        return hook

    def add(self):
        n = 0
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "k_proj"):
                self.handles.append(attn.k_proj.register_forward_hook(self._mk("k"))); n += 1
            if hasattr(attn, "v_proj"):
                self.handles.append(attn.v_proj.register_forward_hook(self._mk("v"))); n += 1
        return n

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
