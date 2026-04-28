"""QuantizedLinear: nn.Module drop-in for nn.Linear with full-IP trellis QAT.

Full incoherence processing (IP): weights are stored in rotated (Hadamard)
basis, and forward pass rotates input activations / un-rotates output so the
matmul happens in rotated space. This makes activation distributions seen by
the matmul near-Gaussian (kurtosis ≈ 3), removing per-layer LR sensitivity.

Holds:
    - W_latent: (out_features, in_features) fp32 nn.Parameter in ROTATED basis
    - sign_l, sign_r: frozen ±1 buffers. Used both for (a) initial W rotation
      at from_linear() and (b) per-forward activation rotation. Same vectors
      on both sides so the math cancels.
    - codebook: frozen (65536, 2) fp32 buffer on CUDA

Forward (W_latent, W_q_cache, matmul all in rotated basis):
    W_q̃ = TrellisQuantSTE_IP(W_latent, codebook)            # Viterbi only
    x̃   = fht(x * sign_r) / sqrt(in_features)               # rotate input
    ỹ   = F.linear(x̃, W_q̃)                                 # rotated matmul
    y   = (fht(ỹ) / sqrt(out_features)) * sign_l + bias     # un-rotate output

Un-rotation order: FHT first, THEN sign multiply (R_out^T = D · H̃ means the
inverse R_out = H̃ · D composes FHT first then sign).

Re-encode schedule knob (reencode_every_n_steps): same as before — cached W_q̃
persists across steps; fresh Viterbi every N-th step.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rht.transform import make_sign_vector
from src.qat.ste import (
    TrellisQuantSTE,
    TrellisQuantSTE_IP,
    apply_rht_gpu,
    make_hyb_codebook_gpu,
)
from src.bcjr.bcjr_quant import bcjr_quantize_weight, BCJR_CHUNK
from src.bcjr.forward_backward import build_pred_succ_tables

try:
    from fast_hadamard_transform import hadamard_transform as _fht
    _HAS_FHT = True
except ImportError:
    _HAS_FHT = False


# Module-level cache for BCJR trellis tables. These are deterministic and
# identical across every QuantizedLinear — sharing them saves ~8 MB × 2
# × num_QLs. For Llama-3.2-1B (112 QLs) that's ~1.8 GB of VRAM recovered.
_BCJR_TABLE_CACHE = {}

def _get_bcjr_tables(device):
    key = str(device)
    if key not in _BCJR_TABLE_CACHE:
        _BCJR_TABLE_CACHE[key] = build_pred_succ_tables(device=device)
    return _BCJR_TABLE_CACHE[key]


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, codebook_gpu,
                 sign_l=None, sign_r=None, bias=True, seed=0,
                 reencode_every_n_steps=1, ip_mode=False,
                 quant_mode="ste", T_init=0.1, bcjr_chunk=BCJR_CHUNK,
                 device="cuda", dtype=torch.float32):
        """
        Args:
            in_features, out_features: like nn.Linear
            codebook_gpu: (65536, 2) float32 cuda tensor. Shared across layers.
            sign_l, sign_r: optional pre-made sign vectors. If None, generated
                from `seed`.
            bias: if True, include learnable bias
            seed: used to derive sign_l/sign_r deterministically
            reencode_every_n_steps: Viterbi cadence (see module docstring)
            ip_mode: if True, run full-IP (W_latent in rotated basis, rotated
                forward). If False (default), pre-IP: W_latent in original
                basis, plain F.linear forward — rotation still happens *inside*
                the quantizer's Viterbi path (via TrellisQuantSTE), but the
                forward sees un-rotated W_q. See findings.md E6/E7 for why
                pre-IP is the current default.
            quant_mode: 'ste' (default) uses TrellisQuantSTE (hard Viterbi +
                straight-through estimator). 'bcjr' uses BCJR forward-backward
                at finite temperature (natively differentiable, soft codeword).
                'bcjr' requires ip_mode=False.
            T_init: initial temperature for BCJR. External annealer can update
                via set_temperature(T). Ignored when quant_mode='ste'.
            bcjr_chunk: tile-chunk size for BCJR (memory / speed tradeoff).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reencode_every = int(reencode_every_n_steps)
        self.ip_mode = bool(ip_mode)
        self.quant_mode = str(quant_mode)
        self.bcjr_chunk = int(bcjr_chunk)
        assert self.quant_mode in ("ste", "bcjr"), f"bad quant_mode {quant_mode}"
        if self.quant_mode == "bcjr":
            assert not self.ip_mode, "BCJR mode requires ip_mode=False (pre-IP data flow)"

        # W_latent is the trainable FP parameter.
        self.W_latent = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )

        # Frozen RHT sign vectors. Shape (out_features,) and (in_features,).
        if sign_l is None:
            sign_l = make_sign_vector(out_features, seed=seed)
        if sign_r is None:
            sign_r = make_sign_vector(in_features, seed=seed + 1)
        # Keep as float32 buffers on the parameter's device.
        self.register_buffer("sign_l",
                             torch.from_numpy(sign_l.astype(np.float32)).to(device))
        self.register_buffer("sign_r",
                             torch.from_numpy(sign_r.astype(np.float32)).to(device))

        # Codebook is a frozen buffer. We don't register it as a module buffer
        # by default (it's shared across layers); caller passes it at init and
        # we store a non-saved reference.
        self.codebook_gpu = codebook_gpu

        # Cached W_q for stale-schedule re-encoding.
        self.register_buffer("_W_q_cache",
                             torch.zeros(out_features, in_features,
                                         dtype=dtype, device=device))
        # Step counter as Python int — used only for modulo branch on the
        # hot path. Keeping it as a GPU tensor would force a GPU→CPU sync on
        # every forward (via .item()), draining pipelining across 112+ QLs.
        self._step_counter = 0

        # Temperature for BCJR. Scalar buffer so external annealers can update
        # it via set_temperature() without needing to touch every layer manually.
        self.register_buffer("T_temp",
                             torch.tensor(float(T_init), dtype=dtype, device=device))
        # BCJR trellis lookup tables — built lazily, not registered (int64, large).
        self._bcjr_preds = None
        self._bcjr_succs = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype,
                                                 device=device))
        else:
            self.register_parameter("bias", None)

        # Default init: Xavier-like for W_latent. Callers doing QAT from a
        # pretrained checkpoint should call `load_fp_weights(W_fp)` to override.
        nn.init.kaiming_uniform_(self.W_latent, a=5 ** 0.5)

    @classmethod
    def from_linear(cls, linear, codebook_gpu, seed=0, **kwargs):
        """Build a QuantizedLinear whose W_latent is the ROTATED version of
        `linear.weight`. Full-IP mode: quantization + forward all happen in
        the rotated basis, so we pre-rotate once at construction.
        """
        out_f, in_f = linear.weight.shape
        bias = linear.bias is not None
        device = linear.weight.device
        dtype = linear.weight.dtype
        qlin = cls(in_f, out_f, codebook_gpu=codebook_gpu, bias=bias,
                   seed=seed, device=device, dtype=dtype, **kwargs)
        with torch.no_grad():
            qlin.load_fp_weights(linear.weight.detach(),
                                 bias=linear.bias.detach() if bias else None)
        return qlin

    def load_fp_weights(self, W_fp, bias=None):
        """Initialize W_latent from an FP weight tensor. In IP mode applies
        Hadamard rotation (W_latent stored in rotated basis). In pre-IP mode
        stores W_fp as-is."""
        with torch.no_grad():
            W_fp_gpu = W_fp.to(device=self.W_latent.device, dtype=torch.float32)
            if self.ip_mode:
                W_fp_gpu = apply_rht_gpu(W_fp_gpu, self.sign_l, self.sign_r)
            self.W_latent.copy_(W_fp_gpu.to(dtype=self.W_latent.dtype))
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.to(self.bias.device, dtype=self.bias.dtype))

    def _ensure_bcjr_tables(self):
        # Use the module-level cache so every QL shares the same tables.
        # Saves ~1.8 GB VRAM on Llama-3.2-1B (112 QLs × 2 tables × 8 MB).
        if self._bcjr_preds is None:
            self._bcjr_preds, self._bcjr_succs = _get_bcjr_tables(
                self.W_latent.device
            )

    def _compute_W_q(self):
        """Run the active quantizer and cache the result.

        quant_mode='ste': TrellisQuantSTE (IP or pre-IP path per ip_mode).
        quant_mode='bcjr': BCJR forward-backward at self.T_temp, pre-IP data flow.
        """
        if self.quant_mode == "bcjr":
            self._ensure_bcjr_tables()
            W_q = bcjr_quantize_weight(
                self.W_latent, self.sign_l, self.sign_r, self.codebook_gpu,
                self.T_temp,
                preds=self._bcjr_preds, succs=self._bcjr_succs,
                chunk_size=self.bcjr_chunk,
            )
        elif self.ip_mode:
            W_q = TrellisQuantSTE_IP.apply(self.W_latent, self.codebook_gpu)
        else:
            W_q = TrellisQuantSTE.apply(
                self.W_latent, self.sign_l, self.sign_r, self.codebook_gpu,
            )
        with torch.no_grad():
            self._W_q_cache.copy_(W_q.detach())
        return W_q

    def set_temperature(self, T):
        """Update BCJR temperature. Accepts python float or 0-d tensor."""
        with torch.no_grad():
            self.T_temp.fill_(float(T))

    def set_quant_mode(self, mode):
        """Switch between 'ste' and 'bcjr' in place. Used by anneal.py to
        flip a layer to BCJR at training start and back to STE for final
        hardening. ip_mode must be False when switching to BCJR."""
        assert mode in ("ste", "bcjr"), f"bad quant_mode {mode}"
        if mode == "bcjr":
            assert not self.ip_mode, "BCJR requires ip_mode=False"
        self.quant_mode = mode

    def _cached_W_q_with_ste_grad(self):
        """Return cached W_q with STE gradient flow attached to W_latent.
        Same basis as W_latent (rotated in IP mode, original in pre-IP)."""
        return self._W_q_cache.detach() + (self.W_latent - self.W_latent.detach())

    def current_W_q(self):
        """Get the W_q (rotated) that would be used this step, honoring the
        schedule. Exposed for tripwires + diagnostics; does NOT advance counter.
        """
        # step is a Python int — no GPU sync.
        if self._step_counter % self.reencode_every == 0:
            return self._compute_W_q()
        return self._cached_W_q_with_ste_grad()

    def _ip_forward(self, x, W_q_rotated):
        """Full-IP matmul: rotate x, matmul, un-rotate y. See module docstring."""
        # W_q is fp32 (codebook dtype). Cast to match x so bf16/fp16 activations
        # don't hit a dtype-mismatch in F.linear. Straight-through gradient
        # to W_latent is preserved because the cast is on the forward path only.
        if W_q_rotated.dtype != x.dtype:
            W_q_rotated = W_q_rotated.to(dtype=x.dtype)
        # Input rotation: x̃ = fht(x * sign_r) / sqrt(in_features)
        x_rot = _fht(x * self.sign_r) / math.sqrt(self.in_features)
        y_rot = F.linear(x_rot, W_q_rotated, bias=None)
        # Output un-rotation: y = (fht(ỹ) / sqrt(out_features)) * sign_l
        y = (_fht(y_rot) / math.sqrt(self.out_features)) * self.sign_l
        if self.bias is not None:
            y = y + self.bias
        return y

    def _plain_forward(self, x, W_q):
        """Pre-IP matmul: plain F.linear, no activation rotation."""
        if W_q.dtype != x.dtype:
            W_q = W_q.to(dtype=x.dtype)
        y = F.linear(x, W_q, bias=None)
        if self.bias is not None:
            y = y + self.bias
        return y

    def forward(self, x):
        W_q = self.current_W_q()
        # Python int increment — no GPU kernel launch for the counter.
        self._step_counter += 1
        if self.ip_mode:
            return self._ip_forward(x, W_q)
        return self._plain_forward(x, W_q)

    def prime_cache(self):
        """Populate _W_q_cache from current W_latent and advance step
        counter by 1. Basis matches self.ip_mode. For BCJR, priming uses
        the STE hard quantizer — we want a crisp initial cache regardless
        of current T. This only affects eval_forward output before the
        first training step and is a no-op on the gradient path."""
        with torch.no_grad():
            if self.ip_mode:
                W_q = TrellisQuantSTE_IP.apply(self.W_latent, self.codebook_gpu)
            else:
                W_q = TrellisQuantSTE.apply(
                    self.W_latent, self.sign_l, self.sign_r, self.codebook_gpu,
                )
            self._W_q_cache.copy_(W_q.detach())
            self._step_counter += 1

    def eval_forward(self, x):
        """Inference path: use cached W_q through the appropriate forward."""
        if self.ip_mode:
            return self._ip_forward(x, self._W_q_cache)
        return self._plain_forward(x, self._W_q_cache)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"reencode_every_n_steps={self.reencode_every}")
