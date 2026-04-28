"""TrellisQuantSTE: autograd.Function wrapping RHT + batched CUDA Viterbi
+ differentiable dequant, with vanilla straight-through estimator backward.

Forward:
    W_latent  --RHT (numpy)-->  W_tilde
    W_tilde   --scale by rms-->  W_tilde_unit
    W_tilde_unit --tile into (B, 256) seqs--> sequences_gpu
    sequences_gpu + codebook  --viterbi CUDA-->  walks, recons
    walks + codebook  --differentiable_dequant (torch)-->  W_q

Backward (vanilla STE):
    dL/dW_latent = dL/dW_q
    (the RHT + Viterbi + inverse RHT composition is replaced by identity
    in the backward direction — the simplest estimator that lets gradients
    reach W_latent at all)

Why STE on the FULL pipeline instead of only the Viterbi step:
- The Viterbi step is the non-differentiable one. In principle we could
  make RHT + inverse RHT differentiable via inverse_rht_torch and only
  STE-through the Viterbi. That gives 'real' gradients through the
  rotations but still STE through the discrete quantization.
- For Phase 0 we take the simpler option: STE through the whole thing.
  This matches the original STE paper and the QAT baselines we compare
  against. We can upgrade to partial-STE in a later phase if needed.
"""
import math

import numpy as np
import torch

from src.rht.transform import apply_rht, apply_inverse_rht
from src.viterbi.cuda_kernel import viterbi_encode_v_batched_cuda

try:
    from fast_hadamard_transform import hadamard_transform as _fht
    _HAS_FHT = True
except ImportError:
    _HAS_FHT = False


# Fixed trellis config for now — matches the CUDA kernel hardcoded params.
L_BITS = 16
K = 2
V = 2
TX = 16
TY = 16
T_SEQ = TX * TY  # 256

# Viterbi backpointers are (n_steps, B, 65536) uint8 = 128*B*65536 bytes.
# On a 12GB GPU, we need B ≤ ~1024 to leave headroom.
# At chunk=512: ~4.3 GB transient per Viterbi call → fragments easily when
# >5 GB of permanent state lives alongside (Adam, latents, caches, snaps).
# At chunk=128: ~1.07 GB transient, far more allocator-friendly; ~4× more
# kernel launches but per-kernel work is smaller so wall-clock is comparable.
VITERBI_CHUNK = 128


def viterbi_encode_chunked(sequences, codebook, chunk=VITERBI_CHUNK):
    """Chunked wrapper around viterbi_encode_v_batched_cuda to fit in VRAM."""
    B, T = sequences.shape
    if B <= chunk:
        return viterbi_encode_v_batched_cuda(sequences, codebook)
    bits_chunks, start_chunks, recons_chunks, mses_chunks = [], [], [], []
    for i in range(0, B, chunk):
        b, s, r, m = viterbi_encode_v_batched_cuda(sequences[i:i+chunk], codebook)
        bits_chunks.append(b)
        start_chunks.append(s)
        recons_chunks.append(r)
        mses_chunks.append(m)
    return (torch.cat(bits_chunks, dim=0),
            torch.cat(start_chunks, dim=0),
            torch.cat(recons_chunks, dim=0),
            torch.cat(mses_chunks, dim=0))


def _tile_for_viterbi(W_tilde_unit_np, Tx=TX, Ty=TY):
    """Reshape an (m, n) matrix into (B, T_seq) sequences for the batched kernel.

    Each block is a contiguous Tx x Ty tile of W, flattened row-major.
    Layout: block index = col_block * n_row_blocks + row_block (column-major
    over blocks — so column-block 0 comes first, matching blockldlq order).

    Returns:
        sequences: (n_col_blocks * n_row_blocks, T_seq) float32
        shape_info: (n_row_blocks, n_col_blocks) for reverse tiling
    """
    m, n = W_tilde_unit_np.shape
    assert m % Tx == 0 and n % Ty == 0, f"shape {(m,n)} not tileable by {(Tx,Ty)}"
    n_row_blocks = m // Tx
    n_col_blocks = n // Ty

    # Per-column-block: slice (m, Ty), reshape (n_row_blocks, Tx*Ty)
    seqs = np.empty((n_col_blocks * n_row_blocks, Tx * Ty), dtype=np.float32)
    for j in range(n_col_blocks):
        col_slice = W_tilde_unit_np[:, j*Ty:(j+1)*Ty]            # (m, Ty)
        seqs[j*n_row_blocks:(j+1)*n_row_blocks] = (
            col_slice.reshape(n_row_blocks, Tx*Ty).astype(np.float32)
        )
    return seqs, (n_row_blocks, n_col_blocks)


def _untile_from_viterbi(recons_np, shape_info, shape_mn, Tx=TX, Ty=TY):
    """Inverse of _tile_for_viterbi."""
    n_row_blocks, n_col_blocks = shape_info
    m, n = shape_mn
    out = np.empty((m, n), dtype=np.float32)
    for j in range(n_col_blocks):
        flat = recons_np[j*n_row_blocks:(j+1)*n_row_blocks]       # (n_row_blocks, Tx*Ty)
        out[:, j*Ty:(j+1)*Ty] = flat.reshape(m, Ty)
    return out


def apply_rht_gpu(W, sign_l_gpu, sign_r_gpu):
    """GPU-native RHT. W, sign_l, sign_r all on CUDA, fp32.

    Matches apply_rht(...) in src.rht.transform: normalized by 1/sqrt(m*n).
    """
    assert _HAS_FHT, "fast_hadamard_transform not installed — build from source"
    m, n = W.shape
    Wr = W * sign_r_gpu.unsqueeze(0)              # (m, n)
    Wr = _fht(Wr)                                  # FWHT on last axis (n)
    Wl = (Wr * sign_l_gpu.unsqueeze(1)).transpose(0, 1).contiguous()  # (n, m)
    Wl = _fht(Wl)
    Wl = Wl.transpose(0, 1).contiguous()
    return Wl / math.sqrt(m * n)


def apply_inverse_rht_gpu(W_tilde, sign_l_gpu, sign_r_gpu):
    assert _HAS_FHT
    m, n = W_tilde.shape
    Wr = _fht(W_tilde)
    Wr = Wr * sign_r_gpu.unsqueeze(0)
    Wl = _fht(Wr.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
    Wl = Wl * sign_l_gpu.unsqueeze(1)
    return Wl / math.sqrt(m * n)


def _tile_for_viterbi_gpu(W, Tx=TX, Ty=TY):
    """GPU-native version of _tile_for_viterbi. Returns (B, T_seq) on GPU."""
    m, n = W.shape
    assert m % Tx == 0 and n % Ty == 0
    n_row_blocks = m // Tx
    n_col_blocks = n // Ty
    # (m, n_col, Ty) -> (n_col, m, Ty) -> (n_col*n_row, Tx*Ty)
    seqs = (W.view(m, n_col_blocks, Ty)
             .permute(1, 0, 2)
             .contiguous()
             .view(n_col_blocks * n_row_blocks, Tx * Ty))
    return seqs, (n_row_blocks, n_col_blocks)


def _untile_from_viterbi_gpu(recons, shape_info, shape_mn, Tx=TX, Ty=TY):
    n_row_blocks, n_col_blocks = shape_info
    m, n = shape_mn
    # (n_col*n_row, Tx*Ty) -> (n_col, m, Ty) -> (m, n_col, Ty) -> (m, n)
    return (recons.view(n_col_blocks, m, Ty)
                 .permute(1, 0, 2)
                 .contiguous()
                 .view(m, n))


def quantize_forward_gpu_no_rht(W_rotated, codebook_gpu, Tx=TX, Ty=TY):
    """Quantize a weight matrix that is ALREADY in rotated (Hadamard) basis.

    Used by the full-IP path: W_latent is stored rotated so activation
    rotation and weight quantization share the same basis. We skip the
    outer RHT wrapping and return W_q still in rotated basis.

    Args:
        W_rotated:    (m, n) fp32 CUDA tensor, assumed already Hadamard-rotated
        codebook_gpu: (65536, 2) fp32 CUDA tensor
    Returns:
        W_q_rotated: (m, n) fp32 CUDA tensor, in same rotated basis
    """
    m, n = W_rotated.shape
    W_scale = W_rotated.pow(2).mean().sqrt().clamp(min=1e-30)
    W_tilde_unit = W_rotated / W_scale
    sequences, shape_info = _tile_for_viterbi_gpu(W_tilde_unit, Tx=Tx, Ty=Ty)
    _, _, recons, _ = viterbi_encode_chunked(sequences, codebook_gpu)
    W_tilde_q_unit = _untile_from_viterbi_gpu(recons, shape_info, (m, n), Tx, Ty)
    return W_tilde_q_unit * W_scale


def quantize_forward_gpu(W_latent, sign_l_gpu, sign_r_gpu, codebook_gpu,
                         Tx=TX, Ty=TY):
    """End-to-end forward quantization, staying on GPU throughout.

    Args:
        W_latent:     (m, n) fp32 CUDA tensor
        sign_l_gpu:   (m,) fp32 CUDA tensor
        sign_r_gpu:   (n,) fp32 CUDA tensor
        codebook_gpu: (65536, 2) fp32 CUDA tensor
    Returns:
        W_q: (m, n) fp32 CUDA tensor
    """
    m, n = W_latent.shape

    W_tilde = apply_rht_gpu(W_latent, sign_l_gpu, sign_r_gpu)

    W_scale = W_tilde.pow(2).mean().sqrt().clamp(min=1e-30)
    W_tilde_unit = W_tilde / W_scale

    sequences, shape_info = _tile_for_viterbi_gpu(W_tilde_unit, Tx=Tx, Ty=Ty)

    _, _, recons, _ = viterbi_encode_chunked(sequences, codebook_gpu)

    W_tilde_q_unit = _untile_from_viterbi_gpu(recons, shape_info, (m, n), Tx, Ty)
    W_tilde_q = W_tilde_q_unit * W_scale
    W_q = apply_inverse_rht_gpu(W_tilde_q, sign_l_gpu, sign_r_gpu)
    return W_q


def quantize_forward_numpy(W_latent_np, sign_l_np, sign_r_np, codebook_gpu,
                           Tx=TX, Ty=TY):
    """Run the full forward quantization pipeline.

    Args:
        W_latent_np: (m, n) float32 numpy on CPU
        sign_l_np: (m,) float32 ±1 numpy
        sign_r_np: (n,) float32 ±1 numpy
        codebook_gpu: (65536, 2) float32 cuda tensor
    Returns:
        W_q_np: (m, n) float32 numpy — final dequantized weight
        W_scale: float — the RMS scale that was applied
        walks: (n_col_blocks * n_row_blocks, T_seq/V) int32 numpy —
               state walks for diagnostics / cache
    """
    m, n = W_latent_np.shape

    # 1. RHT (numpy)
    W_tilde = apply_rht(W_latent_np, sign_l_np, sign_r_np).astype(np.float32)

    # 2. Scale by rms
    W_scale = float(np.sqrt((W_tilde ** 2).mean()))
    if W_scale < 1e-30:
        W_scale = 1.0
    W_tilde_unit = W_tilde / W_scale

    # 3. Tile into (B, T_seq) sequences
    sequences_np, shape_info = _tile_for_viterbi(W_tilde_unit, Tx=Tx, Ty=Ty)

    # 4. Run batched CUDA Viterbi
    sequences_gpu = torch.from_numpy(sequences_np).cuda()
    bitstreams_gpu, start_states_gpu, recons_gpu, _mses_gpu = (
        viterbi_encode_chunked(sequences_gpu, codebook_gpu)
    )
    recons_np = recons_gpu.cpu().numpy()                          # (B, T_seq)

    # 5. Unscale + untile to (m, n)
    W_tilde_q_unit = _untile_from_viterbi(recons_np, shape_info, (m, n),
                                           Tx=Tx, Ty=Ty)
    W_tilde_q = W_tilde_q_unit * W_scale

    # 6. Inverse RHT -> W_q
    W_q = apply_inverse_rht(W_tilde_q.astype(np.float32), sign_l_np, sign_r_np)

    # Assemble walks (state walks) for possible use downstream (caching, analysis)
    # walks shape: (B, n_steps) where n_steps = T_seq / V
    n_col_blocks = shape_info[1]
    n_row_blocks = shape_info[0]
    walks = _replay_walks(
        bitstreams_gpu.cpu().numpy().astype(np.int64),
        start_states_gpu.cpu().numpy().astype(np.int64),
    )

    return W_q.astype(np.float32), W_scale, walks, bitstreams_gpu.cpu().numpy(), \
           start_states_gpu.cpu().numpy()


def _replay_walks(bitstreams, start_states):
    """Replay state walks from bitstreams + start_states. (B, n_steps) int64.

    Bit-compatible with precompute_walk_states in src.quantize.lut_ft.
    """
    B, n_steps = bitstreams.shape
    mask = (1 << L_BITS) - 1
    kV = K * V
    walks = np.zeros_like(bitstreams, dtype=np.int64)
    s = start_states.copy()                                       # (B,)
    for t in range(n_steps):
        s = ((s << kV) | bitstreams[:, t]) & mask
        walks[:, t] = s
    return walks


class TrellisQuantSTE(torch.autograd.Function):
    """Vanilla STE wrapping of the full QTIP forward quantization.

    Usage:
        W_q = TrellisQuantSTE.apply(W_latent, sign_l, sign_r, codebook_gpu)

    W_latent: (m, n) torch tensor (CPU or CUDA), float, requires_grad
    sign_l: (m,) torch tensor or numpy array, ±1 float32
    sign_r: (n,) torch tensor or numpy array, ±1 float32
    codebook_gpu: (65536, 2) float32 cuda tensor

    Returns W_q (m, n) float32 tensor on the same device as W_latent.
    """
    @staticmethod
    def forward(ctx, W_latent, sign_l, sign_r, codebook_gpu):
        in_device = W_latent.device
        in_dtype = W_latent.dtype

        # GPU-native path when everything is on CUDA and FHT is available.
        # Avoids CPU round-trips that thrash RAM and waste PCIe bandwidth.
        use_gpu = _HAS_FHT and W_latent.is_cuda
        if use_gpu:
            sign_l_gpu = _to_cuda_float32(sign_l)
            sign_r_gpu = _to_cuda_float32(sign_r)
            W_q = quantize_forward_gpu(
                W_latent.detach().float(), sign_l_gpu, sign_r_gpu, codebook_gpu,
            )
            return W_q.to(device=in_device, dtype=in_dtype)

        # Fallback: numpy CPU path (PTQ parity).
        W_np = W_latent.detach().float().cpu().numpy()
        sign_l_np = _to_numpy_float32(sign_l)
        sign_r_np = _to_numpy_float32(sign_r)
        W_q_np, W_scale, walks, bitstreams, start_states = quantize_forward_numpy(
            W_np, sign_l_np, sign_r_np, codebook_gpu,
        )
        ctx.W_scale = W_scale
        ctx.walks = walks
        ctx.bitstreams = bitstreams
        ctx.start_states = start_states
        W_q = torch.from_numpy(W_q_np).to(device=in_device, dtype=in_dtype)
        return W_q

    @staticmethod
    def backward(ctx, grad_output):
        # Vanilla straight-through estimator: dL/dW_latent = dL/dW_q.
        # Return None for non-tensor / non-differentiable inputs.
        return grad_output, None, None, None


class TrellisQuantSTE_IP(torch.autograd.Function):
    """STE wrapping Viterbi-only quantization for full-IP mode.

    Assumes W_latent is already in rotated (Hadamard) basis — the activation
    rotation in QuantizedLinear.forward ensures the forward matmul happens in
    the same basis. No outer RHT wrapping here.

    Usage:
        W_q_rotated = TrellisQuantSTE_IP.apply(W_latent_rotated, codebook_gpu)
    """
    @staticmethod
    def forward(ctx, W_latent_rotated, codebook_gpu):
        assert W_latent_rotated.is_cuda, "IP mode requires CUDA"
        assert _HAS_FHT, "fast_hadamard_transform required for IP mode"
        W_q = quantize_forward_gpu_no_rht(
            W_latent_rotated.detach().float(), codebook_gpu,
        )
        return W_q.to(device=W_latent_rotated.device, dtype=W_latent_rotated.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _to_numpy_float32(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _to_cuda_float32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cuda", dtype=torch.float32)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32)).to("cuda")
    return torch.as_tensor(x, dtype=torch.float32, device="cuda")


def make_hyb_codebook_gpu(lut_np, Q=9, L_bits=L_BITS, device="cuda"):
    """Build the full (2^L, V) HYB codebook from a (2^Q, V) LUT, on GPU.

    Mirrors build_differentiable_codebook in src.quantize.lut_ft but returns
    a non-grad-tracked tensor for use as the Viterbi kernel input.
    """
    lut = torch.from_numpy(lut_np.astype(np.float32)).to(device).contiguous()
    n_states = 1 << L_bits
    states = torch.arange(n_states, dtype=torch.int64, device=device)
    x = (states * states + states) & 0xFFFFFFFF
    idx = ((x >> (15 - Q)) & ((1 << Q) - 1)).long()
    sign_flip = ((x >> 15) & 1).bool()
    sign_factor = torch.where(sign_flip, -1.0, 1.0).to(lut.dtype)
    col0 = lut[idx, 0]
    col1 = lut[idx, 1] * sign_factor
    codebook = torch.stack([col0, col1], dim=-1).contiguous()      # (2^L, 2)
    return codebook
