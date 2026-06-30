"""CPU unit tests for KV-cache quantization (no GPU needed).
Covers the REAL TurboQuant path (import + per-head reshape) and the uniform
fallback."""
import torch

from src.eval.kv_quant import quantize_kv_tensor, _import_turboquant


def test_real_turboquant_imports_and_runs():
    """The user-cloned TurboQuant must import and fake-quant at head_dim=64."""
    TQ = _import_turboquant()
    tq3 = TQ(dim=64, bits=3, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(2, 8, 16, 64)
    xh = tq3.forward(x)
    assert xh.shape == x.shape
    e3 = (x - xh).pow(2).mean().item()
    e2 = (x - TQ(dim=64, bits=2, device=torch.device("cpu")).forward(x)).pow(2).mean().item()
    assert e3 < e2, f"3-bit ({e3:.4f}) should beat 2-bit ({e2:.4f})"
    print(f"  real TurboQuant d64: 3b MSE={e3:.4f} < 2b MSE={e2:.4f}")


def test_real_turboquant_per_head_reshape():
    """k_proj output (B,S, n_kv_heads*head_dim) reshaped to per-head vectors,
    quantized, reshaped back — shape preserved, this is exactly _quant's path."""
    TQ = _import_turboquant()
    hd, n_kv = 64, 8
    out = torch.randn(2, 16, n_kv * hd)          # mimic k_proj output
    tq = TQ(dim=hd, bits=3, device=torch.device("cpu"), dtype=torch.float32)
    v = out.reshape(2, 16, n_kv, hd)
    q = tq.forward(v.float()).reshape(out.shape)
    assert q.shape == out.shape
    assert (out - q).pow(2).mean().item() < out.pow(2).mean().item()  # quantization < signal


def test_shape_dtype_preserved():
    x = torch.randn(2, 8, 16, 64, dtype=torch.float32)  # (B, heads, seq, head_dim)
    q = quantize_kv_tensor(x, n_bits=2)
    assert q.shape == x.shape and q.dtype == x.dtype


def test_passthrough_when_full_precision():
    x = torch.randn(4, 64)
    assert torch.equal(quantize_kv_tensor(x, n_bits=16), x)
    assert torch.equal(quantize_kv_tensor(x, n_bits=None), x)


def test_more_bits_lower_error():
    torch.manual_seed(0)
    x = torch.randn(32, 128)
    err = {}
    for b in (2, 3, 4):
        err[b] = (x - quantize_kv_tensor(x, n_bits=b)).pow(2).mean().item()
    assert err[4] < err[3] < err[2], err
    print(f"  KV-quant MSE: 2b={err[2]:.4f} 3b={err[3]:.4f} 4b={err[4]:.4f}")


def test_per_token_independent_scales():
    # a row with a big outlier and a row of small values must use different scales
    x = torch.tensor([[10.0, -10.0, 0.1, -0.1], [0.01, -0.02, 0.015, -0.005]])
    q = quantize_kv_tensor(x, n_bits=2)
    # small row should NOT be crushed to the big row's scale (would zero it out)
    assert q[1].abs().sum() > 0, "per-token scaling failed — small row crushed"


def test_turboquant_bit_allocation():
    # default serving allocation: 3-bit keys, 2-bit values
    x = torch.randn(16, 64)
    qk = quantize_kv_tensor(x, n_bits=3)
    qv = quantize_kv_tensor(x, n_bits=2)
    ek = (x - qk).pow(2).mean().item()
    ev = (x - qv).pow(2).mean().item()
    assert ek < ev, "3-bit keys should be more accurate than 2-bit values"
    print(f"  keys@3b MSE={ek:.4f} < values@2b MSE={ev:.4f} (TurboQuant allocation)")


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
