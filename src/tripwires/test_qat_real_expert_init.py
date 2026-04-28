"""Tripwire Q.1.2: QATExpert init matches PTQ on a real OLMoE expert.

At step 0 (before any gradient update), QATExpert.from_weights produces the
same output as independently running PTQ on the same FP16 expert weights and
manually computing the expert forward with those W_q tensors.

This also measures the *PTQ baseline MSE* against the FP16 teacher Y, which
Q.1.3 must beat.

Sub-tests:
  Q.1.2.1: QATExpert.forward(X) at step 0 matches a reference built by running
           standalone PTQ on gate_up/down + running the expert forward manually.
  Q.1.2.2: PTQ baseline MSE(QATExpert(X), Y_teacher) is finite and well-defined;
           record it for Q.1.3 comparison.
  Q.1.2.3: FP16-vs-PTQ degradation is non-trivial (PTQ's MSE is meaningfully
           worse than FP16) — sanity that we're actually quantizing.

Run: python3 -m src.tripwires.test_qat_real_expert_init
"""
import sys
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM

from src.finetune.collect_activations import collect_layer_activations
from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS, quantize_forward_numpy
from src.qat.qat_expert import QATExpert


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
LAYER_IDX = 0
EXPERT_IDX = 0
BUDGET = 256
SEED = 0


def _make_codebook(seed=0):
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=seed)
    return make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)


def _ptq_weight_numpy(W_fp32, sign_l_np, sign_r_np, cb_gpu):
    W_q_np, _, _, _, _ = quantize_forward_numpy(
        W_fp32, sign_l_np, sign_r_np, cb_gpu,
    )
    return W_q_np


def test_init_matches_ptq_reference(qexp, W_gate_up_fp32, W_down_fp32, X, cb_gpu):
    print("\nQ.1.2.1: QATExpert forward at step 0 == manual PTQ + expert forward")
    print("-" * 60)

    # Reset step counters so both submodules recompute from current W_latent.
    qexp.gate_up._step_counter.zero_()
    qexp.down._step_counter.zero_()

    # Reference: run PTQ on the FP weights using the module's signs so they
    # match exactly.
    sign_l_gu = qexp.gate_up.sign_l.detach().cpu().numpy()
    sign_r_gu = qexp.gate_up.sign_r.detach().cpu().numpy()
    sign_l_d  = qexp.down.sign_l.detach().cpu().numpy()
    sign_r_d  = qexp.down.sign_r.detach().cpu().numpy()

    W_gu_q = _ptq_weight_numpy(W_gate_up_fp32.cpu().numpy().astype(np.float32),
                               sign_l_gu, sign_r_gu, cb_gpu)
    W_d_q  = _ptq_weight_numpy(W_down_fp32.cpu().numpy().astype(np.float32),
                               sign_l_d, sign_r_d, cb_gpu)
    W_gu_q_t = torch.from_numpy(W_gu_q).to("cuda", dtype=torch.float32)
    W_d_q_t  = torch.from_numpy(W_d_q).to("cuda", dtype=torch.float32)

    with torch.no_grad():
        # Reference path
        gate, up = F.linear(X, W_gu_q_t).chunk(2, dim=-1)
        ref = F.linear(F.silu(gate) * up, W_d_q_t)
        # Module path
        y_mod = qexp(X)

    diff = (y_mod - ref).abs().max().item()
    rel  = diff / max(ref.abs().mean().item(), 1e-30)
    print(f"  max |y_module - y_reference|: {diff:.4e}")
    print(f"  relative:                     {rel:.4e}")
    ok = diff < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] module == reference PTQ+forward")
    return ok


def test_ptq_baseline_mse(qexp, X, Y):
    print("\nQ.1.2.2: PTQ baseline MSE (before any QAT training)")
    print("-" * 60)
    qexp.gate_up._step_counter.zero_()
    qexp.down._step_counter.zero_()
    with torch.no_grad():
        y_ptq = qexp(X)
    mse = F.mse_loss(y_ptq, Y).item()
    ok = np.isfinite(mse)
    print(f"  PTQ-init MSE(student(X), Y_teacher): {mse:.6e}")
    print(f"  [{'PASS' if ok else 'FAIL'}] finite baseline")
    return ok, mse


def test_fp16_vs_ptq_gap(W_gate_up_fp32, W_down_fp32, X, Y, ptq_mse):
    print("\nQ.1.2.3: FP16 vs PTQ MSE gap is non-trivial")
    print("-" * 60)
    with torch.no_grad():
        gate, up = F.linear(X, W_gate_up_fp32).chunk(2, dim=-1)
        y_fp = F.linear(F.silu(gate) * up, W_down_fp32)
    fp_mse = F.mse_loss(y_fp, Y).item()
    print(f"  FP32 (exact)  MSE: {fp_mse:.6e}")
    print(f"  PTQ init      MSE: {ptq_mse:.6e}")
    print(f"  ratio (PTQ/FP):   {ptq_mse / max(fp_mse, 1e-30):.2f}")
    # FP should be ~0 (we're computing against captured Y which was produced
    # by the same expert), so any non-trivial PTQ loss >> FP loss.
    ok = ptq_mse > 10 * max(fp_mse, 1e-12)
    print(f"  [{'PASS' if ok else 'FAIL'}] quantization causes real degradation")
    return ok


def main():
    print("=" * 60)
    print("Tripwire Q.1.2: QATExpert init matches PTQ on real expert")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA required")
        sys.exit(1)

    print("Loading FP16 OLMoE (bf16 CPU)...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    cfg = model.config
    rotary = copy.deepcopy(model.model.rotary_emb).to("cuda", dtype=torch.float32)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"\nCollecting layer {LAYER_IDX} activations (budget={BUDGET})...")
    captures = collect_layer_activations(
        model=model, rotary_emb=rotary, layer_idx=LAYER_IDX,
        cfg=cfg, device="cuda", budget_per_expert=BUDGET,
    )
    X_cpu, Y_cpu = captures[EXPERT_IDX]
    X = X_cpu.to("cuda", dtype=torch.float32)
    Y = Y_cpu.to("cuda", dtype=torch.float32)
    print(f"  expert {EXPERT_IDX}: X {tuple(X.shape)}  Y {tuple(Y.shape)}")

    # FP32 expert weights
    experts_mod = model.model.layers[LAYER_IDX].mlp.experts
    W_gate_up = experts_mod.gate_up_proj[EXPERT_IDX].detach().to(
        device="cuda", dtype=torch.float32
    )
    W_down = experts_mod.down_proj[EXPERT_IDX].detach().to(
        device="cuda", dtype=torch.float32
    )
    print(f"  W_gate_up: {tuple(W_gate_up.shape)}  W_down: {tuple(W_down.shape)}")

    print("\nBuilding codebook + QATExpert from FP weights...")
    cb = _make_codebook(seed=SEED)
    qexp = QATExpert.from_weights(
        W_gate_up_fp16=W_gate_up, W_down_fp16=W_down,
        codebook_gpu=cb, seed=SEED,
    )

    results = []
    results.append(("init_matches_ptq",
                    test_init_matches_ptq_reference(qexp, W_gate_up, W_down, X, cb)))
    ok_base, ptq_mse = test_ptq_baseline_mse(qexp, X, Y)
    results.append(("baseline_finite", ok_base))
    results.append(("quant_degrades",
                    test_fp16_vs_ptq_gap(W_gate_up, W_down, X, Y, ptq_mse)))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    print(f"Baseline (for Q.1.3 comparison): PTQ MSE = {ptq_mse:.6e}")
    if all_ok:
        print("Q.1.2 GATE: PASS — QATExpert on real expert inits to PTQ as expected.")
        print("Ready for Q.1.3 (QAT beats PTQ on real expert).")
        sys.exit(0)
    else:
        print("Q.1.2 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
