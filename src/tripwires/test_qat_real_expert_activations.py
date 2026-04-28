"""Tripwire Q.1.1: real-expert activation pipeline sanity.

Sub-tests:
  Q.1.1.1: collect_layer_activations returns (X, Y) with correct shapes
           (n_tokens, hidden) and non-zero content, for layer 0, budget=256.
  Q.1.1.2: running a clean FP16 expert forward on captured X reproduces
           captured Y to within bf16 numerics. This validates that the
           monkey-patched capture hook in collect_activations.py grabs
           the right (input, output) pair.
  Q.1.1.3: at least half the experts got at least 64 tokens
           (routing-diversity sanity — we don't want to pick a never-hit
           expert for Q.1.3 training).

Prereqs:
  cache/model/olmoe-1b-7b-0125/               (FP16 OLMoE)
  cache/hidden_states/layer_00_input/         (via scripts/qat_prep.sh)

Run: python3 -m src.tripwires.test_qat_real_expert_activations
"""
import sys
import copy
import time
import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM

from src.finetune.collect_activations import collect_layer_activations


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
LAYER_IDX = 0
BUDGET = 256


def _load_model_and_rotary():
    print("Loading FP16 OLMoE (bf16 CPU)...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    rotary = copy.deepcopy(model.model.rotary_emb).to(
        device="cuda", dtype=torch.float32,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model, rotary


def test_shapes_and_nonzero(captures, cfg):
    print("\nQ.1.1.1: captures have right shape and are non-zero")
    print("-" * 60)
    hidden = cfg.hidden_size
    n_experts = cfg.num_experts
    ok_all_shapes = True
    ok_all_nonzero = True
    empty_count = 0
    for e in range(n_experts):
        X, Y = captures[e]
        if X is None:
            empty_count += 1
            continue
        shape_ok = (X.dim() == 2 and X.shape[1] == hidden and
                    Y.dim() == 2 and Y.shape[1] == hidden and
                    X.shape[0] == Y.shape[0])
        nonzero_ok = X.abs().mean().item() > 1e-6 and Y.abs().mean().item() > 1e-6
        ok_all_shapes = ok_all_shapes and shape_ok
        ok_all_nonzero = ok_all_nonzero and nonzero_ok
    print(f"  experts with captures: {n_experts - empty_count} / {n_experts}")
    print(f"  all shapes (n, {hidden}): {ok_all_shapes}")
    print(f"  all non-zero:            {ok_all_nonzero}")
    ok = ok_all_shapes and ok_all_nonzero and empty_count < n_experts
    print(f"  [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_teacher_expert_matches(captures, model, cfg):
    """For one well-hit expert, run its FP16 forward on X and confirm Y matches."""
    print("\nQ.1.1.2: FP16 expert forward(X) reproduces captured Y")
    print("-" * 60)
    # Pick the most-hit expert
    best_e, best_n = -1, 0
    for e in range(cfg.num_experts):
        X, _ = captures[e]
        if X is None:
            continue
        if X.shape[0] > best_n:
            best_n = X.shape[0]
            best_e = e
    if best_e < 0:
        print("  no experts captured — FAIL")
        return False
    X, Y = captures[best_e]
    print(f"  expert {best_e}: n_tokens = {X.shape[0]}")

    # Grab FP16 expert weights — fresh fp32 copy on GPU to match collector
    experts_mod = model.model.layers[LAYER_IDX].mlp.experts
    W_gate_up = experts_mod.gate_up_proj[best_e].detach().to(
        device="cuda", dtype=torch.float32
    )
    W_down = experts_mod.down_proj[best_e].detach().to(
        device="cuda", dtype=torch.float32
    )

    X_gpu = X.to("cuda", dtype=torch.float32)
    Y_ref = Y.to("cuda", dtype=torch.float32)

    with torch.no_grad():
        gate, up = F.linear(X_gpu, W_gate_up).chunk(2, dim=-1)
        mlp = F.silu(gate) * up
        Y_pred = F.linear(mlp, W_down)

    diff = (Y_pred - Y_ref).abs().mean().item()
    rel = diff / max(Y_ref.abs().mean().item(), 1e-30)
    print(f"  mean |Y_pred - Y|:  {diff:.4e}")
    print(f"  mean |Y|:           {Y_ref.abs().mean().item():.4e}")
    print(f"  relative error:     {rel:.4e}")
    # Captured Y was cast to fp32 inside the patched forward, but matmul happens
    # in the layer's native dtype upstream. Our replay here is pure fp32 so we
    # expect a small-but-nonzero delta from rounding.
    ok = rel < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] teacher forward reproduces Y")
    return ok


def test_routing_diversity(captures, cfg):
    print("\nQ.1.1.3: routing diversity — enough experts got tokens")
    print("-" * 60)
    counts = []
    for e in range(cfg.num_experts):
        X, _ = captures[e]
        counts.append(0 if X is None else X.shape[0])
    counts.sort(reverse=True)
    well_hit = sum(1 for c in counts if c >= 64)
    print(f"  experts with >= 64 tokens: {well_hit} / {cfg.num_experts}")
    print(f"  top 5 counts: {counts[:5]}")
    print(f"  bottom 5 counts: {counts[-5:]}")
    ok = well_hit >= cfg.num_experts // 2
    print(f"  [{'PASS' if ok else 'FAIL'}] at least half of experts well-hit")
    return ok


def main():
    print("=" * 60)
    print("Tripwire Q.1.1: real-expert activation pipeline")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    model, rotary = _load_model_and_rotary()
    cfg = model.config

    print(f"\nCollecting activations from layer {LAYER_IDX} "
          f"(budget={BUDGET}/expert)...")
    captures = collect_layer_activations(
        model=model, rotary_emb=rotary, layer_idx=LAYER_IDX,
        cfg=cfg, device="cuda", budget_per_expert=BUDGET,
    )

    results = []
    results.append(("shapes_nonzero", test_shapes_and_nonzero(captures, cfg)))
    results.append(("teacher_matches", test_teacher_expert_matches(captures, model, cfg)))
    results.append(("routing_diversity", test_routing_diversity(captures, cfg)))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("Q.1.1 GATE: PASS — real-expert activation pipeline works.")
        print("Ready for Q.1.2 (QATExpert init matches PTQ).")
        sys.exit(0)
    else:
        print("Q.1.1 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
