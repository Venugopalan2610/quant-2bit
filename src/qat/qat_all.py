"""Full-model QAT orchestrator: loop over layers, train + save each.

Per-layer budget (4080 / bf16 FP model):
    - collect teacher I/O:     ~2s (3 shards, cached hidden states)
    - first prime_cache:       ~7 min
    - train N_STEPS=16:        ~30s (batch_seqs=2, 1024 T per seq)
    - 3 mid-train reprimes:    ~21 min
    - final prime_cache:       ~7 min
    - save per-target files:   ~10s
    -----------------------------
    total ~35 min/layer × 16 layers ≈ 9.5 hours

Output: cache/qat/L{NN}/*.pt  + cache/qat/aggregate_stats.pt

Resume: --start-layer N skips layers whose dir already has 4+192=196 files.

Usage:
    python -m src.qat.qat_all                    # all 16 layers, defaults
    python -m src.qat.qat_all --start-layer 5
    python -m src.qat.qat_all --end-layer 2      # smoke: layers 0,1
"""
import os
import sys
import time
import argparse
import copy
import gc

import torch
from transformers import OlmoeForCausalLM

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.qat_layer_driver import qat_one_layer
from src.qat.qat_save import save_qat_layer


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
QAT_DIR = "cache/qat"
NUM_LAYERS = 16
# 4 attn + 64 experts × 3 projs = 196 targets/layer
EXPECTED_FILES_PER_LAYER = 4 + 64 * 3


def _layer_complete(qat_dir, layer_idx):
    d = os.path.join(qat_dir, f"L{layer_idx:02d}")
    if not os.path.isdir(d):
        return False
    return len([f for f in os.listdir(d) if f.endswith(".pt")]) >= EXPECTED_FILES_PER_LAYER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=NUM_LAYERS)
    parser.add_argument("--qat-dir", default=QAT_DIR)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--reprime-every", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="global-norm clip on trainable params; 0 disables")
    parser.add_argument("--prox-lambda", type=float, default=0.01,
                        help="weight of sum ||W_latent - W_q||^2 term; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="retrain layers even if already complete")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    os.makedirs(args.qat_dir, exist_ok=True)

    print("=" * 60)
    print(f"QAT orchestrator  layers [{args.start_layer}, {args.end_layer})")
    print(f"  lr={args.lr}  n_steps={args.n_steps}  "
          f"reprime_every={args.reprime_every}  max_shards={args.max_shards}")
    print(f"  grad_clip={args.grad_clip}  prox_lambda={args.prox_lambda}")
    print(f"  output: {args.qat_dir}")
    print("=" * 60)

    print("\nLoading FP16 OLMoE (bf16 CPU)...", flush=True)
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    cfg = model.config
    rotary = copy.deepcopy(model.model.rotary_emb).to("cuda", dtype=torch.float32)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    layer_stats = []
    grand_start = time.time()
    for layer_idx in range(args.start_layer, args.end_layer):
        print(f"\n{'='*60}\nLayer {layer_idx}\n{'='*60}", flush=True)
        if not args.force and _layer_complete(args.qat_dir, layer_idx):
            print(f"  [L{layer_idx:02d}] already complete, skipping "
                  f"(use --force to retrain)", flush=True)
            continue
        t_layer = time.time()
        qat_layer, stats = qat_one_layer(
            model=model, rotary=rotary, cfg=cfg, codebook_gpu=cb,
            layer_idx=layer_idx,
            lr=args.lr, n_steps=args.n_steps,
            reprime_every=args.reprime_every,
            max_shards=args.max_shards,
            grad_clip=args.grad_clip,
            prox_lambda=args.prox_lambda,
            seed=args.seed,
        )
        save_info = save_qat_layer(
            qat_layer, layer_idx=layer_idx, qat_dir=args.qat_dir,
            intermediate_size=cfg.intermediate_size,
        )
        layer_sec = time.time() - t_layer
        stats["save_info"] = save_info
        stats["seconds"] = layer_sec
        layer_stats.append(stats)

        elapsed = time.time() - grand_start
        done = layer_idx - args.start_layer + 1
        remaining = args.end_layer - layer_idx - 1
        eta_min = (elapsed / done) * remaining / 60
        print(f"  [L{layer_idx:02d}] saved {save_info['n_files']} files "
              f"({save_info['total_bytes']/1e9:.2f} GB)  "
              f"layer {layer_sec/60:.1f} min  "
              f"total {elapsed/60:.1f} min  ETA {eta_min:.0f} min", flush=True)

        # Free QAT layer GPU memory before next iteration
        del qat_layer
        gc.collect()
        torch.cuda.empty_cache()

    stats_path = os.path.join(args.qat_dir, "aggregate_stats.pt")
    torch.save({
        "layer_stats": layer_stats,
        "total_seconds": time.time() - grand_start,
        "layers_processed": [s["layer_idx"] for s in layer_stats],
        "args": vars(args),
    }, stats_path)
    print(f"\nAggregate stats: {stats_path}")
    print(f"Total wall clock: {(time.time() - grand_start)/60:.1f} min")


if __name__ == "__main__":
    main()
