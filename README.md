# quant-olmoe

This repository hosts two related projects on extreme-low-bit LLM
compression:

## 1. BCJR-QAT (Apr 2026)

> *BCJR-QAT: A Differentiable Relaxation of Trellis-Coded Weight Quantization*

A differentiable relaxation of trellis-coded weight quantization that
replaces the non-differentiable Viterbi argmax with the BCJR
forward--backward sum-product algorithm at temperature *T*, producing a
soft codeword equal to the Boltzmann expectation over trellis paths.

**Headline result:** with the right schedule (skip the high-*T* phase to
avoid an overshoot we diagnose), single-layer BCJR-QAT beats QTIP-PTQ on
Llama-3.2-1B layer 4 at 2 bpw by **−0.084 PPL** on WikiText-2
(10.13 vs. 10.22). Multi-layer compounding is super-additive.

- `src/bcjr/` — BCJR kernel (forward--backward, Triton-fused, autograd
  node, parity-verified within fp32 noise; 6.57× speedup over reference)
- `src/qat/` — QAT pipeline:
  `QATDenseDecoderLayer`, `quantized_linear.py`, `train_llama_single_layer.py`,
  `train_e2e_kl.py`
- `src/eval/install_bcjr.py`, `bootstrap_ppl.py` — eval infrastructure
- `scripts/eval_llama_{trajectory,multilayer,step0_baseline}.py` — eval helpers
- `scripts/vast_train_llama_skip_highT.sh` — the headline experiment recipe
- `src/tripwires/test_bcjr_*.py` — kernel parity tests

**Reproducing the headline result** (Llama-3.2-1B layer 4 at 2 bpw):

```bash
# Local 4080: ~10 h for the 4080 baseline (3-step PTQ-init);
# Cloud H100 SXM: ~5 h for the skip-high-T 10-step run that wins.
# See scripts/vast_train_llama_skip_highT.sh for the full recipe.
bash scripts/vast_setup_llama.sh
bash scripts/vast_train_llama_skip_highT.sh

# Then evaluate:
python -m scripts.eval_llama_trajectory \
    --ckpt-dir cache/llama_bcjr_skipT \
    --target-layer 4 \
    --output results/llama_skipT_trajectory.json
```

Quantized snapshots, trajectory checkpoints, and per-window NLLs (for
bootstrap analysis) are released at
[Venugopalan2610](https://huggingface.co/Venugopalan2610) on HuggingFace.

---

## 2. qtip-olmoe (prior work — used as PTQ baseline in the BCJR-QAT paper)

2-bit weight-only quantization of
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0125) on a single
consumer GPU, via per-expert routing-conditioned Hessian calibration on top
of trellis-coded quantization ([QTIP](https://arxiv.org/abs/2406.11235)).
The full pipeline — per-expert Hessian collection, BlockLDLQ trellis
quantization, two per-layer-H ablation baselines, and WikiText-2 / C4 /
lm-eval-harness evaluation — runs end-to-end on a 12 GB consumer GPU in
about 18 hours. Quantization runs strictly per expert on GPU; the paper's
numbers were measured on an RTX 4080 Laptop (12 GB, Ada, compute 8.9)
under WSL2 Ubuntu 22.04.

## Headline results (Table 1)

| Config                                   | Bits | WikiText-2 PPL | C4 PPL |
|------------------------------------------|------|----------------|--------|
| fp16 baseline                            | 16   | 6.65           | 12.24  |
| **Per-expert H (ours)**                  | 2    | **9.09**       | **14.16** |
| Per-layer H, unweighted mean             | 2    | 9.21           | 14.43  |
| Per-layer H, token-weighted mean         | 2    | 9.18           | 14.44  |

Zero-shot downstream accuracy (lm-eval-harness 0.4.11): HellaSwag
acc_norm 64.8 (fp16 68.1), ARC-c 44.3 (46.1), ARC-e 67.9 (70.4),
PIQA 77.9 (79.6). Full numbers in the paper (Section 5.2).

On-disk footprint: 3.3 GB for the 2-bit packed bitstream (8× compression
on quantized tensors), 4.5 GB including the unquantized router + embeddings
+ `lm_head`.

## Hardware and software

| Requirement       | Tested configuration                                    |
|-------------------|---------------------------------------------------------|
| GPU               | NVIDIA RTX 4080 Laptop, 12 GB, compute 8.9 (Ada)        |
| Minimum VRAM      | 12 GB (reproduce.sh asserts ≥ 11.5 GB at startup)       |
| CUDA toolkit      | 12.8                                                    |
| OS                | Ubuntu 22.04 under WSL2 (native Linux should also work) |
| Python            | 3.11                                                    |
| Disk (cache)      | ≈ 200 GB free for hidden states + Hessians (ephemeral)  |

Other ≥12 GB Ampere/Ada cards (RTX 3060 12 GB, 3090, 4090, A10, L4) are
expected to work without changes. `requirements.txt` pins the exact
`transformers` / `torch` / `lm-eval` versions that produced the paper's
numbers; deviating from these is the most common source of ±0.02 PPL
drift.

## Install

```bash
git clone https://github.com/Venugopalan2610/qtip-olmoe.git
cd qtip-olmoe

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the Hadamard transform kernel from source (pip wheel does not
# target CUDA 12.8)
git clone https://github.com/Dao-AILab/fast-hadamard-transform
pip install ./fast-hadamard-transform

# Sanity check
python -c "import torch; \
  print(torch.cuda.get_device_name(0), \
        torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

## Quick start: load the pre-built checkpoint

If you want the numbers but not the 18-hour run, the 4.5 GB quantized
model is published on Hugging Face:

```bash
python -m src.eval.install_quantized --hf-repo Venugopalan2610/qtip-olmoe-2bit
python -m src.eval.run_ppl --config 2bit_noft --dataset wikitext2
python -m src.eval.run_ppl --config 2bit_noft --dataset c4
```

PPL eval is ~10 minutes for both datasets on the 4080 Laptop.

## Full reproduction (all three configs)

```bash
bash scripts/reproduce.sh
```

Runs all four stages, all three quantization configs, PPL eval on both
datasets, and lm-eval-harness on five tasks. Stage wall-clocks on the
RTX 4080 Laptop (your times will scale roughly with TFLOPs):

| Stage                                    | Approx. wall-clock |
|------------------------------------------|-------------------:|
| 1. Calibration data + embedding pass     | 20 min             |
| 2. Per-expert Hessian collection (16L)   | 2.5 hr             |
| 3a. Quantize — per-expert H (ours)       | 3.5 hr             |
| 3b. Quantize — per-layer H unweighted    | 3.5 hr             |
| 3c. Quantize — per-layer H token-weighted| 3.5 hr             |
| 4. PPL eval (4 models × 2 datasets)      | 20 min             |
| 5. lm-eval downstream (2 models × 5 tasks) | 5 hr             |
| 6. Figures (kurtosis, routing imbalance) | 2 min              |
| **Total**                                | **~18 hr**         |

Resume flags skip completed stages:

```bash
bash scripts/reproduce.sh --skip-hessians            # reuse cache/hessians/
bash scripts/reproduce.sh --skip-hessians --skip-quant  # eval only
```

## Per-config reproduction

If you only want one config (e.g. the main result, no ablations):

### Config 1 — Per-expert H (ours, Table 1 row 2)

```bash
# Stage 1: calibration + per-expert Hessians (~2.8 hr)
python -m src.hessian.prepare_calib --n_seqs 2048 --seq_len 1024 --seed 0
python -m src.hessian.embed_pass --shard_size 64 --batch_size 64
python -m src.hessian.collect_all --start-layer 0

# Stage 2: quantize (~3.5 hr)
python -m src.quantize.quantize_all --damp 0.01

# Stage 3: evaluate (~10 min)
python -m src.eval.run_ppl --config 2bit_noft --dataset wikitext2 \
    --quant-dir cache/quantized
python -m src.eval.run_ppl --config 2bit_noft --dataset c4 \
    --quant-dir cache/quantized
```

Expected: WikiText-2 PPL **9.09**, C4 PPL **14.16**.

### Config 2 — Per-layer H, unweighted (Table 1 row 3)

Requires Stage 1 above to have run (same per-expert Hessians are reduced
down to per-layer means).

```bash
python -m src.quantize.compute_per_layer_mean_H
python -m src.quantize.quantize_all_per_layer_H --damp 0.01

python -m src.eval.run_ppl --config 2bit_noft --dataset wikitext2 \
    --quant-dir cache/quantized_per_layer_H
python -m src.eval.run_ppl --config 2bit_noft --dataset c4 \
    --quant-dir cache/quantized_per_layer_H
```

Expected: WikiText-2 PPL **9.21**, C4 PPL **14.43**.

### Config 3 — Per-layer H, token-weighted (Table 1 row 4)

```bash
python -m src.quantize.compute_per_layer_weighted_mean_H
python -m src.quantize.quantize_all_per_layer_weighted_H --damp 0.01

python -m src.eval.run_ppl --config 2bit_noft --dataset wikitext2 \
    --quant-dir cache/quantized_per_layer_weighted_H
python -m src.eval.run_ppl --config 2bit_noft --dataset c4 \
    --quant-dir cache/quantized_per_layer_weighted_H
```

Expected: WikiText-2 PPL **9.18**, C4 PPL **14.44**.

Deviations of more than ±0.02 PPL from these targets typically indicate
a `transformers` or `lm-eval` version mismatch — check `requirements.txt`.

## Architecture

Four stages, one module per stage:

| Stage | Entry point | What it does |
|-------|-------------|--------------|
| 1. Hessian collection | `src/hessian/collect_all.py` | Monkey-patches `OlmoeSparseMoeBlock.forward` to capture routed per-expert activations, then accumulates `H = Xᵀ X` for each of the 2048 expert Hessians (16 layers × 64 experts × 2 projections: fused `gate_up_proj` and `down_proj`). |
| 2. Per-expert quantization | `src/quantize/quantize_all.py` | Wraps BlockLDLQ (`src/quantize/blockldlq.py`) around a trellis codec (`src/codes/`, `src/viterbi/`, `src/cuda/viterbi_kernel.py`). Emits a packed 2-bit bitstream per expert into `cache/quantized/L{00..15}/`. |
| 3. Install-time dequant | `src/eval/install_quantized.py` | Loads the 2-bit bitstream and rematerializes bf16 tensors inside a `transformers`-compatible OLMoE checkpoint. Runtime footprint is therefore bf16 (~14 GB), not 2-bit — see **Limitations**. |
| 4. Evaluation | `src/eval/run_ppl.py`, `src/eval/run_lm_eval.py` | WikiText-2 / C4 perplexity and zero-shot `hellaswag / arc_* / piqa` via `lm-evaluation-harness` 0.4.11. |

The two **per-layer H ablations** in Table 1 live in
`src/quantize/quantize_all_per_layer_H.py` and
`quantize_all_per_layer_weighted_H.py` — same trellis quantizer, different
calibration Hessian (unweighted / token-weighted mean over experts).

Two **exploratory analyses** used in the discussion (Section 5.3 / 5.4)
are also included:

- `src/analysis/measure_expert_kurtosis.py` — per-expert kurtosis
  distribution behind the reconciliation argument in Section 5.4.
  Reproduces Figure 3.
- `src/finetune/` — the LUT fine-tuning sweep of Section 5.3 (negative
  result; kept because it's the cheapest way for a reader to verify the
  conclusion themselves).

`src/tripwires/` contains the ~30 development-time sanity checks used
while building the pipeline (Viterbi correctness, RHT round-trip, BlockLDLQ
on real Hessians, etc.). Each file is documented at the top — they are
not needed for reproduction but are useful if you port the pipeline to a
different architecture.

## Calibration data

The C4 calibration sample is locked by seed (`seed=0`) **and** by token
IDs: the 2048 × 1024-token sample is stored as
`cache/calibration/tokens.npy` after the first run of
`src/hessian/prepare_calib.py`. This protects against Hugging Face
`datasets` version drift — "seed 0 on C4" is reproducible in theory but
not in practice once the upstream dataset is re-versioned. If you need
to regenerate from scratch, delete `cache/calibration/` and re-run; the
SHA-256 of the resulting tokens array is printed to stdout.

## Limitations

### Runtime memory does not match on-disk size

The 3.3 GB bitstream is dequantized back to bf16 at install time, so the
runtime footprint is ~14 GB, not 3.3 GB. On 12 GB VRAM devices,
`accelerate`'s `device_map` CPU-offloads the tail of the model
automatically; our PPL and lm-eval numbers are measured with this split.

### Native 2-bit MoE kernels are follow-up work

QTIP already ships native 2-bit kernels for dense Llama-family models with
>3× speedup over fp16. Porting them to OLMoE's routed-expert layout
requires:

1. Adapt the dense TLUT dequant kernel to the fused `gate_up_proj` layout
   OLMoE uses for its experts.
2. Wire the kernel into `OlmoeSparseMoeBlock.forward` so routed experts
   call the 2-bit kernel in place of the bf16 matmul.
3. End-to-end benchmarking: tokens/sec and peak VRAM against the bf16
   baseline on the same RTX 4080 Laptop hardware.

Estimated scope: 4–8 weeks of engineering on the same hardware. This is
a systems contribution with a different audience from the
calibration-and-quality paper, so it will ship as a companion technical
report.

## Repo layout

```
.
├── README.md                 this file
├── LICENSE                   Apache-2.0
├── NOTICE                    attribution, third-party components
├── CITATION.cff              machine-readable citation
├── requirements.txt          pinned for repro
├── scripts/
│   └── reproduce.sh          one-command repro (~18 hr)
├── src/
│   ├── hessian/              stage 1: per-expert H collection
│   ├── quantize/             stage 2: BlockLDLQ + trellis
│   ├── codes/                reference codebook implementations (1MAD/3INST/HYB)
│   ├── viterbi/              Viterbi trellis encoder (numpy ref + CUDA)
│   ├── cuda/                 custom CUDA kernels (Viterbi forward DP)
│   ├── rht/                  random Hadamard transform (incoherence)
│   ├── eval/                 stages 3–4: install + PPL + lm-eval
│   ├── models/               OLMoE adapter (fused-expert forward)
│   ├── analysis/             Section 5.4 kurtosis analysis
│   ├── finetune/             Section 5.3 LUT FT sweep (negative result)
│   ├── figures/              Figure 2 (routing), Figure 3 (kurtosis)
│   └── tripwires/            development-time sanity checks
└── fast-hadamard-transform/  upstream, BSD-3-Clause, built from source
```

The LaTeX source of the paper is tracked separately in a sister directly. Not visible here.

## Citation (Still publish pending)

```bibtex
@article{iyengar2026bcjrqat,
  title   = {BCJR-QAT: A Differentiable Relaxation of Trellis-Coded
             Weight Quantization},
  author  = {Iyengar, Venugopalan},
  year    = {2026},
  journal = {arXiv preprint}
}
```

Please also cite the upstream QTIP paper that this work is built on top of:

```bibtex
@inproceedings{tseng2024qtip,
  title     = {QTIP: Quantization with Trellises and Incoherence
               Processing},
  author    = {Tseng, Albert and Yao, Qingyao and Kuleshov, Volodymyr and
               De Sa, Christopher},
  booktitle = {NeurIPS},
  year      = {2024}
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) for the full license text and
[`NOTICE`](NOTICE) for attribution. This repository is an independent
clean-room implementation of the algorithms in the QTIP paper; no source
code from the upstream QTIP repository (which is GPL-3.0) is included or
linked.
