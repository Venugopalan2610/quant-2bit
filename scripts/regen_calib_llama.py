"""Re-tokenize C4 calibration with a target model's tokenizer.

OLMoE and Llama have different vocabularies (100K vs 128K), so the existing
cache/calibration/tokens.npy (tokenized for OLMoE) can't be used for Llama.
This script streams C4 English, tokenizes with the target tokenizer, packs
into (n_seqs, seq_len) int32, writes to disk.

Run:
    python3 scripts/regen_calib_llama.py \\
        --model meta-llama/Llama-3.2-1B \\
        --out cache/calibration/tokens_llama.npy \\
        --n-seqs 512 --seq-len 2048
"""
import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID for tokenizer")
    parser.add_argument("--out", required=True, help="Output .npy path")
    parser.add_argument("--n-seqs", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Loading tokenizer: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = getattr(tok, "vocab_size", len(tok))
    print(f"  vocab_size = {vocab}", flush=True)

    target_tokens = args.n_seqs * args.seq_len
    print(f"Streaming C4 until we have {target_tokens:,} tokens ({args.n_seqs} × {args.seq_len})...",
          flush=True)

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    rng = np.random.default_rng(args.seed)

    # Concatenate all tokens with BOS between documents, then chunk.
    buf = []
    n_collected = 0
    n_docs = 0
    for ex in ds:
        ids = tok.encode(ex["text"], add_special_tokens=False)
        if len(ids) == 0:
            continue
        if tok.bos_token_id is not None:
            buf.append(tok.bos_token_id)
        buf.extend(ids)
        n_collected += len(ids) + (1 if tok.bos_token_id is not None else 0)
        n_docs += 1
        if n_docs % 500 == 0:
            print(f"  {n_docs} docs, {n_collected:,} tokens", flush=True)
        if n_collected >= target_tokens + args.seq_len:  # small safety margin
            break

    arr = np.array(buf[:args.n_seqs * args.seq_len], dtype=np.int32)
    arr = arr.reshape(args.n_seqs, args.seq_len)
    print(f"Packed: {arr.shape} {arr.dtype}  from {n_docs} C4 docs", flush=True)

    np.save(args.out, arr)
    meta_path = os.path.splitext(args.out)[0].rstrip(".npy") + ".meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"vocab_size: {vocab}\n")
        f.write(f"n_seqs: {args.n_seqs}\n")
        f.write(f"seq_len: {args.seq_len}\n")
        f.write(f"total_tokens: {args.n_seqs * args.seq_len}\n")
        f.write(f"n_docs: {n_docs}\n")
    print(f"Saved: {args.out}  (meta: {meta_path})", flush=True)


if __name__ == "__main__":
    main()
