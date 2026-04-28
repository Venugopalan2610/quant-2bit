"""Sliding-window perplexity evaluation.

Standard recipe used by QuIP# / QTIP / GPTQ / AWQ:
- Tokenize the entire test stream once
- Slide a window of seq_len tokens (no overlap, stride = seq_len)
- For each window: run forward, sum next-token NLL across all positions
- PPL = exp(total_nll / total_tokens)

This is "non-causally-padded sliding window" — the simplest variant.
We do NOT use overlap for speed.
"""
import math
import time
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_perplexity(
    model,
    token_ids,
    seq_len=2048,
    device="cuda:0",
    max_windows=None,
    verbose=True,
    save_per_window_path=None,
):
    """Compute perplexity on a flat token stream.

    Args:
        model: causal LM with logits output
        token_ids: 1-D LongTensor or list of int, tokenized stream
        seq_len: window size (tokens per forward call)
        device: where to run forward (model must already be on this device)
        max_windows: cap on number of windows (None = all)
        verbose: print progress

    Returns:
        ppl: float, perplexity
        stats: dict with total_nll, total_tokens, n_windows, wall_seconds
    """
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    else:
        token_ids = token_ids.long()

    n_total = token_ids.numel()
    n_windows = n_total // seq_len
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    if verbose:
        print(f"  PPL eval: {n_windows} windows of {seq_len} tokens "
              f"(~{n_windows * seq_len:,} tokens total)")

    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    last_print = t0
    # Per-window NLL + token-count arrays for bootstrap noise estimation.
    per_window_nll = []
    per_window_tokens = []

    model.eval()
    for w in range(n_windows):
        start = w * seq_len
        end = start + seq_len
        window = token_ids[start:end].unsqueeze(0).to(device)  # (1, seq_len)

        out = model(input_ids=window)
        logits = out.logits[0, :-1, :]  # (seq_len-1, vocab)
        targets = window[0, 1:]          # (seq_len-1,)

        # With device_map="auto", logits may land on CPU — align devices.
        if logits.device != targets.device:
            targets = targets.to(logits.device)

        # NLL summed over the window's predictable tokens
        nll = F.cross_entropy(
            logits.float(), targets, reduction="sum",
        ).item()
        total_nll += nll
        total_tokens += targets.numel()
        per_window_nll.append(nll)
        per_window_tokens.append(int(targets.numel()))

        if verbose and time.time() - last_print > 10.0:
            elapsed = time.time() - t0
            rate = (w + 1) / elapsed
            eta = (n_windows - w - 1) / max(rate, 1e-6)
            partial_ppl = math.exp(total_nll / total_tokens)
            print(f"    window {w+1}/{n_windows}  "
                  f"partial PPL={partial_ppl:.3f}  ETA {eta:.0f}s")
            last_print = time.time()

    wall = time.time() - t0
    ppl = math.exp(total_nll / total_tokens)
    stats = {
        "ppl": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "n_windows": n_windows,
        "wall_seconds": wall,
    }
    if save_per_window_path is not None:
        import numpy as np
        np.savez(
            save_per_window_path,
            nll=np.array(per_window_nll, dtype=np.float64),
            tokens=np.array(per_window_tokens, dtype=np.int64),
        )
        if verbose:
            print(f"  per-window NLLs saved to {save_per_window_path}")
    if verbose:
        print(f"  PPL = {ppl:.4f} ({total_tokens:,} tokens, {wall:.0f}s)")
    return ppl, stats


def load_wikitext2_test(tokenizer, max_chars=None):
    """Load wikitext-2-raw test split, tokenize, return flat token ids."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    if max_chars is not None:
        text = text[:max_chars]
    print(f"  wikitext-2 test: {len(text):,} chars")
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"  tokenized to {len(ids):,} tokens")
    return torch.tensor(ids, dtype=torch.long)


def load_c4_validation_sample(tokenizer, target_tokens=300_000):
    """Stream C4 validation, tokenize until we have target_tokens."""
    from datasets import load_dataset
    print(f"  Streaming C4 validation, target ~{target_tokens:,} tokens")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    buf = []
    for doc in ds:
        ids = tokenizer.encode(doc["text"], add_special_tokens=False)
        buf.extend(ids)
        if len(buf) >= target_tokens:
            break
    print(f"  collected {len(buf):,} tokens")
    return torch.tensor(buf[:target_tokens], dtype=torch.long)