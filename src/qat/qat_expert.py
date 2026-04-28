"""QATExpert: QAT analog of QuantizedExpert (LUT-FT) for OLMoE.

Mirrors the OLMoE expert forward:
    gate_up_proj: (2 * intermediate, hidden)  — fused gate+up, chunk(2) on output
    down_proj:    (hidden, intermediate)
    out(x) = down(SiLU(gate) * up)  where  gate,up = chunk(gate_up @ x)

Trainable parameters: W_latent_gate_up, W_latent_down (two dense fp32 tensors).
Frozen: sign vectors, codebook. W_q is derived each forward via TrellisQuantSTE.

This is the dual of QuantizedExpert in src.finetune.quant_expert:
    QuantizedExpert: trainable LUT, frozen walks (PTQ-locked Viterbi solution)
    QATExpert:       trainable W_latent, walks re-derived each step via Viterbi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.qat.quantized_linear import QuantizedLinear


class QATExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, codebook_gpu,
                 seed=0, reencode_every_n_steps=1,
                 device="cuda", dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up = QuantizedLinear(
            in_features=hidden_size,
            out_features=2 * intermediate_size,
            codebook_gpu=codebook_gpu,
            bias=False,
            seed=seed,
            reencode_every_n_steps=reencode_every_n_steps,
            device=device, dtype=dtype,
        )
        self.down = QuantizedLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            codebook_gpu=codebook_gpu,
            bias=False,
            seed=seed + 10_000,
            reencode_every_n_steps=reencode_every_n_steps,
            device=device, dtype=dtype,
        )

    @classmethod
    def from_weights(cls, W_gate_up_fp16, W_down_fp16, codebook_gpu,
                     seed=0, reencode_every_n_steps=1,
                     device="cuda", dtype=torch.float32):
        """Build a QATExpert with W_latent initialized from FP16 teacher weights.

        Args:
            W_gate_up_fp16: (2 * intermediate, hidden) tensor — fused gate+up
            W_down_fp16:    (hidden, intermediate) tensor
        """
        two_inter, hidden = W_gate_up_fp16.shape
        assert two_inter % 2 == 0
        intermediate = two_inter // 2
        assert W_down_fp16.shape == (hidden, intermediate), (
            f"down shape {tuple(W_down_fp16.shape)} != expected "
            f"{(hidden, intermediate)}"
        )
        expert = cls(hidden_size=hidden, intermediate_size=intermediate,
                     codebook_gpu=codebook_gpu, seed=seed,
                     reencode_every_n_steps=reencode_every_n_steps,
                     device=device, dtype=dtype)
        expert.gate_up.load_fp_weights(W_gate_up_fp16)
        expert.down.load_fp_weights(W_down_fp16)
        return expert

    def forward(self, x):
        """x: (n_tokens, hidden) → (n_tokens, hidden)."""
        gate_up = self.gate_up(x)                              # (n, 2*inter)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp = F.silu(gate) * up
        return self.down(mlp)

    def eval_forward(self, x):
        """Inference path: uses cached W_q on both QLs. Requires primed cache."""
        gate_up = self.gate_up.eval_forward(x)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp = F.silu(gate) * up
        return self.down.eval_forward(mlp)

    def prime_cache(self):
        self.gate_up.prime_cache()
        self.down.prime_cache()

    def trainable_parameters(self):
        return [self.gate_up.W_latent, self.down.W_latent]
