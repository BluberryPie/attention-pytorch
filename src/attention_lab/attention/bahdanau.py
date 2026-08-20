import torch
import torch.nn as nn

from .base import AttentionBase


class BahdanauAttention(AttentionBase):
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W1 = nn.Linear(in_features=hidden_dim, out_features=attn_dim)
        self.W2 = nn.Linear(in_features=hidden_dim * 2, out_features=attn_dim)
        self.v = nn.Linear(in_features=attn_dim, out_features=1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Compute raw(unmasked) scores
        # score = v^T @ tanh(W1 @ s_{t-1} + W2 @ h_i)
        scores: torch.Tensor = self.v(
            torch.tanh(self.W1(decoder_hidden).unsqueeze(1) + self.W2(encoder_outputs))
        ).squeeze(-1)  # Shape (batch, seq_len)

        # 2. Mask padding indices(mask value False) to -inf
        scores[~mask] = -torch.inf

        # 3. Softmax over masked scores
        weights = torch.softmax(input=scores, dim=1)  # Shape (batch, seq_len)

        # 4. Take the weighted sum
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights
