import torch
import torch.nn as nn

from attention_lab.attention.base import AttentionBase


class BahdanauDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        attention: AttentionBase,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_id
        )
        self.attention = attention
        self.gru = nn.GRUCell(
            input_size=embed_dim + hidden_dim * 2, hidden_size=hidden_dim
        )
        self.output_layer = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(
        self,
        prev_token: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, weights = self.attention(decoder_hidden, encoder_outputs, mask)
        gru_in = torch.cat(tensors=(self.embedding(prev_token), context), dim=1)
        next_hidden = self.gru(gru_in, decoder_hidden)
        logits = self.output_layer(next_hidden)
        return logits, next_hidden, weights
