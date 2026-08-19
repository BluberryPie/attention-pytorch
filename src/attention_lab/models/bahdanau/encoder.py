import torch
import torch.nn as nn


class BiGRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_id
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.hidden_projection = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim
        )

    def forward(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(source_ids)
        gru_outputs, h_last = self.gru(embeddings)
        # Original paper uses only the backward-direction final state
        decoder_init_h = torch.tanh(self.hidden_projection(h_last[1]))
        return gru_outputs, decoder_init_h
