import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = encoder.embedding.padding_idx

    def forward(
        self,
        source_ids: torch.Tensor,  # Shape (batch, source_len)
        target_ids: torch.Tensor,  # Shape (batch, target_len)
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Run the encoder
        encoder_outputs, decoder_hidden = self.encoder(source_ids)
        # 2. Initialize mask, prev_token for the decoder
        mask = source_ids != self.pad_id
        prev_token = target_ids[:, 0]  # Init with <SOS> ids
        # 3. Run the decoding loop
        len_target_seq = target_ids.shape[1]
        all_logits: list[torch.Tensor] = []
        all_weights: list[torch.Tensor] = []
        for t in range(len_target_seq - 1):
            logits, decoder_hidden, weights = self.decoder(
                prev_token, decoder_hidden, encoder_outputs, mask
            )
            # Accumulate results
            all_logits.append(logits)
            all_weights.append(weights)
            # Apply teacher forcing based on a random experiment
            if torch.rand(1).item() < teacher_forcing_ratio:
                prev_token = target_ids[:, t + 1]
            else:
                prev_token = torch.argmax(input=logits, dim=1)
        # Convert accumulated results into torch.Tensor before return
        logits_stacked: torch.Tensor = torch.stack(
            all_logits, dim=1
        )  # Shape (batch, target_len-1, vocab_size)
        weights_stacked: torch.Tensor = torch.stack(
            all_weights, dim=1
        )  # Shape (batch, target_len-1, source_len)
        return logits_stacked, weights_stacked

    @torch.no_grad()
    def generate(
        self, source_ids: torch.Tensor, sos_id: int, max_len: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        encoder_outputs, decoder_hidden = self.encoder(source_ids)
        mask = source_ids != self.pad_id
        prev_token = torch.full(
            size=(source_ids.shape[0],), fill_value=sos_id, device=source_ids.device
        )
        all_tokens: list[torch.Tensor] = []
        all_weights: list[torch.Tensor] = []
        for _ in range(max_len):
            logits, decoder_hidden, weights = self.decoder(
                prev_token, decoder_hidden, encoder_outputs, mask
            )
            prev_token = torch.argmax(input=logits, dim=1)
            all_tokens.append(prev_token)
            all_weights.append(weights)
        self.train()
        tokens_stacked = torch.stack(all_tokens, dim=1)
        weights_stacked = torch.stack(all_weights, dim=1)
        return tokens_stacked, weights_stacked
