from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AttentionBase(nn.Module, ABC):
    """Common interface for attention mechanisms"""

    @abstractmethod
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the attention context vector & weights for a single decoding step.

        Args:
            decoder_hidden: Shape (batch, hidden_dim). Query.
            encoder_outputs: Shape (batch, seq_len, 2 * hidden_dim). Keys/Values (same tensor for both roles).
            mask: Shape (batch, seq_len). Indicates valid positions (True = real token / False = padding).

        Returns:
            context: Shape (batch, 2 * hidden_dim). Weighted sum of encoder outputs based on the attention distribution.
            weights: Shape (batch, seq_len). Normalized attention scores.
        """
        ...
