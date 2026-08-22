import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from attention_lab.models.seq2seq import Seq2Seq


def train_epoch(
    model: Seq2Seq,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    teacher_forcing_ratio: float,
    clip_norm: float,
    device: torch.device,
) -> float:
    """Run the seq2seq model for a single epoch.

    Args:
        model:
        dataloader:
        optimizer:
        criterion:
        teacher_forcing_ratio:
        clip_norm:
        device:

    Returns:
        The average loss over batches.
    """
    model.train()
    total_loss: torch.Tensor = torch.tensor(
        data=0.0, device=device
    )  # Accumulates running loss
    for source_ids, target_ids in dataloader:
        source_ids = source_ids.to(device)
        target_ids = target_ids.to(device)
        logits, _ = model(
            source_ids, target_ids, teacher_forcing_ratio=teacher_forcing_ratio
        )
        # Compute loss
        # CrossEntropyLoss expects (N, C, d1, ...) with target (N, d1, ...)
        # Permute logits to (batch, vocab_size, target_len - 1)
        logits = logits.permute(0, 2, 1)
        loss = criterion(logits, target_ids[:, 1:])
        total_loss += loss.detach()
        # Optimization step with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

    return (total_loss / len(dataloader)).item()
