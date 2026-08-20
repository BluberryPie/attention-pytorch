import pytest
import torch

from attention_lab.attention.bahdanau import BahdanauAttention

BATCH_SIZE: int = 4
HIDDEN_DIM: int = 8
ATTN_DIM: int = 4
SEQ_LEN: int = 4


@pytest.fixture
def bahdanau_attention() -> BahdanauAttention:
    return BahdanauAttention(hidden_dim=HIDDEN_DIM, attn_dim=ATTN_DIM)


@pytest.fixture
def sample_decoder_hidden() -> torch.Tensor:
    """decoder_hidden tensor of shape (batch, hidden_dim)"""
    return torch.randn(size=(BATCH_SIZE, HIDDEN_DIM))


@pytest.fixture
def sample_encoder_outputs() -> torch.Tensor:
    """encoder_outputs tensor of shape (batch, seq_len, hidden_dim * 2)"""
    return torch.randn(size=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM * 2))


def test_bahdanau_attention_output_shapes(
    bahdanau_attention, sample_decoder_hidden, sample_encoder_outputs
):
    mask: torch.Tensor = torch.full(size=(BATCH_SIZE, SEQ_LEN), fill_value=True)
    context, weights = bahdanau_attention(
        sample_decoder_hidden, sample_encoder_outputs, mask
    )
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM * 2)
    assert weights.shape == (BATCH_SIZE, SEQ_LEN)


def test_bahdanau_attention_weights_sum_to_one(
    bahdanau_attention, sample_decoder_hidden, sample_encoder_outputs
):
    mask: torch.Tensor = torch.full(size=(BATCH_SIZE, SEQ_LEN), fill_value=True)
    _, weights = bahdanau_attention(sample_decoder_hidden, sample_encoder_outputs, mask)
    weights_sum = torch.sum(weights, dim=1)
    # Use allclose over (==) since weights are FP softmax outputs
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum))


def test_bahdanau_attention_masks_padding_positions(
    bahdanau_attention, sample_decoder_hidden, sample_encoder_outputs
):
    mask: torch.Tensor = torch.tensor(
        [[True, True, False, False] for _ in range(BATCH_SIZE)], dtype=torch.bool
    )
    _, weights = bahdanau_attention(sample_decoder_hidden, sample_encoder_outputs, mask)
    padding_weights = weights[:, 2:]
    assert torch.allclose(padding_weights, torch.zeros_like(padding_weights))
    # Assert weights still sum to 1 with paddings
    weights_sum = torch.sum(weights, dim=1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum))
