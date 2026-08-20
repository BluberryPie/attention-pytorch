import pytest
import torch

from attention_lab.attention.bahdanau import BahdanauAttention
from attention_lab.models.bahdanau.decoder import BahdanauDecoder
from attention_lab.models.bahdanau.encoder import BiGRUEncoder

VOCAB_SIZE: int = 8
EMBED_DIM: int = 8
HIDDEN_DIM: int = 8
ATTN_DIM: int = 4
BATCH_SIZE: int = 2
SEQ_LEN: int = 4


@pytest.fixture
def small_bigru_encoder():
    encoder = BiGRUEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    )
    return encoder


@pytest.fixture
def sample_source_ids():
    return torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)


def test_bigru_encoder_output_shapes(small_bigru_encoder, sample_source_ids):
    hidden_dim = small_bigru_encoder.gru.hidden_size
    encoder_outputs, decoder_init_h = small_bigru_encoder(sample_source_ids)
    # Should be (Batch) x (Seq) x (2 x H)
    assert encoder_outputs.shape == (2, 4, 2 * hidden_dim)
    # Should be (Batch) x (H)
    assert decoder_init_h.shape == (2, hidden_dim)


def test_bigru_encoder_decoder_init_h_is_tanh_bounded(
    small_bigru_encoder, sample_source_ids
):
    _, decoder_init_h = small_bigru_encoder(sample_source_ids)
    assert torch.all(decoder_init_h.abs() < 1)


def test_bigru_encoder_zeros_pad_embedding(small_bigru_encoder):
    pad_id: int = small_bigru_encoder.embedding.padding_idx
    assert torch.all(small_bigru_encoder.embedding.weight[pad_id] == 0)


def test_bigru_encoder_uses_backward_direction_for_decoder_init(
    small_bigru_encoder, sample_source_ids
):
    # Call gru directly on the embedded input to get h_n's independently
    embeddings = small_bigru_encoder.embedding(sample_source_ids)
    _, h_last = small_bigru_encoder.gru(embeddings)
    h_last_forward, h_last_backward = h_last[0], h_last[1]
    # Call the full encoder
    _, decoder_init_h = small_bigru_encoder(sample_source_ids)
    assert torch.equal(
        decoder_init_h,
        torch.tanh(small_bigru_encoder.hidden_projection(h_last_backward)),
    )
    assert not torch.equal(
        decoder_init_h,
        torch.tanh(small_bigru_encoder.hidden_projection(h_last_forward)),
    )


@pytest.fixture
def small_bahdanau_attention() -> BahdanauAttention:
    return BahdanauAttention(hidden_dim=HIDDEN_DIM, attn_dim=ATTN_DIM)


@pytest.fixture
def small_bahdanau_decoder(small_bahdanau_attention) -> BahdanauDecoder:
    return BahdanauDecoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        attention=small_bahdanau_attention,
    )


@pytest.fixture
def sample_prev_token() -> torch.Tensor:
    return torch.tensor([1, 2], dtype=torch.long)


@pytest.fixture
def sample_decoder_hidden() -> torch.Tensor:
    return torch.randn(size=(BATCH_SIZE, HIDDEN_DIM))


@pytest.fixture
def sample_encoder_outputs() -> torch.Tensor:
    return torch.randn(size=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM * 2))


@pytest.fixture
def sample_mask() -> torch.Tensor:
    return torch.tensor(
        [[True, True, False, False] for _ in range(BATCH_SIZE)], dtype=torch.bool
    )


def test_bahdanau_decoder_output_shapes(
    small_bahdanau_decoder: BahdanauDecoder,
    sample_prev_token: torch.Tensor,
    sample_decoder_hidden: torch.Tensor,
    sample_encoder_outputs: torch.Tensor,
    sample_mask: torch.Tensor,
):
    logits, next_hidden, weights = small_bahdanau_decoder(
        sample_prev_token, sample_decoder_hidden, sample_encoder_outputs, sample_mask
    )
    assert logits.shape == (BATCH_SIZE, VOCAB_SIZE)
    assert next_hidden.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert weights.shape == (BATCH_SIZE, SEQ_LEN)


def test_bahdanau_decoder_zeros_pad_embedding(small_bahdanau_decoder):
    pad_id: int = small_bahdanau_decoder.embedding.padding_idx
    assert torch.all(small_bahdanau_decoder.embedding.weight[pad_id] == 0)
