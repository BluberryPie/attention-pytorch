import pytest
import torch

from attention_lab.models.bahdanau.encoder import BiGRUEncoder


@pytest.fixture
def small_bigru_encoder():
    vocab_size = 8
    embed_dim = 8
    hidden_dim = 8
    encoder = BiGRUEncoder(
        vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
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
