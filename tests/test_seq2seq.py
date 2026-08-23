import pytest
import torch

from attention_lab.attention.bahdanau import BahdanauAttention
from attention_lab.models.bahdanau.decoder import BahdanauDecoder
from attention_lab.models.bahdanau.encoder import BiGRUEncoder
from attention_lab.models.seq2seq import Seq2Seq

VOCAB_SIZE: int = 8
EMBED_DIM: int = 4
HIDDEN_DIM: int = 4
ATTN_DIM: int = 4
BATCH_SIZE: int = 2
SOURCE_LEN: int = 4
TARGET_LEN: int = 4


@pytest.fixture
def small_bigru_encoder() -> BiGRUEncoder:
    return BiGRUEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
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
def small_bahdanau_seq2seq(small_bigru_encoder, small_bahdanau_decoder) -> Seq2Seq:
    return Seq2Seq(small_bigru_encoder, small_bahdanau_decoder)


@pytest.fixture
def sample_source_ids() -> torch.Tensor:
    return torch.tensor([[4, 5, 6, 7], [6, 5, 0, 0]], dtype=torch.long)


@pytest.fixture
def sample_target_ids() -> torch.Tensor:
    return torch.tensor([[2, 4, 3, 0], [2, 5, 5, 3]], dtype=torch.long)


def test_seq2seq_output_shapes(
    small_bahdanau_seq2seq, sample_source_ids, sample_target_ids
):
    all_logits, all_weights = small_bahdanau_seq2seq(
        sample_source_ids, sample_target_ids
    )
    # Should be (batch, target_len - 1, vocab_size)
    assert all_logits.shape == (2, 3, VOCAB_SIZE)
    # Should be (batch, target_len - 1, source_len)
    assert all_weights.shape == (2, 3, 4)


class FakeDecoder:
    """Fake decoder class to record prev_tokens"""

    def __init__(self):
        self.prev_tokens: list[torch.Tensor] = []

    def __call__(self, prev_token, *args):
        self.prev_tokens.append(prev_token)
        fake_logits = torch.randn(size=(BATCH_SIZE, VOCAB_SIZE))
        fake_decoder_hidden = torch.randn(size=(BATCH_SIZE, HIDDEN_DIM))
        fake_weights = torch.randn(size=(BATCH_SIZE, SOURCE_LEN))
        return fake_logits, fake_decoder_hidden, fake_weights


def test_seq2seq_full_teacher_forcing_uses_ground_truth(
    small_bigru_encoder, sample_source_ids, sample_target_ids
):
    fake_decoder = FakeDecoder()
    seq2seq = Seq2Seq(small_bigru_encoder, fake_decoder)
    seq2seq(sample_source_ids, sample_target_ids, teacher_forcing_ratio=1.0)
    assert torch.equal(
        torch.stack(fake_decoder.prev_tokens, dim=1), sample_target_ids[:, :-1]
    )


def test_seq2seq_no_teacher_forcing_uses_own_predictions(
    small_bigru_encoder, sample_source_ids, sample_target_ids
):
    fake_decoder = FakeDecoder()
    seq2seq = Seq2Seq(small_bigru_encoder, fake_decoder)
    all_logits, _ = seq2seq(
        sample_source_ids, sample_target_ids, teacher_forcing_ratio=0.0
    )
    recorded_prev_tokens = torch.stack(fake_decoder.prev_tokens, dim=1)
    # First recorded value should be sample_target_ids[:, 0]
    assert torch.equal(recorded_prev_tokens[:, 0], sample_target_ids[:, 0])
    # Remaining recorded values should be the argmax results
    assert torch.equal(
        recorded_prev_tokens[:, 1:], torch.argmax(all_logits[:, :-1], dim=-1)
    )


@pytest.fixture
def sos_id() -> int:
    return 3


@pytest.fixture
def max_len() -> int:
    return 10


def test_generate_output_shapes(
    small_bahdanau_seq2seq, sample_source_ids, sos_id, max_len
):
    batch_size: int = sample_source_ids.shape[0]
    source_len: int = sample_source_ids.shape[1]
    tokens, weights = small_bahdanau_seq2seq.generate(
        source_ids=sample_source_ids, sos_id=sos_id, max_len=max_len
    )
    assert tokens.shape == (batch_size, max_len)
    assert weights.shape == (batch_size, max_len, source_len)


def test_generate_does_not_track_gradients(
    small_bahdanau_seq2seq, sample_source_ids, sos_id, max_len
):
    tokens, weights = small_bahdanau_seq2seq.generate(
        source_ids=sample_source_ids, sos_id=sos_id, max_len=max_len
    )
    assert tokens.requires_grad is False
    assert weights.requires_grad is False
    for param in small_bahdanau_seq2seq.parameters():
        assert param.grad is None


def test_generate_restores_train_mode(
    small_bahdanau_seq2seq, sample_source_ids, sos_id, max_len
):
    _ = small_bahdanau_seq2seq.generate(
        source_ids=sample_source_ids, sos_id=sos_id, max_len=max_len
    )
    assert small_bahdanau_seq2seq.training


def test_generate_uses_greedy_decoding(
    small_bigru_encoder, sample_source_ids, sos_id, max_len
):
    fake_decoder = FakeDecoder()
    seq2seq = Seq2Seq(small_bigru_encoder, fake_decoder)
    tokens, _ = seq2seq.generate(sample_source_ids, sos_id=sos_id, max_len=max_len)
    batch_size = sample_source_ids.shape[0]
    # Initial prev_tokens should be the sos_id filled tensor.
    assert torch.equal(
        fake_decoder.prev_tokens[0], torch.full(size=(batch_size,), fill_value=sos_id)
    )
    for t in range(max_len - 1):
        assert torch.equal(tokens[:, t], fake_decoder.prev_tokens[t + 1])
