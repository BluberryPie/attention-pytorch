import pytest
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from attention_lab.attention.bahdanau import BahdanauAttention
from attention_lab.models.bahdanau.decoder import BahdanauDecoder
from attention_lab.models.bahdanau.encoder import BiGRUEncoder
from attention_lab.models.seq2seq import Seq2Seq
from attention_lab.training.loop import evaluate, train_epoch

VOCAB_SIZE: int = 8
EMBED_DIM: int = 4
HIDDEN_DIM: int = 8
ATTN_DIM: int = 4
BATCH_SIZE: int = 2


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
def dataloader() -> DataLoader:
    class FakeDataset(Dataset):
        def __init__(self):
            self.source_ids: torch.Tensor = torch.tensor(
                [[4, 5, 6, 7], [6, 5, 0, 0]], dtype=torch.long
            )
            self.target_ids: torch.Tensor = torch.tensor(
                [[2, 4, 3, 0], [2, 5, 5, 3]], dtype=torch.long
            )

        def __len__(self):
            return self.source_ids.shape[0]

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.source_ids[idx], self.target_ids[idx]

    return DataLoader(FakeDataset(), batch_size=BATCH_SIZE)


@pytest.fixture
def optimizer(small_bahdanau_seq2seq) -> Optimizer:
    return torch.optim.Adam(small_bahdanau_seq2seq.parameters(), lr=1e-3)


@pytest.fixture
def criterion(small_bahdanau_seq2seq) -> nn.Module:
    pad_id: int = small_bahdanau_seq2seq.pad_id
    return nn.CrossEntropyLoss(ignore_index=pad_id)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_train_epoch_returns_float(
    small_bahdanau_seq2seq, dataloader, optimizer, criterion, device
):
    loss = train_epoch(
        model=small_bahdanau_seq2seq,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        teacher_forcing_ratio=0.5,
        clip_norm=5.0,
        device=device,
    )
    assert isinstance(loss, float)


def test_train_epoch_updates_model_parameters(
    small_bahdanau_seq2seq, dataloader, optimizer, criterion, device
):
    # Make sure to use `clone` to get snapshots not references
    params_before_train = {
        k: v.clone() for k, v in small_bahdanau_seq2seq.state_dict().items()
    }
    _ = train_epoch(
        model=small_bahdanau_seq2seq,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        teacher_forcing_ratio=0.5,
        clip_norm=5.0,
        device=device,
    )
    params_after_train = small_bahdanau_seq2seq.state_dict()
    for k, v in params_before_train.items():
        assert not torch.equal(v, params_after_train[k])


def test_evaluate_returns_float(small_bahdanau_seq2seq, dataloader, criterion, device):
    loss = evaluate(
        model=small_bahdanau_seq2seq,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )
    assert isinstance(loss, float)


def test_evaluate_does_not_update_model_parameters(
    small_bahdanau_seq2seq, dataloader, criterion, device
):
    params_before_train = {
        k: v.clone() for k, v in small_bahdanau_seq2seq.state_dict().items()
    }
    _ = evaluate(
        model=small_bahdanau_seq2seq,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )
    params_after_train = small_bahdanau_seq2seq.state_dict()
    for k, v in params_before_train.items():
        assert torch.equal(v, params_after_train[k])


def test_evaluate_restores_train_mode(
    small_bahdanau_seq2seq, dataloader, criterion, device
):
    _ = evaluate(
        model=small_bahdanau_seq2seq,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )
    assert small_bahdanau_seq2seq.training
