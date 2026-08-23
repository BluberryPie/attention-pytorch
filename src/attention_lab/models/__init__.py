from attention_lab.attention import get_attention
from attention_lab.data.vocab import PAD, Vocab
from attention_lab.models.bahdanau.decoder import BahdanauDecoder
from attention_lab.models.bahdanau.encoder import BiGRUEncoder
from attention_lab.models.seq2seq import Seq2Seq
from attention_lab.training.config import TrainConfig


def build_model(vocab: Vocab, config: TrainConfig) -> Seq2Seq:
    variant: str = config.variant
    if variant == "bahdanau":
        encoder = BiGRUEncoder(
            vocab_size=vocab.size,
            embed_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            pad_id=vocab.stoi[PAD],
        )
        attention = get_attention(
            name=variant, hidden_dim=config.hidden_dim, attn_dim=config.attention_dim
        )
        decoder = BahdanauDecoder(
            vocab_size=vocab.size,
            embed_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            attention=attention,
            pad_id=vocab.stoi[PAD],
        )
        return Seq2Seq(encoder, decoder)
    else:
        raise ValueError(f"variant [{variant}] is not supported")
