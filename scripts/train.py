import argparse
import itertools
import logging
import sys
from pathlib import Path

import datasets
from torch.utils.data import DataLoader

from attention_lab.data.dataset import GigawordDataset, make_collate_fn
from attention_lab.data.gigaword import load_gigaword
from attention_lab.data.tokenize import tokenize
from attention_lab.data.vocab import PAD, Vocab
from attention_lab.training.config import TrainConfig


def main():
    # 1. Setup logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").propagate = False
    datasets.utils.logging.set_verbosity_warning()
    logger = logging.getLogger(__name__)

    # 2. Parse `--config <path>` from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    args = parser.parse_args()

    # 3. Load it into TrainConfig
    config_path = Path(args.config)
    try:
        train_config = TrainConfig.from_yaml(config_path)
    except FileNotFoundError:
        logger.error(f"File <{config_path.resolve()}> not found.")
        sys.exit(1)

    # 4. Build/Load the vocab, GigawordDataset and their loaders
    vocab_path: Path = Path(train_config.vocab_path) / "vocab.json"
    if vocab_path.exists():
        logger.info(f"Loading vocab from <{vocab_path.resolve()}>")
        vocab = Vocab.load(vocab_path)
    else:
        logger.info(f"<{vocab_path.resolve()}> not found. Building vocab from scratch.")
        gigaword_train = load_gigaword("train")
        articles, summaries = gigaword_train["article"], gigaword_train["summary"]
        tokenized_data = (
            tokenize(text) for text in itertools.chain(articles, summaries)
        )
        logger.info("This may take some time.")
        vocab = Vocab.build(tokenized_data, max_size=train_config.vocab_size)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)

    train_dataset = GigawordDataset(
        data=load_gigaword("train"),
        vocab=vocab,
        max_source_len=train_config.max_source_len,
    )
    val_dataset = GigawordDataset(
        data=load_gigaword("validation"),
        vocab=vocab,
        max_source_len=train_config.max_source_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(pad_id=vocab.stoi[PAD]),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, collate_fn=make_collate_fn(pad_id=vocab.stoi[PAD])
    )

    # (TODO) 5. Construct the model(encoder, decoder, seq2seq) based on config.variant

    # (TODO) 6. Construct the optimizer and criterion

    # (TODO) 7. Pick a device

    # (TODO) 8. Run the training loop over config.num_epochs


if __name__ == "__main__":
    main()
