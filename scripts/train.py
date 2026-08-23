import argparse
import itertools
import logging
import sys
from pathlib import Path

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_lab.data.dataset import GigawordDataset, make_collate_fn
from attention_lab.data.gigaword import load_gigaword
from attention_lab.data.tokenize import tokenize
from attention_lab.data.vocab import PAD, Vocab
from attention_lab.models import build_model
from attention_lab.models.seq2seq import Seq2Seq
from attention_lab.training.config import TrainConfig
from attention_lab.training.loop import evaluate, train_epoch


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

    # 4. Build/Load the vocab, GigawordDataset and only the validation loader
    vocab_path: Path = Path(train_config.vocab_path) / "vocab.json"
    gigaword_train = load_gigaword("train")
    if vocab_path.exists():
        logger.info(f"Loading vocab from <{vocab_path.resolve()}>")
        vocab = Vocab.load(vocab_path)
    else:
        logger.info(f"<{vocab_path.resolve()}> not found. Building vocab from scratch.")
        articles, summaries = gigaword_train["article"], gigaword_train["summary"]
        tokenized_data = (
            tokenize(text) for text in itertools.chain(articles, summaries)
        )
        logger.info("This may take some time.")
        vocab = Vocab.build(tokenized_data, max_size=train_config.vocab_size)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)

    val_dataset = GigawordDataset(
        data=load_gigaword("validation"),
        vocab=vocab,
        max_source_len=train_config.max_source_len,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, collate_fn=make_collate_fn(pad_id=vocab.stoi[PAD])
    )

    # 5. Construct the model(encoder, decoder, seq2seq) based on config.variant
    try:
        seq2seq: Seq2Seq = build_model(vocab=vocab, config=train_config)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    # 6. Pick a device and move the model to it
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device [{device}] available and selected.")
    seq2seq = seq2seq.to(device)

    # 7. Construct the optimizer and criterion
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=train_config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi[PAD])

    # 8. Run the training loop over config.num_epochs
    Path(train_config.checkpoint_path).mkdir(parents=True, exist_ok=True)
    for epoch in tqdm(range(train_config.num_epochs), desc="Epochs"):
        # Construct the train dataset and loader
        train_dataset = GigawordDataset(
            data=gigaword_train.shuffle().select(range(train_config.subset_size)),
            vocab=vocab,
            max_source_len=train_config.max_source_len,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=make_collate_fn(pad_id=vocab.stoi[PAD]),
        )
        # Run single train epoch and evaluate
        train_loss: float = train_epoch(
            model=seq2seq,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            teacher_forcing_ratio=train_config.teacher_forcing_ratio,
            clip_norm=train_config.gradient_clip_norm,
            device=device,
        )
        eval_loss: float = evaluate(
            model=seq2seq, dataloader=val_loader, criterion=criterion, device=device
        )
        # Checkpoint for every epoch
        checkpoint: dict = {
            "model": seq2seq.state_dict(),
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        }
        torch.save(
            checkpoint, Path(train_config.checkpoint_path) / f"epoch_{epoch + 1}.pt"
        )


if __name__ == "__main__":
    main()
