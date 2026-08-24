import argparse
import itertools
import logging
import sys
from pathlib import Path

import datasets
import torch
from rouge_score.rouge_scorer import RougeScorer
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_lab.data.dataset import GigawordDataset, make_collate_fn
from attention_lab.data.gigaword import load_gigaword
from attention_lab.data.tokenize import tokenize
from attention_lab.data.vocab import EOS, PAD, SOS, Vocab
from attention_lab.models import build_model
from attention_lab.models.seq2seq import Seq2Seq
from attention_lab.training.config import Config


def decode_until_eos(ids: list[int], vocab: Vocab, eos_id: int) -> str:
    try:
        # Slice ids up to the first occurence of <EOS>
        eos_idx: int = ids.index(eos_id)
        ids = ids[:eos_idx]
    except ValueError:
        # If <EOS> not found, leave ids unchanged
        pass
    return " ".join(vocab.decode(ids))


def main():
    # 1. Setup logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").propagate = False
    datasets.utils.logging.set_verbosity_warning()
    logger = logging.getLogger(__name__)

    # 2. Parse `--config` and `--checkpoint` from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint file", required=True
    )
    args = parser.parse_args()

    # 3. Load it into Config
    config_path = Path(args.config)
    try:
        config = Config.from_yaml(config_path)
    except FileNotFoundError:
        logger.error(f"File <{config_path.resolve()}> not found.")
        sys.exit(1)

    # 4. Build/Load the vocab, GigawordDataset and only the validation loader
    vocab_path: Path = Path(config.vocab_path) / "vocab.json"
    if vocab_path.exists():
        logger.info(f"Loading vocab from <{vocab_path.resolve()}>")
        vocab = Vocab.load(vocab_path)
    else:
        gigaword_train = load_gigaword("train")
        logger.info(f"<{vocab_path.resolve()}> not found. Building vocab from scratch.")
        articles, summaries = gigaword_train["article"], gigaword_train["summary"]
        tokenized_data = (
            tokenize(text) for text in itertools.chain(articles, summaries)
        )
        logger.info("This may take some time.")
        vocab = Vocab.build(tokenized_data, max_size=config.vocab_size)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)

    val_dataset = GigawordDataset(
        data=load_gigaword("validation"),
        vocab=vocab,
        max_source_len=config.max_source_len,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, collate_fn=make_collate_fn(pad_id=vocab.stoi[PAD])
    )

    # 5. Pick a device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device [{device}] available and selected.")

    # 6. Build/Load the model and its checkpoint
    try:
        seq2seq: Seq2Seq = build_model(vocab=vocab, config=config)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    seq2seq.load_state_dict(checkpoint["model"])
    seq2seq.to(device)

    # 7. Generation Loop
    sos_id, eos_id = vocab.stoi[SOS], vocab.stoi[EOS]
    rouge_scorer: RougeScorer = RougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    generated_list: list[str] = []
    reference_list: list[str] = []

    for source_ids, target_ids in tqdm(
        val_loader, desc="Generating summaries from validation split"
    ):
        source_ids = source_ids.to(device)
        tokens, weights = seq2seq.generate(
            source_ids=source_ids, sos_id=sos_id, max_len=config.max_target_len
        )
        # Inner loop over each example in the batch
        for i in range(tokens.shape[0]):
            generated: str = decode_until_eos(
                ids=tokens[i].tolist(), vocab=vocab, eos_id=eos_id
            )
            reference: str = decode_until_eos(
                ids=target_ids[i].tolist(), vocab=vocab, eos_id=eos_id
            )
            generated_list.append(generated)
            reference_list.append(reference)


if __name__ == "__main__":
    main()
