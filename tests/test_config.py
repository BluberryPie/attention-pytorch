import textwrap

import pytest

from attention_lab.training.config import TrainConfig


@pytest.mark.parametrize(
    "field, expected",
    [
        ("embedding_dim", 256),
        ("hidden_dim", 512),
        ("attention_dim", 512),
        ("vocab_size", 50_000),
        ("subset_size", 300_000),
        ("max_source_len", 100),
    ],
)
def test_train_config_defaults(field: str, expected: int):
    train_config = TrainConfig()
    assert getattr(train_config, field) == expected


@pytest.mark.parametrize(
    "field, expected",
    [
        ("embedding_dim", 999),
        ("hidden_dim", 512),
        ("attention_dim", 999),
        ("vocab_size", 50_000),
        ("subset_size", 999),
        ("max_source_len", 100),
    ],
)
def test_train_config_from_yaml_overrides_defaults(tmp_path, field: str, expected: int):
    yaml_config = """\
    embedding_dim: 999
    attention_dim: 999
    subset_size: 999
    """
    # Remove leading whitespaces from string
    yaml_config = textwrap.dedent(yaml_config)

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml_config)
    train_config = TrainConfig.from_yaml(config_path)
    assert getattr(train_config, field) == expected
