import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


@torch.no_grad()
def plot_attention_heatmap(
    source_tokens: list[str], generated_tokens: list[str], weights: torch.Tensor
) -> Figure:
    weights = weights.detach().cpu().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(weights)
    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, rotation=90)
    ax.set_yticks(range(len(generated_tokens)))
    ax.set_yticklabels(generated_tokens)
    fig.colorbar(im, ax=ax)
    return fig
