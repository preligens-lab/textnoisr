"""Generate figures for the documentation.

## Pre-requisites

You'll need to install the following packages:

```sh
pip install matplotlib nlpaug
```

If you don't have [Roboto](https://fonts.google.com/specimen/Roboto) installed, the
default font will be used.

## Usage

From the root of the project, run:

```sh
python scripts/generate_figures.py
```

It will download the `rotten_tomatoes` dataset and generate the figures in
`docs/images/`. For more options, see:

```sh
python scripts/generate_figures.py --help
```
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from datasets import load_dataset
from evaluate import load
from nlpaug.augmenter.char import RandomCharAug

from textnoisr.noise import CharNoiseAugmenter
from textnoisr.noise_unbiasing import MAX_SWAP_LEVEL

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
cer = load("cer")

ACTIONS = ("delete", "insert", "substitute", "swap")
STYLE_DIR = Path(__file__).parent.parent / "styles"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "images"
DEFAULT_DATASET = "rotten_tomatoes"
DEFAULT_SPLIT = "train"
DEFAULT_N_SAMPLES = 17


def max_level(actions):
    if "swap" in actions:
        return MAX_SWAP_LEVEL / 1.052 - 0.005
    return 1


def get_cer_from_nlpaug(data, noise_level, actions):
    start = time.time()
    augmented = RandomCharAug(
        actions[0],
        aug_char_p=noise_level,
        aug_char_min=0,
        aug_char_max=None,
        aug_word_min=0,
        aug_word_max=None,
    ).augment(data)
    elapsed = time.time() - start
    return {
        "nlpaug_cer": cer.compute(predictions=augmented, references=data),
        "nlpaug_t": elapsed,
    }


def get_cer_from_textnoisr(data, noise_level, actions):
    if noise_level > max_level(actions):
        return {"textnoisr_cer": None, "textnoisr_t": None}
    start = time.time()
    augmented = CharNoiseAugmenter(
        noise_level=noise_level, actions=actions, seed=43
    ).add_noise(data)
    elapsed = time.time() - start
    return {
        "textnoisr_cer": cer.compute(predictions=augmented, references=data),
        "textnoisr_t": elapsed,
    }


def run_single_benchmark(data, action, n_samples=DEFAULT_N_SAMPLES):
    start = time.time()
    df = pd.DataFrame(
        [
            {
                "noise_level": noise_level,
                **get_cer_from_nlpaug(data, noise_level, [action]),
                **get_cer_from_textnoisr(data, noise_level, [action]),
                "cer": noise_level if noise_level <= max_level([action]) else None,
            }
            for noise_level in np.linspace(0, 1, num=n_samples)
        ]
    )
    elapsed = time.time() - start
    logger.info(
        f"Benchmark ran for action '{action}' and {n_samples} noise levels in"
        f" {elapsed:.2f} seconds"
    )
    return df


def run_benchmark(data, actions=ACTIONS, n_samples=DEFAULT_N_SAMPLES):
    return {
        action: run_single_benchmark(data, action, n_samples=n_samples)
        for action in actions
    }


def plot_cer(df, ax, actions):
    ax.plot(df.noise_level, df.textnoisr_cer, label="textnoisr")
    ax.plot(df.noise_level, df.nlpaug_cer, label="nlpaug")
    ax.axline(
        (0, 0),
        slope=1,
        color="#aaa",
        label="perfect calibration",
        marker=None,
        zorder=-1,
    )

    ax.set_title("Correctness")

    ax.set_ylabel("Character error rate")
    ax.set_xlabel("Noise level")
    ax.set(ylim=(0, ylim_max := 1.05), xlim=(0, 1))
    if "swap" in actions:
        ax.axhspan(max_level(actions), ylim_max)
        ax.text(
            0.01,
            max_level(actions) + 0.01,
            "Max. theoretical CER\nfor `swap` action",
            alpha=0.7,
        )

    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    return ax


def plot_time(df, ax, actions=()):
    ax.plot(df.noise_level, df.textnoisr_t, label="textnoisr")
    ax.plot(df.noise_level, df.nlpaug_t, label="nlpaug")

    ax.set_title("Execution time")

    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_xlabel("Noise level")
    ax.set(ylim=(0, None), xlim=(0, 1))
    if "swap" in actions:
        ax.axvspan(max_level(actions), 1)
        ax.text(
            max_level(actions) + 0.01,
            0.01,
            "Max. theoretical CER\nfor `swap` action",
            alpha=0.7,
        )
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    return ax


def generate_plot(
    df,
    action,
    theme,
    output_dir,
    style_dir=STYLE_DIR,
    dataset_name=None,
    split=None,
    dataset_size=None,
):
    plt.style.use((style_dir / f"textnoisr.{theme}.style").as_posix())
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.tight_layout()
    plot_cer(df, axs[0], [action])
    plot_time(df, axs[1], [action])
    title_parts = [f"action: '{action}'"]
    if dataset_name:
        title_parts.append(f"dataset: '{dataset_name}'")
        if split:
            title_parts.append(f"split: '{split}'")
    if dataset_size:
        title_parts.append(f"dataset size: {dataset_size}k characters")
    title = ", ".join(title_parts).capitalize()
    fig.suptitle(title, y=1.07)
    output_path = output_dir / f"{action}_{theme}.png"
    plt.savefig(output_path, transparent=theme == "dark", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Generated '{output_path}'")


def generate_plots(dfs, **kwargs):
    for action, df in dfs.items():
        for theme in ("light", "dark"):
            generate_plot(df, action=action, theme=theme, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n-samples",
        default=DEFAULT_N_SAMPLES,
        type=int,
        help="Number of noise levels to test",
    )
    parser.add_argument(
        "-o", "--output-dir", default=OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset name")
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split ('train', 'validation' or 'test')",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset, split=args.split)["text"]
    dataset_size = sum(map(len, data)) // 1_000

    dfs = run_benchmark(data, n_samples=args.n_samples)
    generate_plots(
        dfs,
        output_dir=output_dir,
        dataset_name=args.dataset,
        split=args.split,
        dataset_size=dataset_size,
    )


if __name__ == "__main__":
    main()
