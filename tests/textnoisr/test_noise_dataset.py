from math import isclose

import pytest
from datasets import load_dataset as hf_load_dataset
from evaluate import load

from textnoisr import noise, noise_dataset

ABS_TOLERANCE = 1.5e-2
REL_TOLERANCE = 1.5e-2


@pytest.fixture()
def dataset100_text():
    return hf_load_dataset("rotten_tomatoes", split="train")


@pytest.fixture()
def dataset100(dataset100_text):
    def split_tokens(item):
        item["tokens"] = item["text"].split(" ")
        return item

    return dataset100_text.map(split_tokens)


cer = load("cer")


@pytest.mark.nightly
@pytest.mark.parametrize(
    "noise_level,actions",
    [
        (0.001, ["substitute"]),
        (0.001, ["insert"]),
        (0.001, ["delete"]),
        (0.001, ["swap"]),
        (0.001, ["delete", "insert", "substitute", "swap"]),
        (0.01, ["substitute"]),
        (0.01, ["insert"]),
        (0.01, ["delete"]),
        (0.01, ["swap"]),
        (0.01, ["delete", "insert", "substitute", "swap"]),
        (0.1, ["substitute"]),
        (0.1, ["insert"]),
        (0.1, ["delete"]),
        (0.1, ["swap"]),
        (0.1, ["delete", "insert", "substitute", "swap"]),
        (0.15, ["substitute"]),
        (0.15, ["insert"]),
        (0.15, ["delete"]),
        (0.15, ["swap"]),
        (0.15, ["delete", "insert", "substitute", "swap"]),
        (0.20, ["substitute"]),
        (0.20, ["insert"]),
        (0.20, ["delete"]),
        (0.20, ["swap"]),
        (0.20, ["delete", "insert", "substitute", "swap"]),
    ],
)
@pytest.mark.filterwarnings("ignore:jiwer.compute_measures")
def test_add_noise_on_split_into_words(dataset100, noise_level, actions):
    noised_dataset = noise_dataset.add_noise(
        dataset100,
        noise.CharNoiseAugmenter(noise_level=noise_level, actions=actions, seed=42),
    )

    pred = [" ".join(aug_doc["tokens"]) for aug_doc in noised_dataset]
    ref = [" ".join(doc["tokens"]) for doc in dataset100]
    cer_score = cer.compute(predictions=pred, references=ref)
    assert isclose(cer_score, noise_level, abs_tol=ABS_TOLERANCE, rel_tol=REL_TOLERANCE)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "noise_level,actions",
    [
        (0.001, ["substitute"]),
        (0.001, ["insert"]),
        (0.001, ["delete"]),
        (0.001, ["swap"]),
        (0.001, ["delete", "insert", "substitute", "swap"]),
        (0.01, ["substitute"]),
        (0.01, ["insert"]),
        (0.01, ["delete"]),
        (0.01, ["swap"]),
        (0.01, ["delete", "insert", "substitute", "swap"]),
        (0.1, ["substitute"]),
        (0.1, ["insert"]),
        (0.1, ["delete"]),
        (0.1, ["swap"]),
        (0.1, ["delete", "insert", "substitute", "swap"]),
        (0.15, ["substitute"]),
        (0.15, ["insert"]),
        (0.15, ["delete"]),
        (0.15, ["swap"]),
        (0.15, ["delete", "insert", "substitute", "swap"]),
        (0.20, ["substitute"]),
        (0.20, ["insert"]),
        (0.20, ["delete"]),
        (0.20, ["swap"]),
        (0.20, ["delete", "insert", "substitute", "swap"]),
        (0.25, ["substitute"]),
        (0.25, ["insert"]),
        (0.25, ["delete"]),
        (0.25, ["swap"]),
        (0.25, ["delete", "insert", "substitute", "swap"]),
    ],
)
@pytest.mark.filterwarnings("ignore:jiwer.compute_measures")
def test_add_noise_on_text(dataset100_text, noise_level, actions):
    noised_dataset = noise_dataset.add_noise(
        dataset100_text,
        noise.CharNoiseAugmenter(noise_level=noise_level, actions=actions, seed=42),
        feature_name="text",
    )
    pred = [aug_doc["text"] for aug_doc in noised_dataset]
    ref = [doc["text"] for doc in dataset100_text]
    cer_score = cer.compute(predictions=pred, references=ref)
    assert isclose(cer_score, noise_level, abs_tol=ABS_TOLERANCE, rel_tol=REL_TOLERANCE)
