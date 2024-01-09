"""This module tests the functions that add noise to an NLP dataset.

Since these functions are non-deterministic, the following approach is chosen:
* each function is applied N_SAMPLE times to the same input.
* we compute some quantities, for which we know the statistical expectation.
* we compare the resulting stat and the expectation with a given tolerance,
    and set a seed for the experiment to be reproducible.

Notice that when experimenting around, we can be more strict on the tolerance and
set a very high number of samples to ensure the correctness of the functions.
Once the functions are written, we can relax these for the tests not to be too long.
"""

import string
from math import isclose

import pytest
from evaluate import load

import textnoisr.noise_unbiasing as unbias
from textnoisr import noise

cer = load("cer")
ABS_TOLERANCE = 5e-3
REL_TOLERANCE = 5e-3


@pytest.mark.parametrize(
    "p",
    [(0), (0.1), (0.5), (1)],
)
def test__random_char(p):
    noise_augmenter = noise.CharNoiseAugmenter(
        seed=42, noise_level=0, natural_language_swap_correction=1
    )
    example = [
        noise_augmenter._random_char(p, character_set=string.ascii_letters)
        for _ in range(1000000)
    ]

    assert isclose(
        len("".join(example)) / 1000000, p, abs_tol=ABS_TOLERANCE, rel_tol=REL_TOLERANCE
    )
    assert all([len(e) <= 1 for e in example])


@pytest.mark.nightly
@pytest.mark.filterwarnings("ignore:jiwer.compute_measures")
@pytest.mark.parametrize(
    "action,one_token,p",
    [
        ("insert", ["abce", "fghi"], 0.1),
        ("insert", "a", 0),
        ("insert", "a", 0.5),
        ("insert", "a", 0.1),
        ("insert", "a", 1),
        ("insert", "aa", 0.1),
        ("insert", "aaa", 0.1),
        ("insert", "ababa", 0.1),
        ("delete", "a", 0),
        ("delete", "a", 0.5),
        ("delete", "a", 0.1),
        ("delete", "a", 1),
        ("delete", "aa", 0.1),
        ("delete", "aaa", 0.1),
        ("delete", "ababa", 0.1),
        ("substitute", "a", 0),
        ("substitute", "a", 0.5),
        ("substitute", "a", 0.1),
        ("substitute", "a", 1),
        ("substitute", "aa", 0.1),
        ("substitute", "aaa", 0.1),
        ("substitute", "ababa", 0.1),
        ("swap", "ab", 0),
        ("swap", "abc", 0),
        ("swap", 2 * string.ascii_letters, 0.015),
        ("swap", 2 * string.ascii_letters, 0.05),
        ("swap", 2 * string.ascii_letters, 0.45),
        ("swap", 2 * string.ascii_letters, 0.5),
        ("swap", 2 * string.ascii_letters, 0.53),
        ("swap", 2 * string.ascii_letters, unbias.MAX_SWAP_LEVEL),
        ("swap", "ab", 0.1),
        ("swap", "abc", 0.1),
        ("swap", "abcd", 0.1),
        ("swap", "abcde", 0.1),
        ("swap", "abcdef", 0.1),
        ("swap", "abcdefg", 0.1),
        ("swap", "abcdefgh", 0.1),
        ("swap", "abcdefghi", 0.1),
        ("swap", "abcdefghij", 0.1),
        ("swap", "ab", 0.01),
        ("swap", "abc", 0.01),
        ("swap", "abcd", 0.01),
        ("swap", "abcde", 0.01),
        ("swap", "abcdef", 0.01),
        ("swap", "abcdefg", 0.01),
        ("swap", "abcdefgh", 0.01),
        ("swap", "abcdefghi", 0.01),
        ("swap", "abcdefghij", 0.01),
        ("swap", "ab", 0.46),
        ("swap", "abc", 0.46),
        ("swap", "abcd", 0.46),
        ("swap", "abcde", 0.46),
        ("swap", "abcdef", 0.46),
        ("swap", "abcdefg", 0.46),
        ("swap", "abcdefgh", 0.46),
        ("swap", "abcdefghi", 0.46),
        ("swap", "abcdefghij", 0.46),
    ],
)
def test__atomic_random_chars(action, one_token, p):
    """Test the functions for which number of changes is easy to predict."""
    noise_augmenter = noise.CharNoiseAugmenter(
        noise_level=p, actions=[action], seed=42, natural_language_swap_correction=1
    )
    n_sample = 100000

    examples = [one_token] * n_sample

    # We have to do the follwing if/else in order to test `is_split_into_words`:
    if isinstance(one_token, list):
        noised_examples = [" ".join(noise_augmenter.add_noise(e)) for e in examples]
        one_token = " ".join(one_token)
    else:
        noised_examples = [noise_augmenter.add_noise(e) for e in examples]

    n_char = len(one_token)
    # Test on the number of characters of the outputs
    match action:
        case "insert":
            assert all([len(e) >= n_char for e in noised_examples])
        case "delete":
            assert all([len(e) <= n_char for e in noised_examples])
        case "substitute":
            assert all([len(e) == n_char for e in noised_examples])
        case "swap":
            assert all([len(e) == n_char for e in noised_examples])

    # Test on the probability of no change in the input string
    diff = [e == one_token for e in noised_examples]

    proba_nochange = sum(diff) / len(diff)
    if action == "swap":
        expected_proba_nochange = (
            1 - noise.unbias.unbias_swap(p, n_char, natural_language_swap_correction=1)
        ) ** (n_char - 1)
    else:
        expected_proba_nochange = (1 - p) ** n_char

    assert isclose(
        proba_nochange,
        expected_proba_nochange,
        abs_tol=ABS_TOLERANCE,
        rel_tol=REL_TOLERANCE,
    )

    # Test on the expectation value of the Character Error Rate
    average_cer = cer.compute(
        predictions=noised_examples, references=[one_token] * n_sample
    )
    expected_average_cer = p

    assert isclose(
        average_cer,
        expected_average_cer,
        abs_tol=ABS_TOLERANCE,
        rel_tol=REL_TOLERANCE,
    )


@pytest.mark.parametrize(
    "one_token,p,expected_results",
    [
        ("abcd", 0.45, {"abcd", "abdc", "acbd", "bacd", "badc"}),
        ("abcd", 0.3, {"abcd", "abdc", "acbd", "bacd", "badc"}),
        ("abcd", 0, {"abcd"}),
        ("ab", 0.45, {"ab", "ba"}),
        ("ab", 0.3, {"ab", "ba"}),
        ("ab", 0, {"ab"}),
        ("a", 0.3, {"a"}),
        ("", 0.3, {""}),
        ("abcd", 0.6, None),
    ],
)
def test__swap_random_chars(one_token, p, expected_results):
    if p < 0.55:
        noise_augmenter = noise.CharNoiseAugmenter(
            noise_level=p, actions=["swap"], seed=42, natural_language_swap_correction=1
        )
        example = [one_token] * 100000
        noised_example = [noise_augmenter.add_noise(e) for e in example]
        assert set(noised_example) == expected_results
    else:
        with pytest.raises(ValueError):
            noise.CharNoiseAugmenter(
                noise_level=p,
                actions=["swap"],
                seed=42,
                natural_language_swap_correction=1,
            )


def test_reproducibility():
    noise_augmenter_42 = noise.CharNoiseAugmenter(
        noise_level=0.5, seed=42, natural_language_swap_correction=1
    )
    noisy_hello = noise_augmenter_42.add_noise("Hello")
    assert noisy_hello == "HleVlo"
    noisy_hello = noise_augmenter_42.add_noise("Hello")
    assert noisy_hello == "elo"
