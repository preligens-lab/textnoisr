# How this works

## High level design

As can be seen in the source code, there are three main parts:

* The module [textnoisr/noise_dataset.py](api.md#textnoisr.noise_dataset),
    consists on wrapper functions to make this library works seamlessly on the `Dataset` class.

* The module [textnoisr/noise.py](api.md#textnoisr.noise)
    contains a class `CharNoiseAugmenter` that work on the level of a single document.
    This class is basically a wrapper around four methods (one for each action).
    In pseudo-python, the first three ("delete", "insert", "substitute") are one-liners like
    ```
    def ACTION_random_chars(text: str, p: float) -> str:
        return "".join(DO_STUFF(char) if SOME_CONDITION for char in text)
    ```
    It is a little bit more complicated for the "swap" action, since we do not want
    two non-consecutive characters to be swapped. There are two effects to this detail:

    * the code is a little bit more convoluted for "swap" (a dozen of lines instead of two),
    in order to avoid swapping again a character that has already been swapped.
    * this introduces a bias: the Character Error Rate for the case of "swap" does not tends
    to the noise level anymore.
    We need to unbias the noise level beforehand in the "swap" case.
    This is taken into account in the last module.

* The module
    [textnoisr/noise_unbiasing.py](api.md#textnoisr.noise_unbiasing),
    hides the details needed for the Character Error Rate to tend to the noise level.


## Advanced: understanding the unbiasing

As previously said, one important feature of this module is that we want the noise level to be compatible
with the notion of [Character Error Rate](https://huggingface.co/spaces/evaluate-metric/cer).
More precisely, we want (as far as possible)
for the expected value of the Character Error Rate of the output to be the `noise_level` given as input.

Several aspects have been taken into account to enforce this behavior, and several biases have been removed.

### Actions applied successively

When actions are applied successively to the whole text,
the text will be processed several times.
If applied $N$ times with probability $p$, the total probability $P$ to change something will become

$$ P = 1 - (1 - p) ^ N.$$

In our case, $N =\mathtt{len(actions)}$ and $p =\mathtt{noise\_level}$, so that

$$ \mathtt{effective\_noise\_level} = 1 - (1 - \mathtt{noise\_level}) ^ \mathtt{len(actions)} .$$

We have to modify the input $\mathtt{noise\_level}$
in order to get get the expected $\mathtt{effective\_noise\_level}$:

$$ \mathtt{noise\_level} \leftarrow 1.0 - (1.0 - \mathtt{noise\_level}) ^ {1 / \mathtt{len(actions)}}$$

### List of words

Note that the `CharNoiseAugmenter` can add noise to text in the form of list of
    words instead of single string. In that case, we need to apply noise word by word
    in order to have an output list with the same length as the input.
If we applied noise word by word we will decrease the effective Character Error Rate of the whole string
    since no noise will be added on spaces between words.
    So we increase the probability in order to compensate this loss:

$$ \mathtt{noise\_level} \leftarrow  \mathtt{noise\_level}  \times (1 + n_{spaces} / n_{chars}) $$

where $n_{chars}$ is the sum of the number of characters in each word, and $n_{spaces}$ the number
    of spaces between words in the string.

### Action `delete`

No bias to be corrected here.

### Action `insert`

No bias to be corrected here.

### Action `substitute`

No bias to be corrected here, since we ensure that a character is not substituted by itself.

### Action `swap`

Huge bias to be corrected here.

_TL;DR:_ A correction using Markov Chains has been implemented
    for the Character Error Rate to converge to `noise_level`.
    An extra adjustment factor is then applied
    to take into account the structured pattern of natural language.

If you want to know the gory details, you may want to check [this dedicated document](swap_unbiasing.md).


## Conclusion

!!! success "What works"
    The implementation of the `CharNoiseAugmenter` takes all these six aspects into account,
    so the user just have to pass the `noise_level` she/he expects the Character Error Rate to be at the end.

!!! warning
    Be aware that some effects may still remain:

    * When using `["delete", "insert"]`, it is possible to delete a character
        and substitute it with the very same one, resulting in no error where two are naively expected.
    * For very high noise level,
        the computation of the Character Error Rate may not reflect the actual actions performed.
    * A lot of these results is based on the assumptions that the character set is very large.
        This does not really hold for real words: for example,
        swapping the last two characters of `all` results in no change.

Overall, the unit tests show that on a real-world dataset for a noise level of 10%,
    the absolute error between the Character Error Rate and the noise level is less than one percentage point.


*[Character Error Rate]: (=CER) A metric that quantify how noisy is a text. It is number of insert, delete and substitute errors, divided by the total number of characters.
