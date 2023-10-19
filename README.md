# `textnoisr`: Adding random noise to a dataset


[![build-doc](https://github.com/earthcube-lab/textnoisr/actions/workflows/build-doc.yml/badge.svg)](https://github.com/earthcube-lab/textnoisr/actions/workflows/build-doc.yml)
[![code-style](https://github.com/earthcube-lab/textnoisr/actions/workflows/code-style.yml/badge.svg)](https://github.com/earthcube-lab/textnoisr/actions/workflows/code-style.yml)
[![nightly-test](https://github.com/earthcube-lab/textnoisr/actions/workflows/nightly-test.yml/badge.svg)](https://github.com/earthcube-lab/textnoisr/actions/workflows/nightly-test.yml)
[![unit-test](https://github.com/earthcube-lab/textnoisr/actions/workflows/unit-test.yml/badge.svg)](https://github.com/earthcube-lab/textnoisr/actions/workflows/unit-test.yml)



`textnoisr` is a python package that allows to **add random noise to a text dataset**,
and to **control very accurately** the quality of the result.

Here is an example if your dataset consists on the first few lines of [the Zen of python](https://peps.python.org/pep-0020/):

**Raw text**

```
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
...
```

**Noisy text**

```
TheO Zen of Python, by Tim Pfter

BzeautiUful is ebtter than ugly.
Eqxplicin is better than imlicit.
Simple is beateUr than comdplex.
Complex is better than comwlicated.
Flat is bejAter than neseed.
...
```

Four types of "actions" are implemented:

* **insert** a random character, e.g.        STEAM  →  ST<span style="background-color:LightBlue">R</span>EAM,
* **delete** a random character, e.g.        <span style="background-color:Crimson">S</span>TEAM  →  TEAM,
* **substitute** a random character, e.g.    STEA<span style="background-color:LightGreen">M</span>  →  STEA<span style="background-color:LightGreen">L</span>.
* **swap** two **consecutive** characters, e.g.  STE<span style="background-color:Orange">AM</span>  →  STE<span style="background-color:Orange">MA</span>


The general philosophy of the package is that only **one single parameter**
is needed to control the noise level.
This "noise level" is applied character-wise,
and corresponds _roughly_ to the probability for a character to be impacted.

More precisely, this noise level is calibrated so that
the [Character Error Rate](https://huggingface.co/spaces/evaluate-metric/cer)
of a noised dataset converges to this value as the amount of text increases.


**Why a whole package for such a simple task?**

> In the case of inserting, deleting and substituting characters at random with a probability $p$,
> the Character Error Rate is only the average number of those operations,
> so it will converge to the input value $p$ due to the
> [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers).
>
> However, the case of swapping consecutive characters is **not trivial at all** for two reasons:
>
> * First, swapping two characters is not an "atomic operation" with respect to the Character Error Rate metric.
>
> * Second, we do not want to swap repeatedly the same character over and over again
> if the probability to apply the swap action is high:<br>
> <span style="background-color:Orange">ST</span>EAM  →  <span style="background-color:Orange">TS</span>EAM<br>
> T<span style="background-color:Orange">SE</span>AM  →  T<span style="background-color:Orange">ES</span>AM<br>
> TE<span style="background-color:Orange">SA</span>M  →  TE<span style="background-color:Orange">AS</span>M<br>
> TEA<span style="background-color:Orange">SM</span>  →  TEA<span style="background-color:Orange">MS</span><br>
> This would be equivalent to <span style="background-color:Orange">S</span>TEAM  →
> TEAM<span style="background-color:Orange">S</span>, so this cannot be considered "swapping consecutive characters".
> To avoid this behavior, we must avoid swapping a character if it has just been swapped.
> This breaks the independency between one character and the following one,
> and makes the Law of Large Numbers not applicable.
>
> We use Markov Chains to model the swapping of characters.
> This allows us to compute and correct the corresponding bias in order to make itstraightforward
> for the user to get the desired Character Error Rate, as if the Law of Large Number could beapplied!
>
> All the details of this unbiasing [are here](docs/swap_unbiasing.md).
> The goal of this package is for the user to be confident on the result
> without worrying about the implementation details.


---


The documentation follows this plan:

* You may want to follow [a quick tutorial](docs/tutorial.md) to learn the basics of the package,
* The [Results](docs/results.md) page illustrates how **no calibration is needed** in order to add noise to a corpus with a target Character Error Rate.
* The [How this works section](docs/how_this_works.md) explains the mechanisms, and some design choices of this package.
We have been extra careful to explain how some statistical bias have been avoided,
for the package to be both user-friendly and correct.
A [dedicated page](docs/swap_unbiasing.md) deeps dive in the case of the `swap` action.
* The [API Reference](docs/api.md) details all the technical descriptions needed.
