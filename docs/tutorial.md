# Tutorial

The easiest way to use `textnoisr` is to apply it directly to a string.
We also implemented a wrapper to use it on :hugging_face: datasets.

## Add noise to a single string

```pycon
>>> from textnoisr import noise
>>> text = "The duck-billed platypus (Ornithorhynchus anatinus) is a small mammal."
>>> augmenter = noise.CharNoiseAugmenter(noise_level=0.1)
>>> print(augmenter.add_noise(text))
The dhuck-biled plstypus Ornithorhnchus anatinus) is a smaJl mammal.
```

By default, all actions (i.e. `"insert"`, `"swap"`, `"substitute"`, `"delete"`)
are applied to the sentence successively, but we can choose to apply only a subset of them:

```pycon
>>> augmenter = noise.CharNoiseAugmenter(noise_level=0.2, actions=["delete"])
>>> print(augmenter.add_noise(text))
The dckilledplypus Ornithrhynhstinus is a mal mamal.
```

The Character Error Rate of the result should converge to the input value `noise_level`,
although it is not obvious with only one sentence:
the effect of the Law of Large Number will be easily seen
in the next section dealing with whole datasets.


## Add noise to :hugging_face: datasets

For example, let's consider the
[rotten tomatoes](https://huggingface.co/datasets/rotten_tomatoes) dataset:

```pycon
>>> from datasets import load_dataset  # https://huggingface.co/docs/datasets
>>> dataset = load_dataset("rotten_tomatoes", split="train")
>>> print(dataset)
Dataset({
    features: ['text', 'label'],
    num_rows: 8530
})

```

We want to add some noise to the `text` feature of this dataset.
We use the `noise_level` parameter to control how much noisy will be the result:


```pycon
>>> from textnoisr.dataprep import noise_dataset
>>> from textnoisr import noise
>>>
>>> noised_dataset = noise_dataset.add_noise(
>>>      dataset,
>>>      noise.CharNoiseAugmenter(noise_level=0.1),
>>>      feature_name="text",
>>> )
>>> print(f'{dataset["text"][42]!r}')
>>> print(f'{noised_dataset["text"][42]!r}')
'fuller would surely have called this gutsy and at times exhilarating movie a great yarn .'
'fCuller vould surely aJe clled tihs gutsyrndk att ies exhiglrdating mvoie a great yarBA .'
```

Let's compute the Character Error Rate of the resulting dataset,
with respect to the original. As mentioned in the introduction,
it is expected to be close to the input `noise_level`.

```pycon
>>> pred = [aug_doc["text"] for aug_doc in noised_dataset]
>>> ref = [doc["text"] for doc in dataset]
>>>
>>> from evaluate import load  # https://huggingface.co/docs/evaluate
>>>
>>> cer = load("cer")
>>> print(f"Character Error Rate = {cer.compute(predictions=pred, references=ref):.3f}")
Character Error Rate = 0.098
```

By default, all available actions are performed sequentially.
It is possible to perform only some of the available actions though.
Notice that for each of the five following tuples of actions,
the Character Error Rate is close to the `noise_level`:

```pycon
>>> for actions in [
>>>     ["delete", "insert", "substitute", "swap"],
>>>     ["delete"],
>>>     ["insert"],
>>>     ["substitute"],
>>>     ["swap"],
>>> ]:
>>>     noised_dataset = noise_dataset.add_noise(
>>>         dataset,
>>>         noise.CharNoiseAugmenter(noise_level=0.01, actions=actions),
>>>         feature_name="text",
>>>     )
>>>     pred = [aug_doc["text"] for aug_doc in noised_dataset]
>>>     print(f"-------\nAction: {actions!r}")
>>>     print(f'{noised_dataset["text"][42]!r}')
>>>     print(f"Character Error Rate = {cer.compute(predictions=pred, references=ref):.3f}")
-------
Action: ('delete', 'insert', 'substitute', 'swap')
'fuller would surely have called this gutsy and at timese xhilarating movie a great yarn .'
Character Error Rate = 0.010
-------
Action: ('delete')
'fuller would surely have called this gutsy and at timesexhilarating mvie a great yarn .'
Character Error Rate = 0.010
-------
Action: ('insert')
'fuller would surely have called this gutsy and at times exhilarating movie a great yarn .'
Character Error Rate = 0.010
-------
Action: ('substitute')
'fuller wouOd surely have called this gutsy and at times exhilarating movie v great yarn .'
Character Error Rate = 0.010
-------
Action: ('swap')
'fuller would surely have called this gutsy and at times exhliarating movie a great yarn .'
Character Error Rate = 0.010
```

You can also use the `noise_dataset.add_noise` function on a dataset
where text is splitted in words or tokens,
that is a `list[str]` instead of a `str`.

```pycon
>>> def tokenization(example):  # Well, sort of...
>>>     example["tokens"] = example["text"].split()
>>>     return example
>>>
>>>
>>> tokenized_dataset = dataset.map(tokenization, batched=False)
>>>
>>> noised_tokenized_dataset = noise_dataset.add_noise(
>>>     tokenized_dataset,
>>>     noise.CharNoiseAugmenter(noise_level=0.1, actions=actions),
>>>     feature_name="tokens",
>>> )
>>> print(tokenized_dataset[42]["tokens"])
>>> print(noised_tokenized_dataset[42]["tokens"])
['fuller', 'would', 'surely', 'have', 'called', 'this', 'gutsy', 'and', 'at', 'times', 'exhilarating', 'movie', 'a', 'great', 'yarn', '.']
['fuller', 'woudl', 'surely', 'have', 'claled', 'this', 'gusty', 'and', 'at', 'times', 'exhilaratign', 'movie', 'a', 'great', 'yarn', '.']
```

!!! note
    It can be useful to split a dataset before applying noise to each token individually.
    This is the case when adding noise to a dataset already annotated for NER,
    while keeping the 1-1 mapping between annotations and words.
    In this case, you don't want spaces to be deleted, inserted, substituted, or swapped.

*[Character Error Rate]: (=CER) A metric that quantify how noisy is a text. It is number of insert, delete and substitute errors, divided by the total number of characters.

*[NER]: Named Entity Recognition
