# Results

## Result in the ideal case of a string with no repeating characters

As detailed theoretically [here](how_this_works.md),
the four actions have *arbitrarly small errors*
on texts where each letter is unique, like `"01234"` or `"abcdefghijklmnopqrstuvwxyz"`.

This has been extensively unit-tested in `textnoisr/tests/textnoisr/test_noise.py::test__atomic_random_chars` :


```python
...
average_cer = cer.compute(predictions=..., references=...)
...
assert isclose(average_cer, expected_average_cer, abs_tol=ABS_TOLERANCE, rel_tol=REL_TOLERANCE)
```

For arbitrary `ABS_TOLERANCE` and `REL_TOLERANCE`, we are guaranteed that there exists a large enough `n_sample` so that the assertion will not break.

## Benchmark results


That being said, it is interesting to see wether that holds for _natural language_ as well.
For example, the fact that consecutive letters could be equal may lower the Character Error Rate for `swap` action.

`textnoisr` deals with a problem that is partly answered by [nlpaug](https://github.com/makcedward/nlpaug), that has a much larger scope (i.e. NLP augmentation in a broad sense) and does not primarly focus on correctness w.r.t Character Error Rate.

In this section, we present the results of our algorithm, and compare them to the results of `nlpaug`.

We want to check both

* **on the left panel**: the correctness of our approach. The closest to the diagonal, the better it is. This correspond to an absence of bias between the input and the output.
* **on the right panel**: how long it takes to execute the code.


![insert](images/insert_light.png#only-light)
![insert](images/insert_dark.png#only-dark)

![delete](images/delete_light.png#only-light)
![delete](images/delete_dark.png#only-dark)

![substitute](images/substitute_light.png#only-light)
![substitute](images/substitute_dark.png#only-dark)

![swap](images/swap_light.png#only-light)
![swap](images/swap_dark.png#only-dark)
