from datasets import load_dataset
from evaluate import load

from textnoisr import noise, noise_dataset
from textnoisr.noise_unbiasing import MAX_SWAP_LEVEL

cer = load("cer")

# glue-mnli is used to compute correction for English
dataset100_text = load_dataset("glue", "mnli", split="train[:100%]")

noised_dataset = noise_dataset.add_noise(
    dataset100_text,
    noise.CharNoiseAugmenter(
        noise_level=MAX_SWAP_LEVEL,
        actions=("swap",),
        seed=42,
        natural_language_swap_correction=1,
    ),
    feature_name="premise",
)

pred = [aug_doc["premise"] for aug_doc in noised_dataset]
ref = [doc["premise"] for doc in dataset100_text]
cer_score = cer.compute(predictions=pred, references=ref)

print(f"natural_correction = {MAX_SWAP_LEVEL / cer_score:.3f}")
# natural_correction = 1.052
