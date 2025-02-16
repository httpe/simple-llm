# Test script for our simple character level language model

import random

import numpy as np
import torch

from .lm import BiGramModel
from . import llm_samples
from . import sticky

def reset_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

def parse_sample_text(sample_text: str) -> list[str]:
    samples = sample_text.split("\n")
    samples = [x.strip() for x in samples]
    samples = [(x.split(' ')[1] if ' ' in x else x) for x in samples]
    samples = [s for s in samples if len(s) > 0]
    return samples

def test_sticky_rule(n_training: int, n_to_generate: int, min_len: int, max_len: int, stickiness: int):
    # generate some samples from the ground truth model
    ground_true_samples = sticky.generate_samples(n_samples=n_training, min_length=min_len, max_length=max_len, stickiness=stickiness)
    assert all([sticky.validate_samples(stickiness, ground_true_samples)])

    print("Training samples (first 10):")
    print(ground_true_samples[:10])
    print("")

    # baseline samples
    print("Baseline model (random [A-Za-z0-0]) generated samples accuracy:")
    baseline_model = sticky.BaselineModel()
    baseline_samples = baseline_model.generate(n_samples=n_to_generate, max_length=max_len)
    sticky.validate_samples(stickiness, baseline_samples, ground_true_samples)
    print("")

    # validate samples generated from LLMs
    print('--------------------------------------')
    print("")
    print(f"LLM generated samples accuracy:")
    print("")
    for name, sample_text in llm_samples.llm_samples.items():
        print(f"Model: {name}")
        samples = parse_sample_text(sample_text)
        sticky.validate_samples(stickiness, samples, ground_true_samples)
        print("")
    print('--------------------------------------')
    print("")

    # train a bi-gram model, generate some samples and calculate accuracy
    model = BiGramModel(sample_sep=".", dummy_count=0)
    model.train(ground_true_samples)
    generated_samples = model.generate(n_samples=n_to_generate, max_length=max_len)
    print("Bi-gram model generated samples accuracy:")
    sticky.validate_samples(stickiness, generated_samples, ground_true_samples)

    print("")


if __name__ == "__main__":
    # general parameters
    n_training = 100
    n_to_generate = 100
    min_len = 3
    max_len = 10
    
    reset_seeds()
    
    print("")
    test_sticky_rule(n_training, n_to_generate, min_len, max_len, stickiness=1)

    