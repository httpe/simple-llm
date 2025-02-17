# Test script for our simple character level language model

import random

import numpy as np
import torch

from .lm import BiGramModel, TriGramModel, TensorBiGramModel, NeuralNGramModel, train_torch_lm_model, prepare_torch_dataset, sample_torch_lm_model
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
    print("Training samples (first 10):")
    print(ground_true_samples[:10])
    assert all([sticky.validate_samples(stickiness, ground_true_samples)])
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

    # baseline model, simply generate random characters from [A-Za-z0-9]
    print("Baseline model (random [A-Za-z0-0]):")
    baseline_model = sticky.BaselineModel()
    baseline_samples = baseline_model.generate(n_samples=n_to_generate, max_length=max_len)
    print("Samples generated:")
    print(baseline_samples[:10])
    sticky.validate_samples(stickiness, baseline_samples, ground_true_samples)
    print("")

    # train a tri-gram model, this should have 100% accuracy given the stickiness is 1
    print("Naive Counting-based Tri-gram model:")
    model = TriGramModel(sample_sep=".", dummy_count=0)
    model.train(ground_true_samples)
    print("Parameter count: ", model.trigram_count.numel())
    generated_samples = model.generate(n_samples=n_to_generate, max_length=max_len)
    print("Samples generated:")
    print(generated_samples[:10])
    sticky.validate_samples(stickiness, generated_samples, ground_true_samples)
    print("")

    # train a bi-gram model to approximate the tri-gram model
    print("Naive Counting-based Bi-gram model:")
    model = BiGramModel(sample_sep=".", dummy_count=0)
    model.train(ground_true_samples)
    print("Parameter count: ", model.bigram_count.numel())
    generated_samples = model.generate(n_samples=n_to_generate, max_length=max_len)
    print("Samples generated:")
    print(generated_samples[:10])
    sticky.validate_samples(stickiness, generated_samples, ground_true_samples)
    print("")

    # test neural network based bi-gram model, it should be able to approximate the counting-based bi-gram model
    print("Tensor Bi-gram model:")
    look_back = 1
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    tokenizer, dataset = prepare_torch_dataset(ground_true_samples, look_back=look_back)
    model = TensorBiGramModel(vocab_size=tokenizer.vocab_size)
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = train_torch_lm_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
    generated_samples = sample_torch_lm_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    print("Samples generated:")
    print(generated_samples[:10])
    sticky.validate_samples(stickiness, generated_samples, ground_true_samples)
    print("")

    # train a neural network based n-gram model, it should be able to approximate the counting-based tri-gram model given look_back=2
    look_back = 2
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    print(f"Neural {look_back + 1}-gram model model:")
    tokenizer, dataset = prepare_torch_dataset(ground_true_samples, look_back=look_back)
    model = NeuralNGramModel(vocab_size=tokenizer.vocab_size, embed_size=8, hidden_sizes=[8], look_back=look_back)
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = train_torch_lm_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
    generated_samples = sample_torch_lm_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    print(f"Samples generated:")
    print(generated_samples[:10])
    sticky.validate_samples(stickiness, generated_samples, ground_true_samples)

    print("")


if __name__ == "__main__":
    # general parameters
    n_training = 1000
    n_to_generate = 1000
    min_len = 3
    max_len = 10
    
    reset_seeds()
    
    print("")
    test_sticky_rule(n_training, n_to_generate, min_len, max_len, stickiness=1)

    