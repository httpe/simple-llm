# Test script for our simple character level language model

from typing import Callable
import random
import string
import os

import numpy as np
import torch

from .lm import BiGramModel, TriGramModel, TensorBiGramModel, MLPLanguageModel, RNNModel, CharacterTokenizer, prepare_n_gram_dataset, prepare_auto_regressive_dataset, train_torch_n_gram_model, train_auto_regressive_model, sample_torch_n_gram_model, sample_auto_regressive_model
from .transformer import TransformerLM, train_transformer, sample_from_transformer
from .llm_samples import llm_samples
from . import sticky

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def reset_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

def validate_samples(validator: Callable[[str], bool], samples: list[str], existing_samples: list[str]| None = None, model_name: str = "") -> list[bool]:
    n_total = len(samples)
    unique_samples = list(set(samples))
    n_unique = len(unique_samples)
    if existing_samples is None:
        existing_samples = []
    new_samples = [s for s in unique_samples if s not in existing_samples]
    n_new = len(new_samples)

    n_valid = 0
    valid_flags = []
    for sample in new_samples:
        valid = validator(sample)
        valid_flags.append(valid)
        if valid:
            n_valid += 1
        # print(f"{sample}: {valid}")
    
    # print out the statistics
    if model_name != "":
        print(f"{model_name}:")
    print(f"Generated samples (first 10):", new_samples[:10])
    print(f"Valid/New/Unique/Total: {n_valid}/{n_new}/{n_unique}/{n_total}")
    print(f"Accuracy (Valid/New) %: {100 * n_valid/n_new: 0.2f}%")
    print("")

    return valid_flags

def test_sticky_rule(n_training: int, n_validation: int, n_to_generate: int, min_len: int, max_len: int, stickiness: int, use_saved_model: bool, model_save_dir: str = SCRIPT_DIR):
    # validator
    validator = lambda x: sticky.validate_one_sample(stickiness, x)

    # generate some samples from the ground truth model
    reset_seeds()
    training_samples = sticky.generate_samples(n_samples=n_training, min_length=min_len, max_length=max_len, stickiness=stickiness)
    validation_samples = sticky.generate_samples(n_samples=n_validation, min_length=min_len, max_length=max_len, stickiness=stickiness)
    assert all(validate_samples(validator, training_samples, model_name="Ground truth (training)"))
    assert all(validate_samples(validator, validation_samples, training_samples, model_name="Ground truth (validation)"))


    # validate samples generated from LLMs
    if f"stickiness_{stickiness}" in llm_samples:
        for name, samples in llm_samples["stickiness_1"].items():
            validate_samples(validator, samples, training_samples, model_name=f"{name} (LLM)")

    # baseline model, simply generate random characters from [A-Za-z0-9]
    reset_seeds()
    baseline_model = sticky.BaselineModel()
    baseline_samples = baseline_model.generate(n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, baseline_samples, training_samples, model_name="Baseline (random [A-Za-z0-9])")


    # train a bi-gram model to approximate the tri-gram model
    reset_seeds()
    model = BiGramModel(sample_sep=".", dummy_count=0)    
    model.train(training_samples)
    print("Counting-based bi-gram model:")
    print("Parameter count: ", model.bigram_count.numel())
    generated_samples = model.generate(n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)

    # train a tri-gram model, this should have 100% accuracy given the stickiness is 1
    reset_seeds()
    model = TriGramModel(sample_sep=".", dummy_count=0)
    model.train(training_samples)
    print("Counting-based tri-gram model:")
    print("Parameter count: ", model.trigram_count.numel())
    generated_samples = model.generate(n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    

    # use tokenizer for neural network based models
    reset_seeds()
    tokenizer = CharacterTokenizer()
    # a-z A-Z 0-9
    token_samples = [string.ascii_lowercase + string.ascii_uppercase + string.digits]
    tokenizer.train(token_samples)


    # test neural network based bi-gram model, it should be able to approximate the counting-based bi-gram model
    reset_seeds()
    look_back = 1
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    dataset = prepare_n_gram_dataset(training_samples, tokenizer, look_back=look_back)
    model = TensorBiGramModel(vocab_size=tokenizer.vocab_size)
    print("Tensor bi-gram model (train with cross-entropy loss and gradient descent):")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    path = os.path.join(model_save_dir, "tensor_bigram.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_torch_n_gram_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
        torch.save(model.state_dict(), path)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)


    # train a neural network (MLP) based language model
    # it has insufficient context length (look back) to, in theory, achieve a 100% accuracy
    # but should still be able to approximate the counting-based (N-1)-gram model with less parameters
    reset_seeds()
    look_back = stickiness
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    embed_size = 8
    hidden_layer_sizes = [8]
    dataset = prepare_n_gram_dataset(training_samples, tokenizer, look_back=look_back)
    model = MLPLanguageModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_sizes=hidden_layer_sizes, look_back=look_back)
    print(f"Neural language model with {look_back} characters context:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    path = os.path.join(model_save_dir, f"mlp_lm_look_back_{look_back}.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_torch_n_gram_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
        torch.save(model.state_dict(), path)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)


    # train a neural network (MLP) based language model, it should be able to approximate the counting-based N-gram model with much less free parameters
    reset_seeds()
    look_back = stickiness + 1
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    embed_size = 8
    hidden_layer_sizes = [8]
    dataset = prepare_n_gram_dataset(training_samples, tokenizer, look_back=look_back)
    model = MLPLanguageModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_sizes=hidden_layer_sizes, look_back=look_back)
    print(f"Neural language model with {look_back} characters context:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    path = os.path.join(model_save_dir, f"mlp_lm_look_back_{look_back}.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_torch_n_gram_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
        torch.save(model.state_dict(), path)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)


    # Prepare dataset auto-regressive models
    reset_seeds()
    training_dataset = prepare_auto_regressive_dataset(training_samples, tokenizer)
    validation_dataset = prepare_auto_regressive_dataset(validation_samples, tokenizer)


    # train a RNN model with simple recurrent unit
    reset_seeds()
    embed_size = 8
    hidden_state_size = 8
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, use_gru=False)
    print("RNN model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    path = os.path.join(model_save_dir, "rnn.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_auto_regressive_model(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        torch.save(model.state_dict(), path)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)


    # train a RNN model with GRU (Gated Recurrent Unit) unit
    # reduced embedding/hidden size to match the RNN model parameter count
    reset_seeds()
    embed_size = 7 
    hidden_state_size = 7
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, use_gru=True)
    print("GRU model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    path = os.path.join(model_save_dir, "gru.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_auto_regressive_model(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        torch.save(model.state_dict(), path)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)


    # Train a transformer model
    reset_seeds()
    embed_size = 6
    max_context_size = 12
    n_layer = 1
    n_heads = 1
    head_size = embed_size
    ff_hidden_size = 8
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    model = TransformerLM(vocab_size=tokenizer.vocab_size, embed_size=embed_size, max_context_size=max_context_size, n_layer=n_layer, n_heads=n_heads, head_size=head_size, ff_hidden_size=ff_hidden_size)
    print("Transformer model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    path = os.path.join(model_save_dir, "transformer.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_transformer(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
        torch.save(model.state_dict(), path)
    generated_samples = sample_from_transformer(model, tokenizer, max_new_tokens=max_len, n_samples=n_to_generate)
    validate_samples(validator, generated_samples, training_samples)

if __name__ == "__main__":
    # general parameters
    n_training = 1000
    n_validation = 200
    n_to_generate = 1000
    min_len = 3
    max_len = 10
    use_saved_model = True
    
    reset_seeds()
    
    print("")
    stickiness = 1
    model_dir = os.path.join(SCRIPT_DIR, "models", f"stickiness_{stickiness}")
    os.makedirs(model_dir, exist_ok=True)
    test_sticky_rule(n_training, n_validation, n_to_generate, min_len, max_len, stickiness=stickiness, use_saved_model=use_saved_model, model_save_dir=model_dir)

    