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
from . import arithmetics
from . import counting

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def reset_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

def validate_samples(validator: Callable[[str], bool], samples: list[str], existing_samples: list[str]| None = None):
    n_total = len(samples)
    unique_samples = list(set(samples))
    n_unique = len(unique_samples)
    if existing_samples is None:
        existing_samples = []
    new_samples = [s for s in unique_samples if s not in existing_samples]
    n_new = len(new_samples)

    n_valid = 0
    is_valid = []
    valid_samples = []
    for sample in new_samples:
        valid = validator(sample)
        is_valid.append(valid)
        if valid:
            n_valid += 1
            valid_samples.append(sample)
        # print(f"{sample}: {valid}")
    
    print(f"Generated new samples (first 10):", new_samples[:10])
    print(f"Valid new samples (first 10): {valid_samples[:10]}")
    print(f"Valid/New/Unique/Total: {n_valid}/{n_new}/{n_unique}/{n_total}")
    print(f"Accuracy (Valid/New) %: {100 * n_valid/n_new: 0.2f}%")

    return is_valid, valid_samples

def test_sticky_rule(n_training: int, n_validation: int, n_to_generate: int, min_len: int, max_len: int, stickiness: int, strict: bool, use_saved_model: bool, model_save_dir: str = SCRIPT_DIR):
    print("===========================================")
    print(f"Test Sticky Rule {stickiness=} {'strict' if strict else 'loose'}")
    print("===========================================")
    print("")
    
    # validator
    validator = lambda x: sticky.validate_one_sample(stickiness, strict, x)

    # generate some samples from the ground truth model
    reset_seeds()
    print("Ground truth (training):")
    training_samples = sticky.generate_samples(n_samples=n_training, min_length=min_len, max_length=max_len, stickiness=stickiness, strict=strict)
    is_valid, _ = validate_samples(validator, training_samples)
    assert all(is_valid)

    print("Ground truth (training):")
    validation_samples = sticky.generate_samples(n_samples=n_validation, min_length=min_len, max_length=max_len, stickiness=stickiness, strict=strict)
    is_valid, _ = validate_samples(validator, validation_samples, training_samples)
    assert all(is_valid)
    print("")


    # validate samples generated from LLMs
    if f"stickiness_{stickiness}" in llm_samples:
        for name, samples in llm_samples["stickiness_1"].items():
            print("LLM:", name)
            validate_samples(validator, samples, training_samples)
            print("")

    # baseline model, simply generate random characters from [A-Za-z0-9]
    reset_seeds()
    baseline_model = sticky.BaselineModel()
    baseline_samples = baseline_model.generate(n_samples=n_to_generate, max_length=max_len)
    print("Baseline (random [A-Za-z0-9]):")
    validate_samples(validator, baseline_samples, training_samples)
    print("")

    # train a bi-gram model to approximate the tri-gram model
    reset_seeds()
    model = BiGramModel(sample_sep=".", dummy_count=0)    
    model.train(training_samples)
    print("Counting-based bi-gram model:")
    print("Parameter count: ", model.bigram_count.numel()) # 3969
    generated_samples = model.generate(n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")

    # train a tri-gram model, this should have 100% accuracy given the stickiness is 1
    reset_seeds()
    model = TriGramModel(sample_sep=".", dummy_count=0)
    model.train(training_samples)
    print("Counting-based tri-gram model:")
    print("Parameter count: ", model.trigram_count.numel()) # 250047
    generated_samples = model.generate(n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")
    

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
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 4225
    path = os.path.join(model_save_dir, "tensor_bigram.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_torch_n_gram_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
        torch.save(model.state_dict(), path)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


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
    print(f"MLP with {look_back} characters context:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 1177
    path = os.path.join(model_save_dir, f"mlp_lm_look_back_{look_back}.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_torch_n_gram_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
        torch.save(model.state_dict(), path)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


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
    print(f"MLP with {look_back} characters context:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 1241
    path = os.path.join(model_save_dir, f"mlp_lm_look_back_{look_back}.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_torch_n_gram_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
        torch.save(model.state_dict(), path)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


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
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 1249
    path = os.path.join(model_save_dir, "rnn.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_auto_regressive_model(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        torch.save(model.state_dict(), path)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


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
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 1297
    path = os.path.join(model_save_dir, "gru.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_auto_regressive_model(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        torch.save(model.state_dict(), path)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


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
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 1213
    path = os.path.join(model_save_dir, "transformer.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_transformer(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
        torch.save(model.state_dict(), path)
    generated_samples = sample_from_transformer(model, tokenizer, max_new_tokens=max_len, n_samples=n_to_generate)
    validate_samples(validator, generated_samples, training_samples)
    print("")


def unique_numbers_in_counting_samples(samples: list[str], existing_numbers: set[str] | None = None):
    numbers = set()
    for sample in samples:
        parts = sample.split(",")
        for part in parts:
            try:
                n = int(part)
                if existing_numbers is not None and n in existing_numbers:
                    continue
                numbers.add(int(part))
            except:
                pass
    
    if existing_numbers is None:
        print(f"Unique numbers: {len(numbers)}")
    else:
        print(f"New unique numbers: {len(numbers)}")

    return numbers

def test_counting(n_training: int, n_validation: int, n_to_generate: int, max_digits: int, max_count: int, use_saved_model: bool, model_save_dir: str):
    print("===========================================")
    print(f"Test Counting (max digits: {max_digits}, max count: {max_count})")
    print("===========================================")
    print("")
    
    # validator
    validator = lambda x: counting.validate_one_sample(x)

    # generate some samples from the ground truth model
    reset_seeds()
    print("Ground truth (training):")
    training_samples = counting.generate_samples(n_samples=n_training, max_digits=max_digits, max_count=max_count)
    is_valid, _ = validate_samples(validator, training_samples)
    assert all(is_valid)
    training_numbers = unique_numbers_in_counting_samples(training_samples)
    print("")

    print("Ground truth (validation):")
    validation_samples = counting.generate_samples(n_samples=n_validation,  max_digits=max_digits, max_count=max_count)
    is_valid, _ = validate_samples(validator, validation_samples, training_samples)
    assert all(is_valid)
    unique_numbers_in_counting_samples(validation_samples, existing_numbers=training_numbers)
    print("")


    sample_max_len = max_digits * max_count + (max_count - 1) + 3 # max-1: comma, +3: padding, starting and ending tokens


    # train a tri-gram model as baseline
    reset_seeds()
    model = TriGramModel(sample_sep=".", dummy_count=0)
    model.train(training_samples)
    print("Counting-based tri-gram model:")
    print("Parameter count: ", model.trigram_count.numel()) # 1728
    generated_samples = model.generate(n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    unique_numbers_in_counting_samples(generated_samples, existing_numbers=training_numbers)
    print("")

    # use tokenizer for neural network based models
    reset_seeds()
    tokenizer = CharacterTokenizer()
    token_samples = [x for x in string.digits] + [","]
    tokenizer.train(token_samples)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # train a neural network (MLP) based language model
    # it has insufficient context length (look back) to, in theory, achieve a 100% accuracy
    # but should still be able to approximate the counting-based (N-1)-gram model with less parameters
    reset_seeds()
    look_back = 10
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    embed_size = 8
    hidden_layer_sizes = [32, 16]
    dataset = prepare_n_gram_dataset(training_samples, tokenizer, look_back=look_back)
    model = MLPLanguageModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_sizes=hidden_layer_sizes, look_back=look_back)
    print(f"MLP with {look_back} characters context:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 3470
    path = os.path.join(model_save_dir, f"mlp_lm_look_back_{look_back}.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_torch_n_gram_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
        torch.save(model.state_dict(), path)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    unique_numbers_in_counting_samples(generated_samples, existing_numbers=training_numbers)
    print("")


    # Prepare dataset auto-regressive models
    reset_seeds()
    training_dataset = prepare_auto_regressive_dataset(training_samples, tokenizer)
    validation_dataset = prepare_auto_regressive_dataset(validation_samples, tokenizer)


    # train a RNN model with simple recurrent unit
    reset_seeds()
    embed_size = 8
    hidden_state_size = 64
    epoch = 100
    learning_rate = 0.001
    batch_size = 32
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, use_gru=False)
    print("RNN model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 5758
    path = os.path.join(model_save_dir, "rnn.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_auto_regressive_model(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        torch.save(model.state_dict(), path)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


    # train a RNN model with GRU (Gated Recurrent Unit) unit
    # reduced embedding/hidden size to match the RNN model parameter count
    reset_seeds()
    embed_size = 8
    hidden_state_size = 32
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, use_gru=True)
    print("GRU model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 4542
    path = os.path.join(model_save_dir, "gru.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_auto_regressive_model(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        torch.save(model.state_dict(), path)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


    # Train a transformer model
    reset_seeds()
    embed_size = 8
    max_context_size = sample_max_len
    n_layer = 2
    n_heads = 2
    head_size = None
    ff_hidden_size = 64
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    model = TransformerLM(vocab_size=tokenizer.vocab_size, embed_size=embed_size, max_context_size=max_context_size, n_layer=n_layer, n_heads=n_heads, head_size=head_size, ff_hidden_size=ff_hidden_size)
    print("Transformer model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 3150
    path = os.path.join(model_save_dir, "transformer.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_transformer(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
        torch.save(model.state_dict(), path)
    generated_samples = sample_from_transformer(model, tokenizer, max_new_tokens=sample_max_len, n_samples=n_to_generate)
    validate_samples(validator, generated_samples, training_samples)
    unique_numbers_in_counting_samples(generated_samples, existing_numbers=training_numbers)
    print("")


def test_addition(n_training: int, n_validation: int, n_to_generate: int, max_digits: int, use_saved_model: bool, model_save_dir: str):
    print("===========================================")
    print(f"Test Addition (max digits: {max_digits})")
    print("===========================================")
    print("")
    
    # validator
    validator = lambda x: arithmetics.validate_one_sample(x)

    # generate some samples from the ground truth model
    reset_seeds()
    print("Ground truth (training):")
    training_samples = arithmetics.generate_samples(n_samples=n_training, max_digits=max_digits)
    is_valid, _ = validate_samples(validator, training_samples)
    assert all(is_valid)
    print("")

    print("Ground truth (validation):")
    validation_samples = arithmetics.generate_samples(n_samples=n_validation, max_digits=max_digits)
    is_valid, _ = validate_samples(validator, validation_samples, training_samples)
    assert all(is_valid)
    print("")


    sample_max_len = 3*max_digits + 2 + 3 # +2: + and =, +3: padding, starting and ending tokens


    # train a tri-gram model as baseline
    reset_seeds()
    model = TriGramModel(sample_sep=".", dummy_count=0)
    model.train(training_samples)
    print("Counting-based tri-gram model:")
    print("Parameter count: ", model.trigram_count.numel()) # 2197
    generated_samples = model.generate(n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")
    

    # use tokenizer for neural network based models
    reset_seeds()
    tokenizer = CharacterTokenizer()
    token_samples = [x for x in string.digits] + ["+", "="]
    tokenizer.train(token_samples)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # train a neural network (MLP) based language model
    # it has insufficient context length (look back) to, in theory, achieve a 100% accuracy
    # but should still be able to approximate the counting-based (N-1)-gram model with less parameters
    reset_seeds()
    look_back = 10
    epoch = 100
    learning_rate = 0.001
    batch_size = 32
    embed_size = 32
    hidden_layer_sizes = [48,48,32]
    dataset = prepare_n_gram_dataset(training_samples, tokenizer, look_back=look_back)
    model = MLPLanguageModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_sizes=hidden_layer_sizes, look_back=look_back)
    print(f"MLP with {look_back} characters context:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 20303
    path = os.path.join(model_save_dir, f"mlp_lm_look_back_{look_back}.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_torch_n_gram_model(model, dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)
        torch.save(model.state_dict(), path)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


    # Prepare dataset auto-regressive models
    reset_seeds()
    training_dataset = prepare_auto_regressive_dataset(training_samples, tokenizer)
    validation_dataset = prepare_auto_regressive_dataset(validation_samples, tokenizer)


    # train a RNN model with simple recurrent unit
    reset_seeds()
    embed_size = 32
    hidden_state_size = 128
    epoch = 100
    learning_rate = 0.001
    batch_size = 32
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, use_gru=False)
    print("RNN model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 23151
    path = os.path.join(model_save_dir, "rnn.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_auto_regressive_model(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        torch.save(model.state_dict(), path)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


    # train a RNN model with GRU (Gated Recurrent Unit) unit
    # reduced embedding/hidden size to match the RNN model parameter count
    reset_seeds()
    embed_size = 32
    hidden_state_size = 64
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, use_gru=True)
    print("GRU model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 20143
    path = os.path.join(model_save_dir, "gru.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_auto_regressive_model(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        torch.save(model.state_dict(), path)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


    # Train a transformer model
    reset_seeds()
    embed_size = 32
    max_context_size = 15
    n_layer = 2
    n_heads = 4
    head_size = None
    ff_hidden_size = 64
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    model = TransformerLM(vocab_size=tokenizer.vocab_size, embed_size=embed_size, max_context_size=max_context_size, n_layer=n_layer, n_heads=n_heads, head_size=head_size, ff_hidden_size=ff_hidden_size)
    print("Transformer model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 18415
    path = os.path.join(model_save_dir, "transformer.pt")
    if use_saved_model and os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model = train_transformer(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
        torch.save(model.state_dict(), path)

    generated_samples = sample_from_transformer(model, tokenizer, max_new_tokens=sample_max_len, n_samples=n_to_generate)
    validate_samples(validator, generated_samples, training_samples)
    print("")

    # Train a bigger transformer model on GPU
    if torch.cuda.is_available():
        reset_seeds()
        embed_size = 64
        max_context_size = 16
        n_layer = 10
        n_heads = 8
        head_size = None
        ff_hidden_size = 256
        epoch = 100
        learning_rate = 0.001
        batch_size = 256
        model = TransformerLM(vocab_size=tokenizer.vocab_size, embed_size=embed_size, max_context_size=max_context_size, n_layer=n_layer, n_heads=n_heads, head_size=head_size, ff_hidden_size=ff_hidden_size)
        model = model.to("cuda")
        print("(Bigger) Transformer model:")
        print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 501007
        path = os.path.join(model_save_dir, "transformer_bigger.pt")
        if use_saved_model and os.path.exists(path):
            model.load_state_dict(torch.load(path, weights_only=True))
        else:
            model = train_transformer(model, training_dataset, epochs=epoch, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
            torch.save(model.state_dict(), path)

        generated_samples = sample_from_transformer(model, tokenizer, max_new_tokens=sample_max_len, n_samples=n_to_generate)
        validate_samples(validator, generated_samples, training_samples)
        print("")

    

def main():
    # general parameters
    use_saved_model = True
    
    print("")
    stickiness = 1
    strict_stickiness = True
    model_dir = os.path.join(SCRIPT_DIR, "model_checkpoints", f"stickiness_{stickiness}_{'strict' if strict_stickiness else 'loose'}")
    os.makedirs(model_dir, exist_ok=True)
    reset_seeds()
    test_sticky_rule(n_training=1000, n_validation=200, n_to_generate=1000, min_len=3, max_len=10, stickiness=stickiness, strict=strict_stickiness, use_saved_model=use_saved_model, model_save_dir=model_dir)
    print("")

    print("")
    max_digits = 3
    max_count = 3
    model_dir = os.path.join(SCRIPT_DIR, "model_checkpoints", f"counting_{max_digits}_digits_{max_count}_count")
    os.makedirs(model_dir, exist_ok=True)
    reset_seeds()
    test_counting(n_training=500, n_validation=100, n_to_generate=1000, max_digits=max_digits, max_count=max_count, use_saved_model=use_saved_model, model_save_dir=model_dir)
    print("")

    print("")
    max_digits = 2
    model_dir = os.path.join(SCRIPT_DIR, "model_checkpoints", f"addition_{max_digits}")
    os.makedirs(model_dir, exist_ok=True)
    reset_seeds()
    test_addition(n_training=3500, n_validation=1000, n_to_generate=2000, max_digits=max_digits, use_saved_model=use_saved_model, model_save_dir=model_dir)
    print("")


if __name__ == "__main__":
    main()