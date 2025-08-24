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

def reset_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_and_save_model[T: torch.nn.Module](model: T, trainer: Callable[[T, int], T], total_epoch: int, save_every_n_epoch: int, model_dir: str, model_name: str, use_saved_model: bool = True):
    epoch_trained = 0
    
    # read the max epoch trained from disk
    if use_saved_model:
        for file in os.listdir(model_dir):
            if file.startswith(f"{model_name}_epoch_") and file.endswith(".pt"):
                epoch = int(file[len(f"{model_name}_epoch_"):-len(".pt")])
                if epoch <= total_epoch:
                    epoch_trained = max(epoch, epoch_trained)
        path = os.path.join(model_dir, f"{model_name}_epoch_{epoch_trained}.pt")
        if epoch_trained > 0:
            print(f"Loading model from {path}")
            try:
                model.load_state_dict(torch.load(path, weights_only=True))
            except Exception as e:
                epoch_trained = 0
                print(f"Failed to load model from {path}: {e}")
    
    # finish training the model with the remaining epochs
    while total_epoch - epoch_trained > 0:
        epoch_to_train = min(save_every_n_epoch, total_epoch - epoch_trained)
        print(f"Training model from {epoch_trained} epochs to {epoch_trained + epoch_to_train} epochs")
        # reproducibility for continued training
        reset_seeds(epoch_trained)
        model = trainer(model, epoch_to_train)
        epoch_trained += epoch_to_train
        path = os.path.join(model_dir, f"{model_name}_epoch_{epoch_trained}.pt")
        torch.save(model.state_dict(), path)
 
    return model

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

    return list(zip(new_samples, is_valid)), valid_samples

def test_sticky_rule(n_training: int, n_validation: int, n_to_generate: int, min_len: int, max_len: int, stickiness: int, strict: bool, use_saved_model: bool, use_manual_init: bool, model_save_dir: str = SCRIPT_DIR):
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
    llm_output_type = f"stickiness_{stickiness}{'strict' if strict else 'loose'}"
    if llm_output_type in llm_samples:
        for name, samples in llm_samples[llm_output_type].items():
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
    trainer = lambda x, y: train_torch_n_gram_model(x, dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="tensor_bigram", use_saved_model=use_saved_model)
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
    trainer = lambda x, y: train_torch_n_gram_model(x, dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name=f"mlp_lm_look_back_{look_back}", use_saved_model=use_saved_model)
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
    trainer = lambda x, y: train_torch_n_gram_model(x, dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name=f"mlp_lm_look_back_{look_back}", use_saved_model=use_saved_model)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


    # Prepare dataset auto-regressive models
    reset_seeds()
    training_dataset = prepare_auto_regressive_dataset(training_samples, tokenizer)
    validation_dataset = prepare_auto_regressive_dataset(validation_samples, tokenizer)

    # train a specialized NN model
    reset_seeds()
    print("Handcrafted NN model for Sticky Distribution:")
    learning_rate = 0.01
    batch_size = 32
    epoch = 100
    max_context_size = 12
    embed_size = 6
    if use_manual_init:
        model = sticky.StickyNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, max_context_size=max_context_size, use_manual_init=(stickiness, strict))
    else:
        model = sticky.StickyNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, max_context_size=max_context_size, use_manual_init=None) # 998
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if not use_manual_init:
        trainer = lambda x, y: train_transformer(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
        model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="sticky_nn", use_saved_model=use_saved_model)
    generated_samples = sticky.sample_from_sticky_nn_model(model, tokenizer, max_new_tokens=max_len, n_samples=n_to_generate)
    validate_samples(validator, generated_samples, training_samples)
    print("")

    # train a specialized RNN model with simple recurrent unit
    # the initial parameters are set to be able to reproduce the distribution exactly (i.e. 100% accuracy)
    reset_seeds()
    print("Handcrafted RNN model for Sticky Distribution:")
    learning_rate = 0.01
    batch_size = 32
    epoch = 100 # do not train, just use the manual initialization
    if use_manual_init:
        model = sticky.StickyRNNModel(use_manual_init=(stickiness, strict))
    else:
        model = sticky.StickyRNNModel(use_manual_init=None) # 
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 861
    if not use_manual_init:
        trainer = lambda x, y: train_auto_regressive_model(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
        model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="sticky_rnn", use_saved_model=use_saved_model)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")

    # train a RNN model with simple recurrent unit
    reset_seeds()
    embed_size = 8
    hidden_state_size = 8
    epoch = 100
    learning_rate = 0.01
    batch_size = 32
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, cell_type="rnn")
    print("RNN model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 1249
    trainer = lambda x, y: train_auto_regressive_model(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="rnn", use_saved_model=use_saved_model)
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
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, cell_type="gru")
    print("GRU model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 1297
    trainer = lambda x, y: train_auto_regressive_model(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="gru", use_saved_model=use_saved_model)
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
    trainer = lambda x, y: train_transformer(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="transformer", use_saved_model=use_saved_model)
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
    print("")

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
    trainer = lambda x, y: train_torch_n_gram_model(x, dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name=f"mlp_lm_look_back_{look_back}", use_saved_model=use_saved_model)
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
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, cell_type="rnn")
    print("RNN model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 5758
    trainer = lambda x, y: train_auto_regressive_model(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="rnn", use_saved_model=use_saved_model)
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
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, cell_type="gru")
    print("GRU model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 4542
    trainer = lambda x, y: train_auto_regressive_model(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="gru", use_saved_model=use_saved_model)
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
    trainer = lambda x, y: train_transformer(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="transformer", use_saved_model=use_saved_model)
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
    print("")

    # train a neural network (MLP) based language model
    # it has insufficient context length (look back) to, in theory, achieve a 100% accuracy
    # but should still be able to approximate the counting-based (N-1)-gram model with less parameters
    reset_seeds()
    look_back = sample_max_len
    epoch = 100
    learning_rate = 0.001
    batch_size = 32
    embed_size = 32
    hidden_layer_sizes = [48,48,32]
    dataset = prepare_n_gram_dataset(training_samples, tokenizer, look_back=look_back)
    model = MLPLanguageModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_sizes=hidden_layer_sizes, look_back=look_back)
    print(f"MLP with {look_back} characters context:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 21839
    trainer = lambda x, y: train_torch_n_gram_model(x, dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name=f"mlp_lm_look_back_{look_back}", use_saved_model=use_saved_model)
    generated_samples = sample_torch_n_gram_model(model, tokenizer, look_back=look_back, n_samples=n_to_generate, max_length=sample_max_len)
    _, valid_samples = validate_samples(validator, generated_samples, training_samples)
    # print(valid_samples)
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
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, cell_type="rnn")
    print("RNN model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 23151
    trainer = lambda x, y: train_auto_regressive_model(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="rnn", use_saved_model=use_saved_model)
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
    model = RNNModel(vocab_size=tokenizer.vocab_size, embed_size=embed_size, hidden_state_size=hidden_state_size, cell_type="gru")
    print("GRU model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 20143
    trainer = lambda x, y: train_auto_regressive_model(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="gru", use_saved_model=use_saved_model)
    generated_samples = sample_auto_regressive_model(model, tokenizer, n_samples=n_to_generate, max_length=sample_max_len)
    validate_samples(validator, generated_samples, training_samples)
    print("")


    # Train a transformer model
    reset_seeds()
    embed_size = 32
    max_context_size = sample_max_len
    n_layer = 2
    n_heads = 4
    head_size = None
    ff_hidden_size = 64
    epoch = 100
    learning_rate = 0.001
    batch_size = 32
    model = TransformerLM(vocab_size=tokenizer.vocab_size, embed_size=embed_size, max_context_size=max_context_size, n_layer=n_layer, n_heads=n_heads, head_size=head_size, ff_hidden_size=ff_hidden_size)
    print("Transformer model:")
    print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 18287
    trainer = lambda x, y: train_transformer(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
    model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="transformer", use_saved_model=use_saved_model)
    generated_samples = sample_from_transformer(model, tokenizer, max_new_tokens=sample_max_len, n_samples=n_to_generate)
    _, valid_samples = validate_samples(validator, generated_samples, training_samples)
    # print(valid_samples)
    print("")

    # Train a bigger transformer model on GPU
    if torch.cuda.is_available():
        reset_seeds()
        embed_size = 64
        max_context_size = sample_max_len
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
        print("Parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad)) # 500687
        trainer = lambda x, y: train_transformer(x, training_dataset, epochs=y, learning_rate=learning_rate, batch_size=batch_size, ignore_token=tokenizer.pad_token, validation_dataset=validation_dataset)
        model = train_and_save_model(model, trainer, total_epoch=epoch, save_every_n_epoch=100, model_dir=model_save_dir, model_name="transformer_biger", use_saved_model=use_saved_model)
        generated_samples = sample_from_transformer(model, tokenizer, max_new_tokens=sample_max_len, n_samples=n_to_generate)
        _, valid_samples = validate_samples(validator, generated_samples, training_samples)
        # print(valid_samples)
        print("")
    else:
        print("CUDA not available, skipping bigger transformer model training.")
        print("")


def main():
    # general parameters
    use_saved_model = True
    use_manual_init = False
    
    print("")
    stickiness = 1
    strict_stickiness = True
    model_dir = os.path.join(SCRIPT_DIR, "model_checkpoints", f"stickiness_{stickiness}_{'strict' if strict_stickiness else 'loose'}")
    os.makedirs(model_dir, exist_ok=True)
    reset_seeds()
    test_sticky_rule(n_training=1000, n_validation=200, n_to_generate=1000, min_len=3, max_len=10, stickiness=stickiness, strict=strict_stickiness, use_saved_model=use_saved_model, use_manual_init=use_manual_init, model_save_dir=model_dir)
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
    
    print("All tests completed.")


if __name__ == "__main__":
    main()