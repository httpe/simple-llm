# Simple LLM

Simple LLM is a project implementing a GPT-like Large Language Model (LLM) from scratch.

This implementation follows Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) video series as a primary reference.

## Getting Started

To try out the milestone test scripts below, first create a Python virtual environment, and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -v --upgrade pip
pip install -v -r requirements.txt
```

To test our own automated differentiation library:

```bash
python -m simplellm.autograd.test
```

To test our own neural network implementations:

```bash
python -m simplellm.nn.test
```

To test our language model implementations:

```bash
python -m simplellm.lm.test
```

## Milestones

### Automatic Differentiation

Goal: Implement an automatic differentiation library with back-propagation support. The library includes fundamental operations (+, *, ReLU) as building blocks for neural network construction.

Validation: Implementation tested through a linear regression (`y = a*x + b`) on a noiseless dataset using MSE loss. Results validated against ground truth and PyTorch implementation.

Status: **Complete** (see `simplellm/autograd/`)

### Neural Network (Multi-layer Perceptron)

Goal: Implement neural network (MLP) and training algorithms in steps, gradually replace our implementation to that of PyTorch's:

1. Our automatic differentiation library from last stage + our SGD/Adam optimizers
2. PyTorch tensors + our SGD/Adam optimizers written in Pytorch
3. PyTorch NN modules + PyTorch's Adam optimizer

Validation: Implementations tested through quadratic regression (`y = w1*x1**2 + w2*x2**2 + b`) on a noiseless dataset using MSE loss and ReLU activation. Performance compared across all three implementations, with additional comparison between SGD and Adam optimizers.

Status: **Complete** (see `simplellm/nn/`)

### Language Modeling

Goal: Implement and compare various character-level language models using PyTorch.

Model implementations:

1. Statistical n-gram models
   1. Bi-gram and tri-gram models with counting-based approach (implemented)
   2. Bi-gram model optimized through cross-entropy loss and gradient descent (implemented)
2. Neural architectures
   1. N-gram model with character-level embeddings and dense MLP (implemented)
   2. Recurrent Neural Network (Original RNN Cell, GRU cell) (implemented)
   3. Transformer architecture (implemented)
      1. Causal self attention
      2. Multi-head attention
      3. Layer normalization
      4. Position encoding

The training data is synthesized with the following properties:

1. Rule-based generation for interpretability
2. Deterministic validation of distribution membership (Y/N) for any given sample
3. Configurable context length requirements with parameter N
   - N determines minimum n-gram model size required for 100% accuracy

We have implemented 3 synthetic datasets/distributions. In the order of difficulty:

1. `sticky.py`: Generate a mix of lower case (a), upper case (A) and digit (0) characters, such that each class (a/A/0) will appear at least N times consecutively before switching to another class, e.g., "acXYZ13a" (N=2)
2. `counting.py`: Generate variable length counting sequence (consecutive integers separated by comma) for numbers up to N digits, e.g., "32,33,34" (N=2)
3. `arithmetic.py`: Generate addition formulas for numbers up to N digits, e.g. "22+13=35" (N=2)

Evaluation:

1. Train all implemented models on the synthetic dataset
2. Generate new samples from trained models and LLMs prompted with the training samples
3. Validate generated samples against the rules and calculate in-distribution percentage
4. Analyze learned parameters to understand how different architectures capture the underlying distribution

Status: **In Progress** (see `simplellm/lm/`)
