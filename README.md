# Simple LLM

Simple LLM is a project to build a GPT-like LLM from scratch.

The key reference material is Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) video series.

## Milestones

### Automatic Differentiation

Goal: Build an automatic differentiation library with back propagation. It should support operations: +, *, relu, such that we can build neural net on top of it.

Testing: Fit a linear regression `y = a*x + b` on a noiseless dataset with MSE loss. Compare the results of our library with ground truth and that fit by Pytorch.

Status: **Done** (see `autograd/`)

### Neural Network (Multi-layer Perceptron)

Goal: Implement neural network (MLP) and training algorithm for:

1. Our automatic differentiation library (implement SGD/Adam)
2. PyTorch tensor (implement SGD/Adam)
3. PyTorch NN module (Pytorch Adam)

Testing: Fit a quadratic regression `y = w1*x1**2 + w2*x2**2 + b` on a noiseless dataset with MSE loss. Use ReLU as activation function. Compare the loss from the training with the above 3 implementations and SGD vs Adam.

Status: **Done** (see `nn/`)

### Language Modeling

Goal: Implement character level language model using PyTorch with the following methods:

1. Bi-gram model
2. N-gram model by neural network
3. Transformer model
4. Minimal Neural Network model that can be trained to fit the ground truth distribution

Testing: General text samples from some simple rules that can be easily verified. Use those samples as training/testing dataset and test if these models are able to learn the patterns and generate legal out-sample text.

Status: **Ongoing**

