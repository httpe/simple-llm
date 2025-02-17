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

1. Counting based Bi-gram and Tri-gram models
2. N-gram model by neural network
3. Recurrent neural network (RNN) model
4. Transformer model

Testing:

1. Propose a rule-based sample generation algorithm, such that:
   1. It can generate an arbitrary large amount of samples
   2. It is easy to verify if a particular sample is within the distribution / satisfy all the constraints
   3. We can easily adjust the context length required to learn the full distribution, e.g. even in theory, we need least a N-gram model is capture all the constraints set forth by the rule, where N is adjustable
2. Generate samples from the distribution and train various models listed above, plus feed into mainstream LLMs
3. Generate new samples from the trained models (or prompted LLMs) and test how many of them can pass the validation (i.e. are in distribution)
4. Understand how these models actually learned the distribution by looking into the trained parameters

Status: **Ongoing**
