# Testing script for the neural network implementations
# Test by running: python -m simplellm.nn.test

import random
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .nn import TorchTensorMLP, TorchNnMLP, AutoGradMLP, train_auto_grad_model_adam, train_auto_grad_model_sgd, train_torch_nn_model, train_torch_tensor_model_adam, train_torch_tensor_model_sgd

# Custom Dataset class to handle numpy arrays
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Quadratic function to fit: y = 3 * x1^2 + 2 * x2^2 + 100
def generate_data(n: int):
    # sample points for x1 and x2
    axis_n = round(math.sqrt(n))
    x1 = np.linspace(-10, 10, axis_n)
    x2 = np.linspace(-10, 10, axis_n)
    x = np.array(list(zip(*[x.flatten() for x in np.meshgrid(x1, x2)])))

    # actual parameters
    w = np.array([3, 2]).reshape((2,1))
    b = 100.0

    y =  (x ** 2) @ w + b

    return x, y

def get_torch_model_loss(model, dataset):
    criterion = torch.nn.MSELoss()
    train_outputs = model(dataset.X)
    loss = criterion(train_outputs, dataset.y).item()
    return loss

def get_auto_grad_model_loss(model, X, Y):
    y_est = [model(x).data for x in X]
    assert Y.shape[1] == 1
    loss = [(ye-y[0])**2 for ye, y in zip(y_est, Y)]
    total_loss = sum(loss) / len(loss)
    return total_loss

def reset_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def main(n, train_test_split, hidden_sizes, batch_size):
    reset_seeds()

    # Generate data
    X, Y = generate_data(n)
    n = X.shape[0]
    idx = np.random.permutation(n)
    X, Y = X[idx], Y[idx]

    # Train-test split
    split_idx = int(n * train_test_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print("Sample Shapes:")
    print(f"X: {X.shape}, Y: {Y.shape}")
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    
    # Create PyTorch dataset and dataloader
    dataset = NumpyDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = NumpyDataset(X_test, Y_test)

    # Reference model: NN of one hidden layer, with 8 ReLu neurons
    reset_seeds()
    print("Running: TorchNnMLP (Adam)")
    model_nn = TorchNnMLP(input_size=2, hidden_sizes=hidden_sizes, output_size=1)
    model_nn = train_torch_nn_model(dataloader, model=model_nn, epochs=1000, learning_rate=1.0)
    model_nn_loss_train = get_torch_model_loss(model_nn, dataset)
    model_nn_loss_test = get_torch_model_loss(model_nn, test_dataset)
    print(f'TorchNnMLP (Adam) Loss: Train = {model_nn_loss_train:.4f}, Test = {model_nn_loss_test:.4f}')
    # TorchNnMLP (Adam) Loss: Train = 53.5432, Test = 56.4507

    # Model implemented using PyTorch tensors:
    reset_seeds()
    print("Running: TorchTensorMLP (Adam)")
    model_tensor_adam = TorchTensorMLP(input_size=2, hidden_sizes=hidden_sizes, output_size=1)
    model_tensor_adam = train_torch_tensor_model_adam(dataloader, model=model_tensor_adam, epochs=100, learning_rate=1.0)
    model_tensor_loss_adam_train = get_torch_model_loss(model_tensor_adam, dataset)
    model_tensor_loss_adam_test =  get_torch_model_loss(model_tensor_adam, test_dataset)
    print(f'TorchTensorMLP (Adam) Loss: Train = {model_tensor_loss_adam_train:.4f}, Test = {model_tensor_loss_adam_test:.4f}')
    # TorchTensorMLP (Adam) Loss: Train = 61.5234, Test = 63.2537

    reset_seeds()
    print("Running: TorchTensorMLP (SGD)")
    model_tensor_sgd = TorchTensorMLP(input_size=2, hidden_sizes=hidden_sizes, output_size=1)
    model_tensor_sgd = train_torch_tensor_model_sgd(dataloader, model=model_tensor_sgd, epochs=1000, learning_rate=0.01)
    model_tensor_loss_sgd_train = get_torch_model_loss(model_tensor_sgd, dataset)
    model_tensor_loss_sgd_test =  get_torch_model_loss(model_tensor_sgd, test_dataset)
    print(f'TorchTensorMLP (SGD) Loss: Train = {model_tensor_loss_sgd_train:.4f}, Test = {model_tensor_loss_sgd_test:.4f}')
    # TorchTensorMLP (SGD) Loss: Train = 109.6577, Test = 114.5128

    # Model implemented using our autograd:
    reset_seeds()
    print("Running: AutoGradMLP (Adam)")
    model_autograd_adam = AutoGradMLP(input_size=2, hidden_sizes=hidden_sizes, output_size=1)
    model_autograd_adam = train_auto_grad_model_adam(X_train, Y_train, model=model_autograd_adam, epochs=100, learning_rate=1.0, batch_size=batch_size)
    model_autograd_loss_adam_train = get_auto_grad_model_loss(model_autograd_adam, X_train, Y_train)
    model_autograd_loss_adam_test = get_auto_grad_model_loss(model_autograd_adam, X_test, Y_test)
    print(f'AutoGradMLP (Adam) Loss: Train = {model_autograd_loss_adam_train:.4f}, Test = {model_autograd_loss_adam_test:.4f}')
    # AutoGradMLP (Adam) Loss: Train = 76.0929, Test = 75.1556

    reset_seeds()
    print("Running: AutoGradMLP (SGD)")
    model_autograd_sgd = AutoGradMLP(input_size=2, hidden_sizes=hidden_sizes, output_size=1)
    model_autograd_sgd = train_auto_grad_model_sgd(X_train, Y_train, model=model_autograd_sgd, epochs=1000, learning_rate=0.01, batch_size=batch_size)
    model_autograd_loss_sgd_train = get_auto_grad_model_loss(model_autograd_sgd, X_train, Y_train)
    model_autograd_loss_sgd_test = get_auto_grad_model_loss(model_autograd_sgd, X_test, Y_test)
    print(f'AutoGradMLP (SGD) Loss: Train = {model_autograd_loss_sgd_train:.4f}, Test = {model_autograd_loss_sgd_test:.4f}')
    # AutoGradMLP (SGD) Loss: Train = 77.7754, Test = 76.4327


if __name__ == "__main__":
    main(n=10000, batch_size=256, train_test_split=0.8, hidden_sizes=[8])

