# Various implementation of a simple multi-layer perceptron neural network

import random
import math
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..autograd import Value

#############################################################
## Neural Network using our autograd implementation
#############################################################


# A single neuron in the network
class Neuron:
    def __init__(self, input_size: int, w_init_sigma: float, non_linear: bool):
        # KEY IDEA: custom initialization variance, allow for different initialization strategies 
        self.w = [Value(random.normalvariate(0.0, w_init_sigma)) for _ in range(input_size)]
        self.b = Value(random.normalvariate(0.0, w_init_sigma))
        self.non_linear = non_linear
    
    def __call__(self, x):
        r = sum((w*x for w, x in zip(self.w, x)), self.b)
        if self.non_linear:
            r = r.relu()
        return r
            
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.non_linear else 'Linear'}Neuron({len(self.w)})"

# Mimic the PyTorch API
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

# A single layer in the neural network
class Layer(Module):
    def __init__(self, input_size: int, output_size: int, non_linear: bool):
        neurons: list[Neuron] = []
        for _ in range(output_size):
            # KEY IDEA: Kaiming initialization for ReLU neurons, fan-in mode
            # See: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
            w_init_sigma = math.sqrt(2 / input_size)
            neurons.append(Neuron(input_size, w_init_sigma=w_init_sigma, non_linear=non_linear))
        self.neurons = neurons

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

# A multi-layer perceptron
class AutoGradMLP(Module):
    def __init__(self, input_size, hidden_sizes: list[int], output_size: int):
        layers: list[Layer] = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_sizes[i-1]
            layers.append(Layer(layer_input_size, size, non_linear=True))
        if len(hidden_sizes) == 0:
            layer_input_size = input_size
        else:
            layer_input_size = hidden_sizes[-1]
        layers.append(Layer(layer_input_size, output_size, non_linear=False))
        self.layers = layers

    def __call__(self, x):
        # forward pass
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

def train_auto_grad_model_sgd(X: np.ndarray, Y: np.ndarray, model: AutoGradMLP, epochs=1000, learning_rate=0.01, batch_size=256):
    total_samples = X.shape[0]
    if batch_size is not None:
        # KEY IDEA: learning rate must be adjusted by batch size
        # Ref: https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
        learning_rate = learning_rate * (batch_size / total_samples)
    
    params = model.parameters()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        # mini-batch
        permutation = np.random.permutation(total_samples)
        for i in range(0, total_samples, batch_size):
            indices = permutation[i:i+batch_size]
            x_batch, y_batch = X[indices], Y[indices]

            # forward pass
            o = [model(x) for x in x_batch]

            # MSE loss
            assert y_batch.shape[1] == 1
            loss = [(ye-y[0])*(ye-y[0]) for ye, y in zip(o, y_batch)]
            batch_total_loss: Value = sum(loss) # type: ignore

            # # NOTE: L2 regularization
            # l2 = 0.01 * sum([x*x for x in netQ.parameters()])
            # L = batch_total_loss * (1/batch_size) + l2
            L = batch_total_loss * (1/batch_size) 

            # backward pass
            model.zero_grad()
            L.backward()
            
            # Stochastic Gradient Descent
            for p in params:
                p.data -= learning_rate * p.grad

            total_loss += batch_total_loss.data
            
        if (epoch+1) % 10 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Loss: {total_loss/total_samples:.4f}')
            
    return model


def train_auto_grad_model_adam(X: np.ndarray, Y: np.ndarray, model: AutoGradMLP, epochs=1000, learning_rate=0.01, batch_size=256, beta1 = 0.9, beta2 = 0.999, eps = 1e-08):
    total_samples = X.shape[0]

    params = model.parameters()

    # initialize Adam optimizer states
    m = np.zeros(len(params))
    v = np.zeros(len(params))

    for epoch in range(epochs):       
        total_loss = 0.0

        # mini-batch
        permutation = np.random.permutation(total_samples)
        for i in range(0, total_samples, batch_size):
            indices = permutation[i:i+batch_size]
            x_batch, y_batch = X[indices], Y[indices]

            # forward pass
            o = [model(x) for x in x_batch]

            # MSE loss
            assert y_batch.shape[1] == 1
            loss = [(ye-y[0])*(ye-y[0]) for ye, y in zip(o, y_batch)]
            batch_total_loss: Value = sum(loss) # type: ignore

            # # NOTE: L2 regularization
            # l2 = 0.01 * sum([x*x for x in netQ.parameters()])
            # L = batch_total_loss * (1/batch_size) + l2
            L = batch_total_loss * (1/batch_size) 

            # backward pass
            model.zero_grad()
            L.backward()
            
            # Adam optimizer
            for j, p in enumerate(params):
                grad = p.grad
                m[j] = beta1 * m[j] + (1 - beta1) * grad
                v[j] = beta2 * v[j] + (1 - beta2) * grad ** 2
                m_hat = m[j] / (1 - beta1 ** (j + 1))
                v_hat = v[j] / (1 - beta2 ** (j + 1))
                p.data -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            total_loss += batch_total_loss.data

        if (epoch+1) % 1 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Loss: {total_loss/total_samples:.4f}')
    
    return model

#############################################################
## Neural Network using PyTorch Tensor
#############################################################

class TorchLayer:
    def __init__(self, input_size: int, output_size: int, non_linear: bool):
        # Kaiming initialization
        gain = 2.0 if non_linear else 1.0
        w_init_std = gain / math.sqrt(input_size)
        W = torch.randn((input_size, output_size)) * w_init_std
        b = torch.randn(output_size) * 0.01 # break symmetry
        self.W = W
        self.b = b
        self.non_linear = non_linear
    
    def __call__(self, x):
        r = x @ self.W + self.b
        if self.non_linear:
            r = r.relu()
        return r

    def parameters(self):
        return [self.W, self.b]
    

class TorchTensorMLP:
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        layers: list[TorchLayer] = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_sizes[i-1]
            layers.append(TorchLayer(layer_input_size, size, non_linear=True))
        if len(hidden_sizes) == 0:
            layer_input_size = input_size
        else:
            layer_input_size = hidden_sizes[-1]
        layers.append(TorchLayer(layer_input_size, output_size, non_linear=False))
        self.layers = layers

    def __call__(self, x):
        # forward pass
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def train_torch_tensor_model_sgd(dataloader: DataLoader, model: TorchTensorMLP, epochs=100, learning_rate=0.01):
    total_samples = len(dataloader.dataset) # type: ignore
    if dataloader.batch_size is not None:
        # KEY IDEA: learning rate must be adjusted by batch size
        # Ref: https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
        learning_rate = learning_rate * (dataloader.batch_size / total_samples)
    
    # loss function
    criterion = torch.nn.MSELoss()

    params = model.parameters()

    for p in params:
        p.requires_grad = True
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in dataloader:
            batch_size = X_batch.shape[0]

            # forward pass
            outputs = model(X_batch)
            batch_avg_loss = criterion(outputs, y_batch)
            
            # zero out gradients            
            for p in params:
                p.grad = None

            # Backward pass
            batch_avg_loss.backward()
            
            # Stochastic Gradient Descent
            with torch.no_grad():
                for p in params:
                    assert p.grad is not None
                    p -= learning_rate * p.grad

            total_loss += batch_avg_loss.item() * batch_size
            
        if (epoch+1) % 10 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Loss: {total_loss/total_samples:.4f}')
            
    return model

    
# KEY IDEA: use Adam optimizer instead of simple SGD
def train_torch_tensor_model_adam(dataloader: DataLoader, model: TorchTensorMLP, epochs=100, learning_rate=0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-08):
    total_samples = len(dataloader.dataset) # type: ignore

    # loss function
    criterion = torch.nn.MSELoss()

    params = model.parameters()

    for p in params:
        p.requires_grad = True

    # initialize Adam optimizer states
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in dataloader:
            batch_size = X_batch.shape[0]

            # forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # zero out gradients            
            for p in params:
                p.grad = None

            # Backward pass
            loss.backward()
            
            # Adam optimizer
            with torch.no_grad():
                for i, p in enumerate(params):
                    assert p.grad is not None
                    grad = p.grad
                    m[i] = beta1 * m[i] + (1 - beta1) * grad
                    v[i] = beta2 * v[i] + (1 - beta2) * grad ** 2
                    m_hat = m[i] / (1 - beta1 ** (epoch+1))
                    v_hat = v[i] / (1 - beta2 ** (epoch+1))
                    p -= learning_rate * m_hat / (torch.sqrt(v_hat) + eps)
            
            total_loss += loss.item() * batch_size

        if (epoch+1) % 10 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Loss: {total_loss/total_samples:.4f}')
            
    return model


#############################################################
## Neural Network using PyTorch NN Module
#############################################################


# Define the Neural Network
class TorchNnMLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super(TorchNnMLP, self).__init__()

        layers = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_sizes[i-1]
            layers.extend([torch.nn.Linear(layer_input_size, size), torch.nn.ReLU()])
        if len(hidden_sizes) == 0:
            layer_input_size = input_size
        else:
            layer_input_size = hidden_sizes[-1]
        layers.append(torch.nn.Linear(layer_input_size, output_size))

        self.layers_stack = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers_stack(x)

def train_torch_nn_model(dataloader: DataLoader, model: TorchNnMLP, epochs=1000, learning_rate=0.01):
    total_samples = len(dataloader.dataset) # type: ignore
    # Initialize model, loss function, and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in dataloader:
            batch_size = X_batch.shape[0]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size

            # print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Batch Loss: {loss.item():.4f} X[0] = {X_batch[0]}, Y[0] = {y_batch[0]}')

        if (epoch+1) % 10 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Loss: {total_loss/total_samples:.4f}')
            
    return model
