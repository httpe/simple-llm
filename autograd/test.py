import torch
from .autograd import Value
import numpy as np

def test_arithmetic():
    ''' Test a simple arithmetic, compare with PyTorch '''

    def forward(x):
        z = 2 * x + 2 + x # use x twice
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        return x, y

    x0 = torch.Tensor([-4.0]).double()
    x0.requires_grad = True
    x0, y0 = forward(x0)

    x1 = Value(-4.0)
    x1, y1 = forward(x1)

    # forward pass went well
    assert y1.data == y0.data.item()
    # backward pass went well
    assert x0.grad is not None
    assert x1.grad == x0.grad.item()

    print("Arithmetic test passed")

def test_linear_regression():
    ''' Test linear regression '''

    # prepare data
    # y = a * x + b
    x = np.linspace(-10, 10, 100)
    a0 = 3.0
    b0 = 1.0
    y = a0 * x + b0

    # initial values
    a = Value(1.0)
    b = Value(0.0)

    for i in range(1000):
        y_est = [a * x_i + b for x_i in x] # our autograd doesn't support vectorized operations
        loss = [(ye - y)*(ye - y) for ye, y in zip(y_est, y)]
        total_loss = Value(0.0)
        for l in loss:
            total_loss += l
        avg_loss = total_loss * (1/len(loss)) # our autograd doesn't support division
        a.grad = 0.0
        b.grad = 0.0
        avg_loss.backward()

        if i % 100 == 0:
            print(f"{i=}, {a=}, {b=}, {avg_loss=}")
        
        lr = 0.01
        a.data -= lr * a.grad
        b.data -= lr * b.grad

    print(f"Final a: {a.data}, b: {b.data}")
    assert abs(a.data - a0) < 1e-5
    assert abs(b.data - b0) < 1e-5

    print("Linear regression test passed")


if __name__ == "__main__":
    test_arithmetic()
    test_linear_regression()
    print("All tests passed")
