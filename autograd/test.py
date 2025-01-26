import torch
from .autograd import Value

def test_arithmetic():
    ''' Test a simple arithmetic, compare with PyTorch '''

    def foo(x):
        z = 2 * x + 2 + x # use x twice
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        return x, y

    x0 = torch.Tensor([-4.0]).double()
    x0.requires_grad = True
    x0, y0 = foo(x0)

    x1 = Value(-4.0)
    x1, y1 = foo(x1)

    # forward pass went well
    assert y1.data == y0.data.item()
    # backward pass went well
    assert x0.grad is not None
    assert x1.grad == x0.grad.item()

if __name__ == "__main__":
    test_arithmetic()
    print("All tests pass")