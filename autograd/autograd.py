
# Automatic Differentiation Library
# Based on Andraj Karpathy's micrograd

from typing import Callable
import math

class Value:
    '''
    A numeric atom node in the computation graph
    '''

    def __init__(self, data: float | int, label: str="", _children: set["Value"] | None = None, _backward: Callable[["Value"], None] | None = None):
        assert isinstance(data, float) or isinstance(data, int)

        # Node label
        self.label = label
        # Node value
        self.data = data

        # Gradient of this node with respect to the final output
        self.grad = 0.0
        
        # If this node is the output of an operator, store the children (input) nodes
        if _children is None:
            self.children = []
            self._backward = lambda x: None
        else:
            # KEY IDEA: use lambda function + closure to store the backward function, see below
            # When invoked, the function should flow the gradient from this node (self.grad) to its children/inputs ([x.grad for x in self.children]) using chain rule
            assert _backward is not None
            self.children = _children
            self._backward = _backward

    def __repr__(self):
        '''Pretty print the node'''
        label = f"{self.label}: " if self.label != "" else ""
        return f"Value({label}{self.data}, grad={self.grad})"

    def __add__(self, other0: "Value | float | int") -> "Value":
        '''Allow adding two nodes with "+" operator'''

        if isinstance(other0, (float, int)):
            other = Value(other0)
        else:
            other = other0

        r = self.data + other.data

        # Create back-propagation function for flowing output node gradient to input nodes
        # For addition, the gradient should flow 1:1 to the inputs as the derivative of x + y wrt x and y are both 1
        def backward(output: Value):
            # KEY IDEA: use the closure to store the reference to the input nodes
            # KEY IDEA: use += to accumulate the gradient if a child node is used in multiple operations
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad

        # create summation result node, self and other are the children/inputs
        children = set([self, other])
        o = Value(r, _children=children, _backward=backward)
        
        return o
    
    def __radd__(self, left) -> "Value":
        return self + left

    def __sub__(self, other) -> "Value":
        # use plus to implement subtraction
        return self + -1.0 * other
    
    def __rsub__(self, left) -> "Value":
        # use plus to implement subtraction
        return left + (-1.0 * self)
    
    def __mul__(self, other0: "Value | float | int") -> "Value":
        
        if isinstance(other0, (float, int)):
            other = Value(other0)
        else:
            other = other0

        r = self.data * other.data

        # chain rule for multiplication
        def backward(output: "Value"):
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        children = set([self, other])
        o = Value(r, _children=children, _backward=backward)

        return o

    def __rmul__(self, left) -> "Value":
        return self * left

    def tanh(self) -> "Value":
        ''' Tanh activation function '''

        num = math.exp(2 * self.data) - 1
        den = math.exp(2 * self.data) + 1
        r = num / den

        # derivative of tanh is (1 - tanh^2)
        def backward(o: "Value"):
            self.grad += (1 - r ** 2) * o.grad
        children = set([self])
        o = Value(r, _children=set([self]), _backward=backward)
        return o

    def relu(self) -> "Value":
        ''' ReLU activation function '''

        if self.data > 0:
            r = self.data
        else:
            r = 0

        # derivative of relu is 1 if x > 0, 0 otherwise
        def backward(o: "Value"):
            if r > 0:
                self.grad += o.grad
        children: set[Value] = set([self])
        o = Value(r, _children=children, _backward=backward)

        return o

    def backward(self):
        # the derivative of the final output node wrt itself is 1
        self.grad = 1.0

        # perform a topological sort of the calculation graph to find the right order of back-propagation 
        sorted_nodes: list[Value] = []
        visited: set[Value] = set()
        def topological_sort(node: "Value"):
            if node in visited:
                return
            visited.add(node)
            # KEY IDEA: recursively visit all children first before adding the parent node to the sorted list
            for child in node.children:
                topological_sort(child)
            sorted_nodes.append(node)
        topological_sort(self)
        
        # reverse: start from the output to inputs
        for node in reversed(sorted_nodes):
            node._backward(node)
