import math


def _assimilate(x):
    return x if isinstance(x, Value) else Value(x)


# scalar
class Value:
    def __init__(self, data, operands=()):
        self.data = data
        self.grad = 0
        self._operands = set(operands)
        self._backward = lambda: None

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for operand in v._operands:
                    build_topo(operand)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = _assimilate(other)
        out = Value(data=self.data + other.data, operands=(self, other))

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = _assimilate(other)
        out = Value(data=self.data - other.data, operands=(self, other))

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * -1.0

        out._backward = _backward

        return out

    def __rsub__(self, other):
        return _assimilate(other) - self

    def __mul__(self, other):
        other = _assimilate(other)
        out = Value(data=self.data * other.data, operands=(self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):  # x * y^-1
        other = _assimilate(other)
        out = Value(data=self.data / other.data, operands=(self, other))

        def _backward():
            self.grad += out.grad * (1 / other.data)  # 1 / y
            other.grad += out.grad * (-self.data / (other.data**2))  # - (x / y^2)

        out._backward = _backward

        return out

    def __rtruediv__(self, other):
        return _assimilate(other) / self

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "The B230 can only exponentiate constants (int/float)"
        )
        out = Value(data=self.data**other, operands=(self,))

        def _backward():
            self.grad += out.grad * (other * (self.data ** (other - 1)))

        out._backward = _backward

        return out

    # skipping rpow for now: probably will never use it

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, operands=(self,))

        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward
        return out
