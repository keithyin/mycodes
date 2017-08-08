class array(object):
    """Simple Array object that support autodiff."""

    def __init__(self, value, name=None):
        self.value = value
        if name:
            self.grad = lambda g: {name: g}

    def __add__(self, other):
        assert isinstance(other, int)
        ret = array(self.value + other)
        ret.grad = lambda g: self.grad(g)
        return ret

    def __mul__(self, other):
        assert isinstance(other, array)
        ret = array(self.value * other.value)

        def grad(g):
            x = self.grad(g * other.value)
            x.update(other.grad(g * self.value))
            return x

        ret.grad = grad
        return ret

a = array(1, 'a')
b = array(2, 'b')
c = b * a
d = c + 1
print(d.value)
print(d.grad(1))