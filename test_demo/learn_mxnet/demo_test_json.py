import mxnet as mx
a = mx.sym.Variable('a', shape=[2,])
b = mx.sym.Variable('b', shape=[3,])

c = a+b
d = mx.sym.Group([c, b, a])
print(d.tojson())