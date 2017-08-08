import mxnet as mx

data1 = mx.sym.Variable('data1', shape=(1,))
data2 = mx.sym.Variable('data2', shape=(1,))

c = data1 + 1
d = data1 + 1

# group two symbol together
group = mx.sym.Group([c, d])

executor = group.simple_bind(ctx=mx.cpu())

executor.forward(is_train=True)

for val in executor.outputs:
    print(val.asnumpy())
executor.backward([mx.nd.array([1.]), mx.nd.array([1.])])

for grad in executor.grad_arrays:
    print(grad.asnumpy())


