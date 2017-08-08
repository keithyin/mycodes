import mxnet as mx


data2 = mx.sym.Variable('hhh')

data = mx.sym.Variable('data')
fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
out1 = mx.sym.SoftmaxOutput(data=net, name='softmax')
out2 = mx.sym.LinearRegressionOutput(data=net, name='regression')


print(out2.list_arguments())
group = mx.sym.Group([out1, out2])

print(group)

executor = group.simple_bind(mx.cpu(), data=(10, 20))

print(group.list_arguments())
