import mxnet as mx

data = mx.symbol.Variable('data')
net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
net1.list_arguments()
data2 = mx.symbol.Variable('data_2')
net2 = mx.symbol.FullyConnected(data=data2, name='fc2', num_hidden=10)
composed = net2(data_2=net1, name='composed')
print(composed.list_arguments())