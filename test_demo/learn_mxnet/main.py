import mxnet as mx
from mxnet import sym
from mxnet import symbol

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, weight=net,name='fc1', num_hidden=128)
net2 = symbol.FullyConnected(data=net,weight=net, name='fc1', num_hidden=128)

print(sym)
print(symbol)

mx.viz.plot_network(symbol=net).render()
