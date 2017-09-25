import mxnet as mx
from mxnet.gluon import model_zoo
from mxnet.gluon import loss
from mxnet import autograd
from mxnet.gluon import trainer
import time
import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

vgg16 = model_zoo.vision.vgg16(ctx=mx.gpu())
vgg16.initialize(ctx=mx.gpu())
criterion = loss.SoftmaxCrossEntropyLoss()

update = trainer.Trainer(vgg16.collect_params(), optimizer='sgd')
begin = time.time()
bs = 60

for i in range(1000):
    print(i)
    inputs = mx.nd.normal(shape=(bs, 3, 224, 224), ctx=mx.gpu())
    labels = mx.nd.array([0] * bs, ctx=mx.gpu())
    with autograd.record():
        logits = vgg16(inputs)
        loss = criterion(logits, labels)

    loss.backward()
    update.step(batch_size=10)

print("time ", time.time() - begin)

# bs=10, iter=1000, time 10.77, Menory=3441M
# upper bound=60
