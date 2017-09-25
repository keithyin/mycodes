from torchvision import models
from torch import nn
import torch
from torch.autograd import Variable
import time
from torch import optim

vgg16 = models.vgg16(pretrained=False)
vgg16.cuda()
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16.parameters(), lr=0.0001)

begin = time.time()

for i in range(1000):
    print(i)
    bs = 140
    inputs = Variable(torch.FloatTensor(bs, 3, 224, 224)).cuda()
    labels = Variable(torch.LongTensor([1]*bs)).cuda()
    logits = vgg16(inputs)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("time ", time.time() - begin)

# bs=10, iter=1000, time 175.43, Menory=2846M
# upper bound = 140