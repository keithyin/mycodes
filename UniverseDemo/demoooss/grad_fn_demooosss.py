from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
a = Variable(torch.randn(2,3))
b = Variable(torch.randn(3,2))
c = torch.mm(a, b)
print(c.grad_fn)