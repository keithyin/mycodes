from torch import nn
from torch.autograd import Variable
import torch
import time

batch_size = 10
in_feature = 5
seq_len = 400
hidden_size = 30
num_layers = 2

begin = time.time()

# bf=False : 0.16877
# bf = True : 0.128-0.13

rnn = nn.LSTM(input_size=in_feature, hidden_size=hidden_size, num_layers=num_layers,
              batch_first=False)
inputs = Variable(torch.randn(batch_size * seq_len, in_feature), volatile=True)
inputs = inputs.view(batch_size, seq_len, in_feature)

inputs = torch.transpose(inputs, 1, 0)

h0 = Variable(torch.randn(num_layers, batch_size, hidden_size))
c0 = Variable(torch.randn(num_layers, batch_size, hidden_size))
output, hn = rnn(inputs, (h0, c0))

res = []
print(len(output))
for i in range(len(output) - 1):
    res.append(output[i + 1] - output[i])

res = torch.cat(res, dim=0)
res = torch.mean(res, dim=0)
print(res.size())
