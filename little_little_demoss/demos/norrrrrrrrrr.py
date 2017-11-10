import torch

a = torch.rand([4, 6])
norm_a = torch.norm(a, dim=1, p=2)
norm_a = torch.unsqueeze(norm_a, dim=1)
print(a[:, :2])
