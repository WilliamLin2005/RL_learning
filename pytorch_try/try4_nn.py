import torch
import torch.nn as nn
import numpy as np

l=nn.Linear(2, 5)
v=torch.tensor([[1.0,2.0]])
print(l(v))

s=nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 3),
    nn.ReLU(),
    nn.Linear(3,10),
    nn.Dropout(0.5),
    nn.Softmax(dim=1)
)
print(s)
v=torch.tensor([[1.0,2.0], [2.0,3.0]])
print(s(v))
s_sum=s(v).sum()
print(s_sum)
s_sum.backward()
print(s[0].weight.grad)