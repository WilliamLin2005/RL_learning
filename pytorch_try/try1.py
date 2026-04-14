import torch
import numpy as np
a=torch.FloatTensor([[1,2,3],[3,2,1]])
b=np.zeros(shape=(3,2))
b=torch.tensor(b)
n=np.zeros(shape=(3,2),dtype=np.float32)
n=torch.tensor(n)
s=a.sum()
print(a)
print(b)
print(n)
print(s)
print(s.item())