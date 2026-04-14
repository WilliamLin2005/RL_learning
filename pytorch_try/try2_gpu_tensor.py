import torch
import numpy as np
a=torch.FloatTensor([2,3])
print(a)
print(a.device)
gpu_a=a.cuda()
print(gpu_a)
gpu_a+=1
print(gpu_a)
print(gpu_a.device)
c=torch.tensor([4,5],device='cuda')
print(c)
print(c.device)