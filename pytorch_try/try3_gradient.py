import torch
import numpy as np
v1=torch.tensor([1.0,2.0],requires_grad=True)
v2=torch.tensor([3.0,4.0])
v3=v1+v2
v_res=v3.sum()
print(v3)
v_res.backward()
print(v1.grad)
v4=((v1+v2)*2).sum()
v4.backward()
print(v1.grad) #v1.grad是累积的，之前的梯度加上现在的梯度
