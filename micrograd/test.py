import torch 
from micrograd.engine import Value

#My implementation 
n1 = Value(2.0)
w1 = Value(0.0)
n2 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8)
x = n1*w1 + n2*w2 +b
o = x.tanh()
o.backaward()
print(o.data)
print("-------")
print("Gradiants")
print(f"n1={n1.grad}")
print(f"w1={w1.grad}")
print(f"n2={n2.grad}")
print(f"w2={w2.grad}")
print("-------")


#Pytorch
n1 = torch.Tensor([2.0]).double();   n1.requires_grad = True
w1 = torch.Tensor([0.0]).double();   w1.requires_grad = True
n2 = torch.Tensor([-3.0]).double();   n2.requires_grad = True
w2 = torch.Tensor([1.0]).double();   w2.requires_grad = True
b = torch.Tensor([6.8]).double();   b.requires_grad = True
x = n1*w1 + n2*w2 + b
o = torch.tanh(x)
o.backward()
print(o.data.item())
print("-------")
print("Gradiants")
print(f"n1={n1.grad.item()}")
print(f"w1={w1.grad.item()}")
print(f"n2={n2.grad.item()}")
print(f"w2={w2.grad.item()}")
print("-------")
