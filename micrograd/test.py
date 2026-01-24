from micrograd.nn import MLP
import torch
n = MLP(3,[4,4,1])

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]


for k in range(20):
    ys_predict = [n(x) for x in xs]
    loss =0
    for yr,yf in zip(ys,ys_predict):
        loss = (yf-yr)**2+loss

    for p in n.parameters():
        p.grad=0
    loss.backaward()

    for p in n.parameters():
        p.data += -0.05*p.grad
    print(k,loss)


ys_predict = [n(x) for x in xs]
print(ys_predict)



