from micrograd.engine import Value
import numpy as np
class Neuron:
    def __init__(self,ninputs):
        self.w =[Value(np.random.uniform(-1,1)) for _ in range(ninputs)]
        self.b = Value(np.random.uniform(-1,1))
    
    def __call__(self,x):
        act = sum ((wi*xi for xi, wi in zip(x,self.w)),self.b) 
        out = act.tanh()
        return out

class Layer:
    def __init__(self,ninpunts, noutputs):
        self.neurons = [Neuron(ninpunts) for _ in range(noutputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
        
        
class MLP:
    def __init__(self,ninputs,noutputs):
        sz = [ninputs] + noutputs
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(sz)-1)]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
        
