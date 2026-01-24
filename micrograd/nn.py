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



