import numpy as np

class Layer():
    def __init__(self,n_inputs,n_layer):
        self.weights = 0.1*np.random.randn(n_inputs,n_layer)
        self.bias  = np.zeros((1,n_inputs))
    def forward(self,Input):
        self.output = np.dot(Input,self.weights) + self.bias