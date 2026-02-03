import numpy as np

class Relu():
    def forward(self, Inputs):
        return np.maximum(0,Inputs)