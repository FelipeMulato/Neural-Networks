import numpy as np

class Relu():
    def forward(self, Inputs):
        self.ouput = np.maximum(0,Inputs)
    
class Sigmoid():
    def forward(self, Inputs):
        self.ouput = np.power((1-np.exp(-1*Inputs)),-1)

class Tanh():
    def forward(self, Inputs):
        self.ouput =  (1-np.exp(-2*Inputs))/(1+np.exp(-2*Inputs))