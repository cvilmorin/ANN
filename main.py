import numpy as np

class Perceptron():
    def __init__(self,feat_nb,obs_nb):
        self.weight=np.random.rand((1,feat_nb))
        self.biais=np.random.rand((1,1))
    def forward(self,x):
        return self.weight*x +self.biais

perceptron=Perceptron(1,4)
print(perceptron.biais)
