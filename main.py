import numpy as np

class Perceptron():
    def __init__(self,feat_nb,obs_nb):
        self.weight=np.random.rand(1,feat_nb)
        self.biais=np.random.rand(1)

    def forward(self,x):
        return self.weight*x +self.biais

    def softmax(self,Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    #def train(self):


perceptron=Perceptron(1,4)
x=np.random.rand(4,3)
x=x.T
s=perceptron.softmax(perceptron.forward(x))
print(s)


def get_predictions(A2):
    return np.argmax(A2, 0)

print(get_predictions(s))