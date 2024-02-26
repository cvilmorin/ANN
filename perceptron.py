import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np 
from tqdm import tqdm

class Perceptron:
    
    def __init__(self,X,y,epoch,lr):
        self.weight=np.random.randn(X.shape[1],1)
        self.biais=np.random.randn(1)
        self.lr=lr
        self.epoch=epoch
    
    def sigmoid(self,Z):
        return 1/(np.exp(-Z)+1)
    
    def model(self,X):
        Z=X.dot(self.weight)+self.biais
        return self.sigmoid(Z)
    
    def log_loss(self,A,y):
        return -1/len(y)*np.sum(y*np.log(A)+(1-y)*np.log(1-A))
    
    def gradients(self,A,X,y):
        dW=1/len(y)*np.dot(X.T,A-y)
        db=1/len(y)*np.sum(A-y)
        return (dW,db)
    
    def update(self,dW,db):
        weight=self.weight - self.lr*dW
        biais=self.biais - self.lr*db
        return weight,biais

    def predict(self,X):
        A = self.model(X)
        return A >= 0.5


    def fit(self,X,y):
        loss=[]
        for i in tqdm(range(self.epoch)):
            A=self.model(X)
            loss.append(self.log_loss(A,y))
            dW,db=self.gradients(A,X,y)
            self.weight,self.biais=self.update(dW,db)

        y_pred=self.predict(X)
        print(accuracy_score(y,y_pred))
        plt.plot(loss)
        plt.show()
        

