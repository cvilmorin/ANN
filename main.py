from perceptron import Perceptron
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))


choice=input('Choose your ANN')
if choice==1:
    pass
else:
    pass



p=Perceptron(X,y,epoch=10000,lr=1e-1)
p.fit(X,y)
print(X[1],y[1])
print(p.predict(X[1]))
