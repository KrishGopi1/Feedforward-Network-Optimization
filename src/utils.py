from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
def load_mnist():
    mnist = fetch_openml('mnist_784',version=1)
    X = mnist.data.to_numpy()
    Y = mnist.target.astype(int).to_numpy()
    Y = Y.reshape(70000,1)
    X = X/255.0
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=7)
    return X_train.T,X_test.T,Y_train,Y_test