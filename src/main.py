from utils import load_mnist
from loss import *
from train import *
import matplotlib.pyplot as plt
import numpy as np
from model import *


X_train,X_test,Y_train,Y_test = load_mnist()
Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)
layers_dims = [784, 128, 64, 10]

parameters, costs = L_layer_model(X_train,Y_train,layers_dims,num_iterations=100,print_cost=True)

train_preds = predict(X_train, parameters)
test_preds = predict(X_test, parameters)
train_acc = accuracy(train_preds, Y_train)
test_acc = accuracy(test_preds, Y_test)
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")