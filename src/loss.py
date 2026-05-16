import numpy as np

#one hot implementation
def one_hot(Y,num_classes=10):
    Y = np.array(Y).reshape(-1)
    Y_one_hot = np.eye(num_classes)[Y]
    return Y_one_hot.T

#computing cost, for softmax regression
def compute_cost(AL,Y):
    m = Y.shape[1]
    epsilon = 1e-11
    AL = np.clip(AL, epsilon, 1 - epsilon)
    cost = (-1/m)*np.sum(np.multiply(Y,np.log(AL+epsilon)))
    cost = np.squeeze(cost)
    return cost
