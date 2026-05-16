import numpy as np
def l_forward(A,W,b):
    Z = np.dot(W,A) + b
    cache = (A,W,b)
    return Z,cache 
def l_activation_forward(A_before, W, b, act):
    if act=="softmax":
        Z, linear_cache = l_forward(A_before,W,b)
        activation_cache = Z
        Z = Z - np.max(Z,axis = 0,keepdims=True)
        Z_exp = np.exp(Z)
        Z_sum = np.sum(Z_exp,axis=0,keepdims=True)
        A = Z_exp/Z_sum
    elif act=="relu":
        Z, linear_cache = l_forward(A_before,W,b)
        activation_cache = Z
        A = np.maximum(0,Z)
    cache = (linear_cache, activation_cache)
    return A,cache
def l_backward(dZ,cache):
    A_bef,W,b = cache
    m = A_bef.shape[1]
    dW = (1/m)*np.dot(dZ,A_bef.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev,dW,db
def l_activation_backward(dA,cache,act):
    linear_cache,activation_cache = cache
    if act=="relu":
        dZ = dA*(activation_cache>0)
        dA_prev, dW, db = l_backward(dZ,linear_cache)
        return dA_prev,dW,db
