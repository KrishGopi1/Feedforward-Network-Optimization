import numpy as np
from activation import *
import copy

#minibatch gradient descent implementation
def random_mini_batches(X,Y,mini_batch_size):
    m = X.shape[1]
    mini_batches = []
    num_complete_minibatches = int(np.floor(m/mini_batch_size))
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    inc = mini_batch_size

    #iterating and storing mini batches as tuples
    for i in range (0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,i*inc:(i+1)*inc]
        mini_batch_Y = shuffled_Y[:,i*inc:(i+1)*inc]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    
    #for the last mini batch if it is not fully occupied
    if m%mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,inc*num_complete_minibatches:m]
        mini_batch_Y = shuffled_Y[:,inc*num_complete_minibatches:m]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

#initializing adam variables
def initialize_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}
    for l in range(1,L+1):
        v["dW" + str(l)] = np.zeros((parameters['W'+str(l)].shape))
        v["db" + str(l)] = np.zeros((parameters['b'+str(l)].shape))
        s["dW" + str(l)] = np.zeros((parameters['W'+str(l)].shape))
        s["db" + str(l)] = np.zeros((parameters['b'+str(l)].shape))
    return v,s


#forward propagation
def l_model_forward(X, params,keep_prob=1.0,training=True):
    caches = []
    L = len(params)//2
    A = X
    for l in range (1,L):
        A_bef = A
        A,cache = l_activation_forward(A_bef,params["W"+str(l)],params["b"+str(l)],act="relu")

        #Dropout implementation
        if training and keep_prob < 1.0:
            D = np.random.rand(A.shape[0],A.shape[1])
            D = (D<keep_prob)
            A = A*D
            A = A/keep_prob
            cache =(cache,D)
        caches.append(cache)
    #Output layer    
    AL,cache = l_activation_forward(A,params["W"+str(L)],params["b"+str(L)],act="softmax")
    caches.append(cache)
    return AL,caches

#backpropagation function
def l_model_backward(AL,Y,caches,keep_prob=1.0):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    #Final Layer (Softmax)
    current_cache = caches[L-1]
    linear_cache, activation_cache = current_cache
    dZL = AL - Y
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads['db'+str(L)] = l_backward(dZL,linear_cache)

    #Hidden Layers(ReLU)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        actual_cache,D = current_cache
        grads["dA"+str(l+1)] = grads["dA"+str(l+1)]*D
        grads["dA"+str(l+1)] = grads["dA"+str(l+1)]/keep_prob
        grads["dA"+str(l)], grads["dW"+str(l+1)], grads['db'+str(l+1)] = l_activation_backward(grads["dA"+str(l+1)],actual_cache,act='relu')
    
    return grads 

#updating parameters without adam
def update_params(parameters,grads,learning_rate):
    parameters = copy.deepcopy(parameters)
    L = len(parameters)//2
    for l in range(1,L+1):
        parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - learning_rate*grads["db"+str(l)]
    return parameters

#updating parameters with adam
def update_params_adam(parameters,grads,v,s,t,learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-10):
    L = len(parameters)//2
    v_corr = {}
    s_corr = {}
    for l in range(1,L+1):
        v['dW'+str(l)] = beta1*v['dW'+str(l)] + (1-beta1)*grads['dW'+str(l)]
        v['db'+str(l)] = beta1*v['db'+str(l)] + (1-beta1)*grads['db'+str(l)]
        s['dW'+str(l)] = beta2*s['dW'+str(l)] + (1-beta2)*(grads['dW'+str(l)]**2)
        s['db'+str(l)] = beta2*s['db'+str(l)] + (1-beta2)*(grads['db'+str(l)]**2)
        
        #corrected (divide with 1-beta^t)
        v_corr["dW" + str(l)] = v['dW'+str(l)]/(1-(beta1**t))
        v_corr["db" + str(l)] = v['db'+str(l)]/(1-(beta1**t))
        s_corr["dW" + str(l)] = s['dW'+str(l)]/(1-(beta2**t))
        s_corr["db" + str(l)] = s['db'+str(l)]/(1-(beta2**t))

        #parameter updation
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*(v_corr["dW" + str(l)]/(np.sqrt(s_corr["dW" + str(l)])+epsilon))
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*(v_corr["db" + str(l)]/(np.sqrt(s_corr["db" + str(l)])+epsilon))
    return parameters, v, s, v_corr, s_corr

#predict function
def predict(X, parameters):

    AL, _ = l_model_forward(X, parameters,keep_prob=1.0,training=False)

    predictions = np.argmax(AL, axis=0)

    return predictions
#check accuracy
def accuracy(predictions, Y):

    Y_true = np.argmax(Y, axis=0)

    acc = np.mean(predictions == Y_true) * 100

    return acc