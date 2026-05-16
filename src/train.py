import numpy as np
from loss import *
from layers import *
from model import *

def L_layer_model(X, Y, layers_dims, learning_rate = 0.001, num_iterations = 300, mini_batch_size=64,print_cost=False):
   

    np.random.seed(1)
    costs = []                   
    parameters = initialize_params(layers_dims)
    v,s = initialize_adam(parameters)
    t = 0 #adam variable which counts number of steps taken
    
    for i in range(0, num_iterations):
        epoch_cost = 0
        mini_batches = random_mini_batches(X,Y,mini_batch_size)
        
        
        #minibatch loop
        for mini_batch in mini_batches:
            mini_batch_X,mini_batch_Y = mini_batch
            #forward_prop
            AL, caches = l_model_forward(mini_batch_X, parameters,keep_prob=0.8,training=True)
            cost = compute_cost(AL, mini_batch_Y)
            #back_prop
            grads = l_model_backward(AL,mini_batch_Y,caches,keep_prob=0.8)
            t+=1
            parameters,v,s,v_corr,s_corr = update_params_adam(parameters,grads,v,s,t,learning_rate)
            epoch_cost += cost
            
        #Average epoch cost
        epoch_cost = epoch_cost/len(mini_batches)
        costs.append(epoch_cost)
        
        if print_cost and i%10==0 or i==(num_iterations-1) :
            print(f"Cost after epoch {i}: {epoch_cost}")
    return parameters, costs