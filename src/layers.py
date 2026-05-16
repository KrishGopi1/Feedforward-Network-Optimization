import numpy as np
# Initializing parameters (optimize initialization o weights later)

def initialize_params(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range (1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
    return parameters
