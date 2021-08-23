import numpy as np
import matplotlib.pyplot as plt
import json

x_std_test = np.load("resources/x_std_test.npy")
x_train = np.load("resources/x_train.npy")
y_std_test = np.load("resources/y_std_test.npy")
y_train = np.load("resources/y_train.npy")

np.random.seed(1)

def softmax(Z):
    cache = Z
    e = np.exp(Z-np.max(Z)) / (np.exp(Z-np.max(Z)).sum(axis=0, keepdims=True))
    
    return e, cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 

    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def initialize_parameters_deep(layer_dims):
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):
   
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

    #ACA CAMBIE A SOFTMAX
def linear_activation_forward(A_prev, W, b, activation):
   
    #Softmax
    if activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
    caches.append(cache)
    
    assert(AL.shape == (10,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    #cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost=-1*(np.mean(Y*np.log((AL + 1*(np.exp(-8))))))

    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, AL, Y, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "softmax":
        dZ = AL-Y
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,AL,Y, activation = "softmax")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,AL,Y, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X,params):
    p = np.zeros((1,X.shape[1]))

    # Forward propagation
    probas, caches = L_model_forward(X, params)

    for i in range(0,probas.shape[1]):
        ij=0
        w=0
        z=0
        for j in range (0,probas.shape[0]):
            if(probas[j,i]>ij):
                ij=probas[j,i]
                w = i
                z = j
        p[0,w] = z
    return p


# Explore your dataset 
m_train = x_train.shape[0]
#num_px = train_x_orig.shape[1]
m_test = x_std_test.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
#print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(x_std_test.shape))
print ("train_y shape: " + str(y_train.shape))
print ("test_x_orig shape: " + str(x_std_test.shape))
print ("test_y shape: " + str(y_std_test.shape))

# Reshape the training and test examples 
train_x_flatten = x_std_test.reshape(x_std_test.shape[0], -1)   #QUITA LA TRANSPUESTA
test_x_flatten = x_std_test.reshape(x_std_test.shape[0], -1)

layers_dims = [x_train.shape[0], 60, 30, 15, y_train.shape[0]] #  NUMEROS DE CAPAS 
#X;XTRAIN / Y; YTRAIN
#layers_dims = [x.shape[0],]
def L_layer_model(x_train, y_train, layers_dims, learning_rate = 0.0810, num_iterations = 1500, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(x_train, parameters)

        cost = compute_cost(AL, y_train)

        grads = L_model_backward(AL, y_train, caches)

        parameters = update_parameters(parameters, grads, learning_rate)
                
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

parameters = L_layer_model(x_train, y_train, layers_dims, num_iterations = 1500, print_cost = True)


predictions = predict(x_train, parameters)
accuracy = (predictions == np.argmax(y_train, axis=0)).mean() 
print(accuracy)

#print(type(parameters))
#print(parameter['W1'])
#print (parameters['b1'].reshape(-1))

#referencia https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

params = {'dnn_layers':[
            {'n':60, 'activation': 'relu', 'w':parameters['W1'], 'b':parameters['b1'].reshape(-1)},
            {'n':30, 'activation':'relu','w': parameters['W2'],'b':parameters['b2'].reshape(-1)},
            {'n':15, 'activation': 'relu', 'w':parameters['W3'], 'b':parameters['b3'].reshape(-1)},
            {'n':10, 'activation': 'softmax', 'w': parameters['W4'], 'b': parameters['b4'].reshape(-1)}
            ]}

with open('params.json','w') as f:
    json.dump(params, f, indent=4, cls=NumpyEncoder)